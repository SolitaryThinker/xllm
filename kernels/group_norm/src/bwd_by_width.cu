// The backward pass of xllm's group RMS norm and group layer norm, in ONE pass over the data.
//
// dx needs two group statistics (sum(dyhat*xhat), and sum(dyhat) for the layer norm); dgamma and dbeta are sums over
// all rows. A warp that keeps one block of group columns and walks the rows gets all of them from the same load: the
// statistics by shuffles, dgamma/dbeta into registers, combined once at the end (fp32 atomic adds into a scratch
// buffer, then a one-block epilogue that rounds them to the output dtype). So x_or_y and dy are each read once and
// dx written once: 6 bytes an element, the bound for this operation.
//
// Mapping by channels per group, as the forward (2-byte dtypes, cpg % 8 == 0):
//   cpg <= 256          a group is one 16-byte load a lane over L = cpg/8 lanes; a warp keeps W = 32/L group columns
//                       and walks the rows, U row tiles in flight;
//   cpg <= 8192         a group is one CTA of cpg/8 threads (Q vectors a thread), one cross-warp reduction.
// Every other shape is declined and left to the other backward kernels.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "bwd_by_width.h"
#include "width_common.cuh"

#include "group_layer_norm.h"
#include "group_rms_norm.h"

namespace gn {
namespace {

// The conversion, vector and in-warp reduction helpers are shared with the forward: width_common.cuh.
using namespace gn::width;

// dgamma and dbeta are sums over every row, so every CTA of the wide path adds its own into one accumulator, once per
// CTA and not once per row: a cross-CTA combine is where a vector reduction pays. Used as a per-row or per-row-pair
// accumulation the same instruction is several times slower than a register accumulator, because millions of atomic
// transactions contend for the same few thousand L2 slots. Four channels a transaction (`red.global.add.v4.f32`)
// instead of one is a quarter of the read-modify-writes. The address is a multiple of 16 bytes: a thread owns 8
// channels at a multiple of 8, and the group's first channel is a multiple of the group width.
__device__ __forceinline__ void red_add4(float* p, float a, float b, float c, float d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
#else
  atomicAdd(p + 0, a);
  atomicAdd(p + 1, b);
  atomicAdd(p + 2, c);
  atomicAdd(p + 3, d);
#endif
}


__device__ __forceinline__ float block_sum(float v, float* sh, int nwarp, int lane, int wid) {
  v = warp_sum(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  float t = (wid == 0) ? ((lane < nwarp) ? sh[lane] : 0.f) : 0.f;
  if (wid == 0) {
    t = warp_sum(t);
    if (lane == 0) sh[0] = t;
  }
  __syncthreads();
  return sh[0];
}

constexpr int kWarpsPerBlock = 8;  // 256 threads a block
constexpr int kBlocksPerSm = 8;

// ---------------------------------------------------------------- (1) W group columns of a row per warp

template <typename T, int V, int L, int U, bool AFFINE, bool LAYER, bool MEMEFF>
__global__ __launch_bounds__(32 * kWarpsPerBlock) void gn_bwd_warp(
    const T* __restrict__ x_or_y, const T* __restrict__ dy, const float* __restrict__ mean,
    const float* __restrict__ rstd, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ dx,
    float* __restrict__ dgamma, float* __restrict__ dbeta, int64_t rows, int G, int W, int ngb, int cpg_rt) {
  const int cpg = (L > 0) ? V * L : cpg_rt;
  const float inv_cpg = 1.f / static_cast<float>(cpg);
  const int lane = threadIdx.x & 31;
  const int gwid = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  const int nwarps = gridDim.x * kWarpsPerBlock;
  const int k = gwid % ngb;  // the group block this warp keeps: gamma, dgamma and dbeta never change column
  const int j = gwid / ngb;
  const int nj = (nwarps - 1 - k) / ngb + 1;
  if (j >= nj) return;
  const int nlanes = cpg / V;
  const int segi = lane / nlanes;
  const int sub = lane - segi * nlanes;
  const int g = k * W + segi;
  const bool act = (segi < W) && (g < G);  // lanes past W*L (when L does not divide 32) belong to no group
  const int gs = act ? g : 0;
  const int64_t C = static_cast<int64_t>(G) * cpg;
  const int64_t chan = static_cast<int64_t>(gs) * cpg + sub * V;

  float gv[V], bv[V];
  if constexpr (AFFINE) {
    const Vec<T, V> t = load_cached<T, V>(gamma + chan);
#pragma unroll
    for (int i = 0; i < V; ++i) gv[i] = to_f(t.v[i]);
    if constexpr (LAYER) {
      const Vec<T, V> u = load_cached<T, V>(beta + chan);
#pragma unroll
      for (int i = 0; i < V; ++i) bv[i] = to_f(u.v[i]);
    }
  }
  float dga[V], dbe[V];
#pragma unroll
  for (int i = 0; i < V; ++i) {
    dga[i] = 0.f;
    dbe[i] = 0.f;
  }

  auto process = [&](int64_t r, Vec<T, V> xi, const Vec<T, V>& di) {
    const int64_t off = r * C + chan;
    // The statistics belong to the lane's group: a lane past this block's last group (segi >= W, or g >= G) is not in
    // any group, so it reads group gs = 0 rather than past the end of the [rows, G] statistics.
    const float rs = rstd[r * G + gs];
    float mu = 0.f;
    if constexpr (LAYER) mu = mean[r * G + gs];
    float xh[V], dh[V];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
      const float yv = to_f(xi.v[i]);
      const float dv = to_f(di.v[i]);
      float xhat;
      if constexpr (MEMEFF) {
        // y = xhat * gamma + beta: recover xhat from y by undoing the affine transform, without gamma when there is none.
        if constexpr (AFFINE) {
          xhat = (LAYER ? (yv - bv[i]) : yv) / gv[i];
        } else {
          xhat = yv;
        }
      } else {
        xhat = LAYER ? (yv - mu) * rs : yv * rs;
      }
      const float dhat = AFFINE ? dv * gv[i] : dv;
      xh[i] = xhat;
      dh[i] = dhat;
      s1 += dhat * xhat;
      if constexpr (LAYER) s2 += dhat;
      if constexpr (AFFINE) {
        dga[i] += dv * xhat;
        if constexpr (LAYER) dbe[i] += dv;
      }
    }
    float c1 = s1, c2 = s2;
    group_sum2<L>(c1, c2, lane, nlanes);
    c1 *= inv_cpg;
    if constexpr (LAYER) c2 *= inv_cpg;
    Vec<T, V>& out = xi;  // the raw x is dead by now: this row's output reuses its registers
#pragma unroll
    for (int i = 0; i < V; ++i) {
      const float val = dh[i] - xh[i] * c1 - (LAYER ? c2 : 0.f);
      from_f(out.v[i], val * rs);
    }
    if (act) store_stream<T, V>(dx + off, out);
  };

  int64_t r = j;
  for (; r + (U - 1) * nj < rows; r += U * nj) {
    Vec<T, V> xi[U], di[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t off = (r + u * nj) * C + chan;
      xi[u] = load_stream<T, V>(x_or_y + off);
      di[u] = load_stream<T, V>(dy + off);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) process(r + u * nj, xi[u], di[u]);
  }
  for (; r < rows; r += nj) {
    const int64_t off = r * C + chan;
    process(r, load_stream<T, V>(x_or_y + off), load_stream<T, V>(dy + off));
  }

  if constexpr (AFFINE) {
    if (act) {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        atomicAdd(dgamma + chan + i, dga[i]);
        if constexpr (LAYER) atomicAdd(dbeta + chan + i, dbe[i]);
      }
    }
  }
}

// One group of more than 256 channels: a CTA keeps one group column and walks the rows, so dgamma/dbeta stay in
// registers over every row it sees and are combined once, at the end (not one atomic per row and element).
template <typename T, int V, int Q, bool AFFINE, bool LAYER, bool MEMEFF>
__global__ void gn_bwd_wide(const T* __restrict__ x_or_y, const T* __restrict__ dy, const float* __restrict__ mean,
                            const float* __restrict__ rstd, const T* __restrict__ gamma, const T* __restrict__ beta,
                            T* __restrict__ dx, float* __restrict__ dgamma, float* __restrict__ dbeta, int64_t rows,
                            int G, int64_t cpg) {
  const int gi = blockIdx.x % G;  // this CTA's group column
  const int slice = blockIdx.x / G;
  const int nslice = gridDim.x / G;
  const int64_t gi_cpg = static_cast<int64_t>(gi) * cpg;
  const int64_t col = static_cast<int64_t>(threadIdx.x) * V;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * V;
  const float inv_cpg = 1.f / static_cast<float>(cpg);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarp = (blockDim.x + 31) >> 5;
  __shared__ float sh1[2][33], sh2[2][33], sh3[2][33], sh4[2][33];

  float gvv[Q][V], bvv[Q][V];
  if constexpr (AFFINE) {
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      const int64_t ch = gi_cpg + (off < cpg ? off : 0);
      const Vec<T, V> gv = load_cached<T, V>(gamma + ch);
#pragma unroll
      for (int i = 0; i < V; ++i) gvv[q][i] = to_f(gv.v[i]);
      if constexpr (LAYER) {
        const Vec<T, V> bv = load_cached<T, V>(beta + ch);
#pragma unroll
        for (int i = 0; i < V; ++i) bvv[q][i] = to_f(bv.v[i]);
      }
    }
  }
  float dga[Q][V], dbe[Q][V];
#pragma unroll
  for (int q = 0; q < Q; ++q)
#pragma unroll
    for (int i = 0; i < V; ++i) {
      dga[q][i] = 0.f;
      dbe[q][i] = 0.f;
    }

  const int64_t C = static_cast<int64_t>(G) * cpg;
  int64_t r = slice;
  if (r >= rows) return;
  // Two rows a pass: 4 x 16-byte loads a lane outstanding (64 B), and ONE barrier for both rows instead of one a row.
  // When a single row is left the second row repeats it with weight 0, so nothing is counted twice and the second
  // store is skipped.
  for (int64_t it = 0; r < rows; r += 2 * nslice, ++it) {
    const int buf = static_cast<int>(it & 1);  // two shared buffers: no barrier before the next pass overwrites one
    const int64_t r0 = r;
    const bool two = (r + nslice < rows);
    const int64_t r1 = two ? (r + nslice) : r;
    const float w1 = two ? 1.f : 0.f;

    Vec<T, V> xa[Q], da[Q], xb[Q], db[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      const int64_t o = (off < cpg) ? off : 0;
      const int64_t at0 = r0 * C + gi_cpg + o;
      const int64_t at1 = r1 * C + gi_cpg + o;
      xa[q] = load_stream<T, V>(x_or_y + at0);
      da[q] = load_stream<T, V>(dy + at0);
      xb[q] = load_stream<T, V>(x_or_y + at1);
      db[q] = load_stream<T, V>(dy + at1);
    }

    const float rsa = rstd[r0 * G + gi], rsb = rstd[r1 * G + gi];
    float mua = 0.f, mub = 0.f;
    if constexpr (LAYER) {
      mua = mean[r0 * G + gi];
      mub = mean[r1 * G + gi];
    }
    float s1a = 0.f, s2a = 0.f, s1b = 0.f, s2b = 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      if (off < cpg) {
#pragma unroll
        for (int i = 0; i < V; ++i) {
          const float ya = to_f(xa[q].v[i]), yb = to_f(xb[q].v[i]);
          const float dva = to_f(da[q].v[i]), dvb = to_f(db[q].v[i]);
          float ha, hb;
          if constexpr (MEMEFF) {
            if constexpr (AFFINE) {
              ha = (LAYER ? (ya - bvv[q][i]) : ya) / gvv[q][i];
              hb = (LAYER ? (yb - bvv[q][i]) : yb) / gvv[q][i];
            } else {
              ha = ya;
              hb = yb;
            }
          } else {
            ha = LAYER ? (ya - mua) * rsa : ya * rsa;
            hb = LAYER ? (yb - mub) * rsb : yb * rsb;
          }
          const float dha = AFFINE ? dva * gvv[q][i] : dva;
          const float dhb = (AFFINE ? dvb * gvv[q][i] : dvb) * w1;
          s1a += dha * ha;
          s1b += dhb * hb;
          if constexpr (LAYER) {
            s2a += dha;
            s2b += dhb;
          }
          if constexpr (AFFINE) {
            dga[q][i] += dva * ha + dvb * hb * w1;
            if constexpr (LAYER) dbe[q][i] += dva + dvb * w1;
          }
        }
      }
    }
    s1a = warp_sum(s1a);
    s2a = warp_sum(s2a);
    s1b = warp_sum(s1b);
    s2b = warp_sum(s2b);
    if (lane == 0) {
      sh1[buf][wid] = s1a;
      sh2[buf][wid] = s2a;
      sh3[buf][wid] = s1b;
      sh4[buf][wid] = s2b;
    }
    __syncthreads();
    // Every warp sums the block's partials itself: no second barrier, and no warp 0 pass to wait for.
    float c1a = 0.f, c2a = 0.f, c1b = 0.f, c2b = 0.f;
    for (int w = 0; w < nwarp; ++w) {
      c1a += sh1[buf][w];
      c2a += sh2[buf][w];
      c1b += sh3[buf][w];
      c2b += sh4[buf][w];
    }
    c1a *= inv_cpg;
    c2a *= inv_cpg;
    c1b *= inv_cpg;
    c2b *= inv_cpg;

    const int64_t base0 = r0 * C + gi_cpg, base1 = r1 * C + gi_cpg;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      if (off < cpg) {
        Vec<T, V> oa, ob;
#pragma unroll
        for (int i = 0; i < V; ++i) {
          const float ya = to_f(xa[q].v[i]), yb = to_f(xb[q].v[i]);
          const float dva = to_f(da[q].v[i]), dvb = to_f(db[q].v[i]);
          float ha, hb;
          if constexpr (MEMEFF) {
            if constexpr (AFFINE) {
              ha = (LAYER ? (ya - bvv[q][i]) : ya) / gvv[q][i];
              hb = (LAYER ? (yb - bvv[q][i]) : yb) / gvv[q][i];
            } else {
              ha = ya;
              hb = yb;
            }
          } else {
            ha = LAYER ? (ya - mua) * rsa : ya * rsa;
            hb = LAYER ? (yb - mub) * rsb : yb * rsb;
          }
          const float dha = AFFINE ? dva * gvv[q][i] : dva;
          const float dhb = AFFINE ? dvb * gvv[q][i] : dvb;
          from_f(oa.v[i], (dha - ha * c1a - (LAYER ? c2a : 0.f)) * rsa);
          from_f(ob.v[i], (dhb - hb * c1b - (LAYER ? c2b : 0.f)) * rsb);
        }
        store_stream<T, V>(dx + base0 + off, oa);
        if (two) store_stream<T, V>(dx + base1 + off, ob);
      }
    }
  }

  if constexpr (AFFINE) {
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      if (off < cpg) {
        float* dg = dgamma + gi_cpg + off;
        static_assert(V % 4 == 0, "red_add4 covers four channels a transaction, and the loop below steps by 4");
#pragma unroll
        for (int i = 0; i < V; i += 4) {
          red_add4(dg + i, dga[q][i], dga[q][i + 1], dga[q][i + 2], dga[q][i + 3]);
          if constexpr (LAYER)
            red_add4(dbeta + gi_cpg + off + i, dbe[q][i], dbe[q][i + 1], dbe[q][i + 2], dbe[q][i + 3]);
        }
      }
    }
  }
}

// ---------------------------------------------------------------- fp32 partials -> the output dtype

template <typename T>
__global__ void gn_bwd_affine_epilogue(const float* __restrict__ dg, const float* __restrict__ db, T* __restrict__ dgamma,
                                       T* __restrict__ dbeta, int64_t C, bool has_beta) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= C) return;
  from_f(dgamma[i], dg[i]);
  if (has_beta) from_f(dbeta[i], db[i]);
}

// ---------------------------------------------------------------- launch

int sm_count() {
  static int cached = [] {
    int n = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    return n > 0 ? n : 148;
  }();
  return cached;
}

template <typename T, int V, int L, int U, bool AFFINE, bool LAYER, bool MEMEFF>
void launch_warp(const T* x_or_y, const T* dy, const float* mean, const float* rstd, const T* gamma, const T* beta,
                 T* dx, float* dgamma, float* dbeta, int64_t rows, int64_t G, int64_t cpg, cudaStream_t stream) {
  const int l = (L > 0) ? L : static_cast<int>(cpg / V);
  const int W = 32 / l;
  const int ngb = static_cast<int>((G + W - 1) / W);
  const int64_t tasks = rows * ngb;
  int64_t blocks = (tasks + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t cap = static_cast<int64_t>(sm_count()) * kBlocksPerSm;
  const int64_t need = (ngb + kWarpsPerBlock - 1) / kWarpsPerBlock;  // every group block needs at least one warp
  blocks = std::max<int64_t>(need, std::min(blocks, cap));
  gn_bwd_warp<T, V, L, U, AFFINE, LAYER, MEMEFF><<<static_cast<unsigned>(blocks), 32 * kWarpsPerBlock, 0, stream>>>(
      x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma, dbeta, rows, static_cast<int>(G), W, ngb,
      static_cast<int>(cpg));
}

template <typename T, int V, int Q, bool AFFINE, bool LAYER, bool MEMEFF>
void launch_wide(const T* x_or_y, const T* dy, const float* mean, const float* rstd, const T* gamma, const T* beta,
                 T* dx, float* dgamma, float* dbeta, int64_t rows, int64_t G, int64_t cpg, cudaStream_t stream) {
  const int64_t vpt = cpg / V;                        // 16-byte vectors in a group
  const int64_t per = (vpt + Q - 1) / Q;              // vectors a thread holds
  const unsigned threads = static_cast<unsigned>(std::min<int64_t>(1024, ((per + 31) / 32) * 32));
  int per_sm = static_cast<int>(2048 / threads);
  if (per_sm < 1) per_sm = 1;
  int64_t nslice = static_cast<int64_t>(sm_count()) * per_sm / G;  // CTAs holding the same group column
  if (nslice < 1) nslice = 1;
  const int64_t blocks = nslice * G;
  gn_bwd_wide<T, V, Q, AFFINE, LAYER, MEMEFF><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
      x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma, dbeta, rows, static_cast<int>(G), cpg);
}

template <typename T, bool AFFINE, bool LAYER, bool MEMEFF>
void dispatch_shape(const T* x_or_y, const T* dy, const float* mean, const float* rstd, const T* gamma, const T* beta,
                    T* dx, float* dgamma, float* dbeta, int64_t rows, int64_t G, int64_t cpg, cudaStream_t stream) {
  if (cpg % 8 == 0) {
    if (cpg <= 256) {
      switch (cpg / 8) {
        case 3:
          return launch_warp<T, 8, 3, 2, AFFINE, LAYER, MEMEFF>(x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma, dbeta,
                                                                rows, G, cpg, stream);
        case 5:
          return launch_warp<T, 8, 5, 2, AFFINE, LAYER, MEMEFF>(x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma, dbeta,
                                                                rows, G, cpg, stream);
        case 8:
          return launch_warp<T, 8, 8, 2, AFFINE, LAYER, MEMEFF>(x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma, dbeta,
                                                                rows, G, cpg, stream);
        case 10:
          return launch_warp<T, 8, 10, 2, AFFINE, LAYER, MEMEFF>(x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma,
                                                                 dbeta, rows, G, cpg, stream);
        case 16:
          return launch_warp<T, 8, 16, 2, AFFINE, LAYER, MEMEFF>(x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma,
                                                                 dbeta, rows, G, cpg, stream);
        default:
          return launch_warp<T, 8, 0, 2, AFFINE, LAYER, MEMEFF>(x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma, dbeta,
                                                                rows, G, cpg, stream);
      }
    }
    if (cpg <= 8192) {
      const int64_t vpt = cpg / 8;
      if (vpt <= 256) {
        return launch_wide<T, 8, 1, AFFINE, LAYER, MEMEFF>(x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma, dbeta, rows,
                                                           G, cpg, stream);
      }
      return launch_wide<T, 8, 2, AFFINE, LAYER, MEMEFF>(x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma, dbeta, rows, G,
                                                         cpg, stream);
    }
  }
  return launch_wide<T, 8, 4, AFFINE, LAYER, MEMEFF>(x_or_y, dy, mean, rstd, gamma, beta, dx, dgamma, dbeta, rows, G, cpg,
                                                     stream);
}

template <typename T>
bool bwd_typed(int norm, const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean,
               const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta, bool memory_efficient,
               at::Tensor& dx, float* dgamma, float* dbeta, cudaStream_t stream) {
  const int64_t channels = x_or_y.size(-1);
  const int64_t rows = x_or_y.numel() / channels;
  const int64_t G = rstd.size(-1);
  const int64_t cpg = channels / G;
  const bool affine = gamma.numel() > 0;
  if (cpg < 8 || cpg % 8 != 0 || cpg > 8192) return false;

  const T* xp = static_cast<const T*>(x_or_y.data_ptr());
  const T* dyp = static_cast<const T*>(dy.data_ptr());
  const T* gp = affine ? static_cast<const T*>(gamma.data_ptr()) : nullptr;
  const T* bp = (affine && norm != 0) ? static_cast<const T*>(beta.data_ptr()) : nullptr;
  T* dxp = static_cast<T*>(dx.data_ptr());
  const float* mp = norm != 0 ? mean.data_ptr<float>() : nullptr;
  const float* rp = rstd.data_ptr<float>();

  if (norm == 0) {
    if (affine) {
      if (memory_efficient)
        dispatch_shape<T, true, false, true>(xp, dyp, mp, rp, gp, bp, dxp, dgamma, dbeta, rows, G, cpg, stream);
      else
        dispatch_shape<T, true, false, false>(xp, dyp, mp, rp, gp, bp, dxp, dgamma, dbeta, rows, G, cpg, stream);
    } else {
      if (memory_efficient)
        dispatch_shape<T, false, false, true>(xp, dyp, mp, rp, gp, bp, dxp, dgamma, dbeta, rows, G, cpg, stream);
      else
        dispatch_shape<T, false, false, false>(xp, dyp, mp, rp, gp, bp, dxp, dgamma, dbeta, rows, G, cpg, stream);
    }
  } else {
    if (affine) {
      if (memory_efficient)
        dispatch_shape<T, true, true, true>(xp, dyp, mp, rp, gp, bp, dxp, dgamma, dbeta, rows, G, cpg, stream);
      else
        dispatch_shape<T, true, true, false>(xp, dyp, mp, rp, gp, bp, dxp, dgamma, dbeta, rows, G, cpg, stream);
    } else {
      if (memory_efficient)
        dispatch_shape<T, false, true, true>(xp, dyp, mp, rp, gp, bp, dxp, dgamma, dbeta, rows, G, cpg, stream);
      else
        dispatch_shape<T, false, true, false>(xp, dyp, mp, rp, gp, bp, dxp, dgamma, dbeta, rows, G, cpg, stream);
    }
  }
  return true;
}

}  // namespace

bool GroupNormBwdByWidth(int64_t norm, const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean,
                     const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta, bool memory_efficient,
                     at::Tensor& dx, at::Tensor& dgamma_out, at::Tensor& dbeta_out) {
  const int64_t channels = x_or_y.size(-1);
  const int64_t rows = x_or_y.numel() / channels;
  const int64_t G = rstd.size(-1);
  const int64_t cpg = channels / G;
  const bool affine = gamma.numel() > 0;
  const bool has_beta = affine && norm != 0;

  // Taken here: the group widths where this kernel is fastest: the wide groups (more than 256 channels a group, where
  // the mapping is one CTA per group column walking the rows), the widths that are not a power of two (40 and 80
  // channels, 5 and 10 vectors) and 64 channels without an affine transform. Every other shape declines, and the
  // dispatcher tries the next kernel.
  const bool claimed = cpg > 256 || cpg == 40 || cpg == 80 || (cpg == 64 && !affine);
  const bool covered = claimed && cpg >= 8 && cpg % 8 == 0 && cpg <= 8192 &&
                       (x_or_y.scalar_type() == at::kBFloat16 || x_or_y.scalar_type() == at::kHalf) &&
                       x_or_y.is_contiguous() && dy.is_contiguous() && rstd.is_contiguous() && dx.is_contiguous() &&
                       rstd.numel() == rows * G && (!affine || gamma.is_contiguous()) &&
                       (!has_beta || beta.is_contiguous()) &&
                       (norm == 0 || memory_efficient || mean.is_contiguous()) &&
                       (!affine || (dgamma_out.is_contiguous() && (!has_beta || dbeta_out.is_contiguous())));

  if (!covered) return false;

  const at::cuda::OptionalCUDAGuard guard(at::device_of(x_or_y));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // dgamma and dbeta are sums over all rows, combined from every CTA: fp32 partials, one rounding at the end.
  at::Tensor scratch;
  float* dgp = nullptr;
  float* dbp = nullptr;
  if (affine) {
    scratch = at::zeros({2 * channels}, x_or_y.options().dtype(at::kFloat));
    dgp = scratch.data_ptr<float>();
    dbp = dgp + channels;
  }

  bool done = false;
  if (x_or_y.scalar_type() == at::kBFloat16) {
    done = bwd_typed<__nv_bfloat16>(norm, x_or_y, dy, mean, rstd, gamma, beta, memory_efficient, dx, dgp, dbp, stream);
  } else if (x_or_y.scalar_type() == at::kHalf) {
    done = bwd_typed<__half>(norm, x_or_y, dy, mean, rstd, gamma, beta, memory_efficient, dx, dgp, dbp, stream);
  }
  if (!done) return false;

  if (affine) {
    const int64_t blocks = (channels + 255) / 256;
    if (x_or_y.scalar_type() == at::kBFloat16) {
      gn_bwd_affine_epilogue<__nv_bfloat16><<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
          dgp, dbp, static_cast<__nv_bfloat16*>(dgamma_out.data_ptr()),
          has_beta ? static_cast<__nv_bfloat16*>(dbeta_out.data_ptr()) : nullptr, channels, has_beta);
    } else {
      gn_bwd_affine_epilogue<__half><<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
          dgp, dbp, static_cast<__half*>(dgamma_out.data_ptr()),
          has_beta ? static_cast<__half*>(dbeta_out.data_ptr()) : nullptr, channels, has_beta);
    }
  }
  return true;
}

}  // namespace gn
