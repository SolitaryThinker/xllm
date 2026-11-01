// The forward pass of xllm's group RMS norm (norm 0) and group layer norm (norm 1) at memory speed.
//
// A group is `cpg = C / G` contiguous channels of a row; groups are one contiguous run over the whole tensor. The
// mapping is chosen by channels per group, never by the number of groups:
//
//   cpg % 8 == 0 and cpg <= 256   a group is one 16-byte load a lane over L = cpg/8 lanes, so a warp covers W = 32/L
//                                 whole groups of one row. A warp keeps one block of W group columns and walks the
//                                 rows, so gamma (and beta) are loaded once into registers, the statistics are pure
//                                 shuffles (no shared memory, no __syncthreads) and the row loop is unrolled two
//                                 deep so two tiles of the row are in flight.
//   cpg % 256 == 0, cpg <= 2048   a group is one warp: Q = cpg/256 vectors a lane, all of them in registers, one
//                                 warp-level reduction, no shared memory.
//   cpg > 256 otherwise           a group is one CTA of cpg/(8*Q) threads with Q vectors a thread, one cross-warp
//                                 reduction through shared memory.
//   anything else                 a scalar two-pass CTA per group: correct for every dtype and shape, not tuned.
//
// bf16/half in and out, fp32 accumulation, statistics exactly as the interface says (fp32 [rows, G]).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "fwd_by_width.h"
#include "width_common.cuh"

namespace gn {
namespace {

// The conversion, vector and in-warp reduction helpers are shared with the backward: width_common.cuh.
using namespace gn::width;


template <int L>
__device__ __forceinline__ float group_reduce(float s, int lane, int l_rt) {
  if constexpr (L > 0) {
    return group_sum<L>(s, lane);
  } else {
    return group_sum_rt(s, lane, l_rt);
  }
}

// A block-wide sum: every thread ends with the total.
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

// ---------------------------------------------------------------- (1) W groups of a row per warp

constexpr int kWarpsPerBlock = 8;  // 256 threads a block
constexpr int kBlocksPerSm = 8;

template <typename T, int V, int L, bool AFFINE, bool LAYER>
__global__ __launch_bounds__(32 * kWarpsPerBlock) void gn_fwd_warp(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t rows, int G, int W, int ngb, int cpg_rt,
    float eps) {
  const int cpg = (L > 0) ? V * L : cpg_rt;
  const float inv_cpg = 1.f / static_cast<float>(cpg);
  const int lane = threadIdx.x & 31;
  const int gwid = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  const int nwarps = gridDim.x * kWarpsPerBlock;
  const int k = gwid % ngb;        // the group block this warp keeps
  const int j = gwid / ngb;        // this warp's place among the warps holding that block
  const int nj = (nwarps - 1 - k) / ngb + 1;
  if (j >= nj) return;

  const int nlanes = cpg / V;                       // lanes of a group
  const int segi = lane / nlanes;                   // which group of the block
  const int sub = lane - segi * nlanes;             // this lane's vector inside the group
  const int g = k * W + segi;                       // group index inside the row
  // Lanes past W*L (when L does not divide 32) belong to no group of this block: they must not read or write.
  const bool act = (segi < W) && (g < G);
  const int gs = act ? g : 0;                       // an inactive lane reads group 0, in bounds, and writes nothing
  const int64_t C = static_cast<int64_t>(G) * cpg;
  const int64_t chan = static_cast<int64_t>(gs) * cpg + sub * V;

  float gv[V];
  float bv[V];
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

  const int64_t col = sub * V;
  auto process = [&](int64_t r, const Vec<T, V>& in) {
    const int64_t off = r * C + chan;
    float f[V];
#pragma unroll
    for (int i = 0; i < V; ++i) f[i] = to_f(in.v[i]);

    float mu = 0.f;
    float rstd;
    if constexpr (LAYER) {
      float s1 = 0.f;
#pragma unroll
      for (int i = 0; i < V; ++i) s1 += f[i];
      mu = group_reduce<L>(s1, lane, nlanes) * inv_cpg;
      float s2 = 0.f;
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const float d = f[i] - mu;
        s2 += d * d;
      }
      rstd = rsqrtf(group_reduce<L>(s2, lane, nlanes) * inv_cpg + eps);
    } else {
      float s2 = 0.f;
#pragma unroll
      for (int i = 0; i < V; ++i) s2 += f[i] * f[i];
      rstd = rsqrtf(group_reduce<L>(s2, lane, nlanes) * inv_cpg + eps);
    }

    Vec<T, V> out;
#pragma unroll
    for (int i = 0; i < V; ++i) {
      float val = (LAYER ? (f[i] - mu) : f[i]) * rstd;
      if constexpr (AFFINE) val = val * gv[i] + (LAYER ? bv[i] : 0.f);
      from_f(out.v[i], val);
    }
    if (act) {
      store_stream<T, V>(y + off, out);
      if (col == 0) {
        rstd_out[r * G + g] = rstd;
        if constexpr (LAYER) mean_out[r * G + g] = mu;
      }
    }
  };

  // U tiles of the row in flight: the loads are all issued before the first one is consumed.
  constexpr int U = 4;
  int64_t r = j;
  for (; r + (U - 1) * nj < rows; r += U * nj) {
    Vec<T, V> in[U];
#pragma unroll
    for (int u = 0; u < U; ++u) in[u] = load_stream<T, V>(x + (r + u * nj) * C + chan);
#pragma unroll
    for (int u = 0; u < U; ++u) process(r + u * nj, in[u]);
  }
  for (; r < rows; r += nj) process(r, load_stream<T, V>(x + r * C + chan));
}

// ---------------------------------------------------------------- (2) one group per warp, Q vectors a lane

template <typename T, int V, int Q, bool AFFINE, bool LAYER>
__global__ __launch_bounds__(32 * kWarpsPerBlock) void gn_fwd_segwarp(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t tasks, int G, int64_t cpg, float eps) {
  constexpr int kChunk = 32 * V;  // elements a warp loads in one step
  const float inv_cpg = 1.f / static_cast<float>(cpg);
  const int lane = threadIdx.x & 31;
  const int64_t nwarps = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
  const int64_t col = static_cast<int64_t>(lane) * V;

  for (int64_t t = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5); t < tasks; t += nwarps) {
    const int64_t base = t * cpg;
    const int64_t off = base + col;
    const int gi = static_cast<int>(t % G);
    const int64_t gi_cpg = static_cast<int64_t>(gi) * cpg;

    float f[Q][V];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const Vec<T, V> in = load_stream<T, V>(x + off + q * kChunk);
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const float v = to_f(in.v[i]);
        f[q][i] = v;
        s1 += v;
        s2 += v * v;
      }
    }
    float mu = 0.f, var;
    if constexpr (LAYER) {
      mu = warp_sum(s1) * inv_cpg;
      float c2 = 0.f;
#pragma unroll
      for (int q = 0; q < Q; ++q) {
#pragma unroll
        for (int i = 0; i < V; ++i) {
          const float d = f[q][i] - mu;
          c2 += d * d;
        }
      }
      var = warp_sum(c2) * inv_cpg;
    } else {
      var = warp_sum(s2) * inv_cpg;
    }
    const float rstd = rsqrtf(var + eps);

#pragma unroll
    for (int q = 0; q < Q; ++q) {
      // gamma and beta are per channel, so they follow the chunk (they are small and cached: one load a vector).
      float gv[V], bv[V];
      if constexpr (AFFINE) {
        const Vec<T, V> t1 = load_cached<T, V>(gamma + gi_cpg + col + q * kChunk);
#pragma unroll
        for (int i = 0; i < V; ++i) gv[i] = to_f(t1.v[i]);
        if constexpr (LAYER) {
          const Vec<T, V> t2 = load_cached<T, V>(beta + gi_cpg + col + q * kChunk);
#pragma unroll
          for (int i = 0; i < V; ++i) bv[i] = to_f(t2.v[i]);
        }
      }
      Vec<T, V> out;
#pragma unroll
      for (int i = 0; i < V; ++i) {
        float val = (LAYER ? (f[q][i] - mu) : f[q][i]) * rstd;
        if constexpr (AFFINE) val = val * gv[i] + (LAYER ? bv[i] : 0.f);
        from_f(out.v[i], val);
      }
      store_stream<T, V>(y + off + q * kChunk, out);
    }
    if (lane == 0) {
      rstd_out[t] = rstd;
      if constexpr (LAYER) mean_out[t] = mu;
    }
  }
}

// ---------------------------------------------------------------- (3) one group per CTA, Q vectors a thread

template <typename T, int V, int Q, bool AFFINE, bool LAYER>
__global__ void gn_fwd_segcta(const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta,
                              T* __restrict__ y, float* __restrict__ mean_out, float* __restrict__ rstd_out,
                              int64_t tasks, int G, int64_t cpg, float eps) {
  const int64_t t = blockIdx.x;
  const int64_t base = t * cpg;
  const int64_t col = static_cast<int64_t>(threadIdx.x) * V;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * V;
  const int gi = static_cast<int>(t % G);
  const float inv_cpg = 1.f / static_cast<float>(cpg);

  float f[Q][V];
  float s1 = 0.f, s2 = 0.f;
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    const int64_t off = col + q * stride;
    const bool ok = off < cpg;
    const Vec<T, V> in = load_stream<T, V>(x + base + (ok ? off : 0));
#pragma unroll
    for (int i = 0; i < V; ++i) {
      const float v = ok ? to_f(in.v[i]) : 0.f;
      f[q][i] = v;
      s1 += v;
      s2 += v * v;
    }
  }
  __shared__ float sh[33];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarp = (blockDim.x + 31) >> 5;
  float mu = 0.f, var;
  if constexpr (LAYER) {
    mu = block_sum(s1, sh, nwarp, lane, wid) * inv_cpg;
    __syncthreads();
    float c2 = 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t off = col + q * stride;
      if (off < cpg) {
#pragma unroll
        for (int i = 0; i < V; ++i) {
          const float d = f[q][i] - mu;
          c2 += d * d;
        }
      }
    }
    var = block_sum(c2, sh, nwarp, lane, wid) * inv_cpg;
  } else {
    var = block_sum(s2, sh, nwarp, lane, wid) * inv_cpg;
  }
  const float rstd = rsqrtf(var + eps);

#pragma unroll
  for (int q = 0; q < Q; ++q) {
    const int64_t off = col + q * stride;
    if (off < cpg) {
      const int64_t chan = static_cast<int64_t>(gi) * cpg + off;
      float gv[V], bv[V];
      if constexpr (AFFINE) {
        const Vec<T, V> t1 = load_cached<T, V>(gamma + chan);
#pragma unroll
        for (int i = 0; i < V; ++i) gv[i] = to_f(t1.v[i]);
        if constexpr (LAYER) {
          const Vec<T, V> t2 = load_cached<T, V>(beta + chan);
#pragma unroll
          for (int i = 0; i < V; ++i) bv[i] = to_f(t2.v[i]);
        }
      }
      Vec<T, V> out;
#pragma unroll
      for (int i = 0; i < V; ++i) {
        float val = (LAYER ? (f[q][i] - mu) : f[q][i]) * rstd;
        if constexpr (AFFINE) val = val * gv[i] + (LAYER ? bv[i] : 0.f);
        from_f(out.v[i], val);
      }
      store_stream<T, V>(y + base + off, out);
    }
  }
  if (threadIdx.x == 0) {
    rstd_out[t] = rstd;
    if constexpr (LAYER) mean_out[t] = mu;
  }
}

// ---------------------------------------------------------------- (4) any shape: scalar, CTA per group

template <typename T, bool AFFINE, bool LAYER>
__global__ __launch_bounds__(256) void gn_fwd_any(const T* __restrict__ x, const T* __restrict__ gamma,
                                                  const T* __restrict__ beta, T* __restrict__ y,
                                                  float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                  int64_t rows, int64_t G, int64_t cpg, float eps) {
  const int64_t t = blockIdx.x;
  const int64_t r = t / G;
  const int64_t gi = t - r * G;
  const int64_t base = t * cpg;
  const T* xp = x + base;
  T* yp = y + base;
  const float inv_cpg = 1.f / static_cast<float>(cpg);

  float s1 = 0.f, s2 = 0.f;
  for (int64_t i = threadIdx.x; i < cpg; i += blockDim.x) {
    const float f = to_f(xp[i]);
    s1 += f;
    s2 += f * f;
  }
  __shared__ float sh[33];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarp = (blockDim.x + 31) >> 5;
  float mu = 0.f, var;
  if constexpr (LAYER) {
    const float sum = block_sum(s1, sh, nwarp, lane, wid);
    __syncthreads();
    mu = sum * inv_cpg;
    float c2 = 0.f;
    for (int64_t i = threadIdx.x; i < cpg; i += blockDim.x) {
      const float d = to_f(xp[i]) - mu;
      c2 += d * d;
    }
    var = block_sum(c2, sh, nwarp, lane, wid) * inv_cpg;
  } else {
    var = block_sum(s2, sh, nwarp, lane, wid) * inv_cpg;
  }
  const float rstd = rsqrtf(var + eps);
  const T* gp = AFFINE ? gamma + gi * cpg : nullptr;
  const T* bp = (AFFINE && LAYER) ? beta + gi * cpg : nullptr;
  for (int64_t i = threadIdx.x; i < cpg; i += blockDim.x) {
    float val = (LAYER ? (to_f(xp[i]) - mu) : to_f(xp[i])) * rstd;
    if constexpr (AFFINE) val = val * to_f(gp[i]) + (LAYER ? to_f(bp[i]) : 0.f);
    from_f(yp[i], val);
  }
  if (threadIdx.x == 0) {
    rstd_out[t] = rstd;
    if constexpr (LAYER) mean_out[t] = mu;
  }
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

template <typename T, int V, int L, bool AFFINE, bool LAYER>
void launch_warp(const T* x, const T* gamma, const T* beta, T* y, float* mean, float* rstd, int64_t rows, int64_t G,
                 int64_t cpg, float eps, cudaStream_t stream) {
  const int l = (L > 0) ? L : static_cast<int>(cpg / V);
  const int W = 32 / l;
  const int ngb = static_cast<int>((G + W - 1) / W);
  const int64_t tasks = rows * ngb;
  int64_t blocks = (tasks + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t cap = static_cast<int64_t>(sm_count()) * kBlocksPerSm;
  blocks = std::max<int64_t>(1, std::min(blocks, cap));
  gn_fwd_warp<T, V, L, AFFINE, LAYER><<<static_cast<unsigned>(blocks), 32 * kWarpsPerBlock, 0, stream>>>(
      x, gamma, beta, y, mean, rstd, rows, static_cast<int>(G), W, ngb, static_cast<int>(cpg), eps);
}

template <typename T, int V, int Q, bool AFFINE, bool LAYER>
void launch_segwarp(const T* x, const T* gamma, const T* beta, T* y, float* mean, float* rstd, int64_t rows, int64_t G,
                    int64_t cpg, float eps, cudaStream_t stream) {
  const int64_t tasks = rows * G;
  int64_t blocks = (tasks + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t cap = static_cast<int64_t>(sm_count()) * kBlocksPerSm;
  blocks = std::max<int64_t>(1, std::min(blocks, cap));
  gn_fwd_segwarp<T, V, Q, AFFINE, LAYER><<<static_cast<unsigned>(blocks), 32 * kWarpsPerBlock, 0, stream>>>(
      x, gamma, beta, y, mean, rstd, tasks, static_cast<int>(G), cpg, eps);
}

template <typename T, int V, int Q, bool AFFINE, bool LAYER>
void launch_segcta(const T* x, const T* gamma, const T* beta, T* y, float* mean, float* rstd, int64_t rows, int64_t G,
                   int64_t cpg, float eps, cudaStream_t stream) {
  const int64_t vpt = cpg / V;
  const int64_t per = Q;
  const unsigned threads = static_cast<unsigned>(std::min<int64_t>(1024, (vpt + per - 1) / per + 31) / 32 * 32);
  const int64_t blocks = rows * G;
  gn_fwd_segcta<T, V, Q, AFFINE, LAYER><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
      x, gamma, beta, y, mean, rstd, blocks, static_cast<int>(G), cpg, eps);
}

template <typename T, bool AFFINE, bool LAYER>
void launch_any(const T* x, const T* gamma, const T* beta, T* y, float* mean, float* rstd, int64_t rows, int64_t G,
                int64_t cpg, float eps, cudaStream_t stream) {
  const int64_t blocks = rows * G;
  gn_fwd_any<T, AFFINE, LAYER><<<static_cast<unsigned>(blocks), 256, 0, stream>>>(x, gamma, beta, y, mean, rstd, rows,
                                                                                 G, cpg, eps);
}

template <typename T, bool AFFINE, bool LAYER>
void dispatch_shape(const T* x, const T* gamma, const T* beta, T* y, float* mean, float* rstd, int64_t rows, int64_t G,
                    int64_t cpg, float eps, cudaStream_t stream) {
  if constexpr (sizeof(T) == 2) {
    if (cpg % 8 == 0) {
      if (cpg <= 256) {
        switch (cpg / 8) {
          case 1:
            return launch_warp<T, 8, 1, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 2:
            return launch_warp<T, 8, 2, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 3:
            return launch_warp<T, 8, 3, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 4:
            return launch_warp<T, 8, 4, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 5:
            return launch_warp<T, 8, 5, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 6:
            return launch_warp<T, 8, 6, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 8:
            return launch_warp<T, 8, 8, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 10:
            return launch_warp<T, 8, 10, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 12:
            return launch_warp<T, 8, 12, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 16:
            return launch_warp<T, 8, 16, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 20:
            return launch_warp<T, 8, 20, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 24:
            return launch_warp<T, 8, 24, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 32:
            return launch_warp<T, 8, 32, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          default:
            return launch_warp<T, 8, 0, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
        }
      }
      if (cpg % 256 == 0 && cpg <= 2048) {
        switch (cpg / 256) {
          case 2:
            return launch_segwarp<T, 8, 2, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 3:
            return launch_segwarp<T, 8, 3, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 4:
            return launch_segwarp<T, 8, 4, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 5:
            return launch_segwarp<T, 8, 5, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 6:
            return launch_segwarp<T, 8, 6, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 7:
            return launch_segwarp<T, 8, 7, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          case 8:
            return launch_segwarp<T, 8, 8, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
          default:
            break;
        }
      }
      if (cpg <= 8192) {
        const int64_t vpt = cpg / 8;
        if (vpt <= 256) {
          return launch_segcta<T, 8, 1, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
        } else if (vpt <= 512) {
          return launch_segcta<T, 8, 2, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
        } else {
          return launch_segcta<T, 8, 4, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
        }
      }
    }
  }
  return launch_any<T, AFFINE, LAYER>(x, gamma, beta, y, mean, rstd, rows, G, cpg, eps, stream);
}

template <typename T>
void fwd_typed(int norm, const T* x, const T* gamma, const T* beta, T* y, float* mean, float* rstd, int64_t rows,
               int64_t G, int64_t cpg, double eps, cudaStream_t stream) {
  const bool affine = gamma != nullptr;
  if (norm == 0) {
    if (affine)
      dispatch_shape<T, true, false>(x, gamma, beta, y, mean, rstd, rows, G, cpg, static_cast<float>(eps), stream);
    else
      dispatch_shape<T, false, false>(x, gamma, beta, y, mean, rstd, rows, G, cpg, static_cast<float>(eps), stream);
  } else {
    if (affine)
      dispatch_shape<T, true, true>(x, gamma, beta, y, mean, rstd, rows, G, cpg, static_cast<float>(eps), stream);
    else
      dispatch_shape<T, false, true>(x, gamma, beta, y, mean, rstd, rows, G, cpg, static_cast<float>(eps), stream);
  }
}

}  // namespace

bool GroupNormFwdByWidth(int64_t norm, const at::Tensor& x, int64_t num_groups, double eps, const at::Tensor& gamma,
                     const at::Tensor& beta, at::Tensor& y, at::Tensor& mean_out, at::Tensor& rstd_out) {
  const int64_t channels = x.size(-1);
  const int64_t rows = x.numel() / channels;
  const int64_t cpg = channels / num_groups;
  const bool affine = gamma.numel() > 0;
  const bool layer = norm != 0;
  // Taken here: the wide groups (more than 256 channels a group), one group per warp or per CTA, and the two widths
  // that are 5 and 10 16-byte vectors a group (40 and 80 channels), where a warp of 32 lanes holds 6 or 3 whole groups.
  // Every other shape returns false and is left to the other forward kernels.
  if ((x.scalar_type() != at::kBFloat16 && x.scalar_type() != at::kHalf) || cpg % 8 != 0 || cpg > 8192) return false;
  if (!(cpg > 256 || cpg == 40 || cpg == 80)) return false;
  if (!x.is_contiguous() || !y.is_contiguous() || !rstd_out.is_contiguous() || rstd_out.scalar_type() != at::kFloat ||
      (affine && !gamma.is_contiguous()) || (affine && layer && !beta.is_contiguous()) ||
      (layer && (!mean_out.is_contiguous() || mean_out.scalar_type() != at::kFloat || mean_out.numel() != rows * num_groups)) ||
      rstd_out.numel() != rows * num_groups || channels % num_groups != 0) {
    return false;
  }
  if (affine && (gamma.numel() != channels || (layer && beta.numel() != channels))) return false;

  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const void* xp = x.data_ptr();
  void* yp = y.data_ptr();
  const void* gp = affine ? gamma.data_ptr() : nullptr;
  const void* bp = (affine && layer) ? beta.data_ptr() : nullptr;
  float* mp = layer ? mean_out.data_ptr<float>() : nullptr;
  float* rp = rstd_out.data_ptr<float>();

  switch (x.scalar_type()) {
    case at::kBFloat16:
      fwd_typed<__nv_bfloat16>(norm, static_cast<const __nv_bfloat16*>(xp), static_cast<const __nv_bfloat16*>(gp),
                               static_cast<const __nv_bfloat16*>(bp), static_cast<__nv_bfloat16*>(yp), mp, rp, rows,
                               num_groups, cpg, eps, stream);
      break;
    case at::kHalf:
      fwd_typed<__half>(norm, static_cast<const __half*>(xp), static_cast<const __half*>(gp),
                        static_cast<const __half*>(bp), static_cast<__half*>(yp), mp, rp, rows, num_groups, cpg, eps,
                        stream);
      break;
    default:
      return false;
  }
  return true;
}

}  // namespace gn
