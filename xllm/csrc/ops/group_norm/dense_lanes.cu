// Group RMS / layer norm, forward and backward, for the many-group class (a group is 1..32
// vectors of 16 B). See dense_lanes.h for the lane mapping. One file, no shared-memory reduction in the
// normal path: the group's statistics are reduced inside the warp that holds the group.
#include "dense_lanes.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace xllm {
namespace ops {
namespace dense_lanes {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kBlock = 32 * kWarpsPerBlock;

// ---------------------------------------------------------------- data path

__device__ __forceinline__ uint4 ldg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void stg16(void* p, const uint4& v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

template <typename T>
struct Cvt;

template <>
struct Cvt<float> {
  static constexpr int N = 4;
  __device__ __forceinline__ static void unpack(const uint4& v, float* f) {
    f[0] = __uint_as_float(v.x);
    f[1] = __uint_as_float(v.y);
    f[2] = __uint_as_float(v.z);
    f[3] = __uint_as_float(v.w);
  }
  __device__ __forceinline__ static uint4 pack(const float* f) {
    uint4 v;
    v.x = __float_as_uint(f[0]);
    v.y = __float_as_uint(f[1]);
    v.z = __float_as_uint(f[2]);
    v.w = __float_as_uint(f[3]);
    return v;
  }
  __device__ __forceinline__ static float from_float(float x) { return x; }
};

template <>
struct Cvt<__nv_bfloat16> {
  static constexpr int N = 8;
  __device__ __forceinline__ static void unpack(const uint4& v, float* f) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 t = __bfloat1622float2(p[i]);
      f[2 * i] = t.x;
      f[2 * i + 1] = t.y;
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* f) {
    uint4 v;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return v;
  }
  __device__ __forceinline__ static __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }
};

template <>
struct Cvt<__half> {
  static constexpr int N = 8;
  __device__ __forceinline__ static void unpack(const uint4& v, float* f) {
    const __half2* p = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 t = __half22float2(p[i]);
      f[2 * i] = t.x;
      f[2 * i + 1] = t.y;
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* f) {
    uint4 v;
    __half2* p = reinterpret_cast<__half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
    return v;
  }
  __device__ __forceinline__ static __half from_float(float x) { return __float2half(x); }
};

// Sum over the W lanes of this lane's segment, in every lane of the segment. W may be any value in
// [1, 32]: the tree adds a lane only when that lane is still inside the segment, so a segment of 3,
// 5 or 10 lanes needs ceil(log2 W) steps and no shared memory.
__device__ __forceinline__ float seg_sum(float v, int lane, int W) {
  const int lid = lane % W;
#pragma unroll 1
  for (int d = 1; d < W; d <<= 1) {
    const float u = __shfl_down_sync(0xffffffffu, v, d);
    if (lid + d < W) v += u;
  }
  return __shfl_sync(0xffffffffu, v, lane - lid);
}

// ---------------------------------------------------------------- forward

// Q chunks of 16 B a lane: with an even number of vectors a group, Q = 2 halves the lanes a group takes
// (W = V/2), so a warp holds twice as many whole groups -- the segmented reduce is one shuffle shorter and
// the statistics store of the whole warp lands in one 32 B sector instead of half of one. The lane offsets
// stay interleaved (`lid + j*W`), so every instruction is still a contiguous run of a group's chunks.
template <typename T, bool LAYER, bool AFFINE, int ILP, int Q>
__global__ void __launch_bounds__(kBlock) gn_fwd_kernel(const T* __restrict__ x, T* __restrict__ y,
                                                       float* __restrict__ rstd_out, float* __restrict__ mean_out,
                                                       const T* __restrict__ gamma, const T* __restrict__ beta,
                                                       int G, int cpg, int W, float eps, int64_t total_slots) {
  constexpr int N = Cvt<T>::N;
  constexpr int M = Q * N;
  const int lane = threadIdx.x & 31;
  const int gw = 32 / W;
  const int live = gw * W;
  const int soff = lane / W;
  const int lid = lane - soff * W;
  const float inv_n = 1.0f / (float)cpg;
  const int warps = blockDim.x >> 5;
  const int64_t step = (int64_t)ILP * gw;
  const int64_t wstep = (int64_t)gridDim.x * warps * step;
  int64_t s0 = ((int64_t)blockIdx.x * warps + (threadIdx.x >> 5)) * step;

  for (; s0 < total_slots; s0 += wstep) {
    uint4 raw[ILP][Q];
    int64_t off[ILP];
    bool ok[ILP];
    const int64_t jstep = (int64_t)W * N;  // chunk j sits j*W*N elements on: a constant, so one 64-bit address a slot
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      const int64_t slot = s0 + u * gw + soff;
      ok[u] = (lane < live) && (slot < total_slots);
      off[u] = slot * (int64_t)cpg + (int64_t)lid * N;
#pragma unroll
      for (int j = 0; j < Q; ++j) raw[u][j] = ok[u] ? ldg16(x + off[u] + j * jstep) : make_uint4(0, 0, 0, 0);
    }
    float gv[ILP][M], bv[ILP][M];
    if (AFFINE) {
#pragma unroll
      for (int u = 0; u < ILP; ++u) {
        const int64_t slot = s0 + u * gw + soff;
        const int64_t base = (slot % G) * (int64_t)cpg;
#pragma unroll
        for (int j = 0; j < Q; ++j) {
          const int64_t col = base + (int64_t)(lid + j * W) * N;
          Cvt<T>::unpack(ok[u] ? ldg16(gamma + col) : make_uint4(0, 0, 0, 0), gv[u] + j * N);
          if (LAYER) Cvt<T>::unpack(ok[u] ? ldg16(beta + col) : make_uint4(0, 0, 0, 0), bv[u] + j * N);
        }
      }
    }
    float out[ILP][M];
    float rst[ILP], muv[ILP];
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      float v[M];
#pragma unroll
      for (int j = 0; j < Q; ++j) Cvt<T>::unpack(raw[u][j], v + j * N);
      if (LAYER) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < M; ++k) s += v[k];
        const float mu = seg_sum(s, lane, W) * inv_n;
        float c2 = 0.f;
#pragma unroll
        for (int k = 0; k < M; ++k) {
          const float d = v[k] - mu;
          c2 += d * d;
        }
        const float r = rsqrtf(seg_sum(c2, lane, W) * inv_n + eps);
        muv[u] = mu;
        rst[u] = r;
#pragma unroll
        for (int k = 0; k < M; ++k) {
          const float xh = (v[k] - mu) * r;
          out[u][k] = AFFINE ? xh * gv[u][k] + bv[u][k] : xh;
        }
      } else {
        float sq = 0.f;
#pragma unroll
        for (int k = 0; k < M; ++k) sq += v[k] * v[k];
        const float r = rsqrtf(seg_sum(sq, lane, W) * inv_n + eps);
        rst[u] = r;
#pragma unroll
        for (int k = 0; k < M; ++k) {
          const float xh = v[k] * r;
          out[u][k] = AFFINE ? xh * gv[u][k] : xh;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      if (ok[u]) {
#pragma unroll
        for (int j = 0; j < Q; ++j) stg16(y + off[u] + j * jstep, Cvt<T>::pack(out[u] + j * N));
        if (lid == 0) {
          const int64_t slot = s0 + u * gw + soff;
          rstd_out[slot] = rst[u];
          if (LAYER) mean_out[slot] = muv[u];
        }
      }
    }
  }
}

// ---------------------------------------------------------------- backward, no affine

template <typename T, bool LAYER, bool ME, int ILP, int Q>
__global__ void __launch_bounds__(kBlock) gn_bwd_flat_kernel(const T* __restrict__ xr, const T* __restrict__ dy,
                                                             T* __restrict__ dx, const float* __restrict__ rstd,
                                                             const float* __restrict__ mean, int cpg, int W,
                                                             int64_t total_slots) {
  constexpr int N = Cvt<T>::N;
  constexpr int M = Q * N;
  const int lane = threadIdx.x & 31;
  const int gw = 32 / W;
  const int live = gw * W;
  const int soff = lane / W;
  const int lid = lane - soff * W;
  const float inv_n = 1.0f / (float)cpg;
  const int warps = blockDim.x >> 5;
  const int64_t step = (int64_t)ILP * gw;
  const int64_t wstep = (int64_t)gridDim.x * warps * step;
  int64_t s0 = ((int64_t)blockIdx.x * warps + (threadIdx.x >> 5)) * step;

  for (; s0 < total_slots; s0 += wstep) {
    uint4 rx[ILP][Q], rd[ILP][Q];
    int64_t off[ILP];
    bool ok[ILP];
    float r[ILP], mu[ILP];
    const int64_t jstep = (int64_t)W * N;  // chunk j sits j*W*N elements on: a constant, so one 64-bit address a slot
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      const int64_t slot = s0 + u * gw + soff;
      ok[u] = (lane < live) && (slot < total_slots);
      off[u] = slot * (int64_t)cpg + (int64_t)lid * N;
      r[u] = ok[u] ? rstd[slot] : 0.f;
      mu[u] = (LAYER && !ME && ok[u]) ? mean[slot] : 0.f;
#pragma unroll
      for (int j = 0; j < Q; ++j) {
        rx[u][j] = ok[u] ? ldg16(xr + off[u] + j * jstep) : make_uint4(0, 0, 0, 0);
        rd[u][j] = ok[u] ? ldg16(dy + off[u] + j * jstep) : make_uint4(0, 0, 0, 0);
      }
    }
    float out[ILP][M];
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      float xv[M], dv[M], xh[M];
#pragma unroll
      for (int j = 0; j < Q; ++j) {
        Cvt<T>::unpack(rx[u][j], xv + j * N);
        Cvt<T>::unpack(rd[u][j], dv + j * N);
      }
      float s1 = 0.f, s2 = 0.f;
      if (ME) {
#pragma unroll
        for (int k = 0; k < M; ++k) {
          xh[k] = xv[k];
          s2 += dv[k] * xh[k];
          if (LAYER) s1 += dv[k];
        }
      } else if (LAYER) {
#pragma unroll
        for (int k = 0; k < M; ++k) {
          xh[k] = (xv[k] - mu[u]) * r[u];
          s2 += dv[k] * xh[k];
          s1 += dv[k];
        }
      } else {
#pragma unroll
        for (int k = 0; k < M; ++k) {
          xh[k] = xv[k] * r[u];
          s2 += dv[k] * xh[k];
        }
      }
      s2 = seg_sum(s2, lane, W) * inv_n;
      if (LAYER) s1 = seg_sum(s1, lane, W) * inv_n;
#pragma unroll
      for (int k = 0; k < M; ++k) out[u][k] = LAYER ? (dv[k] - s1 - xh[k] * s2) * r[u] : (dv[k] - xh[k] * s2) * r[u];
    }
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      if (ok[u]) {
#pragma unroll
        for (int j = 0; j < Q; ++j) stg16(dx + off[u] + j * jstep, Cvt<T>::pack(out[u] + j * N));
      }
    }
  }
}

// ---------------------------------------------------------------- backward, affine
// A warp owns a fixed set of columns (one tile of gw groups) and walks rows, so dgamma and dbeta
// accumulate in registers. Its partials go to ws[chunk][C] (one writer per element, no atomics) and
// a small second kernel folds the chunks together: x and dy are read once, not twice.

template <typename T, bool LAYER, bool ME, int RB>
__global__ void __launch_bounds__(kBlock) gn_bwd_aff_kernel(const T* __restrict__ xr, const T* __restrict__ dy,
                                                            T* __restrict__ dx, const float* __restrict__ rstd,
                                                            const float* __restrict__ mean, const T* __restrict__ gamma,
                                                            const T* __restrict__ beta, int G, int cpg, int W, int rows,
                                                            int tiles_per_row, int nchunks, int rows_per_chunk, int C,
                                                            float* __restrict__ ws) {
  constexpr int N = Cvt<T>::N;
  const int lane = threadIdx.x & 31;
  const int gw = 32 / W;
  const int soff = lane / W;
  const int lid = lane - soff * W;
  const float inv_n = 1.0f / (float)cpg;
  const int warps = blockDim.x >> 5;
  const int ntasks = tiles_per_row * nchunks;
  const int ntask_warps = gridDim.x * warps;
  for (int t = blockIdx.x * warps + (threadIdx.x >> 5); t < ntasks; t += ntask_warps) {
    const int tile = t % tiles_per_row;
    const int chunk = t / tiles_per_row;
    const int g = tile * gw + soff;
    const int64_t r0 = (int64_t)chunk * rows_per_chunk;
    const int64_t r1 = (r0 + rows_per_chunk < rows) ? r0 + rows_per_chunk : rows;
    // No `continue` per lane: every lane of the warp must reach the shuffles below.
    const bool active = (lane < gw * W) && (g < G);
    const int64_t col = active ? (int64_t)g * cpg + (int64_t)lid * N : 0;
    float gvv[N], bvv[N];
    Cvt<T>::unpack(active ? ldg16(gamma + col) : make_uint4(0, 0, 0, 0), gvv);
    if (LAYER) Cvt<T>::unpack(active ? ldg16(beta + col) : make_uint4(0, 0, 0, 0), bvv);
    float dga[N], dbe[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {
      dga[k] = 0.f;
      dbe[k] = 0.f;
    }
    for (int64_t row = r0; row < r1; row += RB) {
      uint4 rx[RB], rd[RB];
      int64_t off[RB];
      bool ok[RB];
#pragma unroll
      for (int i = 0; i < RB; ++i) {
        ok[i] = active && (row + i < r1);
        const int64_t rw = (row + i < r1) ? row + i : r1 - 1;
        off[i] = rw * (int64_t)C + col;
        rx[i] = ok[i] ? ldg16(xr + off[i]) : make_uint4(0, 0, 0, 0);
        rd[i] = ok[i] ? ldg16(dy + off[i]) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int i = 0; i < RB; ++i) {
        const int64_t rw = (row + i < r1) ? row + i : r1 - 1;
        const float r = ok[i] ? rstd[rw * G + g] : 0.f;
        const float mu = (LAYER && !ME && ok[i]) ? mean[rw * G + g] : 0.f;
        float xv[N], dv[N], xh[N], dh[N];
        Cvt<T>::unpack(rx[i], xv);
        Cvt<T>::unpack(rd[i], dv);
        float s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int k = 0; k < N; ++k) {
          if (ME)
            xh[k] = LAYER ? (xv[k] - bvv[k]) / gvv[k] : xv[k] / gvv[k];
          else
            xh[k] = LAYER ? (xv[k] - mu) * r : xv[k] * r;
          dh[k] = dv[k] * gvv[k];
          s2 += dh[k] * xh[k];
          if (LAYER) s1 += dh[k];
        }
        s2 = seg_sum(s2, lane, W) * inv_n;
        if (LAYER) s1 = seg_sum(s1, lane, W) * inv_n;
        const float wgt = ok[i] ? 1.f : 0.f;
        float out[N];
#pragma unroll
        for (int k = 0; k < N; ++k) {
          out[k] = LAYER ? (dh[k] - s1 - xh[k] * s2) * r : (dh[k] - xh[k] * s2) * r;
          dga[k] += wgt * (dv[k] * xh[k]);
          dbe[k] += wgt * dv[k];
        }
        if (ok[i]) stg16(dx + off[i], Cvt<T>::pack(out));
      }
    }
    if (active) {
      float* w = ws + (int64_t)chunk * C + col;
#pragma unroll
      for (int k = 0; k < N; ++k) w[k] = dga[k];
      if (LAYER) {
        float* w2 = ws + (int64_t)nchunks * C + (int64_t)chunk * C + col;
#pragma unroll
        for (int k = 0; k < N; ++k) w2[k] = dbe[k];
      }
    }
  }
}

// ---------------------------------------------------------------- partial reduction for dgamma / dbeta

template <typename T>
__global__ void gn_dg_reduce_kernel(const float* __restrict__ ws, T* __restrict__ dgamma, T* __restrict__ dbeta,
                                    int C, int nchunks, bool has_beta) {
  const int part = threadIdx.x >> 5;  // 0..7
  const int lcol = threadIdx.x & 31;  // 0..31
  const int col = blockIdx.x * 32 + lcol;
  __shared__ float sdg[8][32];
  __shared__ float sdb[8][32];
  const int per = (nchunks + 7) >> 3;
  const int lo = part * per;
  const int hi = (lo + per < nchunks) ? lo + per : nchunks;
  float s = 0.f, s2 = 0.f;
  if (col < C) {
    const float* w0 = ws + col;
    const float* w1 = ws + (int64_t)nchunks * C + col;
    for (int i = lo; i < hi; ++i) {
      s += w0[(int64_t)i * C];
      if (has_beta) s2 += w1[(int64_t)i * C];
    }
  }
  sdg[part][lcol] = s;
  sdb[part][lcol] = s2;
  __syncthreads();
  if (part == 0 && col < C) {
    float t = 0.f, t2 = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      t += sdg[p][lcol];
      t2 += sdb[p][lcol];
    }
    dgamma[col] = Cvt<T>::from_float(t);
    if (has_beta) dbeta[col] = Cvt<T>::from_float(t2);
  }
}

// ---------------------------------------------------------------- host side

inline int elemsize(at::ScalarType t) {
  if (t == at::kFloat) return 4;
  if (t == at::kBFloat16 || t == at::kHalf) return 2;
  return 0;
}

inline int grid_blocks(int64_t work_units, int64_t units_per_warp, int sm, int blocks_per_sm) {
  const int64_t warps = (work_units + units_per_warp - 1) / units_per_warp;
  int64_t blocks = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t cap = (int64_t)sm * blocks_per_sm;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;
  return (int)blocks;
}

template <typename T>
void fwd_launch(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta, int64_t rows, int G, int cpg,
                int V, bool layer, bool affine, double eps, at::Tensor& y, at::Tensor& mean_out,
                at::Tensor& rstd_out) {
  const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t total_slots = rows * G;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const T* xp = reinterpret_cast<const T*>(x.data_ptr());
  T* yp = reinterpret_cast<T*>(y.data_ptr());
  float* rp = rstd_out.data_ptr<float>();
  float* mp = layer ? mean_out.data_ptr<float>() : nullptr;
  const T* gp = affine ? reinterpret_cast<const T*>(gamma.data_ptr()) : nullptr;
  const T* bp = (affine && layer) ? reinterpret_cast<const T*>(beta.data_ptr()) : nullptr;

  {
    // Chunks a lane: 2 when a group has an even number of 16 B vectors, which halves the lanes a group takes.
    // The affine lane also holds gamma and beta, so it keeps one chunk a lane and the depth that fits.
    const int Q = (!affine && V >= 2 && (V % 2) == 0) ? 2 : 1;
    const int W = V / Q;
    // Slot depth is set by what else a lane holds: the affine path carries gamma/beta, the plain one does not.
    // Chunks in flight a lane (ILP*Q) is what the ring holds, so it stays at 4 plain and 2 affine.
    constexpr int kIlpAff = 2;
    constexpr int kIlpPlain = 4;
    // 64 CTAs a SM unconditionally: with two chunks a lane the long grid is at least as good on every shape this path
    // takes, small ones included.
    const int bps = 64;
    const float feps = (float)eps;
#define GN_FWD(L, A, IL, QQ)                                                                                       \
  gn_fwd_kernel<T, L, A, IL, QQ><<<grid_blocks(total_slots, (int64_t)IL * (32 / W), sm, bps), kBlock, 0, st>>>(    \
      xp, yp, rp, mp, gp, bp, G, cpg, W, feps, total_slots)
    if (layer) {
      if (affine) GN_FWD(true, true, kIlpAff, 1);
      else if (Q == 2) GN_FWD(true, false, kIlpPlain / 2, 2);
      else GN_FWD(true, false, kIlpPlain, 1);
    } else {
      if (affine) GN_FWD(false, true, kIlpAff, 1);
      else if (Q == 2) GN_FWD(false, false, kIlpPlain / 2, 2);
      else GN_FWD(false, false, kIlpPlain, 1);
    }
#undef GN_FWD
  }
}

template <typename T>
void bwd_launch(const at::Tensor& xr, const at::Tensor& dy, const at::Tensor& mean, const at::Tensor& rstd,
                const at::Tensor& gamma, const at::Tensor& beta, int64_t rows, int G, int cpg, int V, int C, bool layer,
                bool affine, bool me, at::Tensor& dx, at::Tensor& dgamma, at::Tensor& dbeta) {
  const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const T* xp = reinterpret_cast<const T*>(xr.data_ptr());
  const T* dyp = reinterpret_cast<const T*>(dy.data_ptr());
  T* dxp = reinterpret_cast<T*>(dx.data_ptr());
  const float* rp = rstd.data_ptr<float>();
  const float* mp = (layer && !me) ? mean.data_ptr<float>() : nullptr;
  const int64_t total_slots = rows * G;

  if (!affine) {
    // Two chunks a lane for an even number of vectors (see the forward): half the lanes a group, one shuffle
    // step fewer, and the whole warp's statistics in one 32 B sector. With ILP = 2 that is 4 chunks in flight a
    // lane where Q = 2, which costs registers (75-92 for bf16) and still fits 2-3 CTAs a SM.
    const int Q = (V >= 2 && (V % 2) == 0) ? 2 : 1;
    const int W = V / Q;
    constexpr int ILP = 2;
    // 64 CTAs a SM, the same number as the forward.
    const int cap = 64;
#define GN_BWD_FLAT(L, M, QQ)                                                                                   \
  gn_bwd_flat_kernel<T, L, M, ILP, QQ><<<grid_blocks(total_slots, (int64_t)ILP * (32 / W), sm, cap), kBlock, 0, \
                                         st>>>(xp, dyp, dxp, rp, mp, cpg, W, total_slots)
    if (layer) {
      if (Q == 2) { if (me) GN_BWD_FLAT(true, true, 2); else GN_BWD_FLAT(true, false, 2); }
      else { if (me) GN_BWD_FLAT(true, true, 1); else GN_BWD_FLAT(true, false, 1); }
    } else {
      if (Q == 2) { if (me) GN_BWD_FLAT(false, true, 2); else GN_BWD_FLAT(false, false, 2); }
      else { if (me) GN_BWD_FLAT(false, true, 1); else GN_BWD_FLAT(false, false, 1); }
    }
#undef GN_BWD_FLAT
    return;
  }

  const int W = V;
  const int gw = 32 / W;
  const int tiles_per_row = (G + gw - 1) / gw;
  // One warp per (column tile, row chunk), and 64 warps' worth of chunks per SM so that the
  // machine is filled by tasks alone (no grid-stride tail); the 256 cap keeps the fp32 partial
  // workspace at <= 8 MB even at C = 8192, where it would otherwise cost more traffic than it saves.
  int64_t nchunks = (int64_t)sm * 64 / tiles_per_row;
  if (nchunks > 256) nchunks = 256;
  if (nchunks < 1) nchunks = 1;
  if (nchunks > rows) nchunks = rows;
  int64_t rows_per_chunk = (rows + nchunks - 1) / nchunks;
  nchunks = (rows + rows_per_chunk - 1) / rows_per_chunk;
  const int64_t tasks = (int64_t)tiles_per_row * nchunks;
  int blocks = (int)((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
  // The block cap is residency, not grid length: this kernel is register-bound (RB slots of x and dy a lane),
  // so 8 CTAs of 256 threads an SM is already past what the register file holds resident, and a longer grid
  // would only add a tail.
  const int cap = sm * 8;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;

  auto ws = at::empty({(int64_t)nchunks * C * (layer ? 2 : 1)}, xr.options().dtype(at::kFloat));
  float* wsp = ws.data_ptr<float>();
  const T* gp = reinterpret_cast<const T*>(gamma.data_ptr());
  const T* bp = layer ? reinterpret_cast<const T*>(beta.data_ptr()) : nullptr;
  constexpr int RB = 4;
#define GN_BWD_AFF(L, M)                                                                                       \
  gn_bwd_aff_kernel<T, L, M, RB><<<blocks, kBlock, 0, st>>>(xp, dyp, dxp, rp, mp, gp, bp, G, cpg, W, (int)rows, \
                                                            tiles_per_row, (int)nchunks, (int)rows_per_chunk, C, wsp)
  if (layer) {
    if (me) GN_BWD_AFF(true, true);
    else GN_BWD_AFF(true, false);
  } else {
    if (me) GN_BWD_AFF(false, true);
    else GN_BWD_AFF(false, false);
  }
#undef GN_BWD_AFF

  const int rb_blocks = (C + 31) / 32;
  T* dgp = reinterpret_cast<T*>(dgamma.data_ptr());
  T* dbp = layer ? reinterpret_cast<T*>(dbeta.data_ptr()) : nullptr;
  gn_dg_reduce_kernel<T><<<rb_blocks, 256, 0, st>>>(wsp, dgp, dbp, C, (int)nchunks, layer);
}

}  // namespace

bool Eligible(const torch::Tensor& x, int64_t num_groups) {
  if (!x.is_cuda() || x.dim() != 3 || !x.is_contiguous()) return false;
  const int es = elemsize(x.scalar_type());
  if (!es) return false;
  const int64_t C = x.size(-1);
  if (C <= 0 || C > (int64_t)2147483647 || num_groups <= 0 || num_groups > (int64_t)2147483647) return false;
  if (C % num_groups != 0) return false;
  const int64_t cpg = C / num_groups;
  if (cpg <= 0 || (cpg * es) % 16 != 0) return false;
  const int64_t V = cpg * es / 16;
  if (V < 1 || V > 32) return false;
  if (x.numel() / C <= 0) return false;
  return true;
}

void Fwd(const torch::Tensor& x, int64_t num_groups, bool layer, const torch::Tensor& gamma, const torch::Tensor& beta,
         double eps, torch::Tensor y, torch::Tensor mean_out, torch::Tensor rstd_out) {
  const int64_t C = x.size(-1);
  const int64_t rows = x.numel() / C;
  const int G = (int)num_groups;
  const int cpg = (int)(C / num_groups);
  const int es = elemsize(x.scalar_type());
  const int V = (int)(cpg * es / 16);
  const bool affine = gamma.numel() > 0;
  switch (x.scalar_type()) {
    case at::kFloat:
      fwd_launch<float>(x, gamma, beta, rows, G, cpg, V, layer, affine, eps, y, mean_out, rstd_out);
      break;
    case at::kBFloat16:
      fwd_launch<__nv_bfloat16>(x, gamma, beta, rows, G, cpg, V, layer, affine, eps, y, mean_out, rstd_out);
      break;
    default:
      fwd_launch<__half>(x, gamma, beta, rows, G, cpg, V, layer, affine, eps, y, mean_out, rstd_out);
      break;
  }
}

void Bwd(const torch::Tensor& x_or_y, const torch::Tensor& dy, int64_t num_groups, bool layer, bool memory_efficient,
         const torch::Tensor& mean, const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
         torch::Tensor dx, torch::Tensor dgamma, torch::Tensor dbeta) {
  const int64_t C = x_or_y.size(-1);
  const int64_t rows = x_or_y.numel() / C;
  const int G = (int)num_groups;
  const int cpg = (int)(C / num_groups);
  const int es = elemsize(x_or_y.scalar_type());
  const int V = (int)(cpg * es / 16);
  const bool affine = gamma.numel() > 0;
  switch (x_or_y.scalar_type()) {
    case at::kFloat:
      bwd_launch<float>(x_or_y, dy, mean, rstd, gamma, beta, rows, G, cpg, V, (int)C, layer, affine, memory_efficient,
                        dx, dgamma, dbeta);
      break;
    case at::kBFloat16:
      bwd_launch<__nv_bfloat16>(x_or_y, dy, mean, rstd, gamma, beta, rows, G, cpg, V, (int)C, layer, affine,
                                memory_efficient, dx, dgamma, dbeta);
      break;
    default:
      bwd_launch<__half>(x_or_y, dy, mean, rstd, gamma, beta, rows, G, cpg, V, (int)C, layer, affine, memory_efficient,
                         dx, dgamma, dbeta);
      break;
  }
}

}  // namespace dense_lanes
}  // namespace ops
}  // namespace xllm
