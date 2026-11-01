// Group RMS / layer norm, backward pass: one pass over x (or y) and dy for dx, dgamma and dbeta.
//
// Per (row, group): xhat (from x, or recovered from y when memory_efficient), dxhat = dy * gamma, the group's two
// sums (dxhat.xhat and, for the layer norm, dxhat), then dx. dgamma and dbeta are sums over every row, so each
// column-stationary warp (small groups) or CTA (wide groups) keeps the partials of the channels it owns in registers
// over its whole row chunk, writes them once to an fp32 workspace indexed by row chunk, and one finalize kernel
// sums the chunks and converts to the output dtype. Nothing reads x or dy twice, so a backward pass moves 3 tensors
// and not 5.
//
// Three mappings, chosen by channels per group:
//   small  16 <= cpg*es <= 512 bytes (VL = 1..32 vectors of 16 B): a group is a VL-lane segment of a warp, so one
//          warp loads 32 vectors = 512 B at a time and a group's sum is one shuffle reduction - no shared memory, no
//          __syncthreads. VL need not be a power of two (24/40/80-channel groups are 3/5/10 vectors).
//          Warps are column-stationary: warp (cs, rc) owns groups [cs*W, cs*W+W) of every row of chunk rc.
//   large  VL > 32: one CTA of 256 threads per (row, group), 8 warps over the group's vectors, the two sums through
//          shared memory (one round trip per row).
//   fallback: one block per (row, group), scalar, shared memory, atomics for dgamma/dbeta -- the groups wider than 512
//          vectors (cpg > 4096) that the wide path does not take. Everything the guard in GroupNormBwdOnePass declines
//          (fp32, unaligned, non-contiguous, a group narrower than 8 elements) is left to the other kernels.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

#include "bwd_one_pass.h"

namespace xllm {
namespace ops {
namespace {

// ---------------------------------------------------------------------------------------- 16-byte vectors and sums

constexpr int kVecElems = 8;      // elements of a 2-byte type in one 16-byte vector
constexpr int kWarpsPerBlock = 8;

template <typename T>
struct __align__(16) Vec8 {
  T e[kVecElems];
};

template <typename T>
struct Cvt;
template <>
struct Cvt<__nv_bfloat16> {
  static __device__ __forceinline__ float f(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 t(float v) { return __float2bfloat16(v); }
};
// Two-byte elements only: `bwd_typed` static-asserts sizeof(T) == 2, so a float caller is a build error rather than a
// silent slower path, and there is no `Cvt<float>` to keep in step with the kernels.
template <>
struct Cvt<__half> {
  static __device__ __forceinline__ float f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half t(float v) { return __float2half(v); }
};

template <typename T>
__device__ __forceinline__ Vec8<T> load8(const T* p) {
  return *reinterpret_cast<const Vec8<T>*>(p);
}
template <typename T>
__device__ __forceinline__ void store8(T* p, const Vec8<T>& v) {
  *reinterpret_cast<Vec8<T>*>(p) = v;
}
template <typename T>
__device__ __forceinline__ void to_f8(const Vec8<T>& v, float* f) {
#pragma unroll
  for (int i = 0; i < kVecElems; ++i) f[i] = Cvt<T>::f(v.e[i]);
}
template <typename T>
__device__ __forceinline__ void from_f8(const float* f, Vec8<T>& v) {
#pragma unroll
  for (int i = 0; i < kVecElems; ++i) v.e[i] = Cvt<T>::t(f[i]);
}

// Sum over all 32 lanes; every lane ends with the total.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Sum over the L lanes of this lane's segment (j = lane % L); every lane of the segment ends with the total. L need
// not be a power of two: with offset `off` a lane folds in its partner only when that partner is inside its own
// segment, which sums a segment of any length <= 32. All 32 lanes of the warp must reach the call.
__device__ __forceinline__ float seg_sum(float v, int j, int L, int lane) {
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    float o = __shfl_down_sync(0xffffffffu, v, off);
    if (off < L && j + off < L) v += o;
  }
  return __shfl_sync(0xffffffffu, v, lane - j);
}

// ---------------------------------------------------------------------------------------- small groups (VL <= 32)

// `ME` (memory_efficient) is a template parameter and not an argument: as a runtime argument it would keep the
// memory-efficient path live in every instantiation (the beta vector of 8 fp32 registers and its divide sequence) for
// the calls that never take it, and that dead state would shrink the row tile that fits in the register budget.
template <typename T, bool LN, bool AFFINE, bool ME, int U>
__global__ __launch_bounds__(256) void bwd_small_k(
    const T* __restrict__ xo, const T* __restrict__ dy, const float* __restrict__ mean,
    const float* __restrict__ rstd, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ dx,
    float* __restrict__ dg, float* __restrict__ db, int64_t rows, int G, int cpg, int L, int ncs, int nrc,
    int64_t rpc, float inv_cpg) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * kWarpsPerBlock + (int)(threadIdx.x >> 5);
  const int cs = warp % ncs;
  const int rc = warp / ncs;
  if (rc >= nrc) return;
  const int W = 32 / L;   // groups one warp covers
  const int sg = lane / L;
  const int j = lane - sg * L;
  const int64_t g0 = (int64_t)cs * W + sg;
  const bool active = (sg < W) && (g0 < G);
  const int gs = active ? (int)g0 : 0;  // a lane outside the warp's groups must not index the stats (g0 can reach
                                        // ncs*W - 1 > G - 1); its value is discarded and never reaches a live segment
  const int64_t C = (int64_t)G * cpg;
  const int64_t off = active ? g0 * cpg + (int64_t)j * kVecElems : 0;  // idle lanes read a valid address, unused
  float ga[kVecElems], ba[kVecElems];
  if (AFFINE) {
    to_f8(load8(gamma + off), ga);
    if (LN && ME) to_f8(load8(beta + off), ba);
  }
  float dga[kVecElems], dba[kVecElems];  // this lane's 8 channels, summed over its whole row chunk
#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    dga[i] = 0.f;
    dba[i] = 0.f;
  }
  const int64_t r0 = (int64_t)rc * rpc;
  int64_t r1 = r0 + rpc;
  if (r1 > rows) r1 = rows;
  // `one` is the whole per-row body. The loop below it loads `U` rows (and their rstd/mean) before any
  // reduction, so those rows' global loads overlap the first row's shuffles instead of queuing behind them -- an
  // unrolled `for` alone does not do that, because the shuffles of row r are a convergence point the compiler will not
  // hoist a load across.
  auto one = [&](Vec8<T> vx, Vec8<T> vd, float rr, float mu, int64_t p) {
    float xf[kVecElems], df[kVecElems];
    to_f8(vx, xf);
    to_f8(vd, df);
    float xh[kVecElems], dh[kVecElems];
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      float x, d;
      if (ME) {
        x = AFFINE ? (xf[i] - (LN ? ba[i] : 0.f)) / ga[i] : xf[i];
      } else {
        x = LN ? (xf[i] - mu) * rr : xf[i] * rr;
      }
      d = AFFINE ? df[i] * ga[i] : df[i];
      xh[i] = x;
      dh[i] = d;
      a += x * d;
      if (LN) b += d;
    }
    const float proj = seg_sum(a, j, L, lane) * inv_cpg;
    const float mb = LN ? seg_sum(b, j, L, lane) * inv_cpg : 0.f;
    if (active) {
      float o[kVecElems];
#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        float t = dh[i] - xh[i] * proj;
        if (LN) t -= mb;
        o[i] = t * rr;
      }
      Vec8<T> ov;
      from_f8(o, ov);
      store8(dx + p, ov);
    }
    if (AFFINE) {
#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        dga[i] += df[i] * xh[i];
        if (LN) dba[i] += df[i];
      }
    }
  };
  int64_t r = r0;
  for (; r + U <= r1; r += U) {
    Vec8<T> vx[U], vd[U];
    float rr[U], mu[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t p = (r + u) * C + off;
      vx[u] = load8(xo + p);
      vd[u] = load8(dy + p);
      rr[u] = rstd[(r + u) * G + gs];
      mu[u] = (LN && !ME) ? mean[(r + u) * G + gs] : 0.f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) one(vx[u], vd[u], rr[u], mu[u], (r + u) * C + off);
  }
  for (; r < r1; ++r) {
    const int64_t p = r * C + off;
    one(load8(xo + p), load8(dy + p), rstd[r * G + gs], (LN && !ME) ? mean[r * G + gs] : 0.f, p);
  }
  if (AFFINE && active) {
    const int64_t ch = (int64_t)rc * C + g0 * cpg + (int64_t)j * kVecElems;
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      dg[ch + i] = dga[i];
      if (LN) db[ch + i] = dba[i];
    }
  }
}

// ---------------------------------------------------------------------------------------- wide groups (VL > 32)

template <typename T, bool LN, bool AFFINE, int NC>
__global__ __launch_bounds__(256) void bwd_large_k(
    const T* __restrict__ xo, const T* __restrict__ dy, const float* __restrict__ mean,
    const float* __restrict__ rstd, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ dx,
    float* __restrict__ dg, float* __restrict__ db, int64_t rows, int G, int cpg, int VL, int64_t rpc, float inv_cpg,
    int mem_eff) {
  const int tid = threadIdx.x;  // 0..255
  const int lane = tid & 31, wid = tid >> 5;
  const int g = blockIdx.x;
  const int rc = blockIdx.y;
  const int64_t C = (int64_t)G * cpg;
  const int64_t base = (int64_t)g * cpg;
  __shared__ float sh[2][kWarpsPerBlock];
  float dga[kVecElems * NC], dba[kVecElems * NC];
#pragma unroll
  for (int i = 0; i < kVecElems * NC; ++i) {
    dga[i] = 0.f;
    dba[i] = 0.f;
  }
  const int64_t r0 = (int64_t)rc * rpc;
  int64_t r1 = r0 + rpc;
  if (r1 > rows) r1 = rows;
  for (int64_t r = r0; r < r1; ++r) {
    const int64_t rp = r * C + base;
    const float rr = rstd[r * G + g];
    const float mu = (LN && !mem_eff) ? mean[r * G + g] : 0.f;
    float xh[kVecElems * NC], dh[kVecElems * NC];
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int k = 0; k < NC; ++k) {
      const int j = tid + 256 * k;
      const int jj = j < VL ? j : VL - 1;
      const int64_t p = rp + (int64_t)jj * kVecElems;
      float xf[kVecElems], df[kVecElems];
      to_f8(load8(xo + p), xf);
      to_f8(load8(dy + p), df);
      if (j < VL) {
        float ga[kVecElems], ba[kVecElems];
        const int64_t pc = base + (int64_t)j * kVecElems;
        if (AFFINE) {
          to_f8(load8(gamma + pc), ga);
          if (LN && mem_eff) to_f8(load8(beta + pc), ba);
        }
#pragma unroll
        for (int i = 0; i < kVecElems; ++i) {
          float x, d;
          if (mem_eff) {
            x = AFFINE ? (xf[i] - (LN ? ba[i] : 0.f)) / ga[i] : xf[i];
          } else {
            x = LN ? (xf[i] - mu) * rr : xf[i] * rr;
          }
          d = AFFINE ? df[i] * ga[i] : df[i];
          xh[k * kVecElems + i] = x;
          dh[k * kVecElems + i] = d;
          a += x * d;
          if (LN) b += d;
          if (AFFINE) {
            dga[k * kVecElems + i] += df[i] * x;
            if (LN) dba[k * kVecElems + i] += df[i];
          }
        }
      }
    }
    a = warp_sum(a);
    b = warp_sum(b);
    if (lane == 0) {
      sh[0][wid] = a;
      sh[1][wid] = b;
    }
    __syncthreads();
    float A = 0.f, B = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      A += sh[0][w];
      B += sh[1][w];
    }
    __syncthreads();  // shared memory is rewritten by the next row
    const float proj = A * inv_cpg;
    const float mb = LN ? B * inv_cpg : 0.f;
#pragma unroll
    for (int k = 0; k < NC; ++k) {
      const int j = tid + 256 * k;
      if (j < VL) {
        float o[kVecElems];
#pragma unroll
        for (int i = 0; i < kVecElems; ++i) {
          float t = dh[k * kVecElems + i] - xh[k * kVecElems + i] * proj;
          if (LN) t -= mb;
          o[i] = t * rr;
        }
        Vec8<T> ov;
        from_f8(o, ov);
        store8(dx + rp + (int64_t)j * kVecElems, ov);
      }
    }
  }
  if (AFFINE) {
#pragma unroll
    for (int k = 0; k < NC; ++k) {
      const int j = tid + 256 * k;
      if (j < VL) {
        const int64_t ch = (int64_t)rc * C + base + (int64_t)j * kVecElems;
#pragma unroll
        for (int i = 0; i < kVecElems; ++i) {
          dg[ch + i] = dga[k * kVecElems + i];
          if (LN) db[ch + i] = dba[k * kVecElems + i];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------- fallback

template <typename T>
__global__ __launch_bounds__(256) void fb_bwd_k(
    const T* __restrict__ xo, const T* __restrict__ dy, const float* __restrict__ mean, const float* __restrict__ rstd,
    const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ dx, float* __restrict__ dg,
    float* __restrict__ db, int64_t G, int cpg, int is_ln, int affine, int mem_eff) {
  const int64_t gi = blockIdx.x;  // row * G + g
  const int64_t row = gi / G;
  const int g = (int)(gi - row * G);
  const int64_t C = (int64_t)G * cpg;
  const int64_t base = row * C + (int64_t)g * cpg;
  const T* xp = xo + base;
  const T* dp = dy + base;
  T* op = dx + base;
  const T* gp = affine ? gamma + (int64_t)g * cpg : nullptr;
  const T* bp = (affine && is_ln && mem_eff) ? beta + (int64_t)g * cpg : nullptr;
  const float rr = rstd[gi];
  const float mu = (is_ln && !mem_eff) ? mean[gi] : 0.f;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  __shared__ float sh[2][kWarpsPerBlock];
  float a = 0.f, b = 0.f;
  for (int i = tid; i < cpg; i += blockDim.x) {
    const float xf = Cvt<T>::f(xp[i]), df = Cvt<T>::f(dp[i]);
    float x, d;
    if (mem_eff) {
      x = affine ? (xf - (bp ? Cvt<T>::f(bp[i]) : 0.f)) / Cvt<T>::f(gp[i]) : xf;
    } else {
      x = is_ln ? (xf - mu) * rr : xf * rr;
    }
    d = affine ? df * Cvt<T>::f(gp[i]) : df;
    a += x * d;
    b += d;
  }
  a = warp_sum(a);
  b = warp_sum(b);
  if (lane == 0) {
    sh[0][wid] = a;
    sh[1][wid] = b;
  }
  __syncthreads();
  if (tid == 0) {
    float A = 0.f, B = 0.f;
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      A += sh[0][w];
      B += sh[1][w];
    }
    sh[0][0] = A;
    sh[1][0] = B;
  }
  __syncthreads();
  const float proj = sh[0][0] / (float)cpg;
  const float mb = is_ln ? sh[1][0] / (float)cpg : 0.f;
  for (int i = tid; i < cpg; i += blockDim.x) {
    const float xf = Cvt<T>::f(xp[i]), df = Cvt<T>::f(dp[i]);
    float x, d;
    if (mem_eff) {
      x = affine ? (xf - (bp ? Cvt<T>::f(bp[i]) : 0.f)) / Cvt<T>::f(gp[i]) : xf;
    } else {
      x = is_ln ? (xf - mu) * rr : xf * rr;
    }
    d = affine ? df * Cvt<T>::f(gp[i]) : df;
    float t = d - x * proj;
    if (is_ln) t -= mb;
    op[i] = Cvt<T>::t(t * rr);
    if (affine) {
      atomicAdd(dg + (int64_t)g * cpg + i, df * x);
      if (is_ln) atomicAdd(db + (int64_t)g * cpg + i, df);
    }
  }
}

// ---------------------------------------------------------------------------------------- finalize

// Sums the nrc fp32 row-chunk partials per channel and converts them to the output dtype. A block covers 32 channels
// with 8 threads on each, so the chunk range is split eight ways inside the block: for C = 4096 that is 128 blocks and
// 1024 warps, so that no block walks a whole column one dependent load at a time. That keeps the finalize at about 4 us
// on the smallest affine shapes, where it is a fixed cost. For
// `nrc < 8` -- the `cpg > 4096` fallback, and the path-2 call with `nrc = 1` -- the same launch is 8x the blocks with
// only warp 0 live, plus a barrier and 2 KB of shared memory to sum a single row of partials.
template <typename T>
__global__ __launch_bounds__(256) void bwd_finalize_k(const float* __restrict__ dg, const float* __restrict__ db,
                                                     T* __restrict__ dgamma, T* __restrict__ dbeta, int C, int nrc,
                                                     int ln) {
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;  // the warp's slice of the row chunks
  const int c = blockIdx.x * 32 + lane;
  const bool live = c < C;  // C is a multiple of 8, not always of 32: skip the work, never the barrier
  __shared__ float sh[2][kWarpsPerBlock][32];
  const int chunk = (nrc + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int lo = w * chunk;
  int hi = lo + chunk;
  if (hi > nrc) hi = nrc;
  float s = 0.f, sb = 0.f;
  if (live) {
    for (int i = lo; i < hi; ++i) {
      s += dg[(int64_t)i * C + c];
      if (ln) sb += db[(int64_t)i * C + c];
    }
  }
  sh[0][w][lane] = s;
  sh[1][w][lane] = sb;
  __syncthreads();
  if (w == 0 && live) {
    float t = sh[0][0][lane], tb = sh[1][0][lane];
#pragma unroll
    for (int k = 1; k < kWarpsPerBlock; ++k) {
      t += sh[0][k][lane];
      tb += sh[1][k][lane];
    }
    dgamma[c] = Cvt<T>::t(t);
    if (ln) dbeta[c] = Cvt<T>::t(tb);
  }
}

// ---------------------------------------------------------------------------------------- plan and launch

struct Plan {
  int path;     // 0 small, 1 wide, 2 fallback
  int VL, L, W;  // 16-byte vectors per group; lanes and groups per warp (path 0)
  int units;    // warps (path 0) or blocks (path 1) along the group axis
  int nrc;      // row chunks
  int64_t rpc;  // rows per chunk
};

// Warps in flight: 32 per SM for the segment mapping (each warp holds U rows x 2 tensors = 64 B per lane in flight),
// and for the wide mapping as many 256-thread CTAs as give ~16 resident CTAs' worth of work per SM -- a CTA per
// (row, group) already carries 8 warps and a shared-memory round trip per row.
constexpr int64_t kTargetWarpsSmall = 4736;
constexpr int64_t kTargetWarpsWide = 2368;

int64_t chunks_for(int64_t rows, int64_t units, int64_t C, int64_t target) {
  int64_t nrc = (target + units - 1) / units;
  const int64_t cap = rows > 4 ? (rows + 3) / 4 : 1;  // at least a few rows per chunk
  if (nrc > cap) nrc = cap;
  const int64_t wcap = C > 0 ? (4 << 20) / C : 1;  // keep the dgamma/dbeta partials under 32 MB
  if (nrc > wcap) nrc = wcap;
  if (nrc < 1) nrc = 1;
  if (nrc > 65535) nrc = 65535;
  return nrc;
}

Plan make_plan(int64_t rows, int64_t G, int64_t cpg, int es) {
  const int64_t C = G * cpg;
  Plan p{};
  p.path = 2;
  p.units = 1;
  p.nrc = 1;
  p.rpc = rows;
  p.VL = (int)(cpg * es / 16);
  if (p.VL >= 1 && p.VL <= 32) {
    p.path = 0;
    p.L = p.VL;
    p.W = 32 / p.VL;
    p.units = (int)((G + p.W - 1) / p.W);
  } else if (p.VL <= 512) {
    p.path = 1;
    p.units = (int)G;
  }
  if (p.path != 2) {
    const int64_t nrc = chunks_for(rows, p.units, C, p.path == 0 ? kTargetWarpsSmall : kTargetWarpsWide);
    p.nrc = (int)nrc;
    p.rpc = (rows + nrc - 1) / nrc;
  }
  return p;
}

template <typename T>
void finalize(const float* dg, const float* db, T* dgout, T* dbout, int C, int nrc, bool ln, cudaStream_t st) {
  bwd_finalize_k<T><<<(C + 31) / 32, 256, 0, st>>>(dg, db, dgout, dbout, C, nrc, ln ? 1 : 0);
}

template <typename T, bool LN, bool AFFINE>
void bwd_paths(const Plan& p, const T* xo, const T* dy, const float* mean, const float* rstd, const T* ga,
               const T* be, T* dx, T* dgout, T* dbout, float* ws, int64_t rows, int64_t G, int64_t cpg, int64_t C,
               float inv_cpg, int mem_eff, cudaStream_t st) {
  float* dg = ws;
  float* db = ws ? ws + (int64_t)p.nrc * C : nullptr;
  if (p.path == 0) {
    const int nctas = (int)((p.units * (int64_t)p.nrc + kWarpsPerBlock - 1) / kWarpsPerBlock);
// Rows in flight per warp: 5 at 8 and 16 vectors, 2 at the remaining widths (1, 2, 4, 32 and the non-power-of-two
// ones), where the idle tail lanes make the wider batch cost more than its loads win. 5 is where registers spill:
// 6 rows are about 30 % slower. `p.units >= 8` keeps the 16-vector rule off shapes with only a few groups, where the
// chunk count already saturates the GPU.
#define GN_ONE_PASS_BWD_SMALL(U, ME)                                                                                \
  bwd_small_k<T, LN, AFFINE, ME, U><<<nctas, 256, 0, st>>>(xo, dy, mean, rstd, ga, be, dx, dg, db, rows, (int)G, \
                                                           (int)cpg, p.L, p.units, p.nrc, p.rpc, inv_cpg)
    const bool me = mem_eff != 0;
    if ((p.VL == 16 && p.units >= 8) || p.VL == 8) {
      me ? GN_ONE_PASS_BWD_SMALL(5, true) : GN_ONE_PASS_BWD_SMALL(5, false);
    } else {
      me ? GN_ONE_PASS_BWD_SMALL(2, true) : GN_ONE_PASS_BWD_SMALL(2, false);
    }
#undef GN_ONE_PASS_BWD_SMALL
    if (AFFINE) finalize<T>(dg, db, dgout, dbout, (int)C, p.nrc, LN, st);
    return;
  }
  if (p.path == 1) {
    const int nc = (p.VL + 255) / 256;  // VL <= 512 here, so 1 or 2 vectors per lane
    dim3 grid((unsigned)p.units, (unsigned)p.nrc);
#define GN_ONE_PASS_BWD_LARGE(NC)                                                                                 \
  bwd_large_k<T, LN, AFFINE, NC><<<grid, 256, 0, st>>>(xo, dy, mean, rstd, ga, be, dx, dg, db, rows, (int)G, \
                                                       (int)cpg, p.VL, p.rpc, inv_cpg, mem_eff)
    if (nc <= 1) {
      GN_ONE_PASS_BWD_LARGE(1);
    } else {
      GN_ONE_PASS_BWD_LARGE(2);
    }
#undef GN_ONE_PASS_BWD_LARGE
    if (AFFINE) finalize<T>(dg, db, dgout, dbout, (int)C, p.nrc, LN, st);
    return;
  }
  fb_bwd_k<T><<<(int)(rows * G), 256, 0, st>>>(xo, dy, mean, rstd, ga, be, dx, ws, ws ? ws + C : nullptr, G, (int)cpg,
                                               LN ? 1 : 0, AFFINE ? 1 : 0, mem_eff);
  if (AFFINE) finalize<T>(ws, ws + C, dgout, dbout, (int)C, 1, LN, st);
}

template <typename T>
void bwd_typed(const Plan& p, const T* xo, const T* dy, const float* mean, const float* rstd, const T* ga,
               const T* be, T* dx, T* dgout, T* dbout, float* ws, int64_t rows, int64_t G, int64_t cpg, int64_t C,
               float inv_cpg, bool ln, bool affine, int mem_eff, cudaStream_t st) {
  static_assert(sizeof(T) == 2, "the vector paths are for 2-byte elements");
  if (ln) {
    if (affine) bwd_paths<T, true, true>(p, xo, dy, mean, rstd, ga, be, dx, dgout, dbout, ws, rows, G, cpg, C, inv_cpg, mem_eff, st);
    else bwd_paths<T, true, false>(p, xo, dy, mean, rstd, ga, be, dx, dgout, dbout, ws, rows, G, cpg, C, inv_cpg, mem_eff, st);
  } else {
    if (affine) bwd_paths<T, false, true>(p, xo, dy, mean, rstd, ga, be, dx, dgout, dbout, ws, rows, G, cpg, C, inv_cpg, mem_eff, st);
    else bwd_paths<T, false, false>(p, xo, dy, mean, rstd, ga, be, dx, dgout, dbout, ws, rows, G, cpg, C, inv_cpg, mem_eff, st);
  }
}

}  // namespace

bool GroupNormBwdOnePass(const torch::Tensor& dy, const torch::Tensor& x_or_y, const torch::Tensor& mean,
                        const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
                        int64_t channels, int64_t num_groups, bool layer, bool memory_efficient, torch::Tensor dx,
                        torch::Tensor dgamma, torch::Tensor dbeta) {
  if (num_groups < 1 || channels < 1 || num_groups > channels) return false;  // before dividing
  if (channels % num_groups != 0) return false;
  const int64_t cpg = channels / num_groups;
  const int64_t rows = x_or_y.numel() / channels;
  const bool affine = gamma.numel() > 0;
  const auto dt = x_or_y.scalar_type();
  const int es = (int)x_or_y.element_size();
  // Only what the vector paths and the fallback both handle: 2-byte elements, a whole number of 16-byte vectors, and
  // contiguous, 16-byte aligned tensors. Anything else is left to the other kernels.
  if (dt != at::kBFloat16 && dt != at::kHalf) return false;
  if ((cpg * es) % 16 != 0 || cpg < kVecElems) return false;
  if (!x_or_y.is_contiguous() || !dy.is_contiguous() || !dx.is_contiguous()) return false;
  if (!rstd.is_contiguous()) return false;
  if (affine && (!gamma.is_contiguous() || (layer && !beta.is_contiguous()))) return false;
  const intptr_t al = reinterpret_cast<intptr_t>(x_or_y.data_ptr()) | reinterpret_cast<intptr_t>(dy.data_ptr()) |
                      reinterpret_cast<intptr_t>(dx.data_ptr()) | reinterpret_cast<intptr_t>(rstd.data_ptr());
  if (al % 16 != 0) return false;
  if (affine) {
    const intptr_t ag = reinterpret_cast<intptr_t>(gamma.data_ptr()) |
                        (layer ? reinterpret_cast<intptr_t>(beta.data_ptr()) : 0);
    if (ag % 16 != 0) return false;
  }
  if (layer && !memory_efficient && mean.numel() == 0) return false;           // nothing to recover x_hat from
  if (layer && memory_efficient && affine && beta.numel() == 0) return false;  // (y - beta) / gamma needs both
  if (layer && !memory_efficient && !mean.is_contiguous()) return false;

  const Plan p = make_plan(rows, num_groups, cpg, es);
  // dgamma and dbeta partials, one fp32 row of C per row chunk (zeroed for the fallback, which adds with atomics).
  torch::Tensor ws;
  float* wsp = nullptr;
  if (affine) {
    ws = p.path == 2 ? at::zeros({2 * channels}, x_or_y.options().dtype(at::kFloat))
                     : at::empty({2 * p.nrc * channels}, x_or_y.options().dtype(at::kFloat));
    wsp = ws.data_ptr<float>();
  }
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int mem_eff = memory_efficient ? 1 : 0;
  if (dt == at::kBFloat16) {
    bwd_typed<__nv_bfloat16>(p, reinterpret_cast<const __nv_bfloat16*>(x_or_y.data_ptr()),
                             reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr()),
                             layer ? mean.data_ptr<float>() : nullptr, rstd.data_ptr<float>(),
                             affine ? reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()) : nullptr,
                             (affine && layer) ? reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()) : nullptr,
                             reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()),
                             affine ? reinterpret_cast<__nv_bfloat16*>(dgamma.data_ptr()) : nullptr,
                             (affine && layer) ? reinterpret_cast<__nv_bfloat16*>(dbeta.data_ptr()) : nullptr, wsp,
                             rows, num_groups, cpg, channels, 1.f / (float)cpg, layer, affine, mem_eff, st);
  } else {
    bwd_typed<__half>(p, reinterpret_cast<const __half*>(x_or_y.data_ptr()),
                      reinterpret_cast<const __half*>(dy.data_ptr()), layer ? mean.data_ptr<float>() : nullptr,
                      rstd.data_ptr<float>(),
                      affine ? reinterpret_cast<const __half*>(gamma.data_ptr()) : nullptr,
                      (affine && layer) ? reinterpret_cast<const __half*>(beta.data_ptr()) : nullptr,
                      reinterpret_cast<__half*>(dx.data_ptr()),
                      affine ? reinterpret_cast<__half*>(dgamma.data_ptr()) : nullptr,
                      (affine && layer) ? reinterpret_cast<__half*>(dbeta.data_ptr()) : nullptr, wsp, rows,
                      num_groups, cpg, channels, 1.f / (float)cpg, layer, affine, mem_eff, st);
  }
  return true;
}

}  // namespace ops
}  // namespace xllm
