// The backward pass of xllm's group timestep decay normalization: the adjoint of the two decaying state statistics
// (mean, var) along the sequence, of the per-step group moments, and of gamma and beta.
//
// Per row b and group k, with the forward pass's state (count c, mean m, var v), d1 = 1 - b1^c, d2 = 1 - b2^c, the
// saved statistics cm = m / d1 and cr = rstd(v / d2 + eps), and dz = dy * gamma, z = (x - cm) * cr:
//
//   dL/dcm  = -cr * sum dz                      dL/dcr = sum dz * z / cr
//   dL/dm   = dL/dcm / d1                       dL/dv  = dL/dcr * (-0.5 * cr^3 / d2)
//   m_t = b1 * m_{t-1} + (1 - b1) * mean_t      (b1 with no reset, 0 at a reset, 1 at a padded step)
//   v_t = b2 * v_{t-1} + (1 - b2) * var_t
// so the adjoint of the state is a reverse affine scan, per statistic, of (A_t, C_t) = (a_t, a_t * dL/dm_t):
//   c_{t-1} = A_t * c_t + C_t,
// with the carry entering at the last step the gradient of the final state (dmean, dvar) and the carry leaving at
// step 0 the gradient of the carried state (dprev_mean, dprev_var). Then, with dL/dmean_t = (1 - b1) * (dL/dm_t + c_t)
// (zero at a padded step) and likewise for var,
//   dx_c = dz_c * cr + P_t + Q_t * z_c,  P_t = (dL/dmean_t) / G,  Q_t = 2 * (dL/dvar_t) / (G * cr_t).
//
// Five memory-bound launches:
//   1. count   the per-row count prefix of every tile (the bias correction 1 - beta^count the backward is not given)
//   2. step    the per-step group adjoints (Am, Av) of every (row, group, step) and the affine aggregate of the tile
//   3. compose the aggregates of a chain into the carry entering each tile, and dprev_mean / dprev_var
//   4. sink    per tile: the reverse scan fused into the element pass, which writes dx and dgamma / dbeta partials
//   5. combine the dgamma / dbeta partials
// Every element of x (or y) and dy is read once and dx written once; the statistics [B, L, K] fp32 are read once.
//
// Gekko's group (G = 64, 16-bit) gets the wide path: 8 channels per lane in one 16-byte load, 8 lanes per group, so
// a warp's one load instruction covers four groups (512 contiguous bytes) and the group reduction is 3 shuffles on
// an 8-lane subgroup. A byte-in-flight bound, not a bandwidth one, sets the pace below that: one 4-byte load per
// lane per step keeps too few bytes in flight.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

#include "bwd_general.h"

namespace xllm {
namespace ops {
namespace {

constexpr int kTC = 32;        // steps per tile
constexpr int kNG = 8;         // groups per CTA on the generic path (one warp each)
constexpr int kThreads = 256;  // = 8 warps
constexpr int kMaxNT = 1024;   // largest number of tiles in one row (L <= 32768)
constexpr int kChunk = 32;     // tiles composed per warp step in the chain kernel

template <class T>
__device__ __forceinline__ float to_f(const T* p) {
  return static_cast<float>(*p);
}

template <class T>
__device__ __forceinline__ void from_f(T* p, float v) {
  *p = static_cast<T>(v);
}

// Eight channels of a group at once: one 16-byte load for the 16-bit formats.
struct F8 {
  float v0, v1, v2, v3, v4, v5, v6, v7;
};

template <class T>
__device__ __forceinline__ F8 load8(const T* p) {
  F8 r{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if constexpr (sizeof(T) == 2) {
    const uint4 w = *reinterpret_cast<const uint4*>(p);
    if constexpr (std::is_same<T, at::BFloat16>::value) {
      r.v0 = __uint_as_float((w.x & 0xffffu) << 16);
      r.v1 = __uint_as_float(w.x & 0xffff0000u);
      r.v2 = __uint_as_float((w.y & 0xffffu) << 16);
      r.v3 = __uint_as_float(w.y & 0xffff0000u);
      r.v4 = __uint_as_float((w.z & 0xffffu) << 16);
      r.v5 = __uint_as_float(w.z & 0xffff0000u);
      r.v6 = __uint_as_float((w.w & 0xffffu) << 16);
      r.v7 = __uint_as_float(w.w & 0xffff0000u);
    } else {
      r.v0 = __half2float(__ushort_as_half((unsigned short)(w.x & 0xffffu)));
      r.v1 = __half2float(__ushort_as_half((unsigned short)(w.x >> 16)));
      r.v2 = __half2float(__ushort_as_half((unsigned short)(w.y & 0xffffu)));
      r.v3 = __half2float(__ushort_as_half((unsigned short)(w.y >> 16)));
      r.v4 = __half2float(__ushort_as_half((unsigned short)(w.z & 0xffffu)));
      r.v5 = __half2float(__ushort_as_half((unsigned short)(w.z >> 16)));
      r.v6 = __half2float(__ushort_as_half((unsigned short)(w.w & 0xffffu)));
      r.v7 = __half2float(__ushort_as_half((unsigned short)(w.w >> 16)));
    }
  }
  return r;
}

__device__ __forceinline__ uint32_t pack16(float x) {
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(x));
}

template <class T>
__device__ __forceinline__ void store8(T* p, const F8& a) {
  if constexpr (sizeof(T) == 2) {
    uint4 w;
    if constexpr (std::is_same<T, at::BFloat16>::value) {
      w.x = pack16(a.v0) | (pack16(a.v1) << 16);
      w.y = pack16(a.v2) | (pack16(a.v3) << 16);
      w.z = pack16(a.v4) | (pack16(a.v5) << 16);
      w.w = pack16(a.v6) | (pack16(a.v7) << 16);
    } else {
      w.x = (uint32_t)__half_as_ushort(__float2half_rn(a.v0)) | ((uint32_t)__half_as_ushort(__float2half_rn(a.v1)) << 16);
      w.y = (uint32_t)__half_as_ushort(__float2half_rn(a.v2)) | ((uint32_t)__half_as_ushort(__float2half_rn(a.v3)) << 16);
      w.z = (uint32_t)__half_as_ushort(__float2half_rn(a.v4)) | ((uint32_t)__half_as_ushort(__float2half_rn(a.v5)) << 16);
      w.w = (uint32_t)__half_as_ushort(__float2half_rn(a.v6)) | ((uint32_t)__half_as_ushort(__float2half_rn(a.v7)) << 16);
    }
    *reinterpret_cast<uint4*>(p) = w;
  }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off, 32);
  return v;
}

// 2**x on the SFU (2 ulp): beta ** count for counts up to 1e6 and beyond.
__device__ __forceinline__ float ex2(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// ---------------------------------------------------------------------------------------------------------------
// 1. The count prefix: the count of valid steps before each tile's first step (prev_count, restarted at a bos).
__global__ void tdn_bwd_count_kernel(const bool* __restrict__ bos, const bool* __restrict__ pad,
                                     const int64_t* __restrict__ prev_count, int64_t* __restrict__ scnt, int L,
                                     int nt) {
  __shared__ int s_nv[kMaxNT], s_hr[kMaxNT], s_na[kMaxNT];
  const int b = blockIdx.x;
  int64_t c = prev_count[b];
  for (int base = 0; base < nt; base += kMaxNT) {
    const int lim = min(kMaxNT, nt - base);
    for (int i = threadIdx.x; i < lim; i += blockDim.x) {
      const int ti = base + i;
      const int t0 = ti * kTC, t1 = min(t0 + kTC, L);
      int nv = 0, hr = 0, na = 0;
      for (int t = t0; t < t1; ++t) {
        const long idx = (long)b * L + t;
        const bool v = (pad == nullptr) || !pad[idx];
        if (!v) continue;
        ++nv;
        if ((bos != nullptr) && bos[idx]) {
          hr = 1;
          na = 1;
        } else {
          ++na;
        }
      }
      s_nv[i] = nv;
      s_hr[i] = hr;
      s_na[i] = na;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i = 0; i < lim; ++i) {
        scnt[(long)b * nt + base + i] = c;
        c = s_hr[i] ? s_na[i] : c + s_nv[i];
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 2. The per-step group adjoints and the tile's affine aggregate.
template <class T>
__global__ void tdn_bwd_step_kernel(const T* __restrict__ x_or_y, const T* __restrict__ dy,
                                    const T* __restrict__ gam, const T* __restrict__ bet,
                                    const float* __restrict__ cm, const float* __restrict__ cr,
                                    const bool* __restrict__ bos, const bool* __restrict__ pad,
                                    const int64_t* __restrict__ scnt, float* __restrict__ Am, float* __restrict__ Av,
                                    float* __restrict__ aggA, float* __restrict__ aggC, float* __restrict__ aggB,
                                    float* __restrict__ aggD, int L, int H, int K, int G, int nt, float b1, float b2,
                                    float lb1, float lb2, bool me, bool vec, long nwarps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool wide = (G == 64) && vec && (H % 8 == 0);
  const int epl = (G + 31) >> 5;
  int ti, kb, b;
  if (wide) {
    const int nwq = (K + 3) >> 2;
    const long qw = (long)blockIdx.x * (blockDim.x >> 5) + warp;
    if (qw >= nwarps) return;
    kb = (int)(qw % nwq);
    const long bt = qw / nwq;
    b = (int)(bt / nt);
    ti = (int)(bt - (long)b * nt);
  } else {
    ti = blockIdx.y;
    kb = blockIdx.x;
    b = blockIdx.z;
  }
  const int t0 = ti * kTC, t1 = min(t0 + kTC, L);
  const long row = (long)b * L * H;

  // ---- Gekko's 64-channel group, 16-bit: 8 channels per lane, 8 lanes per group, 4 groups per warp.
  if (wide) {
    const int sub = lane >> 3, jj = lane & 7;
    const int g = kb * 4 + sub;
    if (g >= K) return;
    const unsigned smask = 0xffu << (sub * 8);
    const int cbase = jj * 8;
    const F8 ga = load8<T>(gam + (long)g * 64 + cbase);
    F8 ba{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (me) ba = load8<T>(bet + (long)g * 64 + cbase);
    const T* xq = x_or_y + row + (long)g * 64 + cbase;
    const T* yq = dy + row + (long)g * 64 + cbase;
    int64_t c2 = scnt[(long)b * nt + ti];
    float accA = 1.f, accC = 0.f, accB = 1.f, accD = 0.f;

    auto one = [&](int tt, const F8& xv, const F8& dv) {
      const long idx = (long)b * L + tt;
      const bool valid = (pad == nullptr) || !pad[idx];
      const bool reset = valid && (bos != nullptr) && bos[idx];
      const float crt = cr[idx * K + g];
      const float cmt = valid ? cm[idx * K + g] : 0.f;
      float s1 = 0.f, s2 = 0.f;
      if (valid) {
        if (me) {
          const float z0 = __fdividef(xv.v0 - ba.v0, ga.v0), z1 = __fdividef(xv.v1 - ba.v1, ga.v1);
          const float z2 = __fdividef(xv.v2 - ba.v2, ga.v2), z3 = __fdividef(xv.v3 - ba.v3, ga.v3);
          const float z4 = __fdividef(xv.v4 - ba.v4, ga.v4), z5 = __fdividef(xv.v5 - ba.v5, ga.v5);
          const float z6 = __fdividef(xv.v6 - ba.v6, ga.v6), z7 = __fdividef(xv.v7 - ba.v7, ga.v7);
          const float d0 = dv.v0 * ga.v0, d1 = dv.v1 * ga.v1, d2 = dv.v2 * ga.v2, d3 = dv.v3 * ga.v3;
          const float d4 = dv.v4 * ga.v4, d5 = dv.v5 * ga.v5, d6 = dv.v6 * ga.v6, d7 = dv.v7 * ga.v7;
          s1 = (d0 + d1) + (d2 + d3) + ((d4 + d5) + (d6 + d7));
          s2 = d0 * z0 + d1 * z1 + d2 * z2 + d3 * z3 + d4 * z4 + d5 * z5 + d6 * z6 + d7 * z7;
        } else {
          const float z0 = (xv.v0 - cmt) * crt, z1 = (xv.v1 - cmt) * crt, z2 = (xv.v2 - cmt) * crt;
          const float z3 = (xv.v3 - cmt) * crt, z4 = (xv.v4 - cmt) * crt, z5 = (xv.v5 - cmt) * crt;
          const float z6 = (xv.v6 - cmt) * crt, z7 = (xv.v7 - cmt) * crt;
          const float d0 = dv.v0 * ga.v0, d1 = dv.v1 * ga.v1, d2 = dv.v2 * ga.v2, d3 = dv.v3 * ga.v3;
          const float d4 = dv.v4 * ga.v4, d5 = dv.v5 * ga.v5, d6 = dv.v6 * ga.v6, d7 = dv.v7 * ga.v7;
          s1 = (d0 + d1) + (d2 + d3) + ((d4 + d5) + (d6 + d7));
          s2 = d0 * z0 + d1 * z1 + d2 * z2 + d3 * z3 + d4 * z4 + d5 * z5 + d6 * z6 + d7 * z7;
        }
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) {
        s1 += __shfl_xor_sync(smask, s1, off, 8);
        s2 += __shfl_xor_sync(smask, s2, off, 8);
      }
      float Amv = 0.f, Avv = 0.f, a1 = 1.f, a2 = 1.f;
      if (valid) {
        if (reset) c2 = 0;
        ++c2;
        const float d1f = -expm1f((float)c2 * lb1);
        const float d2f = -expm1f((float)c2 * lb2);
        Amv = (d1f > 0.f) ? -crt * s1 / d1f : 0.f;
        Avv = (d2f > 0.f) ? -0.5f * s2 * crt * crt / d2f : 0.f;
        a1 = reset ? 0.f : b1;
        a2 = reset ? 0.f : b2;
      }
      if (jj == 0) {
        const long gi = idx * K + g;
        Am[gi] = Amv;
        Av[gi] = Avv;
      }
      accC = accA * (a1 * Amv) + accC;
      accA = accA * a1;
      accD = accB * (a2 * Avv) + accD;
      accB = accB * a2;
    };

    int t = t0;
    for (; t + 1 < t1; t += 2) {
      const F8 x0 = load8<T>(xq + (long)(t + 0) * H);
      const F8 x1 = load8<T>(xq + (long)(t + 1) * H);
      const F8 d0 = load8<T>(yq + (long)(t + 0) * H);
      const F8 d1 = load8<T>(yq + (long)(t + 1) * H);
      one(t + 0, x0, d0);
      one(t + 1, x1, d1);
    }
    for (; t < t1; ++t) {
      const F8 x0 = load8<T>(xq + (long)t * H);
      const F8 d0 = load8<T>(yq + (long)t * H);
      one(t, x0, d0);
    }
    if (jj == 0) {
      const long a = ((long)b * K + g) * nt + ti;
      aggA[a] = accA;
      aggC[a] = accC;
      aggB[a] = accB;
      aggD[a] = accD;
    }
    return;
  }

  // ---- the generic path
  const int g = kb * kNG + warp;
  if (g >= K) return;
  const int cc = lane * epl;
  int64_t c = scnt[(long)b * nt + ti];
  float accA = 1.f, accC = 0.f, accB = 1.f, accD = 0.f;
  const T* gp = gam + (long)g * G + cc;
  const T* bp = bet + (long)g * G + cc;
  const T* xp = x_or_y + row + (long)g * G + cc;
  const T* yp = dy + row + (long)g * G + cc;
  float gv[8], bvv[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    gv[j] = 0.f;
    bvv[j] = 0.f;
  }
  for (int j = 0; j < epl; ++j) {
    if (cc + j < G) {
      gv[j] = to_f(gp + j);
      if (me) bvv[j] = to_f(bp + j);
    }
  }

  for (int t = t0; t < t1; t += 4) {
    float xv[4][8], dv[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (t + i < t1) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (j < epl && cc + j < G) {
            xv[i][j] = to_f(xp + (long)(t + i) * H + j);
            dv[i][j] = to_f(yp + (long)(t + i) * H + j);
          }
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (t + i >= t1) break;
      const long idx = (long)b * L + t + i;
      const bool valid = (pad == nullptr) || !pad[idx];
      const bool reset = valid && (bos != nullptr) && bos[idx];
      const float crt = cr[idx * K + g];
      const float cmt = valid ? cm[idx * K + g] : 0.f;
      float s1 = 0.f, s2 = 0.f;
      if (valid) {
        for (int j = 0; j < epl; ++j) {
          if (cc + j < G) {
            float z;
            if (me) {
              z = __fdividef(xv[i][j] - bvv[j], gv[j]);
            } else {
              z = (xv[i][j] - cmt) * crt;
            }
            const float dz = dv[i][j] * gv[j];
            s1 += dz;
            s2 += dz * z;
          }
        }
      }
      s1 = warp_sum(s1);
      s2 = warp_sum(s2);
      float Amv = 0.f, Avv = 0.f, a1 = 1.f, a2 = 1.f;
      if (valid) {
        if (reset) c = 0;
        ++c;
        const float d1f = -expm1f((float)c * lb1);
        const float d2f = -expm1f((float)c * lb2);
        const float rcp1 = (d1f > 0.f) ? 1.f / d1f : 0.f;
        const float rcp2 = (d2f > 0.f) ? 1.f / d2f : 0.f;
        Amv = -crt * s1 * rcp1;
        Avv = -0.5f * s2 * crt * crt * rcp2;
        a1 = reset ? 0.f : b1;
        a2 = reset ? 0.f : b2;
      }
      const long gi = idx * K + g;
      Am[gi] = Amv;
      Av[gi] = Avv;
      accC = accA * (a1 * Amv) + accC;
      accA = accA * a1;
      accD = accB * (a2 * Avv) + accD;
      accB = accB * a2;
    }
  }

  if (lane == 0) {
    const long a = ((long)b * K + g) * nt + ti;
    aggA[a] = accA;
    aggC[a] = accC;
    aggB[a] = accB;
    aggD[a] = accD;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 3. Compose a chain's tile aggregates into the carry entering each tile (from the end of the sequence).
template <class T>
__global__ void tdn_bwd_compose_kernel(const float* __restrict__ aggA, const float* __restrict__ aggC,
                                       const float* __restrict__ aggB, const float* __restrict__ aggD,
                                       const T* __restrict__ dmean, const T* __restrict__ dvar,
                                       float* __restrict__ stA, float* __restrict__ stB,
                                       T* __restrict__ dprev_mean, T* __restrict__ dprev_var, int K, int nt) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int b = blockIdx.x;
  const int g = blockIdx.y * kNG + warp;
  if (g >= K) return;
  const long chain = ((long)b * K + g) * nt;
  float c1 = to_f(dmean + (long)b * K + g);
  float c2 = to_f(dvar + (long)b * K + g);
  for (int hi = nt - 1; hi >= 0; hi -= kChunk) {
    const int lo = (hi - kChunk + 1 > 0) ? hi - kChunk + 1 : 0;
    const int t = lo + lane;
    const bool has = (lane < hi - lo + 1);
    const float a1 = has ? aggA[chain + t] : 1.f;
    const float b1v = has ? aggC[chain + t] : 0.f;
    const float a2 = has ? aggB[chain + t] : 1.f;
    const float b2v = has ? aggD[chain + t] : 0.f;
    for (int i = hi - lo; i >= 0; --i) {
      const float ra = __shfl_sync(0xffffffffu, a1, i);
      const float rb = __shfl_sync(0xffffffffu, b1v, i);
      const float rc = __shfl_sync(0xffffffffu, a2, i);
      const float rd = __shfl_sync(0xffffffffu, b2v, i);
      if (lane == 0) {
        const long at = chain + lo + i;
        stA[at] = c1;
        stB[at] = c2;
      }
      c1 = ra * c1 + rb;
      c2 = rc * c2 + rd;
    }
  }
  if (lane == 0) {
    from_f<T>(dprev_mean + (long)b * K + g, c1);
    from_f<T>(dprev_var + (long)b * K + g, c2);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 4. Per tile: the reverse scan fused into the element pass, which writes dx and this CTA's dgamma/dbeta partials.
template <class T>
__global__ void tdn_bwd_sink_kernel(const T* __restrict__ x_or_y, const T* __restrict__ dy, const T* __restrict__ gam,
                                    const T* __restrict__ bet, const float* __restrict__ cm,
                                    const float* __restrict__ cr, const bool* __restrict__ bos,
                                    const bool* __restrict__ pad, const float* __restrict__ Am,
                                    const float* __restrict__ Av, const float* __restrict__ stA,
                                    const float* __restrict__ stB, T* __restrict__ dx, float* __restrict__ dgp,
                                    float* __restrict__ dgp2, int L, int H, int K, int G, int nt, int nb, float b1,
                                    float b2, float og1, float og2, bool me, bool vec, long nwarps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool wide = (G == 64) && vec && (H % 8 == 0);
  const int epl = (G + 31) >> 5;
  const float invG = 1.f / (float)G;
  int ti, kb, b;
  if (wide) {
    const int nwq = (K + 3) >> 2;
    const long qw = (long)blockIdx.x * (blockDim.x >> 5) + warp;
    if (qw >= nwarps) return;
    kb = (int)(qw % nwq);
    const long bt = qw / nwq;
    b = (int)(bt / nt);
    ti = (int)(bt - (long)b * nt);
  } else {
    ti = blockIdx.y;
    kb = blockIdx.x;
    b = blockIdx.z;
  }
  const int t0 = ti * kTC, t1 = min(t0 + kTC, L);
  const long row = (long)b * L * H;

  // ---- Gekko's 64-channel group: 8 channels per lane, 8 lanes per group, 4 groups per warp, the carry chain
  //      computed by every lane of the subgroup (it is one fma per step) so nothing waits on a buffer.
  if (wide) {
    const int sub = lane >> 3, jj = lane & 7;
    const int g = kb * 4 + sub;
    if (g >= K) return;
    const unsigned smask = 0xffu << (sub * 8);
    const int cbase = jj * 8;
    const F8 ga = load8<T>(gam + (long)g * 64 + cbase);
    F8 ba{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (me) ba = load8<T>(bet + (long)g * 64 + cbase);
    const long chain = ((long)b * K + g) * nt;
    float c1 = stA[chain + ti], c2 = stB[chain + ti];
    const T* xq = x_or_y + row + (long)g * 64 + cbase;
    const T* yq = dy + row + (long)g * 64 + cbase;
    T* oq = dx + row + (long)g * 64 + cbase;
    float dga = 0.f, dgb = 0.f, dgc = 0.f, dgd = 0.f;
    float dge = 0.f, dgf = 0.f, dgg = 0.f, dgh = 0.f;
    float ega = 0.f, egb = 0.f, egc = 0.f, egd = 0.f;
    float ege = 0.f, egf = 0.f, egg = 0.f, egh = 0.f;

    auto one = [&](int tt, const F8& xv, const F8& dv) {
      const long idx = (long)b * L + tt;
      T* o = oq + (long)tt * H;
      const bool valid = (pad == nullptr) || !pad[idx];
      if (!valid) {
        const F8 z{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        store8<T>(o, z);
        return;
      }
      const bool reset = (bos != nullptr) && bos[idx];
      const float g1 = Am[idx * K + g] + c1;
      const float g2 = Av[idx * K + g] + c2;
      const float crt = cr[idx * K + g];
      // `og1 = (float)(1.0 - beta1)` comes from the host's double, not `1.f - b1`: the subtraction of two numbers of
      // size 1 carries the float beta's own rounding divided by the difference (1.3e-05 relative on beta1, 1.7e-04 on
      // beta2). These two coefficients are the adjoints' weights against `mean` and `var`, so an inaccurate pair would
      // make `dx` the gradient of a forward whose increments differ from the ones that ran; the forward uses the same
      // constants.
      const float p = og1 * g1 * invG;
      const float q = 2.f * og2 * g2 * invG / crt;
      const float cmt = cm[idx * K + g];
      F8 z;
      if (me) {
        const float z0 = __fdividef(xv.v0 - ba.v0, ga.v0), z1 = __fdividef(xv.v1 - ba.v1, ga.v1);
        const float z2 = __fdividef(xv.v2 - ba.v2, ga.v2), z3 = __fdividef(xv.v3 - ba.v3, ga.v3);
        const float z4 = __fdividef(xv.v4 - ba.v4, ga.v4), z5 = __fdividef(xv.v5 - ba.v5, ga.v5);
        const float z6 = __fdividef(xv.v6 - ba.v6, ga.v6), z7 = __fdividef(xv.v7 - ba.v7, ga.v7);
        z = F8{z0, z1, z2, z3, z4, z5, z6, z7};
        dga += dv.v0 * z0; dgb += dv.v1 * z1; dgc += dv.v2 * z2; dgd += dv.v3 * z3;
        dge += dv.v4 * z4; dgf += dv.v5 * z5; dgg += dv.v6 * z6; dgh += dv.v7 * z7;
      } else {
        const float z0 = (xv.v0 - cmt) * crt, z1 = (xv.v1 - cmt) * crt, z2 = (xv.v2 - cmt) * crt;
        const float z3 = (xv.v3 - cmt) * crt, z4 = (xv.v4 - cmt) * crt, z5 = (xv.v5 - cmt) * crt;
        const float z6 = (xv.v6 - cmt) * crt, z7 = (xv.v7 - cmt) * crt;
        z = F8{z0, z1, z2, z3, z4, z5, z6, z7};
        dga += dv.v0 * z0; dgb += dv.v1 * z1; dgc += dv.v2 * z2; dgd += dv.v3 * z3;
        dge += dv.v4 * z4; dgf += dv.v5 * z5; dgg += dv.v6 * z6; dgh += dv.v7 * z7;
      }
      ega += dv.v0; egb += dv.v1; egc += dv.v2; egd += dv.v3;
      ege += dv.v4; egf += dv.v5; egg += dv.v6; egh += dv.v7;
      // The variance's gradient is 2*(x_c - m_t)/G with m_t the *per-step mean of this group's channels*, so the term
      // is q*(z_c - zbar) with zbar = mean_c z_c = cr*(m_t - cummean_t): z alone would center on the running mean.
      float zs = (z.v0 + z.v1) + (z.v2 + z.v3) + ((z.v4 + z.v5) + (z.v6 + z.v7));
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) zs += __shfl_xor_sync(smask, zs, off, 8);
      const float zbar = zs * 0.015625f;  // 1 / 64
      const float qq = q, pz = p - qq * zbar;
      F8 out;
      out.v0 = (dv.v0 * ga.v0) * crt + pz + qq * z.v0;
      out.v1 = (dv.v1 * ga.v1) * crt + pz + qq * z.v1;
      out.v2 = (dv.v2 * ga.v2) * crt + pz + qq * z.v2;
      out.v3 = (dv.v3 * ga.v3) * crt + pz + qq * z.v3;
      out.v4 = (dv.v4 * ga.v4) * crt + pz + qq * z.v4;
      out.v5 = (dv.v5 * ga.v5) * crt + pz + qq * z.v5;
      out.v6 = (dv.v6 * ga.v6) * crt + pz + qq * z.v6;
      out.v7 = (dv.v7 * ga.v7) * crt + pz + qq * z.v7;
      store8<T>(o, out);
      c1 = reset ? 0.f : b1 * g1;
      c2 = reset ? 0.f : b2 * g2;
    };

    int t = t1 - 1;
    for (; t - 1 >= t0; t -= 2) {
      const F8 x0 = load8<T>(xq + (long)(t - 0) * H);
      const F8 x1 = load8<T>(xq + (long)(t - 1) * H);
      const F8 d0 = load8<T>(yq + (long)(t - 0) * H);
      const F8 d1 = load8<T>(yq + (long)(t - 1) * H);
      one(t - 0, x0, d0);
      one(t - 1, x1, d1);
    }
    for (; t >= t0; --t) {
      const F8 x0 = load8<T>(xq + (long)t * H);
      const F8 d0 = load8<T>(yq + (long)t * H);
      one(t, x0, d0);
    }
    const long off0 = (long)b * nt + ti;
    float* pb = dgp + off0 * H + (long)g * 64 + cbase;
    float* pb2 = dgp2 + off0 * H + (long)g * 64 + cbase;
    pb[0] = dga; pb[1] = dgb; pb[2] = dgc; pb[3] = dgd;
    pb[4] = dge; pb[5] = dgf; pb[6] = dgg; pb[7] = dgh;
    pb2[0] = ega; pb2[1] = egb; pb2[2] = egc; pb2[3] = egd;
    pb2[4] = ege; pb2[5] = egf; pb2[6] = egg; pb2[7] = egh;
    return;
  }

  // ---- the generic path: lane 0 of each warp (one group) scans into shared, then every lane does its channels
  const int g = kb * kNG + warp;
  if (g >= K) return;
  const int cc = lane * epl;
  const long chain = ((long)b * K + g) * nt;
  __shared__ float sP[kNG][kTC];
  __shared__ float sQ[kNG][kTC];
  if (lane == 0) {
    float c1 = stA[chain + ti], c2 = stB[chain + ti];
    for (int t = t1 - 1; t >= t0; --t) {
      const long idx = (long)b * L + t;
      const bool valid = (pad == nullptr) || !pad[idx];
      if (!valid) {
        sP[warp][t - t0] = 0.f;
        sQ[warp][t - t0] = 0.f;
        continue;
      }
      const bool reset = (bos != nullptr) && bos[idx];
      const float g1 = Am[idx * K + g] + c1;
      const float g2 = Av[idx * K + g] + c2;
      const float crt = cr[idx * K + g];
      sP[warp][t - t0] = og1 * g1 * invG;
      sQ[warp][t - t0] = __fdividef(2.f * og2 * g2 * invG, crt);
      c1 = reset ? 0.f : b1 * g1;
      c2 = reset ? 0.f : b2 * g2;
    }
  }
  __syncwarp();

  const T* gp = gam + (long)g * G + cc;
  const T* bp = bet + (long)g * G + cc;
  const T* xp = x_or_y + row + (long)g * G + cc;
  const T* yp = dy + row + (long)g * G + cc;
  T* op = dx + row + (long)g * G + cc;
  float gv[8], bvv[8], dgv[8], dbv[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    gv[j] = 0.f;
    bvv[j] = 0.f;
    dgv[j] = 0.f;
    dbv[j] = 0.f;
  }
  for (int j = 0; j < epl && cc + j < G; ++j) {
    gv[j] = to_f(gp + j);
    if (me) bvv[j] = to_f(bp + j);
  }
  for (int t = t0; t < t1; ++t) {
    const long idx = (long)b * L + t;
    T* o = op + (long)t * H;
    const bool valid = (pad == nullptr) || !pad[idx];
    if (!valid) {
      for (int j = 0; j < epl && cc + j < G; ++j) from_f(o + j, 0.f);
      continue;
    }
    const float crt = cr[idx * K + g];
    const float cmt = cm[idx * K + g];
    const float p = sP[warp][t - t0], q = sQ[warp][t - t0];
    float zs[8], zl = 0.f;
    const int nch = min(epl, G - cc);
    for (int j = 0; j < nch; ++j) {
      const float xj = to_f(xp + (long)t * H + j);
      zs[j] = me ? __fdividef(xj - bvv[j], gv[j]) : (xj - cmt) * crt;
      zl += zs[j];
    }
    // 2*(x_c - m_t)/G: the mean of this group's channels at this step, not the running mean cm.
    const float zbar = warp_sum(zl) / (float)G;
    const float pz = p - q * zbar;
    for (int j = 0; j < nch; ++j) {
      const float dvj = to_f(yp + (long)t * H + j);
      dgv[j] += dvj * zs[j];
      dbv[j] += dvj;
      from_f(o + j, (dvj * gv[j]) * crt + pz + q * zs[j]);
    }
  }
  const long off0 = ((long)b * nt + ti) * H + (long)g * G + cc;
  float* pb = dgp + off0;
  float* pb2 = dgp2 + off0;
  for (int j = 0; j < epl && cc + j < G; ++j) {
    pb[j] = dgv[j];
    pb2[j] = dbv[j];
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 5. Combine the dgamma / dbeta partials.
// The partials are [nb][H] where nb = B * nt. One thread a channel would leave the whole reduction in `H/256` blocks,
// too few to read the partials at bandwidth. Two levels instead: stripe the rows over blockIdx.y so the first pass has
// H/256 * 64 blocks and reads the partials at full bandwidth, then reduce the 64 stripe sums.
//
// Level 1: block (channel slice, stripe s) sums its own rows [s*S, s*S+S) and leaves the sum in the stripe's first row.
// A stripe never reads a row another stripe writes, and it writes only after it has read, so no barrier or atomics are
// needed: the rows partition, and h separates the columns. Level 2 reads the stripe rows at stride S*H.
__global__ void tdn_bwd_combine1_kernel(float* __restrict__ dgp, float* __restrict__ dgp2, int H, int nb, int S) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  const long r0 = (long)blockIdx.y * S, r1 = (r0 + S < (long)nb) ? (r0 + S) : (long)nb;
  if (r0 >= (long)nb) return;
  float s = 0.f, s2 = 0.f;
  for (long i = r0; i < r1; ++i) {
    s += dgp[i * H + h];
    s2 += dgp2[i * H + h];
  }
  dgp[r0 * H + h] = s;
  dgp2[r0 * H + h] = s2;
}

template <class T>
__global__ void tdn_bwd_combine2_kernel(const float* __restrict__ dgp, const float* __restrict__ dgp2,
                                        T* __restrict__ dgamma, T* __restrict__ dbeta, int H, int ns, long stride) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  float s = 0.f, s2 = 0.f;
  for (int i = 0; i < ns; ++i) {
    s += dgp[(long)i * stride + h];
    s2 += dgp2[(long)i * stride + h];
  }
  from_f<T>(dgamma + h, s);
  from_f<T>(dbeta + h, s2);
}

}  // namespace

void GroupTimestepDecayNormBwdGeneralCUDA(const torch::Tensor& dy, const torch::Tensor& dmean,
                                          const torch::Tensor& dvar, const torch::Tensor& x_or_y,
                                          const torch::Tensor& prev_count,
                                          const c10::optional<torch::Tensor>& bos_mask,
                                          const c10::optional<torch::Tensor>& padding_mask,
                                          const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                                          const torch::Tensor& gamma, const torch::Tensor& beta, int64_t num_groups,
                                          double beta1, double beta2, double eps, bool memory_efficient,
                                          torch::Tensor dx, torch::Tensor dprev_mean, torch::Tensor dprev_var,
                                          torch::Tensor dgamma, torch::Tensor dbeta) {
  const at::cuda::OptionalCUDAGuard guard(x_or_y.device());
  const int64_t B = x_or_y.size(0), L = x_or_y.size(1), H = x_or_y.size(2);
  const int64_t K = num_groups, G = H / K;
  TORCH_CHECK(K > 0 && G > 0 && H == K * G, "the backward needs H == num_groups * G");
  // The generic path keeps a lane's channels in 8 scalars; 256 channels a group is the widest xllm calls (32..256)
  // and leaves 8 channels a lane over the warp. A wider group would overrun those arrays, so say so.
  TORCH_CHECK(G <= 256, "the backward takes groups of at most 256 channels, got ", G);
  TORCH_CHECK(beta1 > 0.0 && beta1 < 1.0 && beta2 > 0.0 && beta2 < 1.0, "the backward needs 0 < beta < 1");
  // Any length: the steps are tiled, the count prefix and the chain compose walk the tiles in chunks, and the count
  // is int64. `eps` is taken for the interface's sake and not used: cumrstd already carries it.
  const int nt = (int)((L + kTC - 1) / kTC);
  const bool vec16 = (x_or_y.element_size() == 2);
  const bool wide = vec16 && (G == 64) && (H % 8 == 0);
  const int nkb = (int)((K + kNG - 1) / kNG);
  const int nwq = (int)((K + 3) / 4);
  const long nwarps = (long)B * nt * nwq;
  const int nblk = (int)((nwarps + (kThreads / 32) - 1) / (kThreads / 32));
  const int nb = (int)(B * nt);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x_or_y.device());
  auto opts_i = torch::TensorOptions().dtype(torch::kInt64).device(x_or_y.device());

  const bool* bos = bos_mask.has_value() && bos_mask->numel() > 0 ? bos_mask->data_ptr<bool>() : nullptr;
  const bool* pad = padding_mask.has_value() && padding_mask->numel() > 0 ? padding_mask->data_ptr<bool>() : nullptr;

  auto scnt = torch::empty({(long)B * nt}, opts_i);
  auto Am = torch::empty({(long)B * L * K}, opts);
  auto Av = torch::empty({(long)B * L * K}, opts);
  auto aggA = torch::empty({(long)B * K * nt}, opts);
  auto aggB = torch::empty({(long)B * K * nt}, opts);
  auto aggC = torch::empty({(long)B * K * nt}, opts);
  auto aggD = torch::empty({(long)B * K * nt}, opts);
  auto stA = torch::empty({(long)B * K * nt}, opts);
  auto stB = torch::empty({(long)B * K * nt}, opts);
  auto dgp = torch::empty({(long)nb * H}, opts);
  auto dgp2 = torch::empty({(long)nb * H}, opts);

  // Both constants come from the host's double and are the same constant, as in the forward: `og1` is the
  // recurrence's increment and `lb1` is the log of the decay the recurrence actually has, `1 - og1` - so the numerator
  // and the divisor of `1 - beta**count` here cannot drift apart. The bias correction is `-expm1(x)`, not `1 - 2^x`,
  // which is the cancellation-prone rearrangement.
  const float og1 = (float)(1.0 - beta1), og2 = (float)(1.0 - beta2);
  const float lb1 = log1pf(-og1), lb2 = log1pf(-og2);

  tdn_bwd_count_kernel<<<B, kThreads, 0, stream>>>(bos, pad, prev_count.data_ptr<int64_t>(),
                                                   scnt.data_ptr<int64_t>(), (int)L, nt);
  const bool vec_ok = (G % 2 == 0) && (H % 2 == 0);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x_or_y.scalar_type(), "tdn_bwd", [&] {
    const bool vec = vec_ok && (sizeof(scalar_t) == 2);
    const bool wyde = wide && vec;
    const dim3 grid = wyde ? dim3(nblk, 1, 1) : dim3(nkb, nt, B);
    const long nw = wyde ? nwarps : 0;
    tdn_bwd_step_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
        x_or_y.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(), gamma.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(),
        cummean.data_ptr<float>(), cumrstd.data_ptr<float>(), bos, pad, scnt.data_ptr<int64_t>(), Am.data_ptr<float>(),
        Av.data_ptr<float>(), aggA.data_ptr<float>(), aggC.data_ptr<float>(), aggB.data_ptr<float>(),
        aggD.data_ptr<float>(), (int)L, (int)H, (int)K, (int)G, nt, (float)beta1, (float)beta2, lb1, lb2,
        memory_efficient, vec, nw);
    dim3 cgrid(B, nkb, 1);
    tdn_bwd_compose_kernel<scalar_t><<<cgrid, kThreads, 0, stream>>>(
        aggA.data_ptr<float>(), aggC.data_ptr<float>(), aggB.data_ptr<float>(), aggD.data_ptr<float>(),
        dmean.data_ptr<scalar_t>(), dvar.data_ptr<scalar_t>(), stA.data_ptr<float>(), stB.data_ptr<float>(),
        dprev_mean.data_ptr<scalar_t>(), dprev_var.data_ptr<scalar_t>(), (int)K, nt);
    tdn_bwd_sink_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
        x_or_y.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(), gamma.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(),
        cummean.data_ptr<float>(), cumrstd.data_ptr<float>(), bos, pad, Am.data_ptr<float>(), Av.data_ptr<float>(),
        stA.data_ptr<float>(), stB.data_ptr<float>(), dx.data_ptr<scalar_t>(), dgp.data_ptr<float>(),
        dgp2.data_ptr<float>(), (int)L, (int)H, (int)K, (int)G, nt, nb, (float)beta1, (float)beta2, og1, og2,
        memory_efficient, vec, nw);
    const int threads = 256;
    const int nch = (int)((H + threads - 1) / threads);
    // Up to 64 stripes: 64 * H/256 blocks is 384 at 0.9b and 1024 at 7.1b, which is enough to read the partials at
    // bandwidth; the second level then reads 64 rows, not nb. S = ceil(nb/64) makes the last stripe partial, so the
    // number of stripes that hold rows is ceil(nb/S) and not 64: with nb = 129 that is 43 stripes of 3, and launching 64
    // would have the last 21 write past the buffer.
    const int S = (int)((nb + 63) / 64);
    const int ns = (int)((nb + S - 1) / S);
    if (S > 1) {
      tdn_bwd_combine1_kernel<<<dim3(nch, ns), threads, 0, stream>>>(dgp.data_ptr<float>(), dgp2.data_ptr<float>(),
                                                                     (int)H, nb, S);
    }
    tdn_bwd_combine2_kernel<scalar_t><<<nch, threads, 0, stream>>>(
        dgp.data_ptr<float>(), dgp2.data_ptr<float>(), dgamma.data_ptr<scalar_t>(), dbeta.data_ptr<scalar_t>(), (int)H,
        ns, (S > 1) ? (long)S * H : (long)H);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ops
}  // namespace xllm
