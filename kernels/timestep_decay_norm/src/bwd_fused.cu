// The backward pass of group timestep decay normalization (mode 2), one fused kernel over the sequence.
//
// Per row b and group k (G channels), the forward is a decayed running mean/variance with a bias correction, the
// statistics of step t feeding y at step t only. Its adjoint is therefore the same affine recurrence read
// backwards: the gradient of the state after step t obeys
//
//     R_t = alpha_{t+1} * R_{t+1} + C_t ,   R_L = dmean   (the gradient of the final state)
//
// with alpha = 1 at a padded step (the state passes through), 0 at a bos reset (the state restarts) and beta
// otherwise, and C_t the gradient step t injects into the state. A range of steps composes to
// (prod alpha, sum C prod alpha), so tiles of time are published as aggregates and scanned with a decoupled
// look-back from the last tile of the sequence to the first.
//
// Per step, with e = (x - cummean) * cumrstd (or (y - beta) / gamma when memory_efficient), de = dy * gamma and
// A = sum de, B = sum de*e, V = sum v over the group's channels, where v is x (x given) or e (from y) -- the
// per-step group mean of x is V/G in the first case and cummean + V/(G*cumrstd) in the second, and only the
// difference x - that mean enters the variance path:
//
//     dm_t = -A * cumrstd / (1 - beta1**count)      dv_t = -0.5 * B * cumrstd^2 / (1 - beta2**count)
//     dx   = de * cumrstd + (1 - beta1) * R_t / G + (1 - beta2) * 2 * S_t * (x - m_t) / G
//     dgamma += sum dy*e        dbeta += sum dy        dprev = alpha_(-1) * R_0
//
// x (or y) and dy are read twice: once for the moments, once to write dx. The second read is of the tile this CTA
// just read, so it comes out of L2 (the live tiles of the whole grid are a few tens of MiB against 126 MiB of L2).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "bwd_fused.h"

#include "bwd_prep.h"

namespace xllm {
namespace ops {
namespace {

constexpr int kWarpSize = 32;
constexpr int kBwdWarps = 8;  // one warp per group; the CTA covers kBwdWarps adjacent groups of one row
constexpr int kBwdThreads = kWarpSize * kBwdWarps;
constexpr int kBwdKG = kBwdWarps;
// Blocks in the cp.async ring: kD x 4 steps of x and dy in flight, 32 KB a CTA, no register. It is read in two places -
// the kernel's wait/index arithmetic and the launcher's `smem_vec` - so it lives at namespace scope: a kernel indexing
// more slots than the launcher allocates faults with an illegal address.
constexpr int kD = 4;
constexpr int kMaxChannelsPerLane = 8;  // groups of at most 32 * 8 = 256 channels

// ---- scalar conversions -----------------------------------------------------
__device__ __forceinline__ float tf(float x) { return x; }
__device__ __forceinline__ float tf(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ __forceinline__ float tf(const __half& x) { return __half2float(x); }
__device__ __forceinline__ void ts(float& d, float s) { d = s; }
__device__ __forceinline__ void ts(__nv_bfloat16& d, float s) { d = __float2bfloat16(s); }
__device__ __forceinline__ void ts(__half& d, float s) { d = __float2half(s); }

__device__ __forceinline__ void WarpSum3(float& a, float& b, float& c) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, off);
    b += __shfl_xor_sync(0xffffffffu, b, off);
    c += __shfl_xor_sync(0xffffffffu, c, off);
  }
}

// A flag of the look-back is published by one CTA and read by its predecessor: the payload is written before, and
// the flag with release/acquire semantics at GPU scope.
__device__ __forceinline__ int LdAcquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void StRelease(int* p, int v) {  asm volatile("st.release.gpu.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// The look-back flags carry the call's epoch in their upper 30 bits, so a flag left behind by an earlier call reads
// as unpublished (0) instead of as a predecessor's ready state. That is what lets the flags be a buffer held across
// calls rather than one zeroed by a launch: a per-call `at::empty` can hand back a block whose zeros came from a
// tensor that has since been reused, and the failure is silent and shape-dependent.
__device__ __forceinline__ int FlagTag(int epoch, int status) { return (epoch << 2) | status; }
__device__ __forceinline__ int FlagOf(int raw, int epoch) { return ((raw >> 2) == epoch) ? (raw & 3) : 0; }

// 16 B from global straight to shared memory: no register holds the data on the way, which is what lets the depth
// of a pipeline cost shared memory instead of the registers an occupancy limit is made of. The group commit and
// the wait live at the ring, one group a block; the prologue commits exactly `depth` groups and every iteration
// commits exactly one more, so a wait that leaves (depth - 1) groups outstanding retires exactly one group - the
// oldest, which is the one the loop is about to consume.
__device__ __forceinline__ void CpAsync16(void* dst, const void* src) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src) : "memory");
}

// 4 B the same way, for the prologue's fp32 statistics: the same instruction at the only other size those
// columns can use, and the same contract - the data never passes through a register, which is what keeps the register
// count down (hoisting the ring fill above these loops raises it from 74 to 80; see the note at the fill).
__device__ __forceinline__ void CpAsync4(void* dst, const void* src) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s), "l"(src) : "memory");
}

// 8 B, the size of one `s_inv` element (that column is float2 per step, so its element is 8 bytes - the copy is
// size-matched, not an overflow). The prologue uses it for prep's output, which never passes through a register
// either.
__device__ __forceinline__ void CpAsync8(void* dst, const void* src) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(s), "l"(src) : "memory");
}

// dx is written once and never read again by this kernel (or by anything soon after), while x and dy are read by
// pass 1 and then re-read by pass 2 out of the ring. `.cs` marks the line evict-first, so that a dx line entering L2
// does not evict a line pass 2 is about to want; it costs nothing when nobody reads the line. A cache-policy hint
// only: no arithmetic, no registers, no shared memory, no value changed.
__device__ __forceinline__ void StCs128(void* dst, const uint4& v) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

// dgamma/dbeta: every CTA adds its fp32 partials into fg/fb; the conversion into the outputs is
// `FinalizeParamKernel`, one launch later on the same stream. A kernel boundary on one stream orders every CTA's
// reductions before the conversion grid runs, so no CTA needs to elect itself, fence its writes or touch a counter.

// The conversion of the partials into dgamma/dbeta, as a launch of its own. The kernel boundary orders every CTA's
// REDs before this grid runs. It is launched with programmatic stream serialization, so its prologue overlaps the
// primary's last CTAs; `cudaGridDependencySynchronize()` below is the acquire that makes every primary write visible,
// which every secondary configured with programmatic dependent launch must execute.
template <typename T>
__global__ void FinalizeParamKernel(float* __restrict__ fg, float* __restrict__ fb, T* __restrict__ dgamma,
                                    T* __restrict__ dbeta, int nH) {
  cudaGridDependencySynchronize();
  for (int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x); i < nH;
       i += int(blockDim.x) * int(gridDim.x)) {
    T o1, o2;
    ts(o1, fg[i]);
    ts(o2, fb[i]);
    dgamma[i] = o1;
    dbeta[i] = o2;
    fg[i] = 0.f;  // the next call starts from clean partials without a zeroing launch of its own
    fb[i] = 0.f;
  }
}

// ---- where the per-call zeros come from -------------------------------------
// Two buffers have to start at zero before the tile kernel runs: the look-back flags (published and read inside one
// launch, so they cannot hold the previous call's values) and the dgamma/dbeta partials. No launch does it: both are
// held across calls in a per-stream pool - the flags because their tag carries the epoch, so a stale flag reads as
// unpublished, and the partials because `FinalizeParamKernel` writes each one back to zero as it converts it.
// Everything else in this file is written before it is read. The bias-correction reciprocals come from the prep
// module (`GroupTimestepDecayNormBwdPrep`, one CTA per (row, 512-step tile)): the count is a serial scan along t, so
// one CTA a row could not fill 148 SMs.
// ---- the fused backward pass ----
// ME: x_or_y holds y; otherwise x. NC: 32-channel slots per lane, G <= 32 * NC.
template <typename T, bool ME, int NC>
__global__ void __launch_bounds__(kBwdThreads) BwdKernel(
    const T* __restrict__ xs, const T* __restrict__ dys, const float* __restrict__ cms,
    const float* __restrict__ crs, const T* __restrict__ gam, const T* __restrict__ bet,
    const T* __restrict__ dms, const T* __restrict__ dvs, const float2* __restrict__ prep,
    const uint8_t* __restrict__ bos, const uint8_t* __restrict__ pad, T* __restrict__ dxs,
    float* __restrict__ fg, float* __restrict__ fb, T* __restrict__ dgamma,
    T* __restrict__ dbeta, T* __restrict__ dpm, T* __restrict__ dpv,
    float4* __restrict__ aggr, float2* __restrict__ incl, int* __restrict__ flags, int nB, int nL, int nK,
    int nG, int nH, int nt, int tileT, int nCB, int nBK, float b1, float b2, float omb1, float omb2,
    int epoch) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int idx = blockIdx.x;
  const int tileRev = idx / nCB;
  const int chB = idx - tileRev * nCB;
  const int tile = nt - 1 - tileRev;  // tiles are dispatched last-first: a tile's successor is a lower block
  const int b = chB / nBK;
  const int kg = chB - b * nBK;
  const int k = kg * kBwdKG + warp;
  const int t0 = tile * tileT;
  const int t1 = min(nL, t0 + tileT);
  const int nT = t1 - t0;
  const bool active = (warp < kBwdKG) && (k < nK);

  extern __shared__ char smem[];
  float2* s_mom = reinterpret_cast<float2*>(smem);  // [T][KG] per-step dm, dv
  float* s_mv = reinterpret_cast<float*>(s_mom + tileT * kBwdKG);  // per-step group mean of v
  float* s_cm = s_mv + tileT * kBwdKG;
  float* s_cr = s_cm + tileT * kBwdKG;
  T* s_g = reinterpret_cast<T*>(s_cr + tileT * kBwdKG);  // [KG][G], the tile's gamma/beta rows
  T* s_b = s_g + kBwdKG * nG;
  float2* s_inv = reinterpret_cast<float2*>(s_b + kBwdKG * nG);  // [T]
  uint8_t* s_pad = reinterpret_cast<uint8_t*>(s_inv + tileT);    // [T + 1]
  uint8_t* s_bos = s_pad + (tileT + 1);

  for (int i = threadIdx.x; i < kBwdKG * nG; i += kBwdThreads) {
    const int kk = i / nG, cc = i - kk * nG, ka = kg * kBwdKG + kk;
    ts(s_g[i], (ka < nK) ? tf(gam[(size_t)ka * nG + cc]) : 0.f);
    ts(s_b[i], (ka < nK) ? tf(bet[(size_t)ka * nG + cc]) : 0.f);
  }
  for (int i = threadIdx.x; i < nT; i += kBwdThreads) s_inv[i] = prep[(size_t)b * nL + t0 + i];
  for (int i = threadIdx.x; i <= nT; i += kBwdThreads) {
    const int t = t0 + i;
    s_pad[i] = (t < nL && pad) ? pad[(size_t)b * nL + t] : 0;
    s_bos[i] = (t < nL && bos) ? bos[(size_t)b * nL + t] : 0;
  }
  __syncthreads();

  if (active) {
  const int nc = (nG + kWarpSize - 1) >> 5;
  const T* gbase = s_g + warp * nG;
  const T* bbase = s_b + warp * nG;
  const int chain = b * nK + k;
  const size_t slot = (size_t)chain * nt + tile;

  // ---- pass 1: the per-step moments and the tile's aggregate ----
  float dg[NC], db[NC];
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    dg[i] = 0.f;
    db[i] = 0.f;
  }
  float AA1 = 1.f, CC1 = 0.f, AA2 = 1.f, CC2 = 0.f;
  for (int t = t1 - 1; t >= t0; --t) {
    const int tt = t - t0;
    const bool valid = s_pad[tt] == 0;
    float cmt = 0.f, crt = 0.f;
    if (valid) {
      const size_t si = (size_t)b * nL * nK + (size_t)t * nK + k;
      crt = crs[si];
      if (!ME) cmt = cms[si];
    }
    s_cm[tt * kBwdKG + warp] = cmt;
    s_cr[tt * kBwdKG + warp] = crt;

    float sA = 0.f, sB = 0.f, sC = 0.f;
    if (valid) {
      const size_t base = ((size_t)b * nL + t) * nH + (size_t)k * nG;
      const T* xp = xs + base;
      const T* yp = dys + base;
#pragma unroll
      for (int i = 0; i < NC; ++i) {
        const int c = lane + kWarpSize * i;
        if (i < nc && c < nG) {
          const float xv = tf(xp[c]);
          const float dv = tf(yp[c]);
          const float gg = tf(gbase[c]);
          const float e = ME ? (xv - tf(bbase[c])) / gg : (xv - cmt) * crt;
          const float de = dv * gg;
          sA += de;
          sB += de * e;
          sC += ME ? e : xv;
          dg[i] += dv * e;
          db[i] += dv;
        }
      }
      WarpSum3(sA, sB, sC);
    }

    float a1 = 1.f, a2 = 1.f;
    if (t + 1 < nL) {
      const int tp = tt + 1;
      if (s_pad[tp] == 0) {
        if (s_bos[tp]) {
          a1 = 0.f;
          a2 = 0.f;
        } else {
          a1 = b1;
          a2 = b2;
        }
      }
    }
    float dm = 0.f, dv = 0.f;
    if (valid) {
      const float2 inv = s_inv[tt];
      dm = -sA * crt * inv.x;
      dv = -0.5f * sB * crt * crt * inv.y;
    }
    s_mom[tt * kBwdKG + warp] = make_float2(dm, dv);
    s_mv[tt * kBwdKG + warp] = valid ? sC / float(nG) : 0.f;
    CC1 = fmaf(a1, CC1, dm);
    AA1 *= a1;
    CC2 = fmaf(a2, CC2, dv);
    AA2 *= a2;
  }

  {
    float* gp = fg + (size_t)k * nG;
    float* bp = fb + (size_t)k * nG;
#pragma unroll
    for (int i = 0; i < NC; ++i) {
      const int c = lane + kWarpSize * i;
      if (i < nc && c < nG) {
        atomicAdd(gp + c, dg[i]);
        atomicAdd(bp + c, db[i]);
      }
    }
  }
  if (lane == 0) {
    aggr[slot] = make_float4(AA1, CC1, AA2, CC2);
    StRelease(&flags[slot], FlagTag(epoch, 1));
  }

  // ---- decoupled look-back: the adjoint at this tile's first step ----
  float R1 = 0.f, R2 = 0.f;
  if (lane == 0) {
    // Compose tiles to the right until one has its inclusive prefix (flag 2), or the sequence ends (R_L = dmean).
    float ra1 = 1.f, rc1 = 0.f, ra2 = 1.f, rc2 = 0.f;
    for (int j = tile + 1;; ++j) {
      if (j >= nt) {
        R1 = fmaf(ra1, tf(dms[chain]), rc1);
        R2 = fmaf(ra2, tf(dvs[chain]), rc2);
        break;
      }
      const size_t sj = (size_t)chain * nt + j;
      const int f = FlagOf(LdAcquire(&flags[sj]), epoch);
      if (f == 0) {
        __nanosleep(32);
        --j;
      } else if (f == 2) {
        const float2 in = incl[sj];
        R1 = fmaf(ra1, in.x, rc1);
        R2 = fmaf(ra2, in.y, rc2);
        break;
      } else {
        const float4 ag = aggr[sj];
        rc1 = fmaf(ra1, ag.y, rc1);
        ra1 *= ag.x;
        rc2 = fmaf(ra2, ag.w, rc2);
        ra2 *= ag.z;
      }
    }
    const float ms1 = fmaf(AA1, R1, CC1);
    const float ms2 = fmaf(AA2, R2, CC2);
    incl[slot] = make_float2(ms1, ms2);
    StRelease(&flags[slot], FlagTag(epoch, 2));
  }
  R1 = __shfl_sync(0xffffffffu, R1, 0);
  R2 = __shfl_sync(0xffffffffu, R2, 0);

  // ---- pass 2: the local reverse scan and dx ----
  const float invG = 1.f / float(nG);
  float R = R1, S = R2;
  for (int t = t1 - 1; t >= t0; --t) {
    const int tt = t - t0;
    float a1 = 1.f, a2 = 1.f;
    if (t + 1 < nL) {
      const int tp = tt + 1;
      if (s_pad[tp] == 0) {
        if (s_bos[tp]) {
          a1 = 0.f;
          a2 = 0.f;
        } else {
          a1 = b1;
          a2 = b2;
        }
      }
    }
    const float2 mom = s_mom[tt * kBwdKG + warp];
    R = fmaf(a1, R, mom.x);
    S = fmaf(a2, S, mom.y);

    const size_t base = ((size_t)b * nL + t) * nH + (size_t)k * nG;
    T* op = dxs + base;
    if (s_pad[tt] == 0) {
      const float cmt = s_cm[tt * kBwdKG + warp];
      const float crt = s_cr[tt * kBwdKG + warp];
      const float mv = s_mv[tt * kBwdKG + warp];
      const T* xp = xs + base;
      const T* yp = dys + base;
      const float Q = ME ? omb2 * 2.f * S * invG / crt : omb2 * 2.f * S * invG;
      const float P = fmaf(-Q, mv, omb1 * R * invG);
#pragma unroll
      for (int i = 0; i < NC; ++i) {
        const int c = lane + kWarpSize * i;
        if (i < nc && c < nG) {
          const float xv = tf(xp[c]);
          const float dv = tf(yp[c]);
          const float gg = tf(gbase[c]);
          const float e = ME ? (xv - tf(bbase[c])) / gg : (xv - cmt) * crt;
          T o;
          ts(o, fmaf(dv * gg, crt, fmaf(Q, ME ? e : xv, P)));
          op[c] = o;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < NC; ++i) {
        const int c = lane + kWarpSize * i;
        if (i < nc && c < nG) {
          T o;
          ts(o, 0.f);
          op[c] = o;
        }
      }
    }
  }

  if (t0 == 0 && lane == 0) {
    float a1 = 1.f, a2 = 1.f;
    if (s_pad[0] == 0) {
      if (s_bos[0]) {
        a1 = 0.f;
        a2 = 0.f;
      } else {
        a1 = b1;
        a2 = b2;
      }
    }
    T o1, o2;
    ts(o1, a1 * R);
    ts(o2, a2 * S);
    dpm[chain] = o1;
    dpv[chain] = o2;
  }
  }  // active
  // No `cudaTriggerProgrammaticLaunchCompletion()` here, deliberately: the finalize launched after this grid gets the
  // kernel-boundary ordering it needs (that is a stream property and applies to every launch) but not the prologue
  // overlap, which requires the primary to trigger. The trigger is not free here - it moves this general path's widest
  // instantiation from 75 to 80 registers - and this kernel serves only shapes the vector kernel does not take, so
  // the overlap would not pay for the register. On `TDN_BWD_LAUNCH`'s path the trigger's implicit end-of-grid form is
  // what the programmatic launch contract falls back to.
}

// 16 bytes per lane = 8 channels: the unpack/pack of the two 16-bit dtypes.
template <typename T>
struct H8;
template <>
struct H8<__nv_bfloat16> {
  __device__ static __forceinline__ void Unpack(const uint4& r, float* o) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(p[i]);
      o[2 * i] = f.x;
      o[2 * i + 1] = f.y;
    }
  }
  __device__ static __forceinline__ uint4 Pack(const float* o) {
    uint4 r;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(o[2 * i], o[2 * i + 1]);
    return r;
  }
};
template <>
struct H8<__half> {
  __device__ static __forceinline__ void Unpack(const uint4& r, float* o) {
    const __half2* p = reinterpret_cast<const __half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(p[i]);
      o[2 * i] = f.x;
      o[2 * i + 1] = f.y;
    }
  }
  __device__ static __forceinline__ uint4 Pack(const float* o) {
    uint4 r;
    __half2* p = reinterpret_cast<__half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2half2_rn(o[2 * i], o[2 * i + 1]);
    return r;
  }
};

// ---- the fused backward pass, 16 bytes per lane ------------------------------------------------
// G == 8 * LPS: LPS lanes (a power of two, 1..32) cover one step with one 16-byte load each and a warp's load
// covers SPB = 32 / LPS steps of the group -- 8x the bytes in flight of a per-channel load, and the group
// reduction over G channels becomes log2(LPS) shfls on the LPS lanes of a step's subgroup.
// Lane = step_in_block * LPS + channel_block, channels [channel_block * 8, +8).
template <typename T, bool ME, int LPS>
__global__ void __launch_bounds__(kBwdThreads) BwdVecKernel(
    const T* __restrict__ xs, const T* __restrict__ dys, const float* __restrict__ cms,
    const float* __restrict__ crs, const T* __restrict__ gam, const T* __restrict__ bet,
    const T* __restrict__ dms, const T* __restrict__ dvs, const float2* __restrict__ prep,
    const uint8_t* __restrict__ bos, const uint8_t* __restrict__ pad, T* __restrict__ dxs,
    float* __restrict__ fg, float* __restrict__ fb, T* __restrict__ dgamma,
    T* __restrict__ dbeta, T* __restrict__ dpm, T* __restrict__ dpv, float4* __restrict__ aggr,
    float2* __restrict__ incl, int* __restrict__ flags, int nB, int nL, int nK, int nG, int nH, int nt,
    int tileT, int nCB, int nBK, float b1, float b2, float omb1, float omb2, int epoch) {
  constexpr int SPB = kWarpSize / LPS;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int sIdx = lane / LPS;   // the step this lane holds, in a block of SPB
  const int cIdx = lane - sIdx * LPS;
  const int idx = blockIdx.x;
  const int tileRev = idx / nCB;
  const int chB = idx - tileRev * nCB;
  const int tile = nt - 1 - tileRev;
  const int b = chB / nBK;
  const int kg = chB - b * nBK;
  const int k = kg * kBwdKG + warp;
  const int t0 = tile * tileT;
  const int t1 = min(nL, t0 + tileT);
  const int nT = t1 - t0;
  const int nblk = (nT + SPB - 1) / SPB;
  const bool active = k < nK;

  extern __shared__ char smem[];
  float2* s_mom = reinterpret_cast<float2*>(smem);  // [T][KG] per-step dm, dv
  float* s_mv = reinterpret_cast<float*>(s_mom + tileT * kBwdKG);
  float* s_cm = s_mv + tileT * kBwdKG;
  float* s_cr = s_cm + tileT * kBwdKG;
  T* s_g = reinterpret_cast<T*>(s_cr + tileT * kBwdKG);
  T* s_b = s_g + kBwdKG * nG;
  float2* s_inv = reinterpret_cast<float2*>(s_b + kBwdKG * nG);
  float2* s_a = s_inv + tileT;  // the scan's per-step decay factor (a1, a2), one branch-free 8 B load a step
  uint8_t* s_pad = reinterpret_cast<uint8_t*>(s_a + tileT + 1);
  uint8_t* s_bos = s_pad + (tileT + 1);

  // gamma/beta as copies rather than through a register: the same instruction as the statistics' staging, at the
  // only other size these columns can take. A pair is copied only when both of its elements land below `nK`, so the
  // guard is per element and not per pair - `K % kBwdKG != 0` is a shape the interface admits and the over-read it
  // would cause is global. `nG` is a multiple of two on every shape the interface admits; the one-element tail below
  // exists so that an odd `nGam` cannot become an over-read.
  const int nGam = kBwdKG * nG;
  for (int ip = threadIdx.x; ip < (nGam >> 1); ip += kBwdThreads) {
    const int i = 2 * ip;
    const int kk0 = i / nG, cc0 = i - kk0 * nG, ka0 = kg * kBwdKG + kk0;
    const int j = i + 1;
    const int ka1 = kg * kBwdKG + j / nG;
    if (ka1 < nK) {
      CpAsync4(&s_g[i], gam + (size_t)ka0 * nG + cc0);
      if (ME) CpAsync4(&s_b[i], bet + (size_t)ka0 * nG + cc0);
    } else {
      ts(s_g[i], (ka0 < nK) ? tf(gam[(size_t)ka0 * nG + cc0]) : 1.f);
      ts(s_g[j], 1.f);
      if (ME) { ts(s_b[i], (ka0 < nK) ? tf(bet[(size_t)ka0 * nG + cc0]) : 0.f); ts(s_b[j], 0.f); }
    }
  }
  for (int i = 2 * (nGam >> 1); i < nGam; i += kBwdThreads) {  // the odd-`nGam` tail: never taken today
    const int kk = i / nG, cc = i - kk * nG, ka = kg * kBwdKG + kk;
    ts(s_g[i], (ka < nK) ? tf(gam[(size_t)ka * nG + cc]) : 1.f);
    if (ME) ts(s_b[i], (ka < nK) ? tf(bet[(size_t)ka * nG + cc]) : 0.f);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  for (int i = threadIdx.x; i <= nT; i += kBwdThreads) {
    const int t = t0 + i;
    s_pad[i] = (t < nL && pad) ? pad[(size_t)b * nL + t] : 0;
    s_bos[i] = (t < nL && bos) ? bos[(size_t)b * nL + t] : 0;
  }
  __syncthreads();

  // The scan's decay factor is a function of the masks only, and both places a warp walks the tile (the tile
  // aggregate and pass 2's scan) need it per step. Staged once here it is one aligned 8 B load a step, no branch.
  for (int i = threadIdx.x; i <= nT; i += kBwdThreads) {
    float2 a = make_float2(1.f, 1.f);
    if (t0 + i < nL && s_pad[i] == 0) a = s_bos[i] ? make_float2(0.f, 0.f) : make_float2(b1, b2);
    s_a[i] = a;
  }
  // The tile's statistics into shared, coalesced and once: the moment loop's chain then holds no global load
  // besides x and dy, so a block's stall is the one round trip of its own two loads and not a second one.
  // Issued as copies rather than as load-then-store. This loop is the prologue's heaviest staging phase and the
  // only one that runs more than twice a thread: each iteration's shared store would wait on its own strided global
  // read, so the rounds would serialise. As `cp.async` nothing is held in a register on the way and every round is in
  // flight at once; the group is committed here and drained after the prep copy below, so the copies overlap the grid
  // dependency wait and prep's own staging instead of being waited for on the spot.
  for (int i = threadIdx.x; i < nT * kBwdKG; i += kBwdThreads) {
    const int tt = i / kBwdKG, kk = i - tt * kBwdKG, ka = kg * kBwdKG + kk;
    if (ka < nK && s_pad[tt] == 0) {
      const size_t si = (((size_t)b * nL + (t0 + tt)) * nK) + ka;
      CpAsync4(&s_cr[i], crs + si);
      if (ME) {
        s_cm[i] = 0.f;  // the -me edge recovers the mean from y; the staged column stays zero, as before
      } else {
        CpAsync4(&s_cm[i], cms + si);
      }
    } else {
      s_cr[i] = 0.f;
      s_cm[i] = 0.f;
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  // prep's output is the last thing this prologue reads and the only thing it reads that another kernel wrote. The vec
  // launch carries `cudaLaunchAttributeProgrammaticStreamSerialization`, so this grid is resident from the moment prep
  // has launched all of its blocks and the staging above it runs while prep still works; this call releases once prep
  // has completed and flushed, which is what makes the reads below safe - the overlap is bounded by prep's end, not by
  // an earlier trigger (there is no `griddepcontrol.launch_dependents` in the prep kernel). Every thread executes this
  // - the loop below is indexed by threadIdx.x, and a warp that skipped the wait would read a partially written
  // buffer. Without the attribute (and under capture) this is a no-op.
  cudaGridDependencySynchronize();
  // One `s_inv` ELEMENT is 8 bytes: that column is float2 per step, so this copy is size-matched, not an overflow.
  for (int i = threadIdx.x; i < nT; i += kBwdThreads) CpAsync8(&s_inv[i], prep + (size_t)b * nL + t0 + i);
  asm volatile("cp.async.commit_group;" ::: "memory");
  // Drain every group the prologue has committed - phase 1's gamma/beta, phase 4's statistics and phase 6's prep
  // output all retire here, and the condition is `0` rather than a count so it cannot matter how many there are.
  // Nothing of the ring is below this line yet, so the ring's own `wait_group kD-1` never counts a prologue group.
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

  // The ring's shared memory, one slot a block a warp a tensor: 32 uint4 a slot (a lane's 16 B), 2 x kD x 32 a warp.
  // kD (the cp.async ring's depth) is a namespace-scope constant beside kBwdKG: the launcher's smem_vec needs it too.

  // The ring's base must keep its shared-memory provenance: built from a `uintptr_t`, every ring read would compile to
  // a generic LD.E.128 plus a 64-bit IMAD.WIDE instead of an LDS.128, which is ~8 lane-instructions an element over
  // both passes of a kernel whose limiter is the dependent chain. The alignment is computed as an offset and added to
  // a shared pointer, so the provenance survives.
  char* ring_tail = reinterpret_cast<char*>(s_bos + (tileT + 1));
  ring_tail += (16u - (reinterpret_cast<uintptr_t>(ring_tail) & 15u)) & 15u;
  uint4* const s_ring = reinterpret_cast<uint4*>(ring_tail);
  // The fill sits here and not above the prologue: hoisted above the staging loops it extends obase/rgy/rgx across the
  // whole prologue, which raises the registers from 74 to 80 with 8 bytes of stack on both -me instantiations. One
  // CTA's staging is a few hundred ns of DRAM latency, so the gain would be small; the register cliff decides it.
  int obase = (((int)b * nL + t0 + (nblk - 1) * SPB + sIdx) * nH) + (int)k * nG + cIdx * 8;
  const int obstep = SPB * nH;
  // The ring, per warp: kD slots, a slot a lane's 16 B of x and of dy for its step of the block.
  uint4* const rgy = s_ring + (size_t)warp * (2 * kD * 32);
  uint4* const rgx = rgy + kD * 32;
  auto issue = [&](const int nb, const int d) {
    uint4* const dx16 = rgx + d * 32 + lane;
    uint4* const dy16 = rgy + d * 32 + lane;
    const int tt = nb * SPB + sIdx;
    if (nb >= 0 && tt < nT) {
      // The refill address is one integer op off the same 32-bit offset the pass steps: no chain.
      const int off = obase - (nblk - 1 - nb) * obstep;
      CpAsync16(dx16, xs + off);
      CpAsync16(dy16, dys + off);
    } else if (nb >= 0) {
      *dx16 = make_uint4(0, 0, 0, 0);
      *dy16 = make_uint4(0, 0, 0, 0);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  // The guard is needed for correctness: `active` is `k < nK`, and a warp with `k >= nK` carries `(int)k * nG >= H` in
  // obase, so an unguarded `issue` would cp.async from past the row - and past the tensor for the last row's last
  // tile (K % 8 != 0, e.g. K = 1, 4, 100). It also keeps the invariant that no committed cp.async group outlives a
  // warp.
  if (active) {
#pragma unroll
    for (int d = 0; d < kD; ++d) issue(nblk - 1 - d, d);
  }
  if (active) {
  const int chain = b * nK + k;
  const size_t slot = (size_t)chain * nt + tile;
  const uint4 g8 = *reinterpret_cast<const uint4*>(s_g + warp * nG + cIdx * 8);
  float gf[8];
  H8<T>::Unpack(g8, gf);
  // `beta` is not unpacked into registers on the -me edge: it is read from the staging array s_b at the one line that
  // uses it (`e = (y - beta) * rcp(g)`), once in each pass. Keeping it out of registers takes the instantiation from 80
  // registers to 64 (3 CTAs to 4 an SM) on bf16 and on half. Reading both `gf` and `bf` from the staging arrays per
  // element (two shared loads a cell) also reaches 64 registers but is slower; one shared load a cell, with `gf` kept
  // in registers and only `beta` read from s_b, is the form that wins.
  const T* bv = s_b + warp * nG + cIdx * 8;

  // ---- pass 1: the per-step moments and the tile's aggregate ----
  float dg[8], db[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    dg[i] = 0.f;
    db[i] = 0.f;
  }
  // 32-bit element offsets, stepped by a constant: a third to a half of this pass's instructions would be integer
  // address arithmetic otherwise, mostly the 64-bit `((size_t)b*nL + t)*nH` chain rebuilt per step per tensor. One
  // 32-bit offset stepped by SPB*nH a block is one integer op a block, and it costs no register where a
  // pointer per tensor costs four. The host declines any shape with B*L*H >= 2^31. `obstep = SPB * nH` and the
  // ring's `obase - (nblk-1-nb) * obstep` are ints outside that expression; both are bounded by SPB * H = 32 * 65536
  // at the largest product, so they cannot overflow either.
  // nG is a power of two on every fast-path shape, so `1/nG` is exact: the per-step epilogue multiplies by it instead
  // of dividing with IEEE precision (12 issue slots a step inside a 4-lane branch), which removes a division, not a
  // rounding.
  const float kInvG = 1.f / float(nG);
#pragma unroll 1
  for (int blk = nblk - 1; blk >= 0; --blk) {
    const int d = (nblk - 1 - blk) & (kD - 1);
    // `kD - 1` pending, not `kD - 2`: the ring has kD slots and the prologue above committed exactly kD groups, so the
    // group this iteration reads is the oldest of them and leaving kD-1 pending completes exactly that one. It is one
    // more block of lead for the copy, at no cost in bytes, registers or code motion. The invariant it rests on is the
    // prologue's `for (d = 0; d < kD; ++d) issue(...)`: shorten that loop and this becomes a read before its group has
    // arrived. Do not move this call above the refill either - the refill writes the slot this iteration reads (`d`),
    // so it has to stay after the ring payload has been read from shared.
    asm volatile("cp.async.wait_group %0;" ::"n"(kD - 1) : "memory");
    const int tt = blk * SPB + sIdx;
    const int t = t0 + tt;
    const bool inRange = tt < nT;
    // One 32-bit mask load a block and a shuffle, instead of an LDS.U8 a step. `SPB == 4`, so `blk * SPB` is 4-byte
    // aligned inside `s_pad` (8-byte aligned by construction, it follows the float2 arrays); the tail load may read up
    // to 3 bytes into `s_bos`, and it is the `inRange` test - not the buffer - that makes those bytes irrelevant.
    static_assert(kWarpSize / LPS == 4, "the mask load reads one unsigned of s_pad a block: SPB must be 4");
    unsigned pmask = 0u;
    if (lane == 0) pmask = *reinterpret_cast<const unsigned*>(s_pad + blk * SPB);
    pmask = __shfl_sync(0xffffffffu, pmask, 0);
    const bool valid = inRange && (((pmask >> (8 * sIdx)) & 0xffu) == 0u);
    // The ring's payload for this block, from shared: the lane's 16 B of x and of dy.
    const uint4 xr = rgx[d * 32 + lane];
    const uint4 dr = rgy[d * 32 + lane];
    // (the ring refill stays at the end of the body: hoisting it beside these two loads is slower.)
    float cmt = s_cm[(inRange ? tt : 0) * kBwdKG + warp], crt = s_cr[(inRange ? tt : 0) * kBwdKG + warp];
    const float nc = -cmt * crt;  // the x-given recovery's constant term: e = fmaf(x, crt, nc) is one op
    float su = 0.f, sv = 0.f, sw = 0.f;
    if (valid) {
      float xf[8], df[8];
      H8<T>::Unpack(xr, xf);
      H8<T>::Unpack(dr, df);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        // `__fdividef` compiles to MUFU.RCP + FSETP.GEU + FSEL + FMUL (+FCHK a step): a range check this edge does
        // not need, since gf comes from the gamma tensor and is a bf16 normal. The bare reciprocal is one MUFU and a
        // multiply, with no new register (the reciprocal is transient in the element body) and no memory op; carrying
        // the reciprocal in a shared array or in eight live floats is slower.
        float rgf = 0.f;
        if constexpr (ME) {  // `asm volatile` cannot be removed, so on the x-given instantiation this must not exist:
                             // unguarded it would cost 8 dead MUFU.RCP a step on every x-given row
          asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(rgf) : "f"(gf[j]));
        }
        const float e = ME ? (xf[j] - tf(bv[j])) * rgf : fmaf(xf[j], crt, nc);
        const float de = df[j] * gf[j];
        su += de;
        sv += de * e;
        sw += ME ? e : xf[j];
        dg[j] += df[j] * e;
        db[j] += df[j];
      }
    }
#pragma unroll
    for (int off = 1; off < LPS; off <<= 1) {
      su += __shfl_xor_sync(0xffffffffu, su, off);
      sv += __shfl_xor_sync(0xffffffffu, sv, off);
      sw += __shfl_xor_sync(0xffffffffu, sw, off);
    }
    float dm = 0.f, dv = 0.f;
    if (valid) {
      const float2 inv = s_inv[tt];
      dm = -su * crt * inv.x;
      dv = -0.5f * sv * crt * crt * inv.y;
    }
    if (inRange && cIdx == 0) {
      s_mom[tt * kBwdKG + warp] = make_float2(dm, dv);
      s_mv[tt * kBwdKG + warp] = valid ? sw * kInvG : 0.f;
    }
    issue(blk - kD, d);  // refill this slot with the block kD behind: the ring stays full to the end
  }
  asm volatile("cp.async.wait_all;" ::: "memory");  // nothing reads the ring after this, but no group outlives the CTA
  __syncwarp();

  // the tile's aggregate: every lane composes the whole tile from the shared per-step moments (same chain, no shfl).
  float AA1 = 1.f, CC1 = 0.f, AA2 = 1.f, CC2 = 0.f;
  for (int tt = nT - 1; tt >= 0; --tt) {
    const float2 a = s_a[tt + 1];
    const float2 mom = s_mom[tt * kBwdKG + warp];
    CC1 = fmaf(a.x, CC1, mom.x);
    AA1 *= a.x;
    CC2 = fmaf(a.y, CC2, mom.y);
    AA2 *= a.y;
  }

  // dgamma/dbeta of this tile: this lane's 8 channels over its step group, then the step groups of the warp
#pragma unroll
  for (int off = LPS; off < kWarpSize; off <<= 1) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      dg[j] += __shfl_xor_sync(0xffffffffu, dg[j], off);
      db[j] += __shfl_xor_sync(0xffffffffu, db[j], off);
    }
  }
  if (sIdx == 0) {
    float* gp = fg + (size_t)k * nG + cIdx * 8;
    float* bp = fb + (size_t)k * nG + cIdx * 8;
    // The aligned v4 path is taken by both instantiations, with the scalar fallback kept for a misaligned pair. The
    // packed reduction (2 v4 reductions instead of 16 scalars) reassociates the `dgamma`/`dbeta` sums, so these sums
    // are not bit-stable from call to call; `dx` and the four atomic outputs are not either.
    //
    // `fg` is rounded to a whole number of float4s, `4*k*nG` is 256k because `nG` is pinned at 64 by the `G == 64`
    // entry condition, and `32*cIdx` is a multiple of 32 - so both pointers are 16-byte aligned on every shape the
    // interface admits. The test is on the two addresses the asm is about to write and not on a proxy for them, so
    // it cannot go stale if the entry condition widens; the quantity that binds the offset is the flag count.
    if ((((uintptr_t)gp | (uintptr_t)bp) & 15u) == 0) {
#pragma unroll
      for (int j = 0; j < 8; j += 4) {
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(gp + j),
                     "f"(dg[j]), "f"(dg[j + 1]), "f"(dg[j + 2]), "f"(dg[j + 3]) : "memory");
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(bp + j),
                     "f"(db[j]), "f"(db[j + 1]), "f"(db[j + 2]), "f"(db[j + 3]) : "memory");
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        atomicAdd(gp + j, dg[j]);
        atomicAdd(bp + j, db[j]);
      }
    }
  }

  if (lane == 0) {
    aggr[slot] = make_float4(AA1, CC1, AA2, CC2);
    StRelease(&flags[slot], FlagTag(epoch, 1));
  }

  // ---- decoupled look-back: the adjoint at this tile's first step ----
  // A whole warp polls a window of 32 predecessors: their flags, aggregates and inclusives are three independent
  // loads, so a window of 32 tiles costs one memory round trip instead of the one per tile of a serial walk.
  float R1 = 0.f, R2 = 0.f;
  {
    float ra1 = 1.f, rc1 = 0.f, ra2 = 1.f, rc2 = 0.f;
    const float dm0 = tf(dms[chain]), dv0 = tf(dvs[chain]);
    for (int w = tile + 1;; w += kWarpSize) {
      const int j = w + lane;
      const bool inChain = j < nt;
      const size_t sj = (size_t)chain * nt + j;
      const int f = inChain ? FlagOf(LdAcquire(&flags[sj]), epoch) : 2;  // past the chain's end, dmean is the inclusive
      const float4 ag = inChain ? aggr[sj] : make_float4(1.f, 0.f, 1.f, 0.f);
      const float2 inS = inChain ? incl[sj] : make_float2(dm0, dv0);
      const unsigned mk2 = __ballot_sync(0xffffffffu, f == 2);
      const unsigned mk0 = __ballot_sync(0xffffffffu, f == 0);
      const int p = mk2 ? (__ffs(mk2) - 1) : (kWarpSize - 1);
      if (mk0 && (mk2 == 0u || (mk0 & ((1u << p) - 1u)) != 0u)) {
        __nanosleep(32);  // a predecessor nearer than the nearest inclusive has not published yet
        w -= kWarpSize;
        continue;
      }
      // Every lane composes the same window: the partials nearer than the inclusive, then the inclusive itself.
      for (int i = 0; i <= p; ++i) {
        if (i < p) {
          rc1 = fmaf(ra1, __shfl_sync(0xffffffffu, ag.y, i), rc1);
          ra1 *= __shfl_sync(0xffffffffu, ag.x, i);
          rc2 = fmaf(ra2, __shfl_sync(0xffffffffu, ag.w, i), rc2);
          ra2 *= __shfl_sync(0xffffffffu, ag.z, i);
        } else if (mk2) {
          R1 = fmaf(ra1, __shfl_sync(0xffffffffu, inS.x, p), rc1);
          R2 = fmaf(ra2, __shfl_sync(0xffffffffu, inS.y, p), rc2);
        } else {
          rc1 = fmaf(ra1, __shfl_sync(0xffffffffu, ag.y, p), rc1);
          ra1 *= __shfl_sync(0xffffffffu, ag.x, p);
          rc2 = fmaf(ra2, __shfl_sync(0xffffffffu, ag.w, p), rc2);
          ra2 *= __shfl_sync(0xffffffffu, ag.z, p);
        }
      }
      if (mk2) break;
    }
    if (lane == 0) {
      incl[slot] = make_float2(fmaf(AA1, R1, CC1), fmaf(AA2, R2, CC2));
      StRelease(&flags[slot], FlagTag(epoch, 2));
    }
  }

  // ---- pass 2: the local reverse scan and dx ----
  const float invG = 1.f / float(nG);
  float R = R1, S = R2;
  int obase2 = (((int)b * nL + t0 + (nblk - 1) * SPB + sIdx) * nH) + (int)k * nG + cIdx * 8;
  // Pass 2 reads x and dy again through the same ring, so every block of output does not pay its own dependent round
  // trip for two tensors. The ring's slots are free by now - pass 1 waits all of its groups before it publishes its
  // aggregate - so the same ring carries pass 2 at no extra shared memory. The cost of the re-read is the L2 round
  // trip into a register, which only an async copy hides; a prefetch.global.L2 a few blocks ahead hides much less.
  auto issue2 = [&](const int nb, const int d) {
    uint4* const px = rgx + d * 32 + lane;
    uint4* const py = rgy + d * 32 + lane;
    const int ttb = nb * SPB + sIdx;
    if (nb >= 0 && ttb < nT) {
      const size_t pb = ((size_t)b * nL + (t0 + ttb)) * nH + (size_t)k * nG + cIdx * 8;
      CpAsync16(px, xs + pb);
      CpAsync16(py, dys + pb);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
#pragma unroll
  for (int d = 0; d < kD; ++d) issue2(nblk - 1 - d, d);
  for (int blk = nblk - 1; blk >= 0; --blk, obase2 -= obstep) {
    const int rd = (nblk - 1 - blk) & (kD - 1);
    // As in pass 1: kD-1 pending completes exactly the group this iteration reads (the oldest of the kD the prologue
    // committed), giving the copy one more block of lead. Same invariant, same warning about the prologue's loop.
    asm volatile("cp.async.wait_group %0;" ::"n"(kD - 1) : "memory");
    float ro = 0.f, so = 0.f;
    for (int g = SPB - 1; g >= 0; --g) {
      const int tt = blk * SPB + g;
      if (tt < nT) {
        const float2 a = s_a[tt + 1];
        const float2 mom = s_mom[tt * kBwdKG + warp];
        R = fmaf(a.x, R, mom.x);
        S = fmaf(a.y, S, mom.y);
        if (g == sIdx) {
          ro = R;
          so = S;
        }
      }
    }
    const int tt = blk * SPB + sIdx;
    unsigned pmask2 = 0u;  // the same 32-bit mask load and shuffle as pass 1
    if (lane == 0) pmask2 = *reinterpret_cast<const unsigned*>(s_pad + blk * SPB);
    pmask2 = __shfl_sync(0xffffffffu, pmask2, 0);
    if (tt < nT) {
      const int base = obase2;
      if (((pmask2 >> (8 * sIdx)) & 0xffu) == 0u) {
        const float cmt = s_cm[tt * kBwdKG + warp];
        const float crt = s_cr[tt * kBwdKG + warp];
        const float mv = s_mv[tt * kBwdKG + warp];
        const float nc = -cmt * crt;  // the x-given recovery's constant term, once a step
        const float Q = ME ? omb2 * 2.f * so * invG / crt : omb2 * 2.f * so * invG;
        const float P = fmaf(-Q, mv, omb1 * ro * invG);
        const uint4 xr = rgx[rd * 32 + lane];
        const uint4 dr = rgy[rd * 32 + lane];
        float xf[8], df[8], of[8];
        H8<T>::Unpack(xr, xf);
        H8<T>::Unpack(dr, df);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float rgf2 = 0.f;
          if constexpr (ME) {  // as in pass 1: the asm must not exist on the x-given instantiation
            asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(rgf2) : "f"(gf[j]));
          }
          const float e = ME ? (xf[j] - tf(bv[j])) * rgf2 : fmaf(xf[j], crt, nc);
          of[j] = fmaf(df[j] * gf[j], crt, fmaf(Q, ME ? e : xf[j], P));
        }
        const uint4 ov = H8<T>::Pack(of);
        StCs128(dxs + base, ov);
      } else {
        StCs128(dxs + base, make_uint4(0, 0, 0, 0));
      }
    }
    issue2(blk - kD, rd);  // refill this slot with the block kD behind: the ring stays full to the end
  }
  asm volatile("cp.async.wait_all;" ::: "memory");

  if (t0 == 0 && lane == 0) {
    float a1 = 1.f, a2 = 1.f;
    if (s_pad[0] == 0) {
      if (s_bos[0]) {
        a1 = 0.f;
        a2 = 0.f;
      } else {
        a1 = b1;
        a2 = b2;
      }
    }
    T o1, o2;
    ts(o1, a1 * R);
    ts(o2, a2 * S);
    dpm[chain] = o1;
    dpv[chain] = o2;
  }
  }  // active
  // The partials are converted by a launch of their own (FinalizeParamKernel), so no CTA elects itself, fences its
  // own writes or touches a grid counter. The trigger lets that launch's prologue overlap this grid's last CTAs; the
  // finalize's own cudaGridDependencySynchronize() waits for every write made here. BwdVecKernel has exactly one
  // exit, so no branch skips the trigger.
  cudaTriggerProgrammaticLaunchCompletion();
}

int EnvInt(const char* name, int dflt) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return dflt;
  const int got = std::atoi(v);
  return got > 0 ? got : dflt;
}

// Launches the conversion grid with programmatic stream serialization, so its prologue overlaps the primary's last
// CTAs. The attribute is set unconditionally: every path that reaches here has already declined under stream capture
// (the entry's `cudaStreamIsCapturing` guard returns false before any launch), so a secondary that executes
// `cudaGridDependencySynchronize()` is always paired with a launch that carried the attribute.
template <typename TT>
static void LaunchFinalizeKernel(float* fg, float* fb, torch::Tensor& dgamma, torch::Tensor& dbeta, int nH,
                                 cudaStream_t stream) {
  auto kfn = &FinalizeParamKernel<TT>;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(unsigned((nH + 255) / 256));
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  AT_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kfn, fg, fb, reinterpret_cast<TT*>(dgamma.data_ptr()),
                                   reinterpret_cast<TT*>(dbeta.data_ptr()), nH));
}

}  // namespace

bool GroupTimestepDecayNormBwdFused(const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
                                    const torch::Tensor& var_grad, const torch::Tensor& x_or_y,
                                    const torch::Tensor& prev_count,
                                    const c10::optional<torch::Tensor>& bos_mask,
                                    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                                    const torch::Tensor& gamma, const torch::Tensor& beta,
                                    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
                                    double beta1, double beta2, bool memory_efficient,
                                    torch::Tensor x_grad_out, torch::Tensor prev_mean_grad_out,
                                    torch::Tensor prev_var_grad_out, torch::Tensor gamma_grad_out,
                                    torch::Tensor beta_grad_out) {
  const at::cuda::OptionalCUDAGuard guard(x_or_y.device());
  const int64_t B = x_or_y.size(0), L = x_or_y.size(1), H = x_or_y.size(2);
  const int64_t K = num_groups, G = H / K;

  // The predicate that decides whether this entry takes the shape: a group of at most 256 channels, a dtype the
  // kernels cover, and 3-D contiguous operands. It answers false BEFORE anything is written, so the caller can
  // fall through to another path with the outputs untouched.
  if (x_or_y.dim() != 3 || K < 1 || H % K != 0 || G < 1 || G > kWarpSize * kMaxChannelsPerLane) return false;
  // The fast path indexes elements with 32-bit offsets (see BwdVecKernel's prologue). Every such offset is
  // `(b*nL + t)*nH + k*nG + cIdx*8` with b < B, t < L, k*nG + cIdx*8 < H, so it is strictly below B*L*H and every
  // intermediate of that expression is too ((b*nL + t) < B*L, times nH < H). The shared indices are already int.
  // Declining here means the caller falls through to a 64-bit path rather than wrapping.
  if (B * L * H >= 2147483647) return false;
  if (x_or_y.scalar_type() != at::kBFloat16 && x_or_y.scalar_type() != at::kHalf &&
      x_or_y.scalar_type() != at::kFloat)
    return false;
  if (cummean.scalar_type() != at::kFloat || cummean.numel() != B * L * K) return false;
  const auto x = x_or_y.contiguous();
  const auto dy = y_grad.contiguous();
  const auto cm = cummean.contiguous();
  const auto cr = cumrstd.contiguous();
  const auto gam = gamma.contiguous();
  const auto bet = beta.contiguous();
  const auto dme = mean_grad.contiguous();
  const auto dva = var_grad.contiguous();

  const bool has_bos = bos_mask.has_value() && bos_mask->defined() && bos_mask->numel() > 0;
  const bool has_pad = padding_mask.has_value() && padding_mask->defined() && padding_mask->numel() > 0;
  const auto bos_c = has_bos ? bos_mask->contiguous() : at::Tensor();
  const auto pad_c = has_pad ? padding_mask->contiguous() : at::Tensor();
  const uint8_t* bos_p = has_bos ? reinterpret_cast<const uint8_t*>(bos_c.data_ptr()) : nullptr;
  const uint8_t* pad_p = has_pad ? reinterpret_cast<const uint8_t*>(pad_c.data_ptr()) : nullptr;
  // The prep module (which computes the bias-correction reciprocals) is stricter than this entry: bool masks only, and
  // B is one of its grid dimensions. Both are declined here, before anything is written, so the caller falls through
  // to a path that takes them - the same contract every other decline in this entry has.
  if (B > 65535) return false;
  if (has_bos && bos_c.scalar_type() != at::kBool) return false;
  if (has_pad && pad_c.scalar_type() != at::kBool) return false;

  const auto opts_f = x.options().dtype(at::kFloat);
  const auto opts_i = x.options().dtype(at::kInt);

  const int64_t nBK = (K + kBwdKG - 1) / kBwdKG;
  const int64_t nCB = B * nBK;
  // The environment variable TDN_BWD_T overrides the tile length (steps); unset, the length is chosen below.
  int64_t T = EnvInt("TDN_BWD_T", 0);
  if (T == 0) {
    // Tiles of ~4 waves of CTAs, between 16 and 64 steps: 32-64 is fastest (a bigger tile no longer fits
    // the live set in L2, and the look-back chain is longer) and 16 keeps the grid full when chains are few.
    const int64_t want = (nCB * L) / (148 * 4);
    T = ((want + 15) / 16) * 16;
    T = std::min<int64_t>(std::max<int64_t>(T, 16), 64);
  }
  T = std::min<int64_t>(T, std::max<int64_t>(L, 16));
  const int64_t nt = (L + T - 1) / T;
  TORCH_CHECK(nCB * nt <= 2147483647, "tdn bwd: grid too large");

  // A captured CUDA graph would replay with the epoch it was captured at, so every flag would read as unpublished and
  // the look-back would spin to its timeout - a hang, not a wrong number. Decline under capture: the general path
  // serves the call. The same guard keeps at::zeros out of a capture.
  {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    AT_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &cap));
    if (cap != cudaStreamCaptureStatusNone) return false;
  }

  auto prep = at::empty({B, L, 2}, opts_f);
  auto aggr = at::empty({B * K * nt, 4}, opts_f);
  auto incl = at::empty({B * K * nt, 2}, opts_f);
  // The two buffers that must start at zero - the look-back flags and the dgamma/dbeta partials - are held across
  // calls instead of being zeroed by a launch of their own: the flags carry the call's epoch in their tag, so a
  // stale flag reads as unpublished, and `FinalizeParamKernel` writes each partial back to zero as it converts it,
  // with the kernel boundary on this stream ordering that before the next call's tile kernel reads them. Persistent
  // and zero-initialised is what makes that sound: a fresh `at::empty` can return a block whose zeros came from a
  // tensor that has since been reused, which is silent and shape-dependent.
  // A growth or a device change re-allocates to zeros, which is legal at any point because no epoch of the old buffer
  // can be mistaken for one of the new.
  const int64_t nflags = B * K * nt;
  // `fg` sits at `nflags` floats and the packed reduction needs a 16-byte address, so the flag region is rounded up
  // to a whole number of float4s. The other two terms of the offset are already aligned and this is the only one that
  // is not: `4*k*nG` is 256k because `nG` is pinned at 64 by the `G == 64` entry condition, and `32*cIdx` is a
  // multiple of 32. Rounding is a function of the raw value, so the pool's key stays the raw `nflags`: two shapes
  // with the same key always round the same way, and two different keys may round together, which can only refuse a
  // reuse (a slow path), never grant a wrong one.
  const int64_t nflags4 = (nflags + 3) & ~(int64_t)3;
  const int64_t nfbuf = 2 * H;
  // Per-stream pool, the persistent forward's `pool_for(stream)` contract. The flags region is the only buffer that
  // must be per-stream (aggr/incl are per call, and `fg`/`fb` are per call through the counter), but two calls in
  // flight on different streams must not share an epoch space: the older call's look-back would read the newer
  // call's tag as unpublished and wait for ever, a hang rather than a wrong answer. Past the cap, the entry declines
  // to the general path.
  const cudaStream_t cur_stream = at::cuda::getCurrentCUDAStream();
  struct FlagPool {
    bool used = false;
    cudaStream_t stream = nullptr;
    at::Tensor buf;                     // [nflags ints][fg: H][fb: H]
    int epoch_ctr = 0;
    int64_t nflags = -1, H = -1;
  };
  static thread_local FlagPool pools[8];
  FlagPool* pool = nullptr;
  for (int i = 0; i < 8 && pool == nullptr; ++i)
    if (pools[i].used && pools[i].stream == cur_stream) pool = &pools[i];
  for (int i = 0; i < 8 && pool == nullptr; ++i)
    if (!pools[i].used) {
      pools[i].used = true;
      pools[i].stream = cur_stream;
      pool = &pools[i];
    }
  if (pool == nullptr) return false;  // more than 8 streams in flight
  at::Tensor& pbuf = pool->buf;
  if (!pbuf.defined() || pbuf.numel() < nflags4 + nfbuf || pbuf.device() != x.device() ||
      pool->nflags != nflags || pool->H != H) {
    // A new shape moves both regions (fg sits at `nflags`), so the block it lands on is not the block the previous
    // call left clean. Re-zero it and restart the epoch: a shape change is once per shape, not once per call.
    if (pbuf.defined() && pbuf.numel() >= nflags4 + nfbuf && pbuf.device() == x.device()) {
      pbuf.zero_();
    } else {
      pbuf = at::zeros({nflags4 + nfbuf}, opts_f);
    }
    pool->epoch_ctr = 0;
    pool->nflags = nflags;
    pool->H = H;
  }
  if (pool->epoch_ctr >= (1 << 29) - 1) {  // the tag is (epoch << 2) | status: never reuse one still in memory
    pbuf.zero_();
    pool->epoch_ctr = 0;
  }
  int* flags = reinterpret_cast<int*>(pbuf.data_ptr<float>());
  float* fg = pbuf.data_ptr<float>() + nflags4;
  float* fb = fg + H;
  // status 0 is unpublished and the epoch is never 0 (++epoch_ctr runs before the first launch), so an all-zero
  // buffer - a cold process, a re-zero, a memset - is a valid start state for a look-back.
  const int epoch = ++pool->epoch_ctr;

  const auto stream = at::cuda::getCurrentCUDAStream();
  // The bias-correction reciprocals come from the prep module (a grid of one CTA a (row, 512-step tile)). The zeroing
  // the tile kernel needs before it starts - the look-back flags, and the dgamma/dbeta partials which the finalize
  // wrote back to zero - is already in place: no launch does it.
  GroupTimestepDecayNormBwdPrep(bos_mask, padding_mask, prev_count, prep, B, L, beta1, beta2);

  const size_t esz = size_t(x.element_size());
  const size_t smem = size_t(T) * kBwdKG * sizeof(float2) + 3 * size_t(T) * kBwdKG * sizeof(float) +
                      size_t(kBwdKG) * G * 2 * esz + size_t(T) * sizeof(float2) +
                      2 * size_t(T + 1) * sizeof(float) + 2 * size_t(T + 1) * sizeof(uint8_t) + 64;
  // The vec kernel's cp.async ring: 4 slots x 2 tensors x 8 warps x 32 lanes x 16 B, and 16 B of slack for the
  // alignment of the ring after the tail of the tile's other buffers.
  const size_t smem_vec = smem + 2 * size_t(kBwdKG) * kD * 32 * sizeof(uint4) + 16;

  const int NC = G <= 32 ? 1 : (G <= 64 ? 2 : kMaxChannelsPerLane);
  const auto dtype = x.scalar_type();
#define TDN_BWD_LAUNCH(TT, NCP)                                                                            \
  do {                                                                                                     \
    auto kfn = &BwdKernel<TT, false, NCP>;                                                                 \
    auto kfn_me = &BwdKernel<TT, true, NCP>;                                                               \
    if (smem > 48 * 1024) {                                                                                 \
      AT_CUDA_CHECK(cudaFuncSetAttribute((const void*)kfn, cudaFuncAttributeMaxDynamicSharedMemorySize,      \
                                         int(smem)));                                                       \
      AT_CUDA_CHECK(cudaFuncSetAttribute((const void*)kfn_me, cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                                         int(smem)));                                                       \
    }                                                                                                       \
    auto* fn = memory_efficient ? kfn_me : kfn;                                                             \
    fn<<<int(nCB * nt), kBwdThreads, smem, stream>>>(                                                       \
        reinterpret_cast<const TT*>(x.data_ptr()), reinterpret_cast<const TT*>(dy.data_ptr()),              \
        cm.data_ptr<float>(), cr.data_ptr<float>(), reinterpret_cast<const TT*>(gam.data_ptr()),             \
        reinterpret_cast<const TT*>(bet.data_ptr()), reinterpret_cast<const TT*>(dme.data_ptr()),            \
        reinterpret_cast<const TT*>(dva.data_ptr()),                                                        \
        reinterpret_cast<const float2*>(prep.data_ptr<float>()), bos_p, pad_p,                               \
        reinterpret_cast<TT*>(x_grad_out.data_ptr()), fg, fb,                                  \
        reinterpret_cast<TT*>(gamma_grad_out.data_ptr()),                                                  \
        reinterpret_cast<TT*>(beta_grad_out.data_ptr()),                                                    \
        reinterpret_cast<TT*>(prev_mean_grad_out.data_ptr()),                                               \
        reinterpret_cast<TT*>(prev_var_grad_out.data_ptr()),                                                \
        reinterpret_cast<float4*>(aggr.data_ptr<float>()), reinterpret_cast<float2*>(incl.data_ptr<float>()), \
        flags, int(B), int(L), int(K), int(G), int(H), int(nt), int(T), int(nCB), int(nBK),   \
        float(beta1), float(beta2), float(1.0 - beta1), float(1.0 - beta2), epoch);             \
    LaunchFinalizeKernel<TT>(fg, fb, gamma_grad_out, beta_grad_out, int(H), stream);            \
  } while (0)

#define TDN_BWDV_LAUNCH(TT, LP)                                                                            \
  do {                                                                                                     \
    auto kfn = &BwdVecKernel<TT, false, LP>;                                                               \
    auto kfn_me = &BwdVecKernel<TT, true, LP>;                                                             \
    if (smem_vec > 48 * 1024) {                                                                            \
      AT_CUDA_CHECK(cudaFuncSetAttribute((const void*)kfn, cudaFuncAttributeMaxDynamicSharedMemorySize,      \
                                         int(smem_vec)));                                                   \
      AT_CUDA_CHECK(cudaFuncSetAttribute((const void*)kfn_me, cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                                         int(smem_vec)));                                                   \
    }                                                                                                       \
    auto* fn = memory_efficient ? kfn_me : kfn;                                                             \
    /* Programmatic dependent launch: the attribute makes this grid resident from the moment prep has launched all its \
       blocks, so the prologue stages gamma/beta, the masks and the decay factor while prep is still writing; vec's     \
       `cudaGridDependencySynchronize()` (just before it reads prep's buffer) releases on prep's completion and flush,   \
       so the overlap is bounded by prep's *end* and not by any early trigger. The attribute is dropped under capture,  \
       where a launch difference is not allowed; a launch with numAttrs = 0 is the same launch as <<<>>>. */           \
    cudaLaunchConfig_t cfg{};                                                                               \
    cfg.gridDim = dim3(unsigned(nCB * nt));                                                                 \
    cfg.blockDim = dim3(kBwdThreads);                                                                       \
    cfg.dynamicSmemBytes = smem_vec;                                                                        \
    cfg.stream = stream;                                                                                    \
    cudaLaunchAttribute attrs[1];                                                                           \
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;                                              \
    if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap == cudaStreamCaptureStatusNone) {         \
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                     \
      attrs[0].val.programmaticStreamSerializationAllowed = 1;                                              \
      cfg.attrs = attrs;                                                                                     \
      cfg.numAttrs = 1;                                                                                      \
    }                                                                                                       \
    AT_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fn,                                                              \
        reinterpret_cast<const TT*>(x.data_ptr()), reinterpret_cast<const TT*>(dy.data_ptr()),              \
        cm.data_ptr<float>(), cr.data_ptr<float>(), reinterpret_cast<const TT*>(gam.data_ptr()),             \
        reinterpret_cast<const TT*>(bet.data_ptr()), reinterpret_cast<const TT*>(dme.data_ptr()),            \
        reinterpret_cast<const TT*>(dva.data_ptr()),                                                        \
        reinterpret_cast<const float2*>(prep.data_ptr<float>()), bos_p, pad_p,                               \
        reinterpret_cast<TT*>(x_grad_out.data_ptr()), fg, fb,                                          \
        reinterpret_cast<TT*>(gamma_grad_out.data_ptr()),                                                   \
        reinterpret_cast<TT*>(beta_grad_out.data_ptr()),                                                    \
        reinterpret_cast<TT*>(prev_mean_grad_out.data_ptr()),                                               \
        reinterpret_cast<TT*>(prev_var_grad_out.data_ptr()),                                                \
        reinterpret_cast<float4*>(aggr.data_ptr<float>()), reinterpret_cast<float2*>(incl.data_ptr<float>()), \
        flags, int(B), int(L), int(K), int(G), int(H), int(nt), int(T), int(nCB), int(nBK),   \
        float(beta1), float(beta2), float(1.0 - beta1), float(1.0 - beta2), epoch));            \
    LaunchFinalizeKernel<TT>(fg, fb, gamma_grad_out, beta_grad_out, int(H), stream);            \
  } while (0)
  const bool vec = (G == 64) && (dtype != at::kFloat);
  if (vec) {
    if (dtype == at::kBFloat16) TDN_BWDV_LAUNCH(__nv_bfloat16, 8);
    else TDN_BWDV_LAUNCH(__half, 8);
  } else if (dtype == at::kBFloat16) {
    if (NC == 1) TDN_BWD_LAUNCH(__nv_bfloat16, 1);
    else if (NC == 2) TDN_BWD_LAUNCH(__nv_bfloat16, 2);
    else TDN_BWD_LAUNCH(__nv_bfloat16, kMaxChannelsPerLane);
  } else if (dtype == at::kHalf) {
    if (NC == 1) TDN_BWD_LAUNCH(__half, 1);
    else if (NC == 2) TDN_BWD_LAUNCH(__half, 2);
    else TDN_BWD_LAUNCH(__half, kMaxChannelsPerLane);
  } else {
    if (NC == 1) TDN_BWD_LAUNCH(float, 1);
    else if (NC == 2) TDN_BWD_LAUNCH(float, 2);
    else TDN_BWD_LAUNCH(float, kMaxChannelsPerLane);
  }
#undef TDN_BWD_LAUNCH

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

}  // namespace ops
}  // namespace xllm
