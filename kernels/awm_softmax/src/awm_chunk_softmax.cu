// Chunked softmax and logsumexp over time (forward, decode and the fused backward) for the adaptive working memory:
// one pass over key in its physical layout [B, L, K, S], transposed once through shared memory.
//
// Launch shape: one CTA per (batch row, kv head, channel block of ST, chunk). The chunk's ST x C tile lives in shared
// memory, written transposed (time innermost) as it is loaded, so that
//   - the global reads of key are 16 B per thread (ST = 16 channels = 32 B = one sector per row), and
//   - the writes of kkey [B,K,S,L] are 512 B per warp along time, and the reduce pass is a 16 B LDS.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstdint>

#include "awm_chunk_softmax.h"

namespace cg = cooperative_groups;

namespace awm {

constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1e30f;

__device__ __forceinline__ float ex2f(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float tof(__half v) { return __half2float(v); }
__device__ __forceinline__ float tof(float v) { return v; }
__device__ __forceinline__ void fromf(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }
__device__ __forceinline__ void fromf(__half& d, float v) { d = __float2half(v); }
__device__ __forceinline__ void fromf(float& d, float v) { d = v; }

// Shared address of (channel ch, time t): the row is padded to C + PAD, and the 8-element block index of t is XORed
// with a per-channel value so that the transposing stores and the transposing reads of the backward pass spread
// over the banks while every 8-aligned run of 8 times stays contiguous (16 B).
// The group step is 2 blocks: with ch = 8g + j the swizzle is ((j + 2g) mod 8) << 3. At ST = 16, RP = ST/VEC = 2, so
// two lanes of one warp carry the same time and channels that differ by 8 (groups g and g+1). Their swizzles differ
// by 2 * 8 elements = 8 words, and bank = word index mod 32: a group step of 1 block would give bank windows 0..7
// against 4..11, which overlap on 4..7 (conflict degree 1.5); a step of 2 gives 0..7 against 8..15, which are
// disjoint (degree 1.0). At ST = 8 (ch < 8) the group term vanishes and the expression is the plain rotation by j.
// The bank argument is for 2-byte elements (VEC = 8); at fp32 VEC = 4 and the window is a different one.
__device__ __forceinline__ int swz(int ch) { return ((ch + 2 * (ch >> 3)) & 7) << 3; }
__device__ __forceinline__ int toff(int ch, int t, int TP) { return ch * TP + (t ^ swz(ch)); }

// Load one chunk's tile, transposed (thread holds VEC channels of one time step), 4 rows in flight per thread:
// one 16 B LDG per thread is only 8 KiB in flight for a 512-thread CTA, which cannot cover the ~20 KiB an SM needs
// to sustain its share of HBM at 500 ns. With MASKED, a masked step loads nothing (its 16-ch block is one 32 B
// sector no other CTA needs) and its tile entries are -inf: 0 in kkey, nothing in the max and the sum.
template <typename T, int ST, int NT, bool MASKED>
__device__ __forceinline__ void load_tile(T* tile, const T* s0, const bool* mb, int nIter, int nTail, int ti, int si,
                                         long long tstr, int TP, int RPI, int* fFirst = nullptr,
                                         int* fLast = nullptr) {
  constexpr int VEC = 16 / sizeof(T);
  // MASKED only: the first unmasked step and the last masked one among the steps this thread loads. Every step of the
  // slice belongs to exactly one thread (t = ti + i*RPI), so a block-wide min and max of these two say whether the
  // slice's mask is [masked prefix][unmasked suffix] - see fwd_kernel. Free when nobody asks for them.
  int firstF = 0x7fffffff, lastT = -1;
  int i = 0;
  for (; i + 4 <= nIter; i += 4) {
    T v[4][VEC];
    bool msk[4] = {false, false, false, false};
    if (MASKED) {
#pragma unroll
      for (int u = 0; u < 4; ++u) msk[u] = mb[(size_t)(i + u) * RPI];
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (!MASKED || !msk[u]) *(int4*)v[u] = *(const int4*)(s0 + (long long)(i + u) * RPI * tstr);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int t = ti + (i + u) * RPI;
      if (MASKED) {
        if (msk[u]) {
          lastT = t;
#pragma unroll
          for (int j = 0; j < VEC; ++j) fromf(v[u][j], -INFINITY);
        } else if (firstF > t) {
          firstF = t;
        }
      }
#pragma unroll
      for (int j = 0; j < VEC; ++j) tile[toff(si * VEC + j, t, TP)] = v[u][j];
    }
  }
  for (; i < nIter; ++i) {
    const int t = ti + i * RPI;
    const bool sk = MASKED && mb[(size_t)i * RPI];
    T v[VEC];
    if (!sk) *(int4*)v = *(const int4*)(s0 + (long long)i * RPI * tstr);
    if (sk) {
      lastT = t;
#pragma unroll
      for (int j = 0; j < VEC; ++j) fromf(v[j], -INFINITY);
    } else if (MASKED && firstF > t) {
      firstF = t;
    }
#pragma unroll
    for (int j = 0; j < VEC; ++j) tile[toff(si * VEC + j, t, TP)] = v[j];
  }
  // The rounds cover nIter * RPI steps; a slice length that RPI does not divide leaves nTail < RPI of them over, and
  // one guarded row per thread covers exactly those (step nIter*RPI + ti, for every ti < nTail). s0 already carries
  // + ti*tstr, so the row sits at nIter*RPI relative to it. nTail is 0 when RPI divides the slice, and the guard is
  // then never taken; callers therefore need the slice to be a multiple of the reduce round only, not of RPI.
  if (ti < nTail) {
    const int t = ti + nIter * RPI;
    const bool sk = MASKED && mb[(size_t)nIter * RPI];
    T v[VEC];
    if (!sk) *(int4*)v = *(const int4*)(s0 + (long long)nIter * RPI * tstr);
    if (sk) {
      lastT = t;
#pragma unroll
      for (int j = 0; j < VEC; ++j) fromf(v[j], -INFINITY);
    } else if (MASKED && firstF > t) {
      firstF = t;
    }
#pragma unroll
    for (int j = 0; j < VEC; ++j) tile[toff(si * VEC + j, t, TP)] = v[j];
  }
  if (MASKED && fFirst) {
    *fFirst = firstF;
    *fLast = lastT;
  }
}


// ---------------------------------------------------------------- forward (mode 0 chunk, prev chunk, mode 2)
//
// A chunk is cut into P slices along time, one per CTA of a cluster: each rank holds ST x (C/P) on chip, which is
// what lets ST be wide (ST*2 B is the length of one key row read in one go: 32 B at ST=16 costs a quarter of a
// sector per row and only L2 merging brings it back, 128 B at ST=64 needs none) while the tile stays small enough
// for 3 CTAs per SM. The (max, sum) pairs are combined across the ranks through distributed shared memory once,
// after one cluster.sync, and the logsumexp comes out of that combine.

template <typename T, int ST, int NT, int P>
__global__ __launch_bounds__(NT) __cluster_dims__(P, 1, 1) void fwd_kernel(FwdArgs a0, FwdArgs a1) {
  extern __shared__ __align__(16) char smem_raw[];
  const FwdArgs& a = (blockIdx.y == 0) ? a0 : a1;
  const int rank = blockIdx.x % P;
  const int tileIdx = blockIdx.x / P;
  // An inactive CTA (the prev-chunk grid, which has fewer tasks than the chunk grid) must not return: its cluster
  // still has to reach every cluster barrier.
  const bool act = tileIdx < a.nTasks;

  const int SL = a.C / P;  // time steps this rank holds
  const int TP = SL + PAD;
  T* tile = reinterpret_cast<T*>(smem_raw);
  // P x (max, sum): every rank publishes its own pair into every rank's copy, so after one cluster barrier the
  // combine is local and nobody touches another rank's memory again -- which is also what makes it safe to exit.
  float* red = reinterpret_cast<float*>(smem_raw + (size_t)ST * TP * sizeof(T));
  float* mine = red + rank * 2 * ST;
  const cg::cluster_group cluster = cg::this_cluster();

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int NW = NT / 32;
  constexpr int VEC = 16 / sizeof(T);  // elements per 16 B load
  constexpr int RP = ST / VEC;         // threads per time row
  constexpr int RPI = NT / RP;         // rows per load iteration
  constexpr int LV = VEC;              // elements per lane per reduce iteration
  constexpr int LSTR = 32 * LV;        // times covered by one warp iteration
  // The tile address is incremented rather than recomputed per round, in both hot loops. This is exact for every
  // (ST, P, T) the file instantiates: swz(ch) = ((ch + 2*(ch>>3)) & 7) << 3 <= 56 < 64 <= LSTR and LV*lane <= LV*31 <
  // LSTR, so i*LSTR has zeros in every bit that swz or LV*lane can set, and (LV*lane + i*LSTR) ^ swz(ch) equals
  // (LV*lane ^ swz(ch)) + i*LSTR. ST and P do not enter the address (ST sets the loop bounds, P the combine).
  // PTR is a register-budget switch, not a correctness one: with ST = 8 and P > 1 the three running pointers take the
  // kernel from 40 to 57 registers, which drops its residency from 3 CTAs per SM to 2 and costs more than the saved
  // instructions (2064 -> 2016 per moving row) gain.
  constexpr bool PTR = (ST >= 16) || (P == 1);

  int q = act ? tileIdx : 0;
  const int cb = q % a.nbChan;
  q /= a.nbChan;
  const int k = q % a.K;
  q /= a.K;
  const int b = q % a.B;
  const int c = q / a.B;

  const int t0 = c * a.cMul + rank * SL;
  const T* src = (const T*)a.key + b * a.inBStr + (long long)t0 * a.inTStr + k * a.inKStr + (long long)cb * ST;
  const bool* mask = (const bool*)a.mask;
  const int nIter = SL / RPI;  // whole rounds; the band's cells leave a tail (see nTail below)
  const int nL = SL / LSTR;

  // ---- load this rank's slice of the chunk, transposed: thread holds VEC channels of one time step
  int fFirst = 0x7fffffff, fLast = -1;
  if (act) {
    const int si = tid % RP, ti = tid / RP;
    const int nTail = SL - nIter * RPI;  // non-zero for slices that a load round does not divide: C = 6656 at P = 4 gives
                                         // SL = 1664, nIter = 6, nTail = 128. The slice must be a multiple of the
                                         // reduce and write round (32 * 16/sizeof(T)) only; load_tile's guarded row
                                         // covers the tail.
    const T* s0 = src + (long long)ti * a.inTStr + si * VEC;
    const bool* mb = mask ? (mask + b * a.maskBStr + t0 + ti) : nullptr;
    if (mask) {
      load_tile<T, ST, NT, true>(tile, s0, mb, nIter, nTail, ti, si, a.inTStr, TP, RPI, &fFirst, &fLast);
    } else {
      load_tile<T, ST, NT, false>(tile, s0, mb, nIter, nTail, ti, si, a.inTStr, TP, RPI);
    }
  }
  // ---- where the masked prefix ends: a chunk's mask is [masked prefix][unmasked suffix] whenever the segmentation is
  // monotone in time (packed documents). An arbitrary mask is still handled: tBeg then stays 0 and the per-step path
  // below runs unchanged.
  //
  // The reduce and the write then start at the first vector round at or before tBeg - the tile already holds -inf for
  // every masked step, so a round that straddles the boundary contributes nothing - and the write emits zeros for the
  // rounds before it, without reading the tile and without one ex2 per element. When 59-69% of a chunk is prefix that
  // is about 60% of the reduce and write work. Skipping the -inf tile fills of the prefix would save little and is
  // not done.
  __shared__ int wFirstF[NW], wLastT[NW];
  if (mask) {
    int f = fFirst, l = fLast;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      f = min(f, __shfl_xor_sync(0xffffffffu, f, off));
      l = max(l, __shfl_xor_sync(0xffffffffu, l, off));
    }
    if (lane == 0) { wFirstF[warp] = f; wLastT[warp] = l; }
  }
  __syncthreads();
  int tBeg = 0;
  if (mask) {
    int f = 0x7fffffff, l = -1;
    for (int w = 0; w < NW; ++w) { f = min(f, wFirstF[w]); l = max(l, wLastT[w]); }
    // No masked step after the first unmasked one: the prefix ends at f. A slice whose every step is masked leaves
    // `firstF` at INT_MAX in every thread, so f is clamped to SL: that is the all-masked end (empty reduce, zeros
    // written) and it keeps iFirst <= nL, which the write's bounds rest on.
    // The min() is required: without it tBeg = INT_MAX, iFirst = INT_MAX / LSTR, and the prefix zero-fill below would
    // write gigabytes past the end of kkey.
    tBeg = (l < f) ? min(f, SL) : 0;
  }
  const int iFirst = tBeg / LSTR;  // first whole round of the suffix (the rounds below are 32*VEC steps each)

  // ---- online max/sum per channel over this rank's slice, warp per channel
  if (act) {
    for (int ch = warp; ch < ST; ch += NW) {
      const T* row = tile + ch * TP;
      // The tile address is incremented, not recomputed per round (see PTR above): LSTR = 32*LV, swz(ch) < 64 <= LSTR
      // and LV*lane < LSTR, so (LV*lane + i*LSTR) ^ swz(ch) == (LV*lane ^ swz(ch)) + i*LSTR for every i. Where PTR is
      // false the address is recomputed each round.
      const T* rq = row + ((LV * lane) ^ swz(ch)) + (long long)iFirst * LSTR;
      float m = NEG_BIG, s = 0.f;
      for (int i = iFirst; i < nL; ++i) {
        T v[LV];
        if (PTR) { *(int4*)v = *(const int4*)rq; rq += LSTR; }
        else { const int t = LV * lane + i * LSTR; *(int4*)v = *(const int4*)(row + (t ^ swz(ch))); }
        float x[LV], mn = NEG_BIG;
#pragma unroll
        for (int j = 0; j < LV; ++j) { x[j] = tof(v[j]); mn = fmaxf(mn, x[j]); }
        if (mn > m) { s *= ex2f((m - mn) * LOG2E); m = mn; }
        // One FFMA per element instead of FADD + FMUL: ml2 is hoisted per round and ls2 per channel (the write loop
        // below). FOLD is off for ST = 8 with P > 1 (the narrow cells of long chunks), where the extra live scalar
        // takes the kernel from 40 to 56 or 43 registers and from 3 CTAs per SM to 2.
        // Precondition: the fold equals the 3-op form only while |m| * LOG2E is finite, i.e. |m| < 3.4e38 / 1.4427 =
        // 2.36e38. bf16 and fp32 hold values up to 3.39e38, so a legal key can exceed it, and no NaN announces it: an
        // FMA takes the exact product before it rounds, so x*LOG2E stays a finite real and finite + (-inf) is -inf.
        // Once ml2 saturates, every element's exponent is -inf, acc is 0, s stays 0, l is -inf, ls is 0, and for the
        // whole chunk kkey becomes e^x (inf at the maximum, ordinary finite values elsewhere) while lse_out is -inf.
        // Inside the range the result is exact up to rounding: at x == m the exponent is m*LOG2E - fl(m*LOG2E), a
        // nonzero of order |m| * 8.6e-8, so that element is 1 - |m| * 6e-8, and every term of the channel carries a
        // relative error of up to |m| * 5.96e-8 (6.0e-7 at |m| = 10, 6.0e-6 at |m| = 100), which reaches an absolute
        // 1e-4 in lse at |m| ~ 1.7e3.
        // The inputs are not clamped: a clamp costs an instruction and a live scalar in cells that already use all the
        // registers of their residency class (64 = 65536 / (512 * 2)).
        constexpr bool FOLD = (ST >= 16) || (P == 1);
        const float ml2 = m * LOG2E;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < LV; ++j)
          acc += FOLD ? ex2f(fmaf(x[j], LOG2E, -ml2)) : ex2f((x[j] - m) * LOG2E);
        s += acc;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float mB = __shfl_xor_sync(0xffffffffu, m, off);
        float s2 = __shfl_xor_sync(0xffffffffu, s, off);
        float mn = fmaxf(m, mB);
        s = s * ex2f((m - mn) * LOG2E) + s2 * ex2f((mB - mn) * LOG2E);
        m = mn;
      }
      if (lane == 0) { mine[ch] = m; mine[ST + ch] = s; }
    }
    __syncwarp();  // the warp reads back what its lane 0 just wrote (the P > 1 path gets a cluster.sync instead)
  }
  if (P > 1) {
    // publish: each rank hands its own (max, sum) pair to every rank's copy; its own slot is already there. Gated by
    // `act` so that a cluster of the previous-chunk grid's short tail never writes its uninitialised `mine[]` into
    // another rank; the barrier below is outside the `act` branch, because every CTA of a cluster has to reach it and
    // `act` is the same for all of them (`tileIdx = blockIdx.x / P`, and a cluster's ranks share `tileIdx`).
    if (act) {
      for (int dstRank = 0; dstRank < P; ++dstRank) {
        if (dstRank == rank) continue;
        float* dst = cluster.map_shared_rank(red, dstRank) + rank * 2 * ST;
        for (int ch = warp; ch < ST; ch += NW) {
          if (lane == 0) { dst[ch] = mine[ch]; dst[ST + ch] = mine[ST + ch]; }
        }
      }
    }
    cluster.sync();
  }

  // ---- combine the ranks' (max, sum): the chunk's logsumexp, then the softmax of this rank's slice
  if (act) {
    for (int ch = warp; ch < ST; ch += NW) {
      float m = NEG_BIG, s = 0.f;
      for (int r = 0; r < P; ++r) {
        const float* o = red + r * 2 * ST;
        const float mB = o[ch], s2 = o[ST + ch];
        const float mn = fmaxf(m, mB);
        s = s * ex2f((m - mn) * LOG2E) + s2 * ex2f((mB - mn) * LOG2E);
        m = mn;
      }
      const float l = (s > 0.f) ? (m + logf(s)) : -INFINITY;
      const float ls = (l == -INFINITY) ? 0.f : l;
      constexpr bool FOLD = (ST >= 16) || (P == 1);  // forward only, see the reduce above; bwd_kernel folds
                                                      // unconditionally and keeps its register count either way
      const float ls2 = ls * LOG2E;
      if (lane == 0 && rank == 0 && c <= a.lseMaxC) {
        a.lse[b * a.lseBStr + k * a.lseKStr + (long long)(cb * ST + ch) * a.lseNStr + (c + a.slotBase)] = l;
      }
      const T* row = tile + ch * TP;
      T* out = (T*)a.kout + b * a.outBStr + k * a.outKStr + (long long)(cb * ST + ch) * a.outL + t0;
      // the masked prefix: zeros, without reading the tile and without an ex2 per element
      if (iFirst) {
        T z[LV];
#pragma unroll
        for (int j = 0; j < LV; ++j) fromf(z[j], 0.f);
        for (int i = 0; i < iFirst; ++i) *(int4*)(out + LV * lane + i * LSTR) = *(int4*)z;
      }
      // The write loop's two addresses are incremented for the same reason and under the same switch as the reduce's.
      // Both start at round iFirst, not 0: the loop runs iFirst..nL, so the tile must be read from round iFirst.
      const T* rp = row + ((LV * lane) ^ swz(ch)) + (long long)iFirst * LSTR;
      T* op = out + (long long)LV * lane + (long long)iFirst * LSTR;
      for (int i = iFirst; i < nL; ++i) {
        T v[LV], o[LV];
        if (PTR) { *(int4*)v = *(const int4*)rp; rp += LSTR; }
        else { const int t = LV * lane + i * LSTR; *(int4*)v = *(const int4*)(row + (t ^ swz(ch))); }
#pragma unroll
        for (int j = 0; j < LV; ++j)
          fromf(o[j], FOLD ? ex2f(fmaf(tof(v[j]), LOG2E, -ls2)) : ex2f((tof(v[j]) - ls) * LOG2E));
        if (PTR) { *(int4*)op = *(int4*)o; op += LSTR; }
        else { const int t = LV * lane + i * LSTR; *(int4*)(out + t) = *(int4*)o; }
      }
    }
  }
  // Slot 0 of lse_out is the caller's when no prev chunk is passed (the caller writes -inf there), and with a prev
  // chunk the second grid plane writes the prev chunk's logsumexp to it. Storing a zero here when there is no prev
  // chunk would cost seven registers, taking the 16-channel instantiations from 64 to 57, i.e. less load parallelism.
  // The plane is chosen by the launcher's has1, which sets the grid's y extent, and blockIdx.y picks it; slotBase
  // selects the base of lse_out, so fwd_kernel takes no hasPrev flag.
}

// ---------------------------------------------------------------- backward (mode 1)

template <typename T, int ST, int NT>
__global__ __launch_bounds__(NT) void bwd_kernel(BwdArgs a) {
  extern __shared__ __align__(16) char smem_raw[];
  if (blockIdx.x >= a.nTasks) return;

  const int TP = a.C + PAD;
  T* tile = reinterpret_cast<T*>(smem_raw);
  float* sc = reinterpret_cast<float*>(smem_raw + (size_t)ST * TP * sizeof(T));  // lse, then (dlse - sum)

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int NW = NT / 32;
  constexpr int VEC = 16 / sizeof(T);
  constexpr int RP = ST / VEC;
  constexpr int RPI = NT / RP;

  int q = blockIdx.x;
  const int cb = q % a.nbChan;
  q /= a.nbChan;
  const int k = q % a.K;
  q /= a.K;
  const int b = q % a.B;
  const int c = q / a.B;  // 0 .. N-2

  const int t0 = c * a.C;
  const T* src = (const T*)a.key + b * a.inBStr + (long long)t0 * a.inTStr + k * a.inKStr + (long long)cb * ST;
  const bool* mask = (const bool*)a.mask;

  {
    const int si = tid % RP, ti = tid / RP;
    const int nIter = a.C / RPI;                   // full rounds of RPI steps
    const int nTail = a.C - nIter * RPI;           // the rest (< RPI): load_tile's guarded row
    const T* s0 = src + (long long)ti * a.inTStr + si * VEC;
    const bool* mb = mask ? (mask + b * a.maskBStr + t0 + ti) : nullptr;
    if (mask) {
      load_tile<T, ST, NT, true>(tile, s0, mb, nIter, nTail, ti, si, a.inTStr, TP, RPI);
    } else {
      load_tile<T, ST, NT, false>(tile, s0, mb, nIter, nTail, ti, si, a.inTStr, TP, RPI);
    }
  }
  __syncthreads();

  constexpr int LV = VEC;
  constexpr int LSTR = 32 * LV;
  const int nL = a.C / LSTR;

  // ---- per channel: y = exp(x - lse), sum(g * y) over the chunk. The tile already says which steps are masked (a
  // masked step is -inf there), so a group of steps masked whole reads no dkkey at all: its g is multiplied by y = 0.
  for (int ch = warp; ch < ST; ch += NW) {
    const long long lseCh = b * a.lseBStr + k * a.lseKStr + (long long)(cb * ST + ch) * a.lseNStr + (c + 1);
    const float l = a.lse[lseCh];
    const float d = a.dlse[lseCh];
    const float ls = (l == -INFINITY) ? 0.f : l;
    const float ls2 = ls * LOG2E;
    const T* row = tile + ch * TP;
    const T* dk = (const T*)a.dkkey + (((long long)(b * a.K + k) * a.dkKStr + (cb * ST + ch)) * a.dkCL) + t0;
    float acc = 0.f;
    for (int i = 0; i < nL; ++i) {
      const int t = LV * lane + i * LSTR;
      T v[LV], g[LV];
      *(int4*)v = *(const int4*)(row + (t ^ swz(ch)));
      float mx = tof(v[0]);
#pragma unroll
      for (int j = 1; j < LV; ++j) mx = fmaxf(mx, tof(v[j]));
      if (mx != -INFINITY) {
        *(int4*)g = *(const int4*)(dk + t);
#pragma unroll
        for (int j = 0; j < LV; ++j) acc += tof(g[j]) * ex2f(fmaf(tof(v[j]), LOG2E, -ls2));
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) { sc[ch] = ls; sc[ST + ch] = d - acc; }
  }
  __syncthreads();

  // ---- dkey = y * (g + dlse - sum), rounded once, back into the tile. A group of steps masked whole is 0 there,
  // so it reads no dkkey.
  for (int ch = warp; ch < ST; ch += NW) {
    const float A = sc[ST + ch], ls = sc[ch];
    const float ls2 = ls * LOG2E;
    const T* row = tile + ch * TP;
    const T* dk = (const T*)a.dkkey + (((long long)(b * a.K + k) * a.dkKStr + (cb * ST + ch)) * a.dkCL) + t0;
    for (int i = 0; i < nL; ++i) {
      const int t = LV * lane + i * LSTR;
      T v[LV], g[LV], o[LV];
      *(int4*)v = *(const int4*)(row + (t ^ swz(ch)));
      float mx = tof(v[0]);
#pragma unroll
      for (int j = 1; j < LV; ++j) mx = fmaxf(mx, tof(v[j]));
      if (mx == -INFINITY) {
#pragma unroll
        for (int j = 0; j < LV; ++j) fromf(o[j], 0.f);
      } else {
        *(int4*)g = *(const int4*)(dk + t);
#pragma unroll
        for (int j = 0; j < LV; ++j) fromf(o[j], ex2f(fmaf(tof(v[j]), LOG2E, -ls2)) * (tof(g[j]) + A));
      }
      *(int4*)(row + (t ^ swz(ch))) = *(int4*)o;
    }
  }
  __syncthreads();

  // ---- transposed store into dkey [B, L, K, S]: thread takes one time step, ST channels (one or two sectors)
  for (int t = tid; t < a.C; t += NT) {
    T v[ST];
#pragma unroll
    for (int ch = 0; ch < ST; ++ch) v[ch] = tile[toff(ch, t, TP)];
    T* dst = (T*)a.dkey + (long long)(b * a.inBStr + (long long)t0 * a.inTStr) + (long long)t * a.inTStr +
             k * a.inKStr + (long long)cb * ST;
#pragma unroll
    for (int j = 0; j < ST / VEC; ++j) *(int4*)(dst + j * VEC) = *(int4*)(v + j * VEC);
  }
}

// ---------------------------------------------------------------- launchers

template <typename T, int ST, int NT, int P>
static int launch_fwd(const FwdArgs& a0, const FwdArgs& a1, int has1, cudaStream_t st) {
  const size_t sh = (size_t)ST * ((size_t)a0.C / P + PAD) * sizeof(T) + 2 * P * ST * sizeof(float) + 16;
  if (sh > kMaxShared) return 1;
  // The attribute is stored per (function, device) while a `static bool` guard is per process, so the flag is kept
  // per device; a process that uses a second GPU would otherwise launch with the 48 KiB default and fail on every
  // request above it. cudaGetDevice is a thread-local read that runs ahead of the launch.
  static bool done[64] = {};
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 63;
  if (!done[dev]) {
    cudaError_t e = cudaFuncSetAttribute((const void*)fwd_kernel<T, ST, NT, P>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kMaxShared);
    if (e != cudaSuccess) return 2;
    done[dev] = true;
  }
  dim3 g(a0.nTasks * P, has1 ? 2 : 1);
  fwd_kernel<T, ST, NT, P><<<g, NT, sh, st>>>(a0, a1);
  return 0;
}

template <typename T, int ST, int NT>
static int launch_bwd(const BwdArgs& a, cudaStream_t st) {
  const size_t sh = (size_t)ST * (a.C + PAD) * sizeof(T) + 2 * ST * sizeof(float) + 16;
  if (sh > kMaxShared) return 1;
  // per device, not per process - see the note in launch_fwd
  static bool done[64] = {};
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 63;
  if (!done[dev]) {
    cudaError_t e = cudaFuncSetAttribute((const void*)bwd_kernel<T, ST, NT>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kMaxShared);
    if (e != cudaSuccess) return 2;
    done[dev] = true;
  }
  bwd_kernel<T, ST, NT><<<a.nTasks, NT, sh, st>>>(a);
  return 0;
}

// ST: channels per CTA, P: CTAs of a cluster sharing one chunk. (16,2) is the unmasked stream: a 32 B row of key, a
// 66 064 B tile that fits three CTAs per SM by shared memory but is held to two by the register file (64 regs x 512
// threads), and only 2 ranks to synchronize. (8,1) is the masked stream, where most loads are skipped and the call is
// latency bound, so the narrow block with twice the CTAs wins. (32,4) and (16,1) are the fallbacks.
// A wider run of one key row per warp instruction buys read rate (about 4% for the 16 -> 32 channel step on a
// read-only stream), but the tile grows with the width: a (32,2) tile of 132 112 B fits one CTA per SM only, which
// costs more than the width gains. (32,4) splits the chunk four ways to keep the tile small.
template <typename T>
static int launch_fwd_t(const FwdArgs& a0, const FwdArgs& a1, int has1, int ST, int P, cudaStream_t st) {
  if (ST == 16 && P == 2) return launch_fwd<T, 16, 512, 2>(a0, a1, has1, st);
  if (ST == 32 && P == 4) return launch_fwd<T, 32, 512, 4>(a0, a1, has1, st);
  if (ST == 16 && P == 1) return launch_fwd<T, 16, 512, 1>(a0, a1, has1, st);
  if (ST == 8 && P == 1) return launch_fwd<T, 8, 512, 1>(a0, a1, has1, st);
  // The long-chunk cells: 8 channels is the only width whose slice comes back under a split's 68 KiB budget at a
  // 4-byte element (8 x 2056 x 4 + 272 = 66 064 B at P = 4). P = 2 is not among them for a 2-byte key: its budget can
  // only pass where the P = 1 cells already answer (see pick_sp in the binding). For a 4-byte element it is the first
  // candidate and runs against the hard shared-memory limit instead: 102 800 B at C = 6400 up to 201 104 B at
  // C = 12 544, i.e. two CTAs per SM up to C = 7 239 and one above, because above C ~ 6357 no whole-chunk (P = 1)
  // tile fits the limit at all.
  if (ST == 8 && P == 2) return launch_fwd<T, 8, 512, 2>(a0, a1, has1, st);
  if (ST == 8 && P == 4) return launch_fwd<T, 8, 512, 4>(a0, a1, has1, st);
  if (ST == 8 && P == 8) return launch_fwd<T, 8, 512, 8>(a0, a1, has1, st);
  return 1;
}
template <typename T>
static int launch_bwd_t(const BwdArgs& a, int ST, cudaStream_t st) {
  if (ST == 16) return launch_bwd<T, 16, 512>(a, st);
  if (ST == 8) return launch_bwd<T, 8, 512>(a, st);
  return 1;
}

// dtype: 0 bf16, 1 fp16, 2 fp32
int fwd_launch(const FwdArgs& a0, const FwdArgs& a1, int has1, int dtype, int ST, int P, cudaStream_t st) {
  if (dtype == 0) return launch_fwd_t<__nv_bfloat16>(a0, a1, has1, ST, P, st);
  if (dtype == 1) return launch_fwd_t<__half>(a0, a1, has1, ST, P, st);
  if (dtype == 2) return launch_fwd_t<float>(a0, a1, has1, ST, P, st);
  return 1;
}
int bwd_launch(const BwdArgs& a, int dtype, int ST, cudaStream_t st) {
  if (dtype == 0) return launch_bwd_t<__nv_bfloat16>(a, ST, st);
  if (dtype == 1) return launch_bwd_t<__half>(a, ST, st);
  if (dtype == 2) return launch_bwd_t<float>(a, ST, st);
  return 1;
}

}  // namespace awm
