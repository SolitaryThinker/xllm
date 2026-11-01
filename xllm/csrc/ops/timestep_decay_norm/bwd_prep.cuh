// Per-(row, step) bias correction of TDN: 1/(1 - beta**count) for beta1 and beta2, where count is the number of valid
// steps in the step's segment (reset at bos & valid, padded steps leave it alone, prev_count carries in).
//
// The backward pass needs it: the saved statistics are only cummean/cumrstd, so the count cannot be recovered from
// them, and 1 - beta**count has to stay accurate to counts of 1e6 and more (prefill carries counts in the hundreds of
// thousands).
//
// Two ways in:
//   - bias_into_shared<...>(...): the device prologue a tile kernel can call - a tile of (row, steps [t0, t0+n)) gets
//     its per-step (inv1, inv2) in shared memory, no extra launch, no flag, no scratch in HBM. A tile that does not
//     start at 0 recomputes the count prefix [0, t0) with the same block scan and applies it to prev_count, so nothing
//     has to be published or polled; the cost is the masks reread (from L2, once per tile).
//   - prep_kernel<...>: the same thing as a standalone launch writing float2 [B, L].
//
// A segment's aggregate is the monoid (h, a, s): f(c) = h ? s : c + a (h: a reset came at or before the end, a: valid
// steps, s: valid steps since the last reset), which composes associatively - that is what makes it one block scan.
//
// 1 - beta**count is -expm1f(count * logf(beta)): forming exp first loses ~3e-5 relative in 1 - beta**count at
// count = 1 in fp32, expm1 does not.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace xllm {
namespace ops {
namespace prep {

// A segment's aggregate, applied to the count entering it: f(c) = h ? s : c + a.
struct Mon {
  int h, a, s;
};

__device__ __forceinline__ Mon mon_id() { return Mon{0, 0, 0}; }

// x followed by y.
__device__ __forceinline__ Mon mon_cat(Mon x, Mon y) {
  Mon r;
  r.a = x.a + y.a;
  r.h = x.h | y.h;
  r.s = y.h ? y.s : (x.h ? x.s + y.a : 0);
  return r;
}

__device__ __forceinline__ int mon_apply(Mon m, int c) { return m.h ? m.s : c + m.a; }

__device__ __forceinline__ Mon warp_scan_inclusive(Mon v, int lane) {
  Mon inc = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    Mon o;
    o.h = __shfl_up_sync(0xffffffffu, inc.h, d);
    o.a = __shfl_up_sync(0xffffffffu, inc.a, d);
    o.s = __shfl_up_sync(0xffffffffu, inc.s, d);
    if (lane >= d) inc = mon_cat(o, inc);
  }
  return inc;
}

// Exclusive scan over the block's threads, with the block's total in *total when it is not null.
// kThreads is the block size; shm must hold kThreads/32 + 1 Mon. Every thread of the block must call this.
template <int kThreads>
__device__ __forceinline__ Mon block_scan(Mon v, int tid, Mon* shm, Mon* total) {
  constexpr int kWarps = kThreads / 32;
  const int lane = tid & 31, warp = tid >> 5;
  const Mon inc = warp_scan_inclusive(v, lane);
  // The shuffles must be executed by every lane of the warp: only the select is under the lane test.
  Mon ex = mon_id();
  {
    Mon o;
    o.h = __shfl_up_sync(0xffffffffu, inc.h, 1);
    o.a = __shfl_up_sync(0xffffffffu, inc.a, 1);
    o.s = __shfl_up_sync(0xffffffffu, inc.s, 1);
    if (lane > 0) ex = o;
  }
  if (lane == 31) shm[warp] = inc;   // the warp's total is the inclusive value at its last lane
  __syncthreads();
  if (warp == 0) {
    const Mon w = (lane < kWarps) ? shm[lane] : mon_id();
    const Mon wi = warp_scan_inclusive(w, lane);
    Mon we = mon_id();
    {
      Mon o;
      o.h = __shfl_up_sync(0xffffffffu, wi.h, 1);
      o.a = __shfl_up_sync(0xffffffffu, wi.a, 1);
      o.s = __shfl_up_sync(0xffffffffu, wi.s, 1);
      if (lane > 0) we = o;
    }
    if (lane < kWarps) shm[lane] = we;
    if (lane == kWarps - 1) shm[kWarps] = wi;
  }
  __syncthreads();
  const Mon res = mon_cat(shm[warp], ex);
  if (total != nullptr) *total = shm[kWarps];
  return res;
}

// (valid, reset) of a step, at an address already offset by the row.
template <bool kHasBos, bool kHasPad>
__device__ __forceinline__ void step_flags(const bool* __restrict__ bos, const bool* __restrict__ pad,
                                           long long at, bool& valid, bool& reset) {
  valid = !(kHasPad && pad[at]);
  reset = kHasBos && valid && bos[at];
}

// The aggregate of the steps [l, r) of a row (addresses already offset by the row).
template <bool kHasBos, bool kHasPad>
__device__ __forceinline__ Mon row_range(const bool* __restrict__ bos, const bool* __restrict__ pad, long long l,
                                         long long r) {
  Mon m = mon_id();
  for (long long at = l; at < r; ++at) {
    bool valid, reset;
    step_flags<kHasBos, kHasPad>(bos, pad, at, valid, reset);
    if (valid) {
      if (reset) {
        m.h = 1;
        m.s = 1;
      } else if (m.h) {
        ++m.s;
      }
      ++m.a;
    }
  }
  return m;
}

// The count entering step t0 of a row: prev_count plus every valid step before it (a reset before it wins).
// Every thread of the block must call this; it ends with a barrier.
//
// Coalesced, two phases: one strided pass over [0, t0) finds the last reset and the number of valid steps, then (only
// when a reset was found) a second pass counts the valid steps above it. A strided pass matters: with a contiguous
// chunk per thread each warp's 32 lanes are 4 KB apart, so the byte loads cost a sector each.
template <int kThreads, bool kHasBos, bool kHasPad>
__device__ __forceinline__ int tile_start_count(const bool* __restrict__ bos, const bool* __restrict__ pad,
                                               long long row, int t0, int prev_count, int tid, Mon* shm) {
  constexpr int kWarps = kThreads / 32;
  const int lane = tid & 31, warp = tid >> 5;
  int* s = reinterpret_cast<int*>(shm);   // 3 * (kWarps + 1) ints, enough for 2 * kWarps + 2
  if (t0 <= 0) {
    __syncthreads();
    return prev_count;
  }
  if (!kHasBos && !kHasPad) {   // no mask can change it: the count entering step t0 is just the step index
    __syncthreads();
    return prev_count + t0;
  }
  if (kHasBos && !kHasPad) {
    // Only the last reset before t0 matters (a reset wipes every count above it) and every step is valid, so the
    // count entering t0 is t0 - last_reset.
    //
    // Sixteen steps a load. `bos` is a byte array, so one 16-byte read gives 16 steps' flags and the block covers
    // kThreads * 16 = 2048 steps an iteration, instead of chaining dependent batches of 8 single-byte reads until
    // every thread has found a reset. The row's base byte is not necessarily 16-byte aligned, so the vector grid is
    // aligned to the pointer and step i sits at global byte mis + i. Every read below is inside the row: the vector
    // loop starts at chunk 1 when mis > 0, and the <= 15 bytes of the row below the first aligned chunk, and the ones
    // past the last, are walked one byte at a time by lane 0. A contiguous view with a storage offset keeps its
    // misalignment (`expect_contiguous` does not promise alignment), and xllm hands these in as views of a cache or a
    // packed batch.
    const char* const b0 = reinterpret_cast<const char*>(bos + row);
    const unsigned mis = static_cast<unsigned>(reinterpret_cast<uintptr_t>(b0) & 15u);
    const uint4* const q = reinterpret_cast<const uint4*>(b0 - mis);
    const int gtop = int(mis) + t0;            // the range is global bytes [mis, gtop)
    const int cfull = gtop >> 4;               // chunks wholly inside it
    int last = -1;
    // The row's own bytes below its first aligned chunk: [mis, min(16, gtop)). Lane 0 walks at most 15 bytes.
    if (tid == 0 && mis)
      for (int g = int(mis), e = (gtop < 16 ? gtop : 16); g < e; ++g)
        if (b0[g - int(mis)]) last = g - int(mis);
    const int cbeg = (mis ? 1 : 0);             // chunk 0 straddles the row's base when mis > 0
    for (int c = cbeg + tid; c < cfull; c += kThreads) {
      const uint4 v = __ldg(q + c);
      // bytes are 0 or 1, so a byte-wise OR of the four words is non-zero iff some step of the chunk is a bos: one
      // test rejects the 15 chunks in 16 that hold nothing, and only a chunk with a hit pays the 16 comparisons.
      if ((v.x | v.y | v.z | v.w) != 0u) {
        unsigned mask = 0;
        mask |= (v.x & 0x000000ffu) ? 0x0001u : 0u;
        mask |= (v.x & 0x0000ff00u) ? 0x0002u : 0u;
        mask |= (v.x & 0x00ff0000u) ? 0x0004u : 0u;
        mask |= (v.x & 0xff000000u) ? 0x0008u : 0u;
        mask |= (v.y & 0x000000ffu) ? 0x0010u : 0u;
        mask |= (v.y & 0x0000ff00u) ? 0x0020u : 0u;
        mask |= (v.y & 0x00ff0000u) ? 0x0040u : 0u;
        mask |= (v.y & 0xff000000u) ? 0x0080u : 0u;
        mask |= (v.z & 0x000000ffu) ? 0x0100u : 0u;
        mask |= (v.z & 0x0000ff00u) ? 0x0200u : 0u;
        mask |= (v.z & 0x00ff0000u) ? 0x0400u : 0u;
        mask |= (v.z & 0xff000000u) ? 0x0800u : 0u;
        mask |= (v.w & 0x000000ffu) ? 0x1000u : 0u;
        mask |= (v.w & 0x0000ff00u) ? 0x2000u : 0u;
        mask |= (v.w & 0x00ff0000u) ? 0x4000u : 0u;
        mask |= (v.w & 0xff000000u) ? 0x8000u : 0u;
        const int i = ((c << 4) + (31 - __clz(mask))) - int(mis);
        if (i > last) last = i;
      }
    }
    // The at most 15 bytes of the range past the last full chunk, and (when the range is shorter than a chunk) the
    // whole of it; never below `mis`, so the walk stays inside the row.
    if (tid == 0) {
      const int gfrom = (cfull << 4) > int(mis) ? (cfull << 4) : int(mis);
      for (int g = gfrom; g < gtop; ++g)
        if (b0[g - int(mis)]) last = g - int(mis);
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      const int o = __shfl_down_sync(0xffffffffu, last, off);
      if (o > last) last = o;
    }
    if (lane == 0) s[warp] = last;
    __syncthreads();
    if (tid == 0) {
      int r = -1;
      for (int w = 0; w < kWarps; ++w)
        if (s[w] > r) r = s[w];
      s[0] = r;
    }
    __syncthreads();
    const int r = s[0];
    __syncthreads();   // the caller reuses shm after this
    return r >= 0 ? t0 - r : prev_count + t0;
  }
  int cnt = 0, last = -1;
  for (int at = tid; at < t0; at += kThreads) {
    bool valid, reset;
    step_flags<kHasBos, kHasPad>(bos, pad, row + at, valid, reset);
    cnt += valid ? 1 : 0;
    if (reset) last = at;
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    cnt += __shfl_down_sync(0xffffffffu, cnt, off);
    const int o = __shfl_down_sync(0xffffffffu, last, off);
    if (o > last) last = o;
  }
  if (lane == 0) {
    s[warp] = cnt;
    s[kWarps + warp] = last;
  }
  __syncthreads();
  if (tid == 0) {
    int c = 0, r = -1;
    for (int w = 0; w < kWarps; ++w) {
      c += s[w];
      if (s[kWarps + w] > r) r = s[kWarps + w];
    }
    s[2 * kWarps] = c;
    s[2 * kWarps + 1] = r;
  }
  __syncthreads();
  const int c_all = s[2 * kWarps];
  const int r = s[2 * kWarps + 1];
  __syncthreads();   // the caller reuses shm after this
  if (r < 0) return prev_count + c_all;   // no reset before t0
  // A reset at r: the count restarts there (the reset step itself counts 1), then every valid step above it.
  int c2 = 0;
  for (int at = r + 1 + tid; at < t0; at += kThreads) {
    bool valid, reset;
    step_flags<kHasBos, kHasPad>(bos, pad, row + at, valid, reset);
    c2 += valid ? 1 : 0;
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) c2 += __shfl_down_sync(0xffffffffu, c2, off);
  if (lane == 0) s[warp] = c2;
  __syncthreads();
  if (tid == 0) {
    int c = 0;
    for (int w = 0; w < kWarps; ++w) c += s[w];
    s[0] = c;
  }
  __syncthreads();
  const int c_above = s[0];
  __syncthreads();   // the caller reuses shm after this
  return 1 + c_above;
}

// One bias correction and its reciprocal, from a count. alpha = beta**count, inv = 1/(1 - alpha) (0 where that is not
// positive, i.e. count 0).
__device__ __forceinline__ void bias(long long count, float lb, float& alpha, float& inv) {
  const float c = count > (1LL << 23) ? float(1LL << 23) : float(count);  // past this 1 - beta**count is 1 in fp32
  const float d = -expm1f(c * lb);
  alpha = 1.f - d;
  inv = d > 0.f ? 1.f / d : 0.f;
}

// The per-step (inv1, inv2) of a row's steps [t0, t0 + n) into shared memory s_inv[0..n), for a tile kernel's prologue:
// no launch, no scratch in HBM, no flag. Every thread of the block must call it; it ends with a barrier, after which
// any thread may read s_inv. shm must hold kThreads/32 + 1 Mon. Pass c_in_out (any location thread 0 can write, shared
// or global) to also get back the count entering step t0, i.e. the count the tile's own steps continue from.
//
// kZeroOnPad: a padded step writes 0 instead of the state's own 1/(1 - beta**count). With false, a padded step gets
// the true value (the statistics are defined at a padded step, from the unchanged state); with true, inv is already
// the mask a caller would otherwise apply to its own dM/dV (0 * anything = 0, never 0/0: the count is 0 only before
// any valid step, where inv is 0 in either case).
template <int kThreads, bool kHasBos, bool kHasPad, bool kZeroOnPad = false>
__device__ __forceinline__ void bias_into_shared(const bool* __restrict__ bos, const bool* __restrict__ pad,
                                                long long row, int t0, int n, int prev_count, float lb1, float lb2,
                                                float2* s_inv, Mon* shm, int* c_in_out = nullptr) {
  // `row` is not a row index: it is the flat element offset b * L of this row's first step in bos/pad, and every
  // caller passes b * L. The walk below reads bos[row + t] for t in [0, t0). `t0` is the first step this tile covers.
  //
  // Its grid must not be widened to make it cheaper: the walk over [0, t0) is O(t0) and the last tile walks all of L
  // whatever the tiling, so total work grows as 1/T when the tile shrinks. What makes it cheap is reading sixteen mask
  // bytes a load.
  int c_in = tile_start_count<kThreads, kHasBos, kHasPad>(bos, pad, row, t0, prev_count, threadIdx.x, shm);
  // The count entering the tile, for a caller whose own scan needs it (t0 + 1 steps above prev_count, or less after a
  // reset): handed back here so that it need not re-run tile_start_count, which is what this module exists to avoid.
  // Block-uniform by construction; written before shm is reused and read after the function's final barrier.
  if (c_in_out != nullptr && threadIdx.x == 0) *c_in_out = c_in;
  const int tid = threadIdx.x;
  const int per = (n + kThreads - 1) / kThreads;
  const int j0 = tid * per;
  const int j1 = (j0 + per) < n ? (j0 + per) : n;
  Mon mine = (j0 < j1) ? row_range<kHasBos, kHasPad>(bos, pad, row + t0 + j0, row + t0 + j1) : mon_id();
  const Mon ex = block_scan<kThreads>(mine, tid, shm, nullptr);
  int cnt = mon_apply(ex, c_in);
  for (int j = j0; j < j1; ++j) {
    bool valid, reset;
    step_flags<kHasBos, kHasPad>(bos, pad, row + t0 + j, valid, reset);
    if (valid) {
      if (reset) cnt = 0;
      ++cnt;
    }
    float a1, rcp1, a2, rcp2;
    bias(cnt, lb1, a1, rcp1);
    bias(cnt, lb2, a2, rcp2);
    s_inv[j] = (kZeroOnPad && !valid) ? make_float2(0.f, 0.f) : make_float2(rcp1, rcp2);
  }
  __syncthreads();   // the caller reuses shm, and reads s_inv after this
}

// ---- the standalone launch ----

constexpr int kThreads = 512;
constexpr int kTile = 512;
constexpr int kWarps = kThreads / 32;

// out: float2 [B, L] = (inv1, inv2)
template <bool kHasBos, bool kHasPad>
__global__ void __launch_bounds__(kThreads) prep_kernel(const bool* __restrict__ bos, const bool* __restrict__ pad,
                                                        const int64_t* __restrict__ prev_count,
                                                        float2* __restrict__ out, int L, float lb1, float lb2,
                                                        int max_count) {
  __shared__ Mon shm[kWarps + 1];
  __shared__ float2 s_inv[kTile];
  const long long b = blockIdx.y;
  const long long row = b * L;
  const int t0 = blockIdx.x * kTile;
  const int n = L - t0 < kTile ? L - t0 : kTile;
  long long pc = prev_count[b];
  if (pc > max_count) pc = max_count;
  bias_into_shared<kThreads, kHasBos, kHasPad>(bos, pad, row, t0, n, (int)pc, lb1, lb2, s_inv, shm);
  for (int j = threadIdx.x; j < n; j += kThreads) out[row + t0 + j] = s_inv[j];
}

}  // namespace prep
}  // namespace ops
}  // namespace xllm
