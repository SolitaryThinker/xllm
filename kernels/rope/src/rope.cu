// xllm's rotary position embedding in one pass: q and k (or one tensor) rotated in a single launch.
//
// x [B, L, H, R] is read as R / 2 complex numbers (x[..., 2j], x[..., 2j+1]) and multiplied by freqs_cis [L, R / 2]
// (complex64) along L, broadcast over B and H; the backward pass multiplies by the conjugate. Held in fp32 as xllm
// does, then rounded to x's dtype.
//
// The kernel: one CTA per (position, batch, chunk of row slots). Each thread owns one 16-byte vector of a row (VEC
// elements = VEC / 2 complex pairs) and keeps that vector's cos/sin in registers for the whole CTA, so a position's
// table entry is read once per thread and reused over every row it touches. A thread takes RPT consecutive rows of one
// tensor at the same position - q's rows or k's, never a mix: the slot decides which - and issues their loads before
// any of their stores, so RPT requests are in flight per thread instead of one: a single pass over memory needs those
// to keep the pipe full. The row slots of a block are matched to the rows being rotated, so a block does not carry idle
// slots when the heads are few (8 q + 4 k rows, or 4 + 4).
// Consecutive lanes take consecutive vectors of a row, so a warp's lanes cover one row contiguously and the loads are
// fully coalesced 16-byte reads and streaming stores; a thread's other rows are isH apart in the same tensor.
// Inputs are read where the call's strides put them (a view of wider heads, an unbind view, a [B, H, L, D] gradient);
// outputs are written with the output's own strides.
#include "rope.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kMaxThreads = 256;  // the threads a block of rope_kernel may hold: its __launch_bounds__, and the cap
                                  // that npb_max = kMaxThreads / vpr and the position padding keep the block to

// ---- dtype conversion: up to fp32, and back with round-to-nearest-even (what the reference's rounding does) ----
template <typename T>
struct Cvt;
template <>
struct Cvt<__nv_bfloat16> {
  static __device__ __forceinline__ float up(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 dn(float v) { return __float2bfloat16(v); }
};
template <>
struct Cvt<__half> {
  static __device__ __forceinline__ float up(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half dn(float v) { return __float2half(v); }
};
template <>
struct Cvt<float> {
  static __device__ __forceinline__ float up(float v) { return v; }
  static __device__ __forceinline__ float dn(float v) { return v; }
};

struct Params {
  const char* in[2];
  char* out[2];
  int H[2];
  long isB[2], isL[2], isH[2];  // input strides, bytes
  long osB[2], osL[2], osH[2];  // output strides, bytes
  const float2* fr;
  long fsL;  // step between positions of the table, in complex
  long fsP;  // step between the pairs of a position, in complex
  int L;    // positions of the call (a block tile may run past the last one)
  int npbq;  // row slots the first tensor owns; a slot >= npbq belongs to the second
};

// The vector a thread moves in one load: 16 bytes on the fast path, 8 or 4 on the narrower tiers that a width or an
// alignment that is not a multiple of 16 forces. Each tier keeps its own load width, so the bytes in flight fall with
// the tier instead of collapsing to one element a lane (2.7-2.9 TB/s on one element a lane against 6.4 TB/s with the
// vector tiers).
template <int BYTES>
struct WideT {
  static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16, "a vector tier is 4, 8 or 16 bytes");
  using type = void;
};
template <>
struct WideT<16> {
  using type = uint4;
};
template <>
struct WideT<8> {
  using type = uint2;
};
template <>
struct WideT<4> {
  using type = unsigned int;
};

// One thread's row: that vector when the width and the alignment allow it, VEC elements of T when they do not (a
// uint4 member would cost four registers a row for the two elements it uses).
template <typename T, int VEC, bool WIDE>
struct SlotT {
  using W = typename WideT<VEC * (int)sizeof(T)>::type;
  union {
    W r;
    T v[VEC];
  };
};
template <typename T, int VEC>
struct SlotT<T, VEC, false> {
  T v[VEC];
};

// One thread's RPT rows of one position: load them all, then rotate and write them all.
template <typename T, int VEC, bool CONJ, int RPT, bool WIDE>
__global__ __launch_bounds__(kMaxThreads) void rope_kernel(Params p) {
  constexpr int NP = VEC / 2;
  constexpr long VECB = (long)VEC * sizeof(T);
  using WT = typename WideT<VEC * (int)sizeof(T)>::type;
  const int l = blockIdx.x * blockDim.z + threadIdx.z;  // this thread's position in the tile
  if (l >= p.L) return;
  const int b = blockIdx.y;
  const int j = threadIdx.x;   // this thread's vector of the row
  const int rl = threadIdx.y;  // its first row slot
  const int npb = blockDim.y;  // row slots per block
  // This thread's slot in the union slot space: the first npbq slots hold tensor 0's rows, the rest tensor 1's. `t` and
  // the tensor's strides are therefore fixed for all RPT rows this thread owns, so the compiler hoists the t-indexed
  // Params loads out of the row loop and the address walks by isH instead of being rebuilt per row (16% fewer
  // instructions, 1.3% faster on the 40-row cases).
  const int sl = blockIdx.z * npb + rl;
  const int t = sl >= p.npbq;
  const int h0 = sl - (t ? p.npbq : 0);
  const int Ht = p.H[t];

  // This thread's slice of the position's table row: read once, kept in registers for every row.
  float2 cs[NP];
  const float2* ftab = p.fr + (long)l * p.fsL + (long)j * NP * p.fsP;
#pragma unroll
  for (int i = 0; i < NP; ++i) cs[i] = __ldg(ftab + (long)i * p.fsP);

  // The rows this thread owns. A tier with a legal vector keeps them as that vector; the element-wise path (a pointer
  // with no alignment a vector can use) as VEC elements of T.
  SlotT<T, VEC, WIDE> s[RPT];
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    const int h = h0 * RPT + i;
    if (h < Ht) {
      const char* in = p.in[t] + (long)b * p.isB[t] + (long)l * p.isL[t] + (long)h * p.isH[t] + (long)j * VECB;
      if constexpr (WIDE) {
        s[i].r = __ldcs(reinterpret_cast<const WT*>(in));
      } else {
        const T* src = reinterpret_cast<const T*>(in);
#pragma unroll
        for (int k = 0; k < VEC; ++k) s[i].v[k] = src[k];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    const int h = h0 * RPT + i;
    if (h < Ht) {
      char* out = p.out[t] + (long)b * p.osB[t] + (long)l * p.osL[t] + (long)h * p.osH[t] + (long)j * VECB;
      float x[VEC];
#pragma unroll
      for (int k = 0; k < VEC; ++k) x[k] = Cvt<T>::up(s[i].v[k]);
#pragma unroll
      for (int k = 0; k < NP; ++k) {
        const float c = cs[k].x, sn = cs[k].y;
        const float re = x[2 * k], im = x[2 * k + 1];
        // (re + i im) * (c -/+ i sn)
        const float a = CONJ ? fmaf(im, sn, re * c) : fmaf(-im, sn, re * c);
        const float d = CONJ ? fmaf(-re, sn, im * c) : fmaf(re, sn, im * c);
        s[i].v[2 * k] = Cvt<T>::dn(a);
        s[i].v[2 * k + 1] = Cvt<T>::dn(d);
      }
      if constexpr (WIDE) {
        __stcs(reinterpret_cast<WT*>(out), s[i].r);
      } else {
        T* dst = reinterpret_cast<T*>(out);
#pragma unroll
        for (int k = 0; k < VEC; ++k) dst[k] = s[i].v[k];
      }
    }
  }
}

// A width the shaped launch cannot cover from one block (`R / vec > kMaxThreads`, for example past 2048 channels at
// bf16/fp16 on the 16-byte tier): one pair a thread, the row's pairs walked in strides of the block. It is outside the
// shapes xllm rotates (its widest head rotates 256 channels) and exists so that any width is computed instead of
// refused. It is the fallback under `rope_wide_vec`: it takes the calls whose tier is the element one (a base pointer
// or a stride with no pair alignment), where two elements at a time is the widest load that exists, and the calls whose
// row slots pass what grid.z holds.
template <typename T, bool CONJ>
__global__ __launch_bounds__(kMaxThreads) void rope_wide(Params p, const int npairs) {
  constexpr long PB = 2 * sizeof(T);  // bytes of a complex pair
  const int l = blockIdx.x;
  const int b = blockIdx.y;
  const int H0 = p.H[0], rows = p.H[0] + p.H[1];
  // `long` on the row index: `rows` is at most 2^31 - 1 (binding.cpp refuses more) and the stride is
  // `blockDim.y * gridDim.z`, at most 2 * 65,535 = 131,070 once `zs` is capped. With an `int` index, a row taken past
  // 2^31 - 1 - 131,070 would wrap negative and still satisfy `r < rows`: the loop would read and write at a negative
  // row, an illegal address. The comparison stays with `int rows` because that value cannot exceed 2^31 - 1.
  for (long r = (long)blockIdx.z * blockDim.y + threadIdx.y; r < rows; r += blockDim.y * gridDim.z) {
    const int t = r < H0 ? 0 : 1;
    const int h = r < H0 ? (int)r : (int)(r - H0);  // exact: r < rows <= 2^31 - 1
    const char* in = p.in[t] + (long)b * p.isB[t] + (long)l * p.isL[t] + (long)h * p.isH[t];
    char* out = p.out[t] + (long)b * p.osB[t] + (long)l * p.osL[t] + (long)h * p.osH[t];
    for (int j = threadIdx.x; j < npairs; j += blockDim.x) {
      const float2 cs = __ldg(p.fr + (long)l * p.fsL + (long)j * p.fsP);
      const T* src = reinterpret_cast<const T*>(in + (long)j * PB);
      const float x0 = Cvt<T>::up(src[0]), x1 = Cvt<T>::up(src[1]);
      const float c = cs.x, sn = cs.y;
      const float a = CONJ ? fmaf(x1, sn, x0 * c) : fmaf(-x1, sn, x0 * c);
      const float d = CONJ ? fmaf(-x0, sn, x1 * c) : fmaf(x0, sn, x1 * c);
      T* dst = reinterpret_cast<T*>(out + (long)j * PB);
      dst[0] = Cvt<T>::dn(a);
      dst[1] = Cvt<T>::dn(d);
    }
  }
}

template <typename T, bool CONJ>
void launch_wide(const Params& p, const int rows, const int L, const int B, const int npairs, cudaStream_t stream) {
  const int tx = 128, ty = 2;  // 256 threads: 128 pair lanes over 2 rows, four warps of lanes
  // `long` on the sum's left operand: up to 2^31 - 1 rows are admitted, and `rows + ty - 1` in `int` overflows at
  // exactly that value. `zs` would go negative, the clamp below would set it to 1, and the launch would still cover
  // every row (the loop strides by `blockDim.y * gridDim.z`) but in 2^30 serial passes. Host-side only, so the device
  // code does not change.
  int zs = (int)(((long)rows + ty - 1) / ty);
  if (zs < 1) zs = 1;
  // `grid.z` holds 65535 blocks and the row count is bounded only by the int that carries it, so a call with many
  // heads has more row slots than a grid dimension holds, and an uncapped launch would fail with
  // `cudaErrorInvalidValue`. Capping is enough because the loop above strides by `blockDim.y * gridDim.z`: every row
  // is visited once, wherever the block that owns it sits. A non-positive row count is a no-op rather than a launch.
  if (rows < 1) return;
  if (zs > 65535) zs = 65535;
  rope_wide<T, CONJ><<<dim3(L, B, zs), dim3(tx, ty, 1), 0, stream>>>(p, npairs);
}

// The wide path's vector body: a row too wide for one block is covered by several blocks along blockIdx.z, each
// taking kMaxThreads vectors of it: one 16-byte (or 8-byte, or 4-byte) vector a thread, the bytes in flight of the
// shaped launch's fast path, instead of the pair a thread `rope_wide` moves. The pair body runs at about 3.3 TB/s on a
// 4096-channel row whatever the head count, the rate the pair tier gets on a narrow row too: four bytes a lane, not
// the row's width, is what limits it.
//
// RPT rows a thread, as the shaped launch does it: one table read serves RPT rows. At R = 4096 a row's vector carries
// 16 B of data and its NP = 4 pairs carry 32 B of table, so a table read per row is twice the data it moves and one
// read per four rows is 32 B for 64 B.
//
// This body is reached only when `vpr > kMaxThreads` (the wide trigger) and the tier ladder admitted a vector of `VEC`
// elements, which proves that every row base is aligned to the vector this thread loads, because `aligned()` tests the
// base pointer and all six strides that way. VEC is 8, 4 or 2 elements, so every tier the shaped launch has (16, 8 and
// 4 bytes at bf16/fp16; 16 and 8 at fp32) is served here too, and `vec == 2` is the 4-byte vector body with four rows
// a thread. The element tier (no pair alignment) is the only call that keeps `rope_wide`. The table is deliberately
// read as `float2` (`__ldg` of 8 bytes, stepped by `fsP`), exactly as the pair body reads it, so a pair-strided table
// or a slice that starts anywhere in its storage takes this path with no table-alignment condition.
template <typename T, bool CONJ, int VEC, int RPT>
__global__ __launch_bounds__(kMaxThreads) void rope_wide_vec(Params p, const int vpr, const int rows,
                                                             const int slots) {
  constexpr int NP = VEC / 2;
  using S = SlotT<T, VEC, true>;
  const int slot = blockIdx.z % slots;   // this block's first row: slot * RPT
  const int tile = blockIdx.z / slots;   // which kMaxThreads-vector tile of the row
  const int jv = tile * kMaxThreads + (int)threadIdx.x;
  if (jv >= vpr) return;
  const float2* fr = p.fr + (long)blockIdx.x * p.fsL + (long)jv * NP * p.fsP;
  float2 cs[NP];
#pragma unroll
  for (int i = 0; i < NP; ++i) cs[i] = __ldg(fr + (long)i * p.fsP);
  const int H0 = p.H[0];
  const int r0 = slot * RPT;
  const int nr = rows - r0 < RPT ? rows - r0 : RPT;  // the last slot of a tensor may hold fewer rows
  S s[RPT];
  const char* src[RPT];
  char* dst[RPT];
#pragma unroll
  for (int i = 0; i < RPT; ++i) {  // every row's vector in flight before any of them is stored
    if (i >= nr) break;
    const int r = r0 + i;
    const int t = r < H0 ? 0 : 1;
    const int h = r < H0 ? r : r - H0;
    src[i] = p.in[t] + (long)blockIdx.y * p.isB[t] + (long)blockIdx.x * p.isL[t] + (long)h * p.isH[t];
    dst[i] = p.out[t] + (long)blockIdx.y * p.osB[t] + (long)blockIdx.x * p.osL[t] + (long)h * p.osH[t];
    s[i].r = reinterpret_cast<const S*>(src[i])[jv].r;
  }
#pragma unroll
  for (int i = 0; i < RPT; ++i) {  // indexed by the unrolled loop constant, not by a runtime counter
    if (i >= nr) break;
#pragma unroll
    for (int k = 0; k < NP; ++k) {
      const float x0 = Cvt<T>::up(s[i].v[2 * k]), x1 = Cvt<T>::up(s[i].v[2 * k + 1]);
      const float c = cs[k].x, sn = cs[k].y;
      s[i].v[2 * k] = Cvt<T>::dn(CONJ ? fmaf(x1, sn, x0 * c) : fmaf(-x1, sn, x0 * c));
      s[i].v[2 * k + 1] = Cvt<T>::dn(CONJ ? fmaf(-x0, sn, x1 * c) : fmaf(x0, sn, x1 * c));
    }
    reinterpret_cast<S*>(dst[i])[jv].r = s[i].r;
  }
}

template <typename T, bool CONJ, int VEC>
void launch_wide_vec(Params& p, const int rows, const int L, const int B, const int R, cudaStream_t stream) {
  const int vpr = R / VEC;  // exact: the tier ladder admits a vector only when it divides the row
  const int tiles = (vpr + kMaxThreads - 1) / kMaxThreads;
  constexpr int RPT = 4;    // rows a thread: one table read serves four rows (see the kernel's header comment)
  const int slots = (int)(((long)rows + RPT - 1) / RPT);  // `long`: rows is admitted up to 2^31 - 1
  const long z = (long)slots * tiles;
  // More row-tile pairs than grid.z holds, or more rows than the pair body's own grid.z can take (it packs two rows
  // into each of 65535 z slots, so 131070 rows is its ceiling too): the pair body takes the call.
  if (z > 65535 || rows > 131070) {
    if (CONJ)
      launch_wide<T, true>(p, rows, L, B, R / 2, stream);
    else
      launch_wide<T, false>(p, rows, L, B, R / 2, stream);
    return;
  }
  rope_wide_vec<T, CONJ, VEC, RPT><<<dim3(L, B, (unsigned)z), dim3(kMaxThreads), 0, stream>>>(p, vpr, rows, slots);
}

// Rows a thread keeps in flight, and the block shape that goes with it: one position per block over
// ceil(H[0] / RPT) + ceil(H[1] / RPT) row slots - a tensor's last slot carries up to RPT-1 idle rows when its head
// count is not a multiple of RPT, trimmed by the kernel's `h < Ht` guard - split over blockIdx.z when there are more
// slots than one block holds.
// A small call (decode: a few hundred kilobytes) does not have the bytes to fill the GPU from one pass, so there
// every thread takes one 16-byte chunk and the blocks stay small and many: its time is the launch and one dependent
// read, not the stream rate.
template <typename T, int VEC, bool CONJ, bool WIDE>
void launch_shaped(Params& p, const int vpr, const int L, const int B, const bool small, cudaStream_t stream) {
  const int npb_max = kMaxThreads / vpr < 1 ? 1 : kMaxThreads / vpr;
  const int rpt = small ? 1 : 4;
  // `long` on each sum's left operand, for the same reason as `launch_wide`'s `zs`: `binding.cpp` admits a single
  // tensor of up to 2^31 - 1 heads, and `H + rpt - 1` in `int` overflows from `H = 2^31 - 3`, which makes `npbq`
  // negative and the launch refused by the driver on a call the interface accepted. Host-side only.
  const int npbq = (int)(((long)p.H[0] + rpt - 1) / rpt);
  const int npbk = (int)(((long)p.H[1] + rpt - 1) / rpt);
  p.npbq = npbq;
  int npb = npbq + npbk;
  if (npb > npb_max) npb = npb_max;
  if (npb < 1) npb = 1;
  // Positions a block covers: enough of them that the block is whole warps of work rather than carrying idle row
  // slots, and no more threads than that needs. A block of 48 threads (12 rows, 4 a thread, vpr 16) leaves 16 lanes of
  // a warp with nothing to do, so two positions make it 96 threads of real work (three warps); 96 rather than 192 is
  // the faster of the two on the shapes that reach it.
  const int per_pos = vpr * npb;
  int pos = 1;
  if (!small) {
    for (int cand = 2; cand <= 8 && per_pos * pos < 96; cand *= 2) {
      if ((per_pos * cand) % 32 != 0) continue;
      if (per_pos * cand > kMaxThreads) break;
      pos = cand;
      if (per_pos * pos >= 96) break;
    }
    // Then fill the last warp. A block that is not a whole number of warps issues its memory instructions with lanes
    // that have nothing to do, and on the narrow tiers that is most of the block: at `per_pos = 50` (R = 100, two row
    // slots, three heads a position) no candidate clears the doubling loop's `% 32` filter (100 % 32 = 4, 200 % 32 = 8,
    // 400 % 32 = 16), so the loop stops at `pos = 1`: 50 threads, 14 of its 64 lanes idle. This search takes `pos = 5`,
    // 250 threads with 6 idle. The waste is not monotone in `pos`, so every candidate under the cap is tried rather
    // than stopping at the first that improves: at `per_pos = 75` the pick is `pos = 2` (150 threads, 10 idle) and at
    // `per_pos = 27` it is `pos = 7` (189 threads, 3 idle).
    //
    // The search differs from the doubling loop only where `per_pos < 96` has no whole-warp candidate there; above 96
    // the doubling loop does not run and both reduce to `per_pos % 32 != 0`. Where it fires, it picks the block with
    // fewer idle lanes, and where the doubling loop would leave a block under a warp the pick is far faster. At
    // `R = 36` bf16 with three rows a position (`vpr 9`, `npb 2`, `per_pos 18`) the doubling loop stops at `pos = 1`:
    // one block of 18 threads a position, `grid.x` 231424 at L = 231424, block 9x2x1. The search takes `pos = 7`, 126
    // threads, `grid.x` 33061, block 9x2x7: 7x fewer blocks, and a gain that grows with the grid (2.7x at L = 32768,
    // 3.1x at 65536, 3.4x at 127232, 3.4-3.6x from 231424 up). At `per_pos 50` the pick gains about 4% on a 100 MB
    // window and about 31% on a cold 410 MB one. R = 128 (`per_pos` 32) and R = 256 (`per_pos` 64) are served by the
    // doubling loop, which is why that loop is kept.
    //
    // For every shape xllm rotates, `per_pos * pos` after the doubling loop is a multiple of 32 (`per_pos` is 160, 128,
    // 80, 64, 48 or 32), so the search does not run and blocks are 96 to 256 threads, whole warps; the decode calls are
    // `small`, where `pos` stays 1. For R = 102 the answer depends on the row slots: with three (`per_pos = 153`) the
    // cap is one and there is nothing to choose; with one (`per_pos = 51`) the search takes `pos` 5, 255 threads and 1
    // idle lane; with two (`per_pos = 102`), `pos` 2, 204 threads and 20 idle.
    if (per_pos * pos % 32 != 0) {
      // Clamped to `blockDim.z`'s 64 as well as to the thread bound: at `per_pos <= 3` the thread bound alone would
      // allow a `pos` the launch cannot express (a zero-waste candidate exists at 16 or 32, which is a property of the
      // arithmetic and not of this line, so the clamp states the bound itself).
      const int by_threads = kMaxThreads / per_pos;
      const int cap = by_threads < 64 ? by_threads : 64;
      int best = pos, best_waste = 32 - (per_pos * pos) % 32;
      for (int cand = 1; cand <= cap; ++cand) {
        const int waste = (32 - (per_pos * cand) % 32) % 32;
        if (waste < best_waste) {
          best_waste = waste;
          best = cand;
        }
      }
      pos = best;
    }
  }
  // Positions past the end of the call do no work, so a block holding more of them than there are is idle by
  // construction. This matters only where `1 < L < pos`, which xllm's shapes do not reach.
  if (pos > L) pos = L < 1 ? 1 : L;
  const int zs = (npbq + npbk + npb - 1) / npb;
  // The row slots live in `grid.z` (65535) and `binding.cpp` bounds only B, L and the total head count: a call with
  // very many heads has more slots than that, and this body has no `z` stride (`sl = blockIdx.z * npb + rl`), so the
  // launch would fail with `cudaErrorInvalidValue`. The cap is on slots, so the head count that crosses it depends on
  // the row's width through `npb`: `rows > 262140 * npb`, i.e. 67,107,840 rows at R = 2 bf16 (`vpr` 1, `npb` 256) and
  // 2,100,000 rows at R = 256 (`vpr` 32, `npb` 8), which is 128 MB and 512 MB a tensor. The pair body strides over
  // `gridDim.z` and its `zs` is capped, so it carries such a call correctly at pair-tier speed, the same fallback as
  // for a row too wide for one block.
  if (zs > 65535) {
    if (CONJ)
      launch_wide<T, true>(p, p.H[0] + p.H[1], L, B, vpr * VEC / 2, stream);
    else
      launch_wide<T, false>(p, p.H[0] + p.H[1], L, B, vpr * VEC / 2, stream);
    return;
  }
  // `L` is caller-controlled and `binding.cpp` accepts up to 2^31 - 1, so this numerator must not be an `int` sum: it
  // overflows for `L >= 2^31 - pos + 1`, which makes `grid.x` negative and fails the launch with
  // `cudaErrorInvalidValue`. Host-side only.
  // The device is safe because `blockIdx.x * blockDim.z <= ceil(L/pos) * pos - pos <= L - 1`, the lane
  // `l = blockIdx.x * blockDim.z + threadIdx.z` reaches at most `L + pos - 2`, and the lanes past `p.L` are dropped by
  // `if (l >= p.L) return;`. With the product in range, `ceil(L/pos) * pos <= 2^31` exactly when `pos` divides 2^31 -
  // 32, 16, 8, the values this chooser picks for any shape the device can allocate (`per_pos` 1, 2, 4) - so `l` stays
  // at most 2^31 - 1 and the `int` holds. A `pos` that does not divide 2^31 (`per_pos` 9 -> 28, `per_pos` 25 -> 5)
  // would let `l` pass 2^31 - 1 with the guard unable to absorb it; that needs R >= 6 with >= 10 heads at this `L`,
  // i.e. more than 250 GiB a tensor.
  const dim3 grid((int)(((long)L + pos - 1) / pos), B, zs);
  const dim3 block(vpr, npb, pos);
  if (rpt == 4)
    rope_kernel<T, VEC, CONJ, 4, WIDE><<<grid, block, 0, stream>>>(p);
  else
    rope_kernel<T, VEC, CONJ, 1, WIDE><<<grid, block, 0, stream>>>(p);
}

template <typename T>
void launch_typed(Params& p, const int vec, const bool wide, const int rows, const int L, const int B, const int R,
                  const bool conj, const bool small, cudaStream_t stream) {
  constexpr int V16 = 16 / sizeof(T);
  const int vpr = R / vec;
  if (vpr > kMaxThreads) {  // a row wider than one block can cover: the wide path, which has no such limit
    // The vector body on every tier the shaped launch has, the pair body only for the element tier. `vec` is what the
    // ladder proved legal for this call (pointer and all six strides aligned to it, and it divides R), so each of these
    // arms is reached only when its vector load is well defined; `wide` distinguishes the pair tier from the element
    // one, which is why the 4-byte arm needs it and the wider ones do not. `if constexpr` keeps the arms a dtype cannot
    // reach out of the binary: fp32's V16 is 4, so there is no 8-byte-element body to compile for it.
    if (vec == V16) {
      if (conj)
        launch_wide_vec<T, true, V16>(p, rows, L, B, R, stream);
      else
        launch_wide_vec<T, false, V16>(p, rows, L, B, R, stream);
      return;
    }
    if constexpr (V16 != 4) {  // 2-byte dtypes: the 8-byte tier
      if (vec == 4) {
        if (conj)
          launch_wide_vec<T, true, 4>(p, rows, L, B, R, stream);
        else
          launch_wide_vec<T, false, 4>(p, rows, L, B, R, stream);
        return;
      }
    }
    if (wide) {  // the pair tier, 4 bytes at bf16/fp16 and 8 at fp32: a vector body like any other, four rows a thread
      if (conj)
        launch_wide_vec<T, true, 2>(p, rows, L, B, R, stream);
      else
        launch_wide_vec<T, false, 2>(p, rows, L, B, R, stream);
      return;
    }
    if (conj)
      launch_wide<T, true>(p, rows, L, B, R / 2, stream);
    else
      launch_wide<T, false>(p, rows, L, B, R / 2, stream);
    return;
  }
  if (vec == V16) {
    if (conj)
      launch_shaped<T, V16, true, true>(p, vpr, L, B, small, stream);
    else
      launch_shaped<T, V16, false, true>(p, vpr, L, B, small, stream);
  } else if (vec == 4 && V16 != 4) {
    // The 8-byte tier: a width that is a multiple of four elements but not of eight, or a pointer only 8-byte aligned.
    if (conj)
      launch_shaped<T, 4, true, true>(p, vpr, L, B, small, stream);
    else
      launch_shaped<T, 4, false, true>(p, vpr, L, B, small, stream);
  } else if (wide) {
    // The 4-byte tier in 2-byte dtypes, the 8-byte one in fp32: one uint (or uint2) a lane.
    if (conj)
      launch_shaped<T, 2, true, true>(p, vpr, L, B, small, stream);
    else
      launch_shaped<T, 2, false, true>(p, vpr, L, B, small, stream);
  } else {
    // No vector is legal here (a pointer two bytes off 16, say): one element at a time.
    if (conj)
      launch_shaped<T, 2, true, false>(p, vpr, L, B, small, stream);
    else
      launch_shaped<T, 2, false, false>(p, vpr, L, B, small, stream);
  }
}

}  // namespace

void rope_launch(bool conj, int B, int L, int R, const void* freqs, long fsL, long fsP, int dtype, RopeTens t1,
                 RopeTens t2, cudaStream_t stream) {
  if (B <= 0 || L <= 0 || R <= 0 || (t1.heads <= 0 && t2.heads <= 0)) return;
  const int es = dtype == 2 ? 4 : 2;
  // Whether a vector of v elements (v * es bytes) can be taken from every row of this tensor: both pointers and all
  // six strides must be multiples of the vector's width.
  const auto aligned = [es](const RopeTens& t, const long v) {
    if (t.heads <= 0) return true;
    const long b = v * es;
    if (reinterpret_cast<uintptr_t>(t.in) % b || reinterpret_cast<uintptr_t>(t.out) % b) return false;
    if (t.sB % v || t.sL % v || t.sH % v) return false;
    if (t.oB % v || t.oL % v || t.oH % v) return false;
    return true;
  };
  // Widest vector a thread can move first: 16 bytes, then 8, then 4, then one element at a time. Stepping down one
  // tier instead of jumping straight to the pair path keeps a width that is not a multiple of the 16-byte vector
  // (R % 8 != 0 in bf16, R % 4 != 0 in fp32) at half the bytes in flight rather than a quarter.
  int vec = 2;
  bool wide = false;
  for (int tier = 16; tier >= 4; tier /= 2) {
    const int v = tier / es;
    if (v < 2) continue;  // a pair is the narrowest thing the rotation can express
    if (R % v) continue;
    if (!aligned(t1, v) || !aligned(t2, v)) continue;
    vec = v;
    wide = true;
    break;
  }

  Params p{};
  p.in[0] = static_cast<const char*>(t1.in);
  p.out[0] = static_cast<char*>(t1.out);
  p.H[0] = t1.heads;
  p.in[1] = static_cast<const char*>(t2.in);
  p.out[1] = static_cast<char*>(t2.out);
  p.H[1] = t2.heads;
  const RopeTens* ts[2] = {&t1, &t2};
  for (int i = 0; i < 2; ++i) {
    p.isB[i] = ts[i]->sB * es;
    p.isL[i] = ts[i]->sL * es;
    p.isH[i] = ts[i]->sH * es;
    p.osB[i] = ts[i]->oB * es;
    p.osL[i] = ts[i]->oL * es;
    p.osH[i] = ts[i]->oH * es;
  }
  p.fr = static_cast<const float2*>(freqs);
  p.fsL = fsL;
  p.fsP = fsP;
  p.L = L;
  const int rows = t1.heads + t2.heads;
  // A call that moves a few hundred kilobytes (decode) cannot fill the GPU from one pass: it is a launch and one
  // dependent read, so it gets one 16-byte chunk a thread and many small blocks instead of rows in flight.
  const bool small = (long long)B * L * rows * R * es * 2 <= (8LL << 20);
  if (dtype == 0)
    launch_typed<__nv_bfloat16>(p, vec, wide, rows, L, B, R, conj, small, stream);
  else if (dtype == 1)
    launch_typed<__half>(p, vec, wide, rows, L, B, R, conj, small, stream);
  else
    launch_typed<float>(p, vec, wide, rows, L, B, R, conj, small, stream);
}

