// Backward of the group RMS norm and group layer norm with a persistent, warp-specialized TMA pipeline.
//
// One persistent CTA per SM, tiles handed out on demand (one a CTA to start, then from a counter in global memory, so
// the CTAs end together). Warp 0 is the producer: one lane streams tiles of TR whole rows into a ring of S
// shared-memory stages with 1D bulk copies (cp.async.bulk, completion counted on the stage's `full` mbarrier):
// x_or_y and dy (TR x C each: one contiguous byte range per tensor), rstd and, for the layer norm from x, mean
// (TR x G fp32). Warps 1..NW are consumers; each owns fixed columns of the row for the whole kernel:
//   narrow (V = cpg/8 <= 32 chunks of 16 B a group): a warp-chunk is GW = 32/V whole groups, one chunk a lane;
//     warp (c, wr) owns warp-chunks c + q*NWc (q < Q) and the rows wr, wr + NWr, ... of each tile; the group sums
//     are segmented shuffles (any V, 3/5/10 included);
//   wide (V > 32): warp c owns group c, Q = ceil(V/32) chunks a lane, full-warp sums, two reads of shared memory.
// Fixed columns keep gamma and the dgamma/dbeta partials in registers for the whole kernel. dx goes out with
// st.global.cs. x_or_y and dy are read from HBM once. With an affine transform the CTA partials are combined in
// shared memory (one slot a warp-row, added in order), written to a [grid, C] fp32 workspace, and after a grid
// barrier (cooperative launch, one CTA per SM) every CTA sums a slice of the columns over all partials in a fixed
// order and writes dgamma (and dbeta): the result is the same bits run to run.
//
// The consumer is what bounds this design (the producer waits on `empty` most of the time), so its per-row path is
// written for instruction count: the mode (norm, affine, memory_efficient) and the group width are compile-time,
// gamma is fp32 in registers, and x_hat is formed once a chunk as a*x + b.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

#include <map>
#include <mutex>
#include <utility>

#include "bwd_tma.h"
#include "tma.cuh"

namespace gn_tma {
namespace {

namespace cg = cooperative_groups;

struct BwdArgs {
  const uint8_t* x;    // x or y [R, C]
  const uint8_t* dy;   // [R, C]
  const float* mean;   // [R, G] (layer norm from x)
  const float* rstd;   // [R, G]
  const uint4* gamma;  // [C] (affine)
  uint4* dx;           // [R, C]
  float* part;         // [grid, nred * C] (affine)
  void* dgamma;
  void* dbeta;
  int* ctr;            // [2]: the next tile past the first grid, the CTAs done (zero at entry and at exit)
  long long R;
  int C, G, V, GW, U, TR, S, NWc, NWr, NWg, num_tiles;
  int stage_bytes, off_dy, off_rs, off_mu;
  float inv_cpg;
};

// ---- Tiles: the first gridDim.x tiles go one a CTA, every later one to the CTA whose producer asks first, from a
// counter in global memory. Round-robin tiles would leave the CTAs finishing far apart when the ring is latency-bound
// and the SMs' paths to memory differ, so most CTAs would idle at the dgamma grid barrier; taken on demand, a CTA on
// a slower path takes fewer tiles and every CTA ends within the tiles it has in flight. The tile
// index travels to the consumers in the stage (tidx[s], written before the arrive that completes the stage's phase;
// -1: no tile left). ----
__device__ __forceinline__ int next_tile(const BwdArgs& a, int /*t: the tile just issued*/) {
  return (int)gridDim.x + atomicAdd(a.ctr, 1);
}

__device__ __forceinline__ void end_of_tiles(int* tidx, uint64_t* full, int s) {
  tidx[s] = -1;
  mbar_arrive(&full[s]);  // a plain arrive with no bytes completes the phase the consumers wait on
}

__device__ __forceinline__ int stage_tile(const int* tidx, int s) { return reinterpret_cast<const volatile int*>(tidx)[s]; }

// Shared memory: a control block at fixed offsets first (full[8], empty[8], the stage tiles[8]: addressed with no
// register), then the ring of S <= kMaxStages stages, then the split kernel's row partials.
constexpr int kMaxStages = 8;
constexpr int kCtlBytes = 256;
static_assert(kMaxStages * (2 * 8 + 4) <= kCtlBytes, "control block");

template <typename T>
__device__ __forceinline__ void store_param(void* p, int i, float v);
template <>
__device__ __forceinline__ void store_param<__nv_bfloat16>(void* p, int i, float v) {
  reinterpret_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
}

// Sum over the group's lanes: compile-time power-of-two width VK (xor butterfly), or VK = 0 for a run-time width.
template <int VK>
__device__ __forceinline__ float gsum(float v, int V, int gl, int head, bool pow2) {
  if constexpr (VK > 0) {
#pragma unroll
    for (int o = 1; o < VK; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
  } else {
    return seg_sum(v, V, gl, head, pow2);
  }
}
template <int VK>
__device__ __forceinline__ float2 gsum2(float2 v, int V, int gl, int head, bool pow2) {
  if constexpr (VK > 0) {
#pragma unroll
    for (int o = 1; o < VK; o <<= 1) {
      v.x += __shfl_xor_sync(0xffffffffu, v.x, o);
      v.y += __shfl_xor_sync(0xffffffffu, v.y, o);
    }
    return v;
  } else {
    return seg_sum2(v, V, gl, head, pow2);
  }
}

// Per-lane constants of one owned chunk: gamma, fp32 (affine).
template <bool AFFINE, bool MEMEFF>
struct ChunkConst {
  float g[AFFINE ? 8 : 1];
};

// One chunk of one row: x_hat (8), g*dy (8), dy (8) in fp32. The instantiated modes are RMS and the layer norm with
// gamma from x (x_hat from the saved statistics) and RMS without gamma from y (x_hat = y).
template <typename T, bool LAYER, bool AFFINE, bool MEMEFF>
__device__ __forceinline__ void chunk_vals(const uint4& xv, const uint4& dv, const ChunkConst<AFFINE, MEMEFF>& k,
                                           float rs, float mu, float (&xh)[8], float (&gd)[8], float (&d)[8]) {
  static_assert(!(AFFINE && MEMEFF), "x_hat = (y - beta) / gamma is not instantiated: those calls decline");
  float x[8];
  unpack8<T>(xv, x);
  unpack8<T>(dv, d);
  if constexpr (MEMEFF) {
#pragma unroll
    for (int e = 0; e < 8; ++e) xh[e] = x[e];
  } else {
    const float b = LAYER ? -mu * rs : 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) xh[e] = LAYER ? fmaf(x[e], rs, b) : x[e] * rs;
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    if constexpr (AFFINE) gd[e] = k.g[e] * d[e];
    else gd[e] = d[e];
  }
}

// ---- dgamma / dbeta: CTA partials, then a grid-wide column sum; called by every thread of the CTA after the main
// loop. Bit-reproducible: every warp-row stores its columns in its own slot (each column has exactly one owner lane
// per warp-row), the slots are added in warp-row order, and the grid sum adds the CTA partials in a fixed order. The
// slots and the column-slice scratch reuse the ring (host checks they fit). ----
template <typename T, bool LAYER, int Q>
__device__ __forceinline__ void affine_epilogue(const BwdArgs& a, uint8_t* smem, int warp, int NW, const bool (&ok)[Q],
                                                const int (&ch)[Q], const float (&accg)[Q][8],
                                                const float (&accb)[LAYER ? Q : 1][8]) {
  const int nred = LAYER ? 2 : 1;
  const int NC = nred * a.C;
  float* red = reinterpret_cast<float*>(smem);  // [NWr][NC]
  __syncthreads();                              // every stage consumed: the ring is free
  if (warp >= 1 && warp <= NW) {
    float* mine = red + (size_t)((warp - 1) / a.NWc) * NC;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      if (ok[q]) {
        float4* g4 = reinterpret_cast<float4*>(mine + ch[q] * 8);
        g4[0] = make_float4(accg[q][0], accg[q][1], accg[q][2], accg[q][3]);
        g4[1] = make_float4(accg[q][4], accg[q][5], accg[q][6], accg[q][7]);
        if constexpr (LAYER) {
          float4* b4 = reinterpret_cast<float4*>(mine + a.C + ch[q] * 8);
          b4[0] = make_float4(accb[q][0], accb[q][1], accb[q][2], accb[q][3]);
          b4[1] = make_float4(accb[q][4], accb[q][5], accb[q][6], accb[q][7]);
        }
      }
    }
  }
  __syncthreads();
  float* mypart = a.part + (size_t)blockIdx.x * NC;
  for (int i = threadIdx.x; i < NC; i += blockDim.x) {
    float v = red[i];
    for (int w = 1; w < a.NWr; ++w) v += red[(size_t)w * NC + i];
    mypart[i] = v;
  }
  cg::this_grid().sync();
  // Every producer has taken its last tile (before its CTA's first barrier above): the counter goes back to zero.
  if (blockIdx.x == 0 && threadIdx.x == 0) a.ctr[0] = 0;

  const int per = (NC + gridDim.x - 1) / gridDim.x;
  const int j0 = blockIdx.x * per;
  const int j1 = min(NC, j0 + per);
  const int ncols = j1 - j0;
  if (ncols <= 0) return;
  const int nthr = blockDim.x;
  if (ncols * 2 > nthr) {
    for (int j = j0 + threadIdx.x; j < j1; j += nthr) {
      float sum = 0.f;
      for (int i = 0; i < (int)gridDim.x; ++i) sum += a.part[(size_t)i * NC + j];
      if (j < a.C) store_param<T>(a.dgamma, j, sum);
      else store_param<T>(a.dbeta, j - a.C, sum);
    }
    return;
  }
  const int slices = nthr / ncols;
  const int tcol = threadIdx.x % ncols, tsl = threadIdx.x / ncols;
  __syncthreads();
  if (tsl < slices) {
    float sum = 0.f;
    for (int i = tsl; i < (int)gridDim.x; i += slices) sum += a.part[(size_t)i * NC + j0 + tcol];
    red[tsl * ncols + tcol] = sum;
  }
  __syncthreads();
  if (threadIdx.x < ncols) {
    float tot = 0.f;
    for (int i = 0; i < slices; ++i) tot += red[i * ncols + threadIdx.x];
    const int j = j0 + threadIdx.x;
    if (j < a.C) store_param<T>(a.dgamma, j, tot);
    else store_param<T>(a.dbeta, j - a.C, tot);
  }
}

template <typename T, bool LAYER, bool AFFINE, bool MEMEFF, bool WIDE, int Q, int VK, int RR = 2>
__global__ void __launch_bounds__(WIDE ? 288 : 544, 1) gn_bwd_tma_kernel(const BwdArgs a) {
  extern __shared__ __align__(1024) uint8_t smem_all[];
  uint8_t* smem = smem_all + kCtlBytes;  // the ring (and, after it, the epilogue's scratch)
  constexpr bool NEED_MEAN = LAYER && !MEMEFF;
  const int S = a.S;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_all);
  uint64_t* empty = full + kMaxStages;
  int* tidx = reinterpret_cast<int*>(empty + kMaxStages);  // the tile in each stage
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = a.NWc * a.NWr;
  const int C8 = a.C >> 3;

  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NW);
    }
    fence_mbar_init();
  }
  __syncthreads();

  float accg[AFFINE ? Q : 1][8], accb[(AFFINE && LAYER) ? Q : 1][8];
  int ch[Q];
  bool ok[Q];
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    ch[q] = 0;
    ok[q] = false;
  }
#pragma unroll
  for (int q = 0; q < (AFFINE ? Q : 1); ++q)
#pragma unroll
    for (int e = 0; e < 8; ++e) accg[q][e] = 0.f;
#pragma unroll
  for (int q = 0; q < ((AFFINE && LAYER) ? Q : 1); ++q)
#pragma unroll
    for (int e = 0; e < 8; ++e) accb[q][e] = 0.f;

  if (warp == 0) {
    if (lane == 0) {
      int s = 0;
      uint32_t ph = 0;
      int t = blockIdx.x;
      for (int k = 0;; ++k) {
        if (k >= S) mbar_wait(&empty[s], ph ^ 1u);
        if (t >= a.num_tiles) {
          end_of_tiles(tidx, full, s);
          break;
        }
        tidx[s] = t;
        const long long r0 = (long long)t * a.TR;
        const int rows = (int)min((long long)a.TR, a.R - r0);
        const uint32_t xb = (uint32_t)rows * a.C * 2u;
        const uint32_t sb = (uint32_t)rows * a.G * 4u;
        uint8_t* st = smem + (size_t)s * a.stage_bytes;
        mbar_arrive_expect_tx(&full[s], 2u * xb + sb * (NEED_MEAN ? 2u : 1u));
        bulk_g2s(st, a.x + r0 * a.C * 2, xb, &full[s]);
        bulk_g2s(st + a.off_dy, a.dy + r0 * a.C * 2, xb, &full[s]);
        bulk_g2s(st + a.off_rs, a.rstd + r0 * a.G, sb, &full[s]);
        if (NEED_MEAN) bulk_g2s(st + a.off_mu, a.mean + r0 * a.G, sb, &full[s]);
        t = next_tile(a, t);  // its latency hides behind the wait for the next free stage
        if (++s == S) { s = 0; ph ^= 1u; }
      }
    }
  } else if (warp <= NW) {
    const int cw = warp - 1;
    const int cidx = cw % a.NWc, wr = cw / a.NWc;
    const int V = a.V;
    int grp[Q];
    ChunkConst<AFFINE, MEMEFF> kc[Q];
    const int gl = WIDE ? 0 : lane % V;
    const int head = lane - gl;
    const bool pow2 = (V & (V - 1)) == 0;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      int gg;
      bool okq;
      int c;
      if (!WIDE) {
        const int wc = cidx + q * a.NWc;
        gg = wc * a.GW + lane / V;
        okq = wc < a.U && lane < a.GW * V && gg < a.G;
        c = wc * a.GW * V + lane;
      } else {
        gg = cidx;
        const int j = q * 32 + lane;
        okq = j < V;
        c = gg * V + j;
      }
      ok[q] = okq;
      ch[q] = okq ? c : 0;
      grp[q] = okq ? gg : 0;
      if constexpr (AFFINE) {
        uint4 graw = make_uint4(Pack<T>::kOnes, Pack<T>::kOnes, Pack<T>::kOnes, Pack<T>::kOnes);
        if (okq) graw = a.gamma[ch[q]];
        unpack8<T>(graw, kc[q].g);
      }
    }

    int s = 0;
    uint32_t ph = 0;
    for (;;) {
      mbar_wait(&full[s], ph);
      const int t = stage_tile(tidx, s);
      if (t < 0) break;
      const uint8_t* st = smem + (size_t)s * a.stage_bytes;
      const long long r0 = (long long)t * a.TR;
      const int rows = (int)min((long long)a.TR, a.R - r0);
      // Warp-row wr takes rows wr, wr + NWr, ... of the tile, RR of them at a time. TR is a multiple of RR * NWr
      // (host), so every warp-row gets the same number of rows in every tile but the short last one.
      int r = wr;
      const uint4* xs = reinterpret_cast<const uint4*>(st) + r * C8;
      const uint4* ds = reinterpret_cast<const uint4*>(st + a.off_dy) + r * C8;
      const float* rss = reinterpret_cast<const float*>(st + a.off_rs) + r * a.G;
      const float* mus = reinterpret_cast<const float*>(st + a.off_mu) + r * a.G;
      uint4* dxr = a.dx + (r0 + r) * C8;
      const int xstep = a.NWr * C8, sstep = a.NWr * a.G;
      // Narrow groups: RR rows of this warp at once, so that their chains (smem load -> sums -> shuffles -> store)
      // overlap; one row at a time left the consumer warps latency-bound.
      constexpr int RRk = WIDE ? 1 : RR;
      for (; r < rows; r += RRk * a.NWr, xs += RRk * xstep, ds += RRk * xstep, rss += RRk * sstep,
                       mus += RRk * sstep, dxr += RRk * xstep) {
        if constexpr (!WIDE) {
#pragma unroll
          for (int q = 0; q < Q; ++q) {
            float xh[RR][8], gd[RR][8], d[RR][8], rs[RR];
            float2 sm[RR];
#pragma unroll
            for (int j = 0; j < RR; ++j) {
              const bool rv = r + j * a.NWr < rows;
              const int jo = rv ? j : 0;  // a missing last row recomputes row 0 and stores nothing
              rs[j] = rss[jo * sstep + grp[q]];
              const float mu = NEED_MEAN ? mus[jo * sstep + grp[q]] : 0.f;
              chunk_vals<T, LAYER, AFFINE, MEMEFF>(xs[jo * xstep + ch[q]], ds[jo * xstep + ch[q]], kc[q], rs[j], mu,
                                                   xh[j], gd[j], d[j]);
              float2 s0 = make_float2(0.f, 0.f), s1 = s0;
#pragma unroll
              for (int e = 0; e < 8; e += 2) {
                s0.x = fmaf(gd[j][e], xh[j][e], s0.x);
                s1.x = fmaf(gd[j][e + 1], xh[j][e + 1], s1.x);
                if (LAYER) { s0.y += gd[j][e]; s1.y += gd[j][e + 1]; }
              }
              sm[j] = make_float2(s0.x + s1.x, s0.y + s1.y);
            }
#pragma unroll
            for (int j = 0; j < RR; ++j) {
              if constexpr (LAYER) sm[j] = gsum2<VK>(sm[j], V, gl, head, pow2);
              else sm[j].x = gsum<VK>(sm[j].x, V, gl, head, pow2);
            }
#pragma unroll
            for (int j = 0; j < RR; ++j) {
              if (ok[q] && r + j * a.NWr < rows) {
                const float coef1 = sm[j].x * a.inv_cpg;
                const float coef2 = LAYER ? sm[j].y * a.inv_cpg : 0.f;
                float o[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) o[e] = rs[j] * fmaf(-xh[j][e], coef1, LAYER ? gd[j][e] - coef2 : gd[j][e]);
                st_cs_v4(dxr + j * xstep + ch[q], pack8<T>(o));
                // dgamma/dbeta are taken after the shuffle chain: taking them before it is slower, although it
                // removes a small spill.
                if constexpr (AFFINE) {
#pragma unroll
                  for (int e = 0; e < 8; ++e) {
                    accg[q][e] = fmaf(d[j][e], xh[j][e], accg[q][e]);
                    if (LAYER) accb[q][e] += d[j][e];
                  }
                }
              }
            }
          }
        } else {
          const float rs = rss[grp[0]];
          const float mu = NEED_MEAN ? mus[grp[0]] : 0.f;
          float2 sm = make_float2(0.f, 0.f);
#pragma unroll
          for (int q = 0; q < Q; ++q) {
            if (ok[q]) {
              float xh[8], gd[8], d[8];
              chunk_vals<T, LAYER, AFFINE, MEMEFF>(xs[ch[q]], ds[ch[q]], kc[q], rs, mu, xh, gd, d);
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                sm.x = fmaf(gd[e], xh[e], sm.x);
                if (LAYER) sm.y += gd[e];
              }
            }
          }
          sm.x = warp_sum(sm.x);
          if (LAYER) sm.y = warp_sum(sm.y);
          const float coef1 = sm.x * a.inv_cpg;
          const float coef2 = LAYER ? sm.y * a.inv_cpg : 0.f;
#pragma unroll
          for (int q = 0; q < Q; ++q) {
            if (ok[q]) {
              float xh[8], gd[8], d[8], o[8];
              chunk_vals<T, LAYER, AFFINE, MEMEFF>(xs[ch[q]], ds[ch[q]], kc[q], rs, mu, xh, gd, d);
#pragma unroll
              for (int e = 0; e < 8; ++e) o[e] = rs * fmaf(-xh[e], coef1, LAYER ? gd[e] - coef2 : gd[e]);
              st_cs_v4(dxr + ch[q], pack8<T>(o));
              if constexpr (AFFINE) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                  accg[q][e] = fmaf(d[e], xh[e], accg[q][e]);
                  if (LAYER) accb[q][e] += d[e];
                }
              }
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
      if (++s == S) { s = 0; ph ^= 1u; }
    }
  }

  if constexpr (AFFINE) {
    affine_epilogue<T, LAYER, Q>(a, smem, warp, NW, ok, ch, accg, accb);
  } else {
    // No grid barrier here: the last CTA out sets the tile counter back to zero for the next call on this stream
    // (every producer has taken its last tile before its CTA counts itself done).
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      if (atomicAdd(a.ctr + 1, 1) == (int)gridDim.x - 1) {
        __threadfence();
        a.ctr[0] = 0;
        a.ctr[1] = 0;
      }
    }
  }
}

__device__ __forceinline__ void named_bar(int id, int nthreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}

// ---- Split groups: a group of V > 32 chunks spread over NWg warps of QW chunks a lane (cpg 2048 and up, which the
// one-warp wide mapping cannot hold in registers). A row's two sums are a warp sum per warp, then the group's NWg
// partials through shared memory: pass 1 over this warp's rows of the tile writes one partial per (row, warp), one
// named barrier per warp-row and tile, pass 2 re-reads x and dy from the stage, adds the group's partials and writes
// dx. The partials are double-buffered by tile parity: a warp can only start writing a buffer again after the next
// tile's barrier, which every warp of its warp-row reaches only after its pass 2 over that buffer. ----
template <typename T, bool LAYER, int QW>
__global__ void __launch_bounds__(544, 1) gn_bwd_tma_split_kernel(const BwdArgs a) {
  extern __shared__ __align__(1024) uint8_t smem_all[];
  uint8_t* smem = smem_all + kCtlBytes;  // the ring
  constexpr int NRED = LAYER ? 2 : 1;
  const int S = a.S;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_all);
  uint64_t* empty = full + kMaxStages;
  int* tidx = reinterpret_cast<int*>(empty + kMaxStages);                         // the tile in each stage
  float* part_rows = reinterpret_cast<float*>(smem + (size_t)S * a.stage_bytes);  // [2][TR][NWc][NRED]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = a.NWc * a.NWr;
  const int C8 = a.C >> 3;

  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NW);
    }
    fence_mbar_init();
  }
  __syncthreads();

  float accg[QW][8], accb[LAYER ? QW : 1][8];
  int ch[QW];
  bool ok[QW];
#pragma unroll
  for (int q = 0; q < QW; ++q) {
    ch[q] = 0;
    ok[q] = false;
#pragma unroll
    for (int e = 0; e < 8; ++e) accg[q][e] = 0.f;
  }
#pragma unroll
  for (int q = 0; q < (LAYER ? QW : 1); ++q)
#pragma unroll
    for (int e = 0; e < 8; ++e) accb[q][e] = 0.f;

  if (warp == 0) {
    if (lane == 0) {
      int s = 0;
      uint32_t ph = 0;
      int t = blockIdx.x;
      for (int k = 0;; ++k) {
        if (k >= S) mbar_wait(&empty[s], ph ^ 1u);
        if (t >= a.num_tiles) {
          end_of_tiles(tidx, full, s);
          break;
        }
        tidx[s] = t;
        const long long r0 = (long long)t * a.TR;
        const int rows = (int)min((long long)a.TR, a.R - r0);
        const uint32_t xb = (uint32_t)rows * a.C * 2u;
        const uint32_t sb = (uint32_t)rows * a.G * 4u;
        uint8_t* st = smem + (size_t)s * a.stage_bytes;
        mbar_arrive_expect_tx(&full[s], 2u * xb + sb * (LAYER ? 2u : 1u));
        bulk_g2s(st, a.x + r0 * a.C * 2, xb, &full[s]);
        bulk_g2s(st + a.off_dy, a.dy + r0 * a.C * 2, xb, &full[s]);
        bulk_g2s(st + a.off_rs, a.rstd + r0 * a.G, sb, &full[s]);
        if (LAYER) bulk_g2s(st + a.off_mu, a.mean + r0 * a.G, sb, &full[s]);
        t = next_tile(a, t);
        if (++s == S) { s = 0; ph ^= 1u; }
      }
    }
  } else if (warp <= NW) {
    const int cw = warp - 1;
    const int cc = cw % a.NWc, wr = cw / a.NWc;
    const int g = cc / a.NWg, sub = cc % a.NWg;
    ChunkConst<true, false> kc[QW];
#pragma unroll
    for (int q = 0; q < QW; ++q) {
      const int j = sub * 32 + lane + q * 32 * a.NWg;
      ok[q] = j < a.V;
      ch[q] = ok[q] ? g * a.V + j : 0;
      uint4 graw = make_uint4(Pack<T>::kOnes, Pack<T>::kOnes, Pack<T>::kOnes, Pack<T>::kOnes);
      if (ok[q]) graw = a.gamma[ch[q]];
      unpack8<T>(graw, kc[q].g);
    }
    const int bufsz = a.TR * a.NWc * NRED;
    int s = 0;
    uint32_t ph = 0;
    for (int k = 0;; ++k) {
      mbar_wait(&full[s], ph);
      const int t = stage_tile(tidx, s);
      if (t < 0) break;
      const uint8_t* st = smem + (size_t)s * a.stage_bytes;
      const long long r0 = (long long)t * a.TR;
      const int rows = (int)min((long long)a.TR, a.R - r0);
      const uint4* xs = reinterpret_cast<const uint4*>(st);
      const uint4* ds = reinterpret_cast<const uint4*>(st + a.off_dy);
      const float* rss = reinterpret_cast<const float*>(st + a.off_rs);
      const float* mus = reinterpret_cast<const float*>(st + a.off_mu);
      float* buf = part_rows + (size_t)(k & 1) * bufsz;
      // pass 1: this warp's partial sums of each of its rows
      for (int r = wr; r < rows; r += a.NWr) {
        const float rs = rss[r * a.G + g];
        const float mu = LAYER ? mus[r * a.G + g] : 0.f;
        float2 sm = make_float2(0.f, 0.f);
#pragma unroll
        for (int q = 0; q < QW; ++q) {
          if (ok[q]) {
            float xh[8], gd[8], d[8];
            chunk_vals<T, LAYER, true, false>(xs[r * C8 + ch[q]], ds[r * C8 + ch[q]], kc[q], rs, mu, xh, gd, d);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              sm.x = fmaf(gd[e], xh[e], sm.x);
              if (LAYER) sm.y += gd[e];
            }
          }
        }
        sm.x = warp_sum(sm.x);
        if (LAYER) sm.y = warp_sum(sm.y);
        if (lane == 0) {
          float* p = buf + (r * a.NWc + cc) * NRED;
          p[0] = sm.x;
          if (LAYER) p[1] = sm.y;
        }
      }
      named_bar(1 + wr, 32 * a.NWc);
      // pass 2: the group's sums, then dx and the dgamma/dbeta partials
      for (int r = wr; r < rows; r += a.NWr) {
        const float rs = rss[r * a.G + g];
        const float mu = LAYER ? mus[r * a.G + g] : 0.f;
        const float* p = buf + (r * a.NWc + g * a.NWg) * NRED;
        float s1 = 0.f, s2 = 0.f;
        for (int w = 0; w < a.NWg; ++w) {
          s1 += p[w * NRED];
          if (LAYER) s2 += p[w * NRED + 1];
        }
        const float coef1 = s1 * a.inv_cpg;
        const float coef2 = LAYER ? s2 * a.inv_cpg : 0.f;
        uint4* dxr = a.dx + (r0 + r) * C8;
#pragma unroll
        for (int q = 0; q < QW; ++q) {
          if (ok[q]) {
            float xh[8], gd[8], d[8], o[8];
            chunk_vals<T, LAYER, true, false>(xs[r * C8 + ch[q]], ds[r * C8 + ch[q]], kc[q], rs, mu, xh, gd, d);
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = rs * fmaf(-xh[e], coef1, LAYER ? gd[e] - coef2 : gd[e]);
            st_cs_v4(dxr + ch[q], pack8<T>(o));
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              accg[q][e] = fmaf(d[e], xh[e], accg[q][e]);
              if (LAYER) accb[q][e] += d[e];
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
      if (++s == S) { s = 0; ph ^= 1u; }
    }
  }
  affine_epilogue<T, LAYER, QW>(a, smem, warp, NW, ok, ch, accg, accb);
}

constexpr int kSmemBudget = 220 * 1024;  // ring + barriers, at most (B300: 227 KB a block)

bool aligned16(const at::Tensor& t) { return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0; }

// The tile counter of the calls on one (device, stream): two ints, zeroed once on that stream when first used; every
// call leaves them at zero (the grid barrier of the dgamma combine, or the last CTA out), and the calls on one stream
// run one after another, so a call always starts from zero. One counter a stream keeps concurrent streams apart.
int* tile_counter(const at::Tensor& like, int dev, cudaStream_t stream) {
  static std::mutex mu;
  static std::map<std::pair<int, cudaStream_t>, at::Tensor*> counters;  // never freed: one small tensor a stream
  std::lock_guard<std::mutex> lock(mu);
  auto it = counters.find({dev, stream});
  if (it == counters.end())
    it = counters.emplace(std::make_pair(dev, stream), new at::Tensor(at::zeros({4}, like.options().dtype(at::kInt))))
             .first;
  return it->second->data_ptr<int>();
}

// Launch geometry and the device it runs on.
struct Launch {
  int dev, grid, threads, optin;
  size_t smem;
  cudaStream_t stream;
};

// Returns false, with nothing launched, if the device refuses the shared-memory opt-in or the launch (a cooperative
// grid larger than what fits at once, for instance); the error is cleared so the caller's fallback runs clean.
template <typename T, bool LAYER, bool AFFINE, bool MEMEFF, bool WIDE, int Q, int VK, int RR>
bool launch(const BwdArgs& a, const Launch& L) {
  auto kern = gn_bwd_tma_kernel<T, LAYER, AFFINE, MEMEFF, WIDE, Q, VK, RR>;
  static unsigned long long opted = 0;  // devices this kernel has the opt-in on
  const unsigned long long bit = L.dev < 64 ? (1ull << L.dev) : 0ull;
  if (!(opted & bit)) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L.optin) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    opted |= bit;
  }
  cudaError_t err;
  if (AFFINE) {
    BwdArgs copy = a;
    void* args[] = {&copy};
    err = cudaLaunchCooperativeKernel((void*)kern, dim3(L.grid), dim3(L.threads), args, L.smem, L.stream);
  } else {
    kern<<<L.grid, L.threads, L.smem, L.stream>>>(a);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return true;
}

// The instantiated set: bf16, the three modes xllm's models call (RMS affine from x, RMS without affine from y,
// layer norm affine from x); narrow with group widths of 8 and 16 chunks at compile time and any other at run time,
// one or two warp-chunks a warp; wide (one warp a group) with 4..6 chunks a lane for RMS affine from x. Everything
// else declines and takes another path.
// Rows a consumer warp handles at once (narrow groups): with one warp-chunk a warp, two rows overlap their dependent
// chains (smem load -> sums -> shuffles -> store); with two warp-chunks a warp, two rows spill at the 96-register
// cap of 17 warps, so one row.
constexpr int rows_at_once(int Q) { return Q >= 2 ? 1 : 2; }

template <bool LAYER, bool AFFINE, bool MEMEFF, int Q, int VK>
bool launch_rr(const BwdArgs& a, const Launch& L) {
  using T = __nv_bfloat16;
  return launch<T, LAYER, AFFINE, MEMEFF, false, Q, VK, rows_at_once(Q)>(a, L);
}

template <bool LAYER, bool AFFINE, bool MEMEFF>
bool launch_narrow(const BwdArgs& a, int Q, const Launch& L) {
  const int VK = (a.V == 8 || a.V == 16) ? a.V : 0;
  if (Q == 1) {
    if (VK == 8) return launch_rr<LAYER, AFFINE, MEMEFF, 1, 8>(a, L);
    if (VK == 16) return launch_rr<LAYER, AFFINE, MEMEFF, 1, 16>(a, L);
    return launch_rr<LAYER, AFFINE, MEMEFF, 1, 0>(a, L);
  }
  if (Q == 2) {
    if (VK == 8) return launch_rr<LAYER, AFFINE, MEMEFF, 2, 8>(a, L);
    if (VK == 16) return launch_rr<LAYER, AFFINE, MEMEFF, 2, 16>(a, L);
    return launch_rr<LAYER, AFFINE, MEMEFF, 2, 0>(a, L);
  }
  return false;
}

// Split groups: RMS with gamma from x, QW chunks a lane; always cooperative (the dgamma grid barrier).
template <int QW>
bool launch_split_q(const BwdArgs& a, const Launch& L) {
  auto kern = gn_bwd_tma_split_kernel<__nv_bfloat16, false, QW>;
  static unsigned long long opted = 0;
  const unsigned long long bit = L.dev < 64 ? (1ull << L.dev) : 0ull;
  if (!(opted & bit)) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L.optin) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    opted |= bit;
  }
  BwdArgs copy = a;
  void* args[] = {&copy};
  if (cudaLaunchCooperativeKernel((void*)kern, dim3(L.grid), dim3(L.threads), args, L.smem, L.stream) !=
      cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return true;
}

bool launch_split(const BwdArgs& a, int QW, const Launch& L) {
  if (QW == 2) return launch_split_q<2>(a, L);
  if (QW == 3) return launch_split_q<3>(a, L);
  if (QW == 4) return launch_split_q<4>(a, L);
  return false;
}

bool launch_wide(const BwdArgs& a, int Q, const Launch& L) {
  using T = __nv_bfloat16;
  switch (Q) {
    case 4: return launch<T, false, true, false, true, 4, 0, 1>(a, L);
    case 5: return launch<T, false, true, false, true, 5, 0, 1>(a, L);
    case 6: return launch<T, false, true, false, true, 6, 0, 1>(a, L);
    default: return false;
  }
}

}  // namespace

bool backward(bool layer, int64_t G, bool memeff, const at::Tensor& x_or_y, const at::Tensor& dy,
              const at::Tensor& mean, const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
              at::Tensor& dx, at::Tensor& dgamma, at::Tensor& dbeta) {
  static const bool off = [] { const char* e = std::getenv("GN_TMA"); return e && e[0] == '0'; }();
  if (off) return false;
  const auto dt = x_or_y.scalar_type();
  if (dt != at::kBFloat16) return false;
  if (dy.scalar_type() != dt || dx.scalar_type() != dt) return false;
  if (x_or_y.dim() < 2 || !x_or_y.is_contiguous() || !dy.is_contiguous() || !dx.is_contiguous()) return false;
  const int64_t C = x_or_y.size(-1);
  if (C <= 0 || G <= 0 || C % G) return false;
  const int64_t R = x_or_y.numel() / C;
  if (R <= 0 || dy.numel() != R * C || dx.numel() != R * C) return false;
  const int64_t cpg = C / G;
  if (C % 8 || cpg % 8 || C > 8192) return false;
  if ((R * G) % 4) return false;
  const bool affine = gamma.numel() > 0;
  // The instantiated modes.
  const bool mode_ok = (!layer && affine && !memeff) || (!layer && !affine && memeff) || (layer && affine && !memeff);
  if (!mode_ok) return false;
  const bool need_mean = layer && !memeff;
  if (rstd.scalar_type() != at::kFloat || !rstd.is_contiguous() || rstd.numel() != R * G || !aligned16(rstd))
    return false;
  if (need_mean && (mean.scalar_type() != at::kFloat || !mean.is_contiguous() || mean.numel() != R * G ||
                    !aligned16(mean)))
    return false;
  if (affine) {
    if (gamma.scalar_type() != dt || !gamma.is_contiguous() || gamma.numel() != C || !aligned16(gamma)) return false;
    if (dgamma.numel() != C || dgamma.scalar_type() != dt) return false;
    if (layer && (beta.scalar_type() != dt || !beta.is_contiguous() || beta.numel() != C || !aligned16(beta) ||
                  dbeta.numel() != C || dbeta.scalar_type() != dt))
      return false;
  }
  if (!aligned16(x_or_y) || !aligned16(dy) || !aligned16(dx)) return false;

  const int V = (int)(cpg / 8);
  const bool wide = V > 32;
  int GW = 1, U = 1, Q = 1, NWc = 1, NWr = 1, NWg = 1;
  bool split = false;
  if (!wide) {
    GW = 32 / V;
    U = (int)((G + GW - 1) / GW);
    if (U > 32) return false;
    Q = U <= 16 ? 1 : 2;
    NWc = (U + Q - 1) / Q;
    NWr = std::max(1, 16 / NWc);
  } else {
    if (layer || !affine || memeff) return false;
    Q = (V + 31) / 32;
    // One warp a group (Q = 4..6 chunks a lane, 8 consumer warps, no cross-warp sum) unless the group is a whole
    // number of 64-chunk units: then the split mapping fills 16 warps with no idle lane and is faster, while at 160
    // chunks a split leaves lanes idle and loses. Groups wider than 6 chunks a lane can only be split.
    const bool one_warp = Q >= 4 && Q <= 6 && G <= 8;
    if (one_warp && !(V >= 128 && V % 64 == 0)) {
      NWc = (int)G;
      NWr = std::max(1, 8 / NWc);
    } else {
      if (V < 128) return false;
      // Split: a group over NWg warps of QW (2, 3 or 4) chunks a lane; the QW that fills more of the 16 consumer
      // warps, the smallest on a tie (192 chunks: 16 warps at 3 and 4 with no idle chunk slot at 3, 12 warps at 2).
      int best_w = 0;
      for (int qw : {2, 3, 4}) {
        const int ng = (V + 32 * qw - 1) / (32 * qw);
        if ((int64_t)ng * G > 16) continue;
        const int nc = (int)G * ng;
        int nr = 1;  // warp-rows: a power of two (TR is a multiple of it), at most 8 named barriers
        while (nr * 2 * nc <= 16 && nr < 8) nr *= 2;
        if (nc * nr > best_w) {
          best_w = nc * nr;
          Q = qw;
          NWg = ng;
          NWc = nc;
          NWr = nr;
        }
      }
      if (best_w == 0) return false;
      split = true;
    }
  }
  // ---- launch geometry, on x_or_y's device ----
  const c10::cuda::CUDAGuard guard(x_or_y.device());
  const int dev = x_or_y.get_device();
  const DevInfo info = dev_info(dev);
  if (info.sms <= 0) return false;
  // 1 KB for the barriers; 4 KB more for the split kernel's row partials ([2][TR][NWc][nred] floats)
  const int budget = std::min(kSmemBudget, info.smem_optin) - 1024 - (split ? 4096 : 0);
  const int RR = wide ? 1 : rows_at_once(Q);
  const int NW = NWc * NWr;
  // Rows a tile: ~48 KB of each of x_or_y and dy for rows of 16 KB, ~40 KB otherwise (16 KB tiles are 4-10 % slower);
  // a multiple of RR * NWr (every
  // warp-row the same rows); the statistics' bytes (rows * G * 4) a multiple of 16; at least two stages.
  const int unit = RR * NWr;
  const int tb = C >= 8192 ? 49152 : 40960;
  const int nst = need_mean ? 2 : 1;
  auto stage_of = [&](int tr) {
    const int s = 2 * tr * (int)C * 2 + tr * (int)G * 4 * nst;
    return (s + 127) / 128 * 128;
  };
  // rounded up to a multiple of unit (for example 10 rows -> 16 with S = 2 stages, which is faster than 8 rows with S = 4)
  int TR = std::max(unit, (tb / (2 * (int)C) + unit - 1) / unit * unit);
  while ((TR * G) % 4) TR += unit;
  while (budget / stage_of(TR) < 2) {
    int t = TR - unit;
    while (t > 0 && (t * G) % 4) t -= unit;
    if (t <= 0) return false;
    TR = t;
  }
  const int stage = stage_of(TR);
  const int S = std::min(8, budget / stage);
  const int xbytes = TR * (int)C * 2;
  const int sbytes = TR * (int)G * 4;
  const long long num_tiles = (R + TR - 1) / TR;
  const int grid = (int)std::min<long long>(info.sms, num_tiles);
  const int nred = layer ? 2 : 1;
  // The CTA combine of dgamma/dbeta reuses the ring: one slot of nred * C floats a warp-row, then (the column-slice
  // sum) one float a thread.
  const size_t scratch = std::max<size_t>((size_t)NWr * nred * C, (size_t)32 * (1 + NWc * NWr)) * sizeof(float);
  if (affine && scratch > (size_t)S * stage) return false;
  const size_t prow = split ? (size_t)2 * TR * NWc * nred * sizeof(float) : 0;
  if (prow > 4096) return false;
  const size_t smem = (size_t)kCtlBytes + (size_t)S * stage + prow;

  BwdArgs a{};
  a.x = reinterpret_cast<const uint8_t*>(x_or_y.data_ptr());
  a.dy = reinterpret_cast<const uint8_t*>(dy.data_ptr());
  a.mean = need_mean ? mean.data_ptr<float>() : nullptr;
  a.rstd = rstd.data_ptr<float>();
  a.gamma = affine ? reinterpret_cast<const uint4*>(gamma.data_ptr()) : nullptr;
  a.dx = reinterpret_cast<uint4*>(dx.data_ptr());
  a.R = R;
  a.C = (int)C;
  a.G = (int)G;
  a.V = V;
  a.GW = GW;
  a.U = U;
  a.TR = TR;
  a.NWg = NWg;
  a.S = S;
  a.NWc = NWc;
  a.NWr = NWr;
  a.num_tiles = (int)num_tiles;
  a.stage_bytes = stage;
  a.off_dy = xbytes;
  a.off_rs = 2 * xbytes;
  a.off_mu = 2 * xbytes + sbytes;
  a.inv_cpg = 1.0f / (float)cpg;
  // The CTA partials of dgamma/dbeta, [grid, nred * C] fp32, from the caching allocator on the current stream: no
  // cudaMalloc after the first calls, and safe on any stream and under graph capture (the block is the stream's
  // until this call's kernel is done with it).
  at::Tensor part;
  if (affine) {
    part = at::empty({(int64_t)grid * nred * C}, x_or_y.options().dtype(at::kFloat));
    a.part = part.data_ptr<float>();
    a.dgamma = dgamma.data_ptr();
    a.dbeta = layer ? dbeta.data_ptr() : nullptr;
  }
  const Launch L{dev, grid, 32 * (1 + NW), info.smem_optin, smem, at::cuda::getCurrentCUDAStream()};
  a.ctr = tile_counter(x_or_y, dev, L.stream);
  if (split) return launch_split(a, Q, L);
  if (wide) return launch_wide(a, Q, L);
  if (!layer && affine) return launch_narrow<false, true, false>(a, Q, L);
  if (!layer && !affine) return launch_narrow<false, false, true>(a, Q, L);
  return launch_narrow<true, true, false>(a, Q, L);
}

}  // namespace gn_tma
