// The single-pass forward of the group timestep decay normalization (modes 0 and 1), for groups of 64 channels and
// 2-byte elements. Any other shape returns false and the caller takes another path.
//
// One launch, persistent CTAs. A tile is (one row, kKc groups, kTileT steps) of x, streamed once into shared memory
// with cp.async (16 B per lane), and y is written from there: x is read once and never re-read. Per tile:
//   1. the per-step moments m, v of the kKc groups: one lane takes one whole step - 64 channels as eight 16-byte loads
//      from a 528-byte-padded slot, reduced by an in-lane add tree, no shuffle - and kWarpsPerGroup warps split the
//      tile's steps between them,
//   2. each of the first kKc warps takes one group: 32 lanes = 32 chunks of 4 steps, a chunk's (n, Pm, Pv) map, a warp
//      inclusive scan over the chunks = this tile's aggregate, published as a 16-byte epoch-tagged record,
//   3. a decoupled look-back of kLookbackWindow-wide windows over the predecessors' records (aggregate or inclusive
//      prefix) gives the exclusive prefix; the tile's inclusive prefix is published back,
//   4. every lane takes its own chunk's incoming state from the exclusive chunk prefix and scans its 4 steps itself,
//      so the scan over a tile is 4 serial steps per lane, not kTileT steps in one lane,
//   5. y from shared memory, once; the [B, L, K] fp32 statistics in one coalesced pass (mode 1).
//
// The maps are affine: (r, n, Pm, Pv) per group, whose composition is
//   r = E.r | L.r ; n = L.r ? L.n : E.n + L.n ; P = L.r ? L.P : b^(L.n) E.P + L.P
// for the map L applied after the map E, and b^n = exp2f(n log2 b).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <type_traits>

#include "fwd_persistent.h"


namespace xllm {
namespace ops {
namespace {

// The look-back's window: how many records a lane may name in one walk round. The scan that folds those records is
// log2(kLookbackWindow) shuffles instead of 5, and the step-back that walks toward the nearest inclusive prefix
// advances kLookbackWindow records at a time instead of 32. A narrow window does not remove the look-back: the poll,
// the composition, the tag protocol, the publish schedule and the record layout are the same, and the value is
// identical for every arrival pattern because the walk still steps back until it finds an inclusive prefix. The cost
// is depth: a walk that has to step back r records needs r / window rounds. 4 is the best width: the
// variable that matters is the number of tiles that miss the fast path, not the number of chains per SM.
constexpr int kLookbackWindow = 4;

constexpr int kKc = 4;                     // groups per tile (one warp per group in the scan)
constexpr int kTileT = 128;                // steps per tile
constexpr int kStages = 1;                 // tile buffers in shared memory
// Warps per CTA. The shared tile fixes CTAs an SM at 3, so this is the only knob that changes warps an SM: 4 gives 12
// warps a SM, 8 gives 24. The two heavyweight phases (the moments and the normalize) split by (group, half the tile),
// so all 8 warps work; the scan stays one warp per group (warps 0-3) because it is a per-group serial chain, and the
// statistics write moves to warp kKc so it overlaps the normalize instead of running inside the chain warps. 8 beats
// 4 where a row is far from the memory pipe's peak (the thin rows); the wide rows need bytes, not warps. 16 would
// still be correct (512 threads and 3 CTAs is 1536, under the 2048 an SM allows, so the shared memory budget does not
// change) at the cost of 12 idle warps in the scan phase.
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kWarpsPerGroup = kWarps / kKc;     // warps sharing one group's steps in the moments (2 at 8 warps)
static_assert(kWarps >= kKc && kWarps % kKc == 0,
              "kWarps must be a multiple of kKc (and at least kKc): a smaller or non-multiple value silently breaks "
              "the moments loop and the scan guard instead of failing to build");
constexpr int kStatsWarp = (kKc < kWarps) ? kKc : 0;  // the statistics write, off the chain warps
constexpr int kG = 64;                     // channels per group (fast path)
constexpr int kCPS = 8;                    // channel elements per lane (16 bytes)
constexpr int kLPG = kG / kCPS;            // lanes per group (8)
constexpr int kStepElems = kKc * kG;       // elements of one step of a tile (256)
// A step's slot in the tile buffer is padded from 256 to 264 elements (512 -> 528 B) on purpose: the moments phase
// gives one whole step to one lane, so consecutive lanes are a step apart, and at a 512-B stride (128 words) every
// lane would land on shared bank 0 - a 32-way conflict, i.e. 32x slower instead of 8x fewer dependent levels. At
// 528 B (132 words) the eight lanes of an LDS.128 phase differ by 4 banks each and cover all 32.
constexpr int kStepSlot = kStepElems + 8;
constexpr int kStepBytes = kStepSlot * 2;  // 528
constexpr int kTileBytes = kTileT * kStepBytes;
constexpr int kChunk = kTileT / 32;        // steps per lane in the scan (4)
constexpr int kAuxFloats = 4 * kTileT * kKc + kKc * 3;
// kStages is 1 and stays 1: with 3 CTAs an SM the inter-CTA overlap already hides the load, and halving the tile to
// fit two stages doubles every per-tile fixed cost (it is slower). Every buffer here is indexed by the
// stage (x = stile_s, the masks = tbos/tpad); sm_m/sm_v/sm_cm/sm_cr are produced and consumed inside one iteration,
// so they stay single-buffered. The next tile's copy is hidden another way: it is issued inside the y pass, slot by
// slot (see the y loop).
constexpr int kMaskBuf = 2;  // the y pass writes the next tile's masks while this tile still reads the current ones
static_assert(kMaskBuf >= kStages,
              "the mask buffers are indexed by the stage that will read them: two stages in flight would need two "
              "buffer sets, and `issue()` below fills exactly one per stage");
constexpr int kSmemBytes =
    kStages * kTileBytes + (int)(kAuxFloats * sizeof(float)) + kMaskBuf * 2 * kTileT;
// ---- conversions ----------------------------------------------------------------------------------------------
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ void from_f32(float v, __nv_bfloat16& o) { o = __float2bfloat16(v); }
__device__ __forceinline__ void from_f32(float v, __half& o) { o = __float2half(v); }

// ---- no packed fp32 arithmetic -----------------------------------------------------------------------------------
// sm_100+ has packed fp32 arithmetic on a 64-bit register pair (`FADD2`/`FMUL2`/`FFMA2`). The moments and the
// normalize are written with plain scalar operations on purpose: the packed forms (with 4-byte shared loads, or with
// 16-byte loads and the subtraction folded into an `fma` against -cm) halve the instruction count and not the time
// and were slower, so the fp32 pipe is not what this kernel is short of.

// ---- the look-back record ------------------------------------------------------------------------------------
// One 16-byte record per (tile, group), written twice: first as the tile's aggregate (status 1), then as its
// inclusive prefix (status 2). A reader accepts a record only when its tag carries this launch's epoch, and reads
// the payload and the tag in ONE aligned 16-byte access, which is what makes the aggregate -> inclusive overwrite
// safe without a lock: a reader sees the old record or the new one, never half of each. That is the only reason the
// protocol works, so keep both accesses a single vector ld/st (`.v4.u32`); split either into four 32-bit accesses and
// the look-back silently accepts a half-written prefix. Because the 16-byte access carries payload and tag together,
// no fence is needed. (`ld.acquire.gpu`/`st.release.gpu` would be slower: the acquire load blocks the look-back's
// lane batching, so the plain `.cg` forms are used, with the atomicity argument above as the contract.)
struct __align__(16) Rec {
  float pm, pv;
  unsigned int nr;   // n | r << 31
  unsigned int tag;  // epoch << 2 | status (1 aggregate, 2 inclusive)
};

__device__ __forceinline__ void ld_rec(const Rec* p, Rec& r) {
  unsigned a, b, c, d;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
  r.pm = __uint_as_float(a);
  r.pv = __uint_as_float(b);
  r.nr = c;
  r.tag = d;
}
__device__ __forceinline__ void st_rec(Rec* p, const Rec& r) {
  asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(__float_as_uint(r.pm)),
               "r"(__float_as_uint(r.pv)), "r"(r.nr), "r"(r.tag)
               : "memory");
}

// compose(E, L): the map L applied after the map E. b1n = b1^(L.n), b2n = b2^(L.n).
__device__ __forceinline__ void compose(const Rec& E, const Rec& L, float b1n, float b2n, Rec& out) {
  const bool er = (E.nr >> 31) != 0, lr = (L.nr >> 31) != 0;
  const unsigned en = E.nr & 0x7fffffffu, ln = L.nr & 0x7fffffffu;
  out.nr = (lr ? ln : en + ln) | ((er || lr) ? 0x80000000u : 0u);
  out.pm = lr ? L.pm : b1n * E.pm + L.pm;
  out.pv = lr ? L.pv : b2n * E.pv + L.pv;
}

__device__ __forceinline__ void identity(Rec& r) {
  r.pm = 0.f;
  r.pv = 0.f;
  r.nr = 0u;
}

// ---- async copy -----------------------------------------------------------------------------------------------
// The `"memory"` clobber is load-bearing and not decoration: the same lane refills the 16 B it has just read out of
// shared memory (the refill inside the store phase), so an inline asm the compiler believes touches no memory could be
// scheduled above the LDS whose slot it overwrites, and the wrong number that follows would be a one-tile-late
// corruption, not a fault. The backward kernel's CpAsync16 carries the same clobber for the same reason. Its price is
// nil: the asm carries no operand the compiler could have re-ordered usefully across.
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }

// ---- the kernel ------------------------------------------------------------------------------------------------
template <typename T, bool NOWALK>
__global__ void __launch_bounds__(kThreads) fwd_persistent_kernel(
    const T* __restrict__ x, const unsigned char* __restrict__ bos, const unsigned char* __restrict__ pad,
    const long long* __restrict__ prev_count, const T* __restrict__ prev_mean, const T* __restrict__ prev_var,
    const T* __restrict__ gamma_w, const T* __restrict__ beta_w, T* __restrict__ y,
    long long* __restrict__ count_out, T* __restrict__ mean_out, T* __restrict__ var_out,
    float* __restrict__ cummean, float* __restrict__ cumrstd, Rec* __restrict__ recs, const unsigned epoch,
    const int B, const int L, const int H, const int K, const int cpr, const int chains, const int ntp,
    const int ntiles, const float b1, const float b2, const float eps, const float ln1, const float ln2,
    const float lb1, const float lb2) {
  extern __shared__ char smem_raw[];
  T* stile = reinterpret_cast<T*>(smem_raw);
  float* sm_m = reinterpret_cast<float*>(smem_raw + (size_t)kStages * kTileBytes);
  float* sm_v = sm_m + kTileT * kKc;
  float* sm_cm = sm_v + kTileT * kKc;
  float* sm_cr = sm_cm + kTileT * kKc;
  unsigned char* sm_bos = reinterpret_cast<unsigned char*>(sm_cr + kTileT * kKc);  // [kMaskBuf][kTileT]
  unsigned char* sm_pad = sm_bos + kMaskBuf * kTileT;

  const int cta = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gl = lane / kLPG;   // group within a warp-step
  const int sub = lane % kLPG;  // this lane's kCPS channels of that group

  // one tile's x and masks into stage s, as one cp.async group
  auto issue = [&](int tile, int s) {
    if (tile < ntiles) {
      const int j = tile / chains;
      const int ch = tile - j * chains;
      const int b = ch / cpr, kb = ch - b * cpr;
      const int g0 = kb * kKc, t0 = j * kTileT;
      const int nb = min(kTileT, L - t0);
      const int ng = min(kKc, K - g0);
      const int cps = ng * kG / kCPS;
      const size_t xbase = ((size_t)b * L + t0) * (size_t)H + (size_t)g0 * kG;
      T* dst = stile + (size_t)s * kTileT * kStepSlot;
      for (int i = warp; i < nb; i += kWarps) {
        for (int p = lane; p < cps; p += 32) {
          cp_async16(dst + (size_t)i * kStepSlot + p * kCPS, x + xbase + (size_t)i * H + p * kCPS);
        }
      }
      for (int i = threadIdx.x; i < nb; i += kThreads) {
        // This store exists twice: here (the prologue) and in the tile loop's tail, which fills the next tile's masks
        // from the y pass. The two must stay in step: same clamp by the tile's own length, same `mask ? ... : 0`
        // form, same row index (b*L + t0 here, mrow_n there). Changing one and not the other gives a wrong result only
        // on the tiles that follow a bos or a padded step.
        sm_bos[s * kTileT + i] = bos ? bos[(size_t)b * L + t0 + i] : (unsigned char)0;
        sm_pad[s * kTileT + i] = pad ? pad[(size_t)b * L + t0 + i] : (unsigned char)0;
      }
    }
    cp_commit();
  };

  issue(cta, 0);
  // kStages > 1 would need a second stage, a second mask buffer set and a second issue() here. The static_assert
  // above fails the build if kStages is raised past kMaskBuf, which is what the mask buffers' alternation assumes
  // (the y pass fills buffer (it&1)^1 for the next tile, which is read at it+1).

  for (int tile = cta, it = 0; tile < ntiles; tile += gridDim.x, ++it) {
    const int s = it % kStages;
    const int j = tile / chains;
    const int ch = tile - j * chains;
    const int b = ch / cpr;
    const int kb = ch - b * cpr;
    const int g0 = kb * kKc;
    const int t0 = j * kTileT;
    const int nb = min(kTileT, L - t0);
    const bool last = (j == ntp - 1);
    const bool okw = (g0 + gl) < K;  // this lane's group exists
    const int mbu = it & 1;  // the mask buffers alternate: the y pass fills the other one for the next tile
    const unsigned char* tbos = sm_bos + mbu * kTileT;
    const unsigned char* tpad = sm_pad + mbu * kTileT;
    const T* stile_s = stile + (size_t)s * kTileT * kStepSlot;

    // gamma and beta of this lane's channels: constant over the tile, fetched before the wait
    float gam[kCPS], bet[kCPS];
    {
      const T* gp = gamma_w + (size_t)(g0 + gl) * kG + sub * kCPS;
      const T* bp = beta_w + (size_t)(g0 + gl) * kG + sub * kCPS;
      if (okw) {
#pragma unroll
        for (int i = 0; i < kCPS; ++i) {
          gam[i] = to_f32(gp[i]);
          bet[i] = to_f32(bp[i]);
        }
      }
    }
    // this stage's cp.async group is the older of those in flight
    asm volatile("cp.async.wait_group %0;" ::"n"(kStages - 1));
    __syncthreads();

    // ---- 1. the moments of every step: ONE LANE PER STEP, the whole 64-channel group in that lane, no shuffle.
    // kWarpsPerGroup warps share a group's steps (warp / kKc picks the half of the tile, warp % kKc the group), so all
    // kWarps warps work here and each covers kTileT / kWarpsPerGroup steps in kTileT / (32 * kWarpsPerGroup) dependent
    // iterations. The scan below reads both halves of its group, so this phase ends on a CTA barrier.
    {
      const int wgg = warp % kKc;
      const int wpart = warp / kKc;
      const bool okwg = (g0 + wgg) < K;
      if (okwg) {
        constexpr float kInv = 1.f / (float)kG;
        const T* gsrc = stile_s + (size_t)wgg * kG;
        // One step a lane a turn. Two steps in flight (both computed before either store) gain nothing: the
        // unrolled inner loop already gives the scheduler what it needs.
        for (int base = wpart * 32; base < nb; base += 32 * kWarpsPerGroup) {
          const int t = base + lane;
          if (t < nb) {
            const T* p = gsrc + (size_t)t * kStepSlot;
            // fp16 only: sum the step shifted by its first channel. The values that cause the c/s cancellation are
            // the ones near the reference, and their differences from it are exactly representable in fp32, while an
            // outlier reference leaves the estimate no worse than the unshifted form; it costs one subtraction an
            // element. Without it fp16 loses cumrstd accuracy from c/s ~256. bf16 needs no shift - 64 bf16 values
            // within two binades have an exact sum of squares in fp32, and the final fmaf rounds once - so there `ref`
            // is zero and this is the plain two-pass form, bit for bit.
            float ref = 0.f;
            if constexpr (std::is_same_v<T, __half>) ref = to_f32(p[0]);
            float s0 = 0.f, q = 0.f;
#pragma unroll
            for (int c = 0; c < kG; c += 8) {
              const float d0 = to_f32(p[c + 0]) - ref, d1 = to_f32(p[c + 1]) - ref;
              const float d2 = to_f32(p[c + 2]) - ref, d3 = to_f32(p[c + 3]) - ref;
              const float d4 = to_f32(p[c + 4]) - ref, d5 = to_f32(p[c + 5]) - ref;
              const float d6 = to_f32(p[c + 6]) - ref, d7 = to_f32(p[c + 7]) - ref;
              s0 += ((d0 + d1) + (d2 + d3)) + ((d4 + d5) + (d6 + d7));
              q += fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3, fmaf(d4, d4, fmaf(d5, d5,
                  fmaf(d6, d6, d7 * d7)))))));
            }
            const float mn = ref + s0 * kInv;
            const float sh = s0 * kInv;  // the shift's mean, and all the variance needs
            sm_m[wgg * kTileT + t] = mn;
            sm_v[wgg * kTileT + t] = fmaxf(fmaf(-sh, sh, q * kInv), 0.f);
          }
        }
      }
      // Outside `if (okwg)` on purpose: every warp of the CTA must reach the barrier, and `okwg` is only warp-uniform,
      // so moving this inside that branch would make the barrier divergent.
      if constexpr (kWarpsPerGroup > 1) __syncthreads();  // the scan reads both halves of its group
      else __syncwarp();                                  // one warp owns the group: nothing to wait for
    }

    // ---- 2. one warp per group: the chunk maps, the warp scan, the tile's aggregate, the look-back, the scan
    const int gg = warp;
    const bool okg = (warp < kKc) && (g0 + gg) < K;  // warps kKc..kWarps-1 idle: this phase is per-group serial
    if (okg) {
      const int c0 = lane * kChunk;
      Rec rl;
      {
        float pm = 0.f, pv = 0.f;
        int n = 0, r = 0;
        const int t1 = min(c0 + kChunk, nb);
        for (int t = c0; t < t1; ++t) {
          // Branchless form: the three cases are one expression each in fp32, so the per-step branch is a select.
          // Bit-identical to the three-arm `if / else if` for finite state: fmaf(0,pm,u) = u = (1-b1)*m for a reset,
          // fmaf(b1,pm,u) for a valid step, fmaf(1,pm,0) = pm for a padded one, in fp32 and with no reassociation.
          // Two exceptions that finite inputs do not reach: fmaf(0,pm,u) is NaN where pm is inf or NaN (the reset arm
          // would write u regardless), and fmaf(1,-0.f,0) = +0.f where a padded step would keep -0.
          const bool valid = tpad[t] == 0;
          const bool reset = valid && tbos[t] != 0;
          const float m = sm_m[gg * kTileT + t], v = sm_v[gg * kTileT + t];
          const float u1 = valid ? (1.f - b1) * m : 0.f;
          const float u2 = valid ? (1.f - b2) * v : 0.f;
          pm = fmaf(reset ? 0.f : (valid ? b1 : 1.f), pm, u1);
          pv = fmaf(reset ? 0.f : (valid ? b2 : 1.f), pv, u2);
          n = reset ? 1 : (valid ? n + 1 : n);
          r = r || reset;
        }
        rl.pm = pm;
        rl.pv = pv;
        rl.nr = (unsigned)n | (r ? 0x80000000u : 0u);
      }
      // inclusive scan of the chunks: chunk lane k is later in time than chunk k-1
      Rec pfx = rl;
#pragma unroll
      for (int m = 1; m < 32; m <<= 1) {
        Rec inc;
        inc.pm = __shfl_up_sync(0xffffffffu, pfx.pm, m);
        inc.pv = __shfl_up_sync(0xffffffffu, pfx.pv, m);
        inc.nr = __shfl_up_sync(0xffffffffu, pfx.nr, m);
        if (lane >= m) {
          Rec nx;
          compose(inc, pfx, exp2f((float)(pfx.nr & 0x7fffffffu) * lb1),
                  exp2f((float)(pfx.nr & 0x7fffffffu) * lb2), nx);  // inc earlier in time, pfx later
          pfx = nx;
        }
      }
      // this chunk's exclusive prefix = the inclusive prefix of the chunk before it
      Rec excl;
      excl.pm = __shfl_up_sync(0xffffffffu, pfx.pm, 1);
      excl.pv = __shfl_up_sync(0xffffffffu, pfx.pv, 1);
      excl.nr = __shfl_up_sync(0xffffffffu, pfx.nr, 1);
      if (lane == 0) identity(excl);

      // this tile's aggregate (the last chunk's inclusive prefix) goes out as a record
      Rec agg;
      agg.pm = __shfl_sync(0xffffffffu, pfx.pm, 31);
      agg.pv = __shfl_sync(0xffffffffu, pfx.pv, 31);
      agg.nr = __shfl_sync(0xffffffffu, pfx.nr, 31);
      Rec* base = &recs[(size_t)tile * kKc + gg];
      if (lane == 0) {
        Rec pub = agg;
        pub.tag = (epoch << 2) | 1u;
        st_rec(base, pub);
      }

      // the exclusive prefix of the tile: windows of 32 predecessors, latest first
      Rec acc;
      identity(acc);
      int wbase = j;
      bool done = (j == 0);
      // Fast path: ask the nearest predecessor first. Above ~2 waves it published its inclusive prefix long ago, so one
      // 16-byte load answers the whole look-back - where the window below loads kLookbackWindow records and runs a
      // compose tree. Below that wave count the first poll waits for it, and if what arrives is only an aggregate this
      // falls through to the window walk unchanged. Folding the poll into the window's lane 0 is slower: every tile
      // would pay the window's record loads and a ballot before it can decide, while the poll's hit is a cheap check
      // whose second read of the same record is an L1 hit.
      if (!done) {
        Rec r0;
        const Rec* p0 = &recs[((size_t)(j - 1) * chains + ch) * kKc + gg];
        while (true) {
          ld_rec(p0, r0);
          if ((r0.tag >> 2) == epoch && (!NOWALK || (r0.tag & 3u) == 2u)) break;
          __nanosleep(32);
        }
        if ((r0.tag & 3u) == 2u) {
          acc = r0;  // every lane read the same address, so acc is as uniform as the walk leaves it
          done = true;
        }
      }
      while (!done) {
        // The load guard and the ballot mask in one line: a lane at or above the window names no record. `pi = -1`
        // makes `pi >= 0` false below, so `incl` is false and the ballot cannot name that lane, and the load is not
        // issued either.
        const int pi = (lane < kLookbackWindow) ? (wbase - 1 - lane) : -1;
        Rec rc;
        if (pi >= 0) {
          const Rec* p = &recs[((size_t)pi * chains + ch) * kKc + gg];
          while (true) {
            ld_rec(p, rc);
            if ((rc.tag >> 2) == epoch) break;
            __nanosleep(32);
          }
        } else {
          identity(rc);
        }
        const bool incl = (pi >= 0) && ((rc.tag & 3u) == 2u);
        const unsigned imask = __ballot_sync(0xffffffffu, incl);
        const int lowest = imask ? (__ffs(imask) - 1) : -1;
        Rec cur = rc;
#pragma unroll
        for (int m = 1; m < kLookbackWindow; m <<= 1) {
          Rec inc;
          inc.pm = __shfl_up_sync(0xffffffffu, cur.pm, m);
          inc.pv = __shfl_up_sync(0xffffffffu, cur.pv, m);
          inc.nr = __shfl_up_sync(0xffffffffu, cur.nr, m);
          if (lane >= m) {
            Rec nx;
            compose(cur, inc, exp2f((float)(inc.nr & 0x7fffffffu) * lb1),
                    exp2f((float)(inc.nr & 0x7fffffffu) * lb2), nx);
            cur = nx;
          }
        }
        // `take` must name a lane the window actually covered: `kLookbackWindow - 1`, not 31. A lane at 31 holds the
        // identity when the window is narrow, so the fallback would compose the identity: correct whenever an
        // inclusive prefix is found, wrong when it is not.
        const int take = (lowest >= 0) ? lowest : (kLookbackWindow - 1);
        Rec win;
        win.pm = __shfl_sync(0xffffffffu, cur.pm, take);
        win.pv = __shfl_sync(0xffffffffu, cur.pv, take);
        win.nr = __shfl_sync(0xffffffffu, cur.nr, take);
        // compose(win, acc) is `win` while acc is the identity: `acc.nr` is the whole word, so a record that carries a
        // reset has a non-zero value even at count zero and takes the compose path - the branch skips only the true
        // identity, never a segment start (a bos at a tile boundary).
        if (acc.nr != 0u) {
          Rec nx;
          compose(win, acc, exp2f((float)(acc.nr & 0x7fffffffu) * lb1),
                  exp2f((float)(acc.nr & 0x7fffffffu) * lb2), nx);  // win earlier, acc later
          acc = nx;
        } else {
          acc = win;
        }
        if (lowest >= 0 || wbase - kLookbackWindow <= 0) {
          done = true;
        } else {
          wbase -= kLookbackWindow;
        }
      }

      // the incoming state of the tile; then this tile's inclusive prefix goes back out
      const int g = g0 + gg;
      const float mean0 = to_f32(prev_mean[(size_t)b * K + g]);
      const float v0 = to_f32(prev_var[(size_t)b * K + g]);
      const bool tr = (acc.nr >> 31) != 0;
      const int np = (int)(acc.nr & 0x7fffffffu);
      const long long bcnt = prev_count[b];
      const float tin1 = tr ? 0.f : exp2f((float)np * lb1);
      const float tin2 = tr ? 0.f : exp2f((float)np * lb2);
      const float Min = fmaf(tin1, mean0, acc.pm);
      const float Vin = fmaf(tin2, v0, acc.pv);
      const long long cin = tr ? (long long)np : bcnt + (long long)np;
      // The inclusive prefix is published after those three loads, not above them: the round trip is not on the
      // successor's critical path, and moving the store up puts its cost in front of the replay's first load.
      if (lane == 0) {
        Rec incl;
        compose(acc, agg, exp2f((float)(agg.nr & 0x7fffffffu) * lb1),
                exp2f((float)(agg.nr & 0x7fffffffu) * lb2), incl);
        incl.tag = (epoch << 2) | 2u;
        st_rec(base, incl);
      }

      // this chunk's incoming state, from its exclusive prefix applied to the tile's incoming state
      const bool er = (excl.nr >> 31) != 0;
      const int en = (int)(excl.nr & 0x7fffffffu);
      float M = er ? excl.pm : fmaf(exp2f((float)en * lb1), Min, excl.pm);
      float V = er ? excl.pv : fmaf(exp2f((float)en * lb2), Vin, excl.pv);
      long long cnt = er ? (long long)en : cin + (long long)en;
      // The bias correction is carried as `d = 1 - beta^count`, never as `p = beta^count`: at small counts p is within
      // 1e-3 of 1, so `1 - p` cancels ~3 digits of it (cummean off by 1.4e-5 relative, ~100x fp32's own error, and y
      // fails from c/s ~1000). d is exact at a reset (1 - beta, by Sterbenz) and updates as an affine step,
      // `d <- beta*d + (1 - beta)`, the same recurrence as M, so M/d stays consistent.
      const float om1 = 1.f - b1, om2 = 1.f - b2;
      float d1 = -expm1f((float)cnt * ln1);
      float d2 = -expm1f((float)cnt * ln2);
      const int t1 = min(c0 + kChunk, nb);
      for (int t = c0; t < t1; ++t) {
        // Branchless form of the replay, with the same identity argument as the map and the same two exceptions
        // (bit-identical for finite state): fmaf(0,x,u) is NaN where x is inf/NaN and the reset arm would write u
        // regardless, and fmaf(a,d,0) with a = 1 turns a -0.f (`-expm1f(0)`) into +0.f where a padded step would keep
        // the -0. Applied to all five recurrences. d1/d2 ride the same a1/a2: their reset value is om1/om2 and a padded
        // step must be the identity, which fmaf(a1,d1,valid ? om1 : 0.f) gives in all three cases. `cnt` is an integer
        // select and exact.
        const bool valid = tpad[t] == 0;
        const bool reset = valid && tbos[t] != 0;
        const float m = sm_m[gg * kTileT + t], v = sm_v[gg * kTileT + t];
        const float a1 = reset ? 0.f : (valid ? b1 : 1.f);
        const float a2 = reset ? 0.f : (valid ? b2 : 1.f);
        const float u1 = valid ? om1 * m : 0.f;
        const float u2 = valid ? om2 * v : 0.f;
        cnt = reset ? 1 : (valid ? cnt + 1 : cnt);
        M = fmaf(a1, M, u1);
        V = fmaf(a2, V, u2);
        d1 = fmaf(a1, d1, valid ? om1 : 0.f);
        d2 = fmaf(a2, d2, valid ? om2 : 0.f);
        const bool okc = (d1 > 0.f) && (d2 > 0.f);
        sm_cm[t * kKc + gg] = okc ? __fdividef(M, d1) : 0.f;
        sm_cr[t * kKc + gg] = rsqrtf((okc ? __fdividef(V, d2) : 0.f) + eps);
      }
      // the state after this tile, from the tile's own inclusive prefix
      if (last && lane == 0) {
        const bool ir = (agg.nr >> 31) != 0;
        const int in_ = (int)(agg.nr & 0x7fffffffu);
        const float Mf = ir ? agg.pm : fmaf(exp2f((float)in_ * lb1), Min, agg.pm);
        const float Vf = ir ? agg.pv : fmaf(exp2f((float)in_ * lb2), Vin, agg.pv);
        from_f32(Mf, mean_out[(size_t)b * K + g]);
        from_f32(Vf, var_out[(size_t)b * K + g]);
        if (kb == 0) count_out[b] = ir ? (long long)in_ : cin + (long long)in_;
      }
    }
    __syncthreads();

    // ---- 3. the [B, L, K] fp32 statistics, one 16-byte store per step when the groups allow it (one warp, so each
    // address is written once)
    if (cummean != nullptr && warp == kStatsWarp) {
      if ((K % kKc) == 0) {
        for (int t = lane; t < nb; t += 32) {
          const float4 a = make_float4(sm_cm[t * kKc + 0], sm_cm[t * kKc + 1], sm_cm[t * kKc + 2],
                                       sm_cm[t * kKc + 3]);
          const float4 bq = make_float4(sm_cr[t * kKc + 0], sm_cr[t * kKc + 1], sm_cr[t * kKc + 2],
                                        sm_cr[t * kKc + 3]);
          const size_t si = ((size_t)b * L + t0 + t) * (size_t)K + (size_t)g0;
          *reinterpret_cast<float4*>(cummean + si) = a;
          *reinterpret_cast<float4*>(cumrstd + si) = bq;
        }
      } else {
        for (int t = lane; t < nb; t += 32) {
          const size_t si = ((size_t)b * L + t0 + t) * (size_t)K + (size_t)g0;
          for (int q = 0; q < kKc; ++q) {
            if (g0 + q < K) {
              cummean[si + q] = sm_cm[t * kKc + q];
              cumrstd[si + q] = sm_cr[t * kKc + q];
            }
          }
        }
      }
    }

    // ---- 4. y
    // Two shapes of this loop are slower and are not used: pointer-carried addresses (`q += kWarps*H` instead of
    // rebuilding the index), and a `has_pad`/`has_bos` kernel flag that short-circuits the per-step mask read so that
    // a maskless call pays no `LDS.U8` and no test.
    // ---- the next tile's copy, issued slot by slot inside the store phase --------------------------------------------
    // The load is hidden by making it fly during the y pass instead of after it: each lane refills exactly the 16 B it
    // has just read, so no lane overwrites a slot another lane is still using and the trailing barrier is unnecessary.
    // The y body reads the element at `gl*kG + sub*kCPS` and this lane is `lane = gl*kLPG + sub` with
    // `kLPG*kCPS == kG` (8 * 8 == 64), so `lane*kCPS == gl*kG + sub*kCPS` exactly: the refill is the same address
    // as the read, not an approximation of it. If kCPS, kLPG or kG changes, the two stop being the same address, the
    // copy would refill a slot some other lane still has to read, and the failure would be a wrong number only on
    // the tiles after a reuse, not a build error. Keep kLPG*kCPS == kG.
    const int tn = tile + kStages * gridDim.x;
    int nb_n = 0, cps_n = 0, mrow_n = 0;
    const T* xn = x;  // this lane's source base for the next tile, lane*kCPS already folded in
    if (tn < ntiles) {
      const int jn = tn / chains, chn = tn - jn * chains;
      const int bn = chn / cpr, kbn = chn - bn * cpr;
      const int g0n = kbn * kKc, t0n = jn * kTileT;
      nb_n = min(kTileT, L - t0n);
      const int ngn = min(kKc, K - g0n);
      cps_n = ngn * kG / kCPS;
      xn = x + ((size_t)bn * L + t0n) * (size_t)H + (size_t)g0n * kG + lane * kCPS;
      mrow_n = bn * L + t0n;
    }
    const int kYend = (nb > nb_n) ? nb : nb_n;
    // The per-lane refill below addresses the stage this tile was loaded into; with two stages in flight the tile's
    // slots and the refill's slots would be different stages and the barrier above would publish the wrong one.
    static_assert(kStages == 1, "the per-lane refill addresses this tile's stage");
#pragma unroll 4
    for (int i = warp; i < kYend; i += kWarps) {
      if (i < nb && okw) {
        const bool valid = tpad[i] == 0;
        const T* p = stile_s + (size_t)i * kStepSlot + gl * kG + sub * kCPS;
        const float cm = sm_cm[i * kKc + gl], cr = sm_cr[i * kKc + gl];
        const float ncr = -cm * cr;  // hoisted once a step, outside the 8-channel loop
        T* q = y + (((size_t)b * L + t0 + i) * (size_t)H + (size_t)(g0 + gl) * kG + sub * kCPS);
        __align__(16) T o[kCPS];
        // The padded-step test stays a per-element select, not a branch: ptxas already predicates the select for free
        // inside the unrolled loop, and a branch stops it scheduling the eight loads as one block.
#pragma unroll
        for (int c = 0; c < kCPS; ++c) {
          const float v = fmaf(fmaf(to_f32(p[c]), cr, ncr), gam[c], bet[c]);  // = (x - cm)*cr*gam + bet
          from_f32(valid ? v : 0.f, o[c]);
        }
        // y is written once and never read back, and it is a plain store: an evict-first hint is not used, because
        // its effect scales with the size of the write stream and not with what it evicts, so it pays only where y is
        // a small part of the traffic.
        const uint4 ov = *reinterpret_cast<const uint4*>(o);
        asm volatile(
            "{\n\t.reg .u64 gt;\n\tcvta.to.global.u64 gt, %4;\n\tst.global.v4.u32 [gt], {%0, %1, %2, %3};\n}" ::"r"(
                ov.x),
            "r"(ov.y), "r"(ov.z), "r"(ov.w), "l"(q)
            : "memory");
      }
      if (i < nb_n && lane < cps_n) {
        // The slot this iteration has just read, refilled from the next tile: `xn + i*H` is one add off a per-tile
        // base (one base per tile, not a pointer carried across the unrolled iterations, which is slower).
        // `stile_s` alone because kStages == 1: the tile in flight is this one's.
        cp_async16(stile + (size_t)s * kTileT * kStepSlot + (size_t)i * kStepSlot + lane * kCPS,
                   xn + (size_t)i * H);
      }
    }
    cp_commit();  // one cp.async group per tile
    {
      const int mn = (it & 1) ^ 1;
      for (int i = threadIdx.x; i < nb_n; i += kThreads) {
        sm_bos[mn * kTileT + i] = bos ? bos[(size_t)mrow_n + i] : (unsigned char)0;
        sm_pad[mn * kTileT + i] = pad ? pad[(size_t)mrow_n + i] : (unsigned char)0;
      }
    }
    // No trailing barrier: every lane refilled exactly the 16 B it had just read, so no lane overwrites a slot another
    // lane is still reading, and the next iteration's barrier (behind cp.async.wait_group) publishes the mask bytes.
  }
}

// ---- host -----------------------------------------------------------------------------------------------------
// The look-back needs one 16-byte record per (tile, group), published and read inside the launch. It is never
// memset per call: a record counts only when its tag carries the current launch's epoch, and the epoch strictly
// increases, so a record left by an earlier call is ignored (a fresh allocation is zeroed once, which no epoch can
// match). The pool is kept per CUDA stream: two forwards in flight on different streams must not overwrite each
// other's records or share an epoch, or the older call's spinner waits for a record that will never come back.
// More than kMaxPools concurrent streams returns false and the caller takes another path for the call.
constexpr int kMaxPools = 8;
struct Pool {
  cudaStream_t st;
  int device;
  Rec* p;
  size_t cap;
  unsigned epoch;
};
thread_local Pool g_pools[kMaxPools];

Pool* pool_for(cudaStream_t st) {
  int device = -1;
  AT_CUDA_CHECK(cudaGetDevice(&device));
  for (int i = 0; i < kMaxPools; ++i) {
    if (g_pools[i].p != nullptr && g_pools[i].st == st && g_pools[i].device == device) return &g_pools[i];
  }
  for (int i = 0; i < kMaxPools; ++i) {
    if (g_pools[i].p == nullptr && g_pools[i].cap == 0u) {
      g_pools[i].st = st;
      g_pools[i].device = device;
      return &g_pools[i];
    }
  }
  return nullptr;
}

bool ensure_recs(Pool* q, size_t need, cudaStream_t st) {
  if (need <= q->cap) return true;
  size_t cap = q->cap ? q->cap : (size_t)(1 << 16);
  while (cap < need) cap *= 2;
  if (q->p) cudaFree(q->p);
  q->p = nullptr;
  q->cap = 0;
  if (cudaMalloc((void**)&q->p, cap * sizeof(Rec)) != cudaSuccess) {
    q->p = nullptr;
    return false;
  }
  // a fresh buffer is garbage: clear it once so no tag can be mistaken for a published record
  if (cudaMemsetAsync(q->p, 0, cap * sizeof(Rec), st) != cudaSuccess) return false;
  q->cap = cap;
  return true;
}

// torch instantiates data_ptr only for its own scalar types
template <typename T>
struct TorchScalar;
template <>
struct TorchScalar<__nv_bfloat16> {
  using type = at::BFloat16;
};
template <>
struct TorchScalar<__half> {
  using type = at::Half;
};
template <typename T>
const T* tptr(const torch::Tensor& t) {
  return reinterpret_cast<const T*>(t.data_ptr<typename TorchScalar<T>::type>());
}
template <typename T>
T* tptr_m(torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr<typename TorchScalar<T>::type>());
}

template <typename T>
bool launch_persistent(const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
               const c10::optional<torch::Tensor>& padding_mask, const torch::Tensor& prev_count,
               const torch::Tensor& prev_mean, const torch::Tensor& prev_var, const torch::Tensor& gamma,
               const torch::Tensor& beta, int64_t num_groups, double beta1, double beta2, double eps,
               torch::Tensor& y, torch::Tensor& count_out, torch::Tensor& mean_out, torch::Tensor& var_out,
               float* cummean, float* cumrstd) {
  const int B = (int)x.size(0), L = (int)x.size(1), H = (int)x.size(2);
  const int K = (int)num_groups;
  const int cpr = (K + kKc - 1) / kKc;
  const int chains = B * cpr;
  const int ntp = (L + kTileT - 1) / kTileT;
  const int ntiles = chains * ntp;
  static thread_local uint64_t smem_set = 0;
  const int device = x.get_device();
  if (device >= 64) return false;
  const uint64_t device_bit = uint64_t{1} << device;
  if (!(smem_set & device_bit)) {
    // BOTH instantiations: cudaFuncSetAttribute is PER-FUNCTION and a template instantiation is a
    // function. Without the second line the NOWALK=true launch dies on a 48 KB default limit.
    AT_CUDA_CHECK(cudaFuncSetAttribute(fwd_persistent_kernel<T, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    AT_CUDA_CHECK(cudaFuncSetAttribute(fwd_persistent_kernel<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    smem_set |= device_bit;
  }
  // BOTH instantiations are queried and the smaller occupancy is the one the one-wave grid is built on: the
  // look-back's invariant below is that every CTA is resident, and under the dispatch either instantiation may be
  // the one launched, so sizing the grid for only one of them would be a hang waiting for the two to diverge.
  int per_sm_a = 0, per_sm_b = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_a, fwd_persistent_kernel<T, false>, kThreads, kSmemBytes);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_b, fwd_persistent_kernel<T, true>, kThreads, kSmemBytes);
  int per_sm = (per_sm_a < per_sm_b) ? per_sm_a : per_sm_b;
  if (per_sm < 1) per_sm = 1;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // The scan requires all CTAs to be resident together. A cooperative launch enforces that invariant even
  // when another stream or process is using the device; an ordinary occupancy-sized grid does not guarantee it.
  if (!at::cuda::getCurrentDeviceProperties()->cooperativeLaunch) return false;
  int grid = sms * per_sm;
  if (grid > ntiles) grid = ntiles;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // This kernel declines a CUDA graph capture: its look-back waits on CTAs it assumes co-resident, and a captured
  // graph is replayed later, under a grid whose residency is not controlled. Nothing is written before this return,
  // so TdnFwdGeneral (per-call scratch, no cross-CTA wait) takes the call.
  {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) return false;
  }
  Pool* pool = pool_for(stream);
  if (pool == nullptr) return false;  // too many streams in flight: the caller takes another path
  if (!ensure_recs(pool, (size_t)ntiles * kKc, stream)) return false;
  if (++pool->epoch >= (1u << 30)) {  // the tag holds 30 bits of epoch; wrap with a cleared buffer
    pool->cap = 0;
    if (!ensure_recs(pool, (size_t)ntiles * kKc, stream)) return false;
    pool->epoch = 1;
  }
  const unsigned epoch = pool->epoch;
  Rec* recs = pool->p;
  const unsigned char* bos = bos_mask.has_value() ? (const unsigned char*)bos_mask->data_ptr<bool>() : nullptr;
  const unsigned char* pad =
      padding_mask.has_value() ? (const unsigned char*)padding_mask->data_ptr<bool>() : nullptr;
  const float b1 = (float)beta1, b2 = (float)beta2;
  // The dispatch: a launch with enough chains and enough tiles per chain takes the predecessor's inclusive prefix
  // straight out of the fast path and never walks (NOWALK). The predicate is read off this launcher's own `chains`
  // and `ntp`.
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = kSmemBytes;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  if (chains >= 48 && ntp >= 64) {
    AT_CUDA_CHECK(cudaLaunchKernelEx(&config, fwd_persistent_kernel<T, true>,
        tptr<T>(x), bos, pad, (const long long*)prev_count.data_ptr<int64_t>(), tptr<T>(prev_mean),
        tptr<T>(prev_var), tptr<T>(gamma), tptr<T>(beta), tptr_m<T>(y),
        (long long*)count_out.data_ptr<int64_t>(),
        tptr_m<T>(mean_out), tptr_m<T>(var_out), cummean, cumrstd, recs, epoch, B, L, H, K, cpr, chains, ntp,
        ntiles, b1, b2, (float)eps, log1pf(b1 - 1.f), log1pf(b2 - 1.f), log2f(b1), log2f(b2)));
  } else {
    AT_CUDA_CHECK(cudaLaunchKernelEx(&config, fwd_persistent_kernel<T, false>,
        tptr<T>(x), bos, pad, (const long long*)prev_count.data_ptr<int64_t>(), tptr<T>(prev_mean),
        tptr<T>(prev_var), tptr<T>(gamma), tptr<T>(beta), tptr_m<T>(y),
        (long long*)count_out.data_ptr<int64_t>(),
        tptr_m<T>(mean_out), tptr_m<T>(var_out), cummean, cumrstd, recs, epoch, B, L, H, K, cpr, chains, ntp,
        ntiles, b1, b2, (float)eps, log1pf(b1 - 1.f), log1pf(b2 - 1.f), log2f(b1), log2f(b2)));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

}  // namespace

namespace {
// every wide access in this kernel is 16 bytes: cp.async from x, one 16-B store to y, float4 bumps on the stats
inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u; }
inline bool opt_bool_mask(const c10::optional<torch::Tensor>& m, int64_t numel) {
  return !m.has_value() || !m->defined() ||
         (m->is_cuda() && m->scalar_type() == at::kBool && m->is_contiguous() && m->numel() == numel);
}
}  // namespace

bool TdnFwdPersistent(const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
                 const c10::optional<torch::Tensor>& padding_mask, const torch::Tensor& prev_count,
                 const torch::Tensor& prev_mean, const torch::Tensor& prev_var, const torch::Tensor& gamma,
                 const torch::Tensor& beta, int64_t num_groups, double beta1, double beta2, double eps,
                 torch::Tensor y, torch::Tensor count_out, torch::Tensor mean_out, torch::Tensor var_out,
                 const c10::optional<torch::Tensor>& cummean, const c10::optional<torch::Tensor>& cumrstd) {
  if (!x.is_cuda() || x.dim() != 3) return false;
  const int B = (int)x.size(0), L = (int)x.size(1), H = (int)x.size(2);
  const int K = (int)num_groups;
  if (L < 2 || B < 1 || K < 1) return false;
  if ((long long)H != (long long)K * kG) return false;  // groups of exactly 64 channels
  const auto dt = x.scalar_type();
  if (dt != at::kBFloat16 && dt != at::kHalf) return false;
  // every tensor this entry touches, by dtype, shape and layout: a silent accept here is a wrong answer
  if (!x.is_contiguous() || !y.is_contiguous() || !gamma.is_contiguous() || !beta.is_contiguous()) return false;
  if (y.scalar_type() != dt || gamma.scalar_type() != dt || beta.scalar_type() != dt) return false;
  if (y.numel() != x.numel() || gamma.numel() != (int64_t)H || beta.numel() != (int64_t)H) return false;
  if (count_out.scalar_type() != at::kLong || count_out.numel() != (int64_t)B) return false;
  if (mean_out.scalar_type() != dt || var_out.scalar_type() != dt) return false;
  if (mean_out.numel() != (int64_t)B * K || var_out.numel() != (int64_t)B * K) return false;
  if (prev_count.scalar_type() != at::kLong || prev_count.numel() != (int64_t)B) return false;
  if (prev_mean.scalar_type() != dt || prev_var.scalar_type() != dt) return false;
  if (prev_mean.numel() != (int64_t)B * K || prev_var.numel() != (int64_t)B * K) return false;
  if (!prev_count.is_contiguous() || !prev_mean.is_contiguous() || !prev_var.is_contiguous()) return false;
  if (!aligned16(x.data_ptr()) || !aligned16(y.data_ptr())) return false;
  if (!opt_bool_mask(bos_mask, (int64_t)B * L) || !opt_bool_mask(padding_mask, (int64_t)B * L)) return false;
  const bool want_stats = cummean.has_value() && cummean->defined();
  if (want_stats) {
    if (!cumrstd.has_value() || !cumrstd->defined()) return false;
    if (cummean->scalar_type() != at::kFloat || cumrstd->scalar_type() != at::kFloat) return false;
    if (!cummean->is_contiguous() || !cumrstd->is_contiguous()) return false;
    if (cummean->numel() < (int64_t)B * L * K || cumrstd->numel() < (int64_t)B * L * K) return false;
    if (!aligned16(cummean->data_ptr()) || !aligned16(cumrstd->data_ptr())) return false;
  }
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  float* cma = want_stats ? cummean->data_ptr<float>() : nullptr;
  float* crd = want_stats ? cumrstd->data_ptr<float>() : nullptr;
  if (x.scalar_type() == at::kBFloat16)
    return launch_persistent<__nv_bfloat16>(x, bos_mask, padding_mask, prev_count, prev_mean, prev_var, gamma, beta,
                                    num_groups, beta1, beta2, eps, y, count_out, mean_out, var_out, cma, crd);
  return launch_persistent<__half>(x, bos_mask, padding_mask, prev_count, prev_mean, prev_var, gamma, beta, num_groups,
                           beta1, beta2, eps, y, count_out, mean_out, var_out, cma, crd);
}

}  // namespace ops
}  // namespace xllm
