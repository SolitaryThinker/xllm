// Mode 1 (backward) of the adaptive working memory: dkey from key, key_mask, lse, dkkey and dlse.
//
//   dkey[b,t,k,s] = y_t * (g_t - dot + dlse[b,k,s,c+1])     for t in chunk c = 0 .. N-2
//   y_t           = exp(x_t - lse[b,k,s,c+1])               (0 where key_mask[b,t])
//   dot           = sum_{t in chunk c} g_t * y_t            (fp32)
//   g_t           = dkkey[b,k,s,c*C + t]
//
// lse is an input, so no max and no sum of exp is recomputed: y is one exp2 of key and the saved value.
// dkey is zero in the last chunk and at masked positions, which falls out of y_t = 0 there.
// A masked step is loaded only when it shares a turn of the loop with an unmasked one: the test is per turn, not per
// step, so what 57-66% of a training chunk's steps save is whole turns.
//
// Write contract: on this path the kernel writes dkey's unmasked positions of chunks 0 .. N-2 and nothing else.
// dkey's last chunk and its masked positions must already be zero when the caller passes dkey; a buffer that holds
// other values there keeps them. (When run() falls back to the ATen path for a shape this path declines, that path
// writes the zeros itself.) Writing the masked positions here instead would cost 2.3-5.2 us per call as a third
// whole-row pass (11.8-21.0% on the six training backward shapes), or 35% inside the write kernel's partial-line
// stores.
//
// dot runs along time, so it needs every step of a chunk before any of dkey can be written. Two streaming kernels do
// it, with no barrier between CTAs (a cluster-cooperative version spent 34% of its time in cluster.sync):
//
//   awm_dot   one pass over the chunk: key and dkkey in, the dot's partials out (nc of them per channel).
//   awm_write one pass over the chunk: key and dkkey again (out of L2), the nc partials summed, dkey written.
//
// A CTA owns TCH channels of one chunk and a run of its steps; a thread owns NCH of the channels of one step, where
// NCH is 16 B of key (8 channels at 2 bytes, 4 at 4). 16 channels a CTA is 32 B of a 128 B line of key at a 2-byte
// element, so four CTAs share a line and it is fetched once from DRAM and serves all four out of L2; at a 4-byte
// element the same 16 channels are 64 B and two CTAs share the line. 32 channels a CTA (two CTAs share a line) is
// 3.6-22.4% slower on the six training backward shapes, and 64 channels (one whole line at 2 bytes) is slower still:
// a wider lane group puts more steps in flight per warp, and the line traffic it saves is already absorbed by L2.
//
// The mask is tested once per turn of the loop, warp-uniform (the masked steps of a row form contiguous runs), and
// the load comes after it, so a turn whose steps are all masked costs two mask bytes and nothing else.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

constexpr int NCAP = 32;    // most pieces a chunk is cut into along time (a dot partial each)

namespace {

template <typename ET> constexpr int NCH_OF = 16 / (int)sizeof(ET);  // channels of a step one thread loads: 16 B of key
constexpr int TCH = 16;     // channels of a step handled by one CTA (32 B); wider blocks are slower
constexpr int NTHR = 128;   // threads per CTA
constexpr int CTAS = 8;     // CTAs per SM the launch bounds ask for, which is 64 registers a thread: 8 x 128 threads x 64
                            // regs is the whole 64 K register file, so the bounds hold the allocator at exactly that cap
                            // instead of letting it spend a second CTA's registers on one CTA's spills
constexpr int UNROLL = 2;   // step groups a thread loads per turn of the loop
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float ex2f(float v) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

template <typename ET> struct Cvt;
template <> struct Cvt<nv_bfloat16> {
  static __device__ __forceinline__ float f(nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ nv_bfloat16 e(float v) { return __float2bfloat16(v); }
};
template <> struct Cvt<__half> {
  static __device__ __forceinline__ float f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half e(float v) { return __float2half(v); }
};
template <> struct Cvt<float> {
  static __device__ __forceinline__ float f(float v) { return v; }
  static __device__ __forceinline__ float e(float v) { return v; }
};

// NCH channels of a step of key/dkey: 16 contiguous bytes.
template <typename ET>
__device__ __forceinline__ void ldpart(const ET* p, ET* out) {
  *reinterpret_cast<uint4*>(out) = __ldcg(reinterpret_cast<const uint4*>(p));
}
template <typename ET>
__device__ __forceinline__ void stpart(ET* p, const ET* in) {
  __stcs(reinterpret_cast<uint4*>(p), *reinterpret_cast<const uint4*>(in));
}
// One element of dkkey, time innermost: a warp of consecutive lanes reads one channel of 32 steps at once.
template <typename ET>
__device__ __forceinline__ ET ldcg1(const ET* p) {
  if constexpr (sizeof(ET) == 4) return __ldcg(reinterpret_cast<const float*>(p));
  else return __ldcg(p);
}

// The dot of a chunk, in nc pieces along time: awm_dot then awm_write.
template <typename ET, int CTA = CTAS>
__global__ void __launch_bounds__(NTHR, CTA) awm_dot(const ET* __restrict__ key, const unsigned char* __restrict__ mask,
                        const float* __restrict__ lse, const ET* __restrict__ dkkey, float* __restrict__ dots,
                        int L, int K, int S, int C, int N, int nc, int slice, long long dstride) {
  // PDL: the dependent grid (awm_write) may start launching now. It does not read `dots` until it has passed its own
  // `griddepcontrol.wait`, which sits at that read and nowhere earlier, so nothing here is ordered by this line.
  // The placement is load-bearing in both directions: the dependent grid is launched once every CTA of this grid has
  // executed this instruction or completed, so putting it first - ahead of every branch - makes the launch happen
  // while this grid is still running, and keeps the consumer's waiting blocks from taking SMs from producer blocks
  // that have not started. An early `return` above this line would not break the launch (the dependent grid would
  // wait for that CTA to complete) but would serialize it. A `dots` load hoisted above the wait in awm_write would
  // turn that wait into a race.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  constexpr int NCH = NCH_OF<ET>;
  constexpr int NLANE = TCH / NCH;    // lanes that share a step
  constexpr int NROWG = NTHR / NLANE; // steps the CTA covers at once
  constexpr int NW = NTHR / 32;
  __shared__ float s_part[NW][NLANE][NCH];
  __shared__ float s_slice[TCH];

  const int tid = threadIdx.x;
  const int p = tid % NLANE;   // which NCH channels of the step
  const int rg = tid / NLANE;  // which step of the CTA's NROWG
  const int j0 = p * NCH;
  const int bz = blockIdx.z;
  const int b = bz / K, k = bz - b * K;
  const int c = blockIdx.y;
  const int cb = blockIdx.x / nc;
  const int rank = blockIdx.x - cb * nc;
  const int s0 = cb * TCH;
  const int t0 = rank * slice;
  const int t1 = min(t0 + slice, C);
  const int DLL = L - C;
  const long long ch = (long long)(b * K + k) * S + s0 + j0;  // this thread's first channel

  float nl[NCH];
#pragma unroll
  for (int jj = 0; jj < NCH; ++jj) nl[jj] = -lse[(ch + jj) * N + (c + 1)] * LOG2E;

  const long long keybase = ((long long)b * L + (long long)c * C) * K + k;  // * S + s later
  const long long dkbase = ch * DLL + (long long)c * C;
  const unsigned char* mp = mask ? mask + (long long)b * L + (long long)c * C : nullptr;
  float acc[NCH];
#pragma unroll
  for (int jj = 0; jj < NCH; ++jj) acc[jj] = 0.f;

  for (int rr = t0 + rg; rr < t1; rr += NROWG * UNROLL) {
    // The mask first, so a turn of steps that are all masked costs two mask bytes and no load at all; the test is
    // warp-uniform because a row's masked steps are one contiguous run.
    bool use[UNROLL];
    int tl[UNROLL];
    bool m_all = mp != nullptr;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int t = rr + u * NROWG;
      tl[u] = min(t, t1 - 1);  // in the slice and in bounds, so the load below is unconditional; `use` picks the value
      const bool m = mp != nullptr && t < t1 && mp[tl[u]];
      use[u] = (t < t1) && !m;
      m_all &= m || t >= t1;
    }
    if (mp != nullptr && __all_sync(0xffffffffu, m_all)) continue;
    ET xv[UNROLL][NCH];
    ET gv[UNROLL][NCH];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      ldpart(key + (keybase + (long long)tl[u] * K) * S + s0 + j0, xv[u]);
      const ET* gp = dkkey + dkbase + (long long)tl[u];
#pragma unroll
      for (int jj = 0; jj < NCH; ++jj) gv[u][jj] = ldcg1(gp + (long long)jj * DLL);
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      if (use[u]) {
#pragma unroll
        for (int jj = 0; jj < NCH; ++jj)
          acc[jj] = fmaf(ex2f(fmaf(Cvt<ET>::f(xv[u][jj]), LOG2E, nl[jj])),
                         Cvt<ET>::f(gv[u][jj]), acc[jj]);
      }
    }
  }

  // the thread's partials, the warp's (the lanes that share its channels), the CTA's
#pragma unroll
  for (int jj = 0; jj < NCH; ++jj) {
    float v = acc[jj];
#pragma unroll
    for (int m = NLANE; m < 32; m <<= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    if ((tid & 31) == p) s_part[tid >> 5][p][jj] = v;
  }
  __syncthreads();
  if (tid < TCH) {
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) tot += s_part[w][tid / NCH][tid % NCH];
    s_slice[tid] = tot;
  }
  __syncthreads();
  if (tid < TCH)
    dots[(long long)rank * dstride + (((long long)(b * K + k) * S + s0 + tid) * (N - 1) + c)] = s_slice[tid];
}

// dkey's one write pass: y again from key, g from dkkey, the dot summed from the nc partials.
template <typename ET, int CTA = CTAS>
__global__ void __launch_bounds__(NTHR, CTA) awm_write(const ET* __restrict__ key, const unsigned char* __restrict__ mask,
                          const float* __restrict__ lse, const float* __restrict__ dlse,
                          const ET* __restrict__ dkkey, const float* __restrict__ dots, ET* __restrict__ dkey,
                          int L, int K, int S, int C, int N, int nc, int nt, long long dstride) {
  constexpr int NCH = NCH_OF<ET>;
  constexpr int NLANE = TCH / NCH;
  constexpr int NROWG = NTHR / NLANE;
  __shared__ float s_c[TCH];
  __shared__ float s_dot[NCAP * TCH];
  const int tid = threadIdx.x;
  const int p = tid % NLANE;
  const int rg = tid / NLANE;
  const int j0 = p * NCH;
  const int bz = blockIdx.z;
  const int b = bz / K, k = bz - b * K;
  const int c = blockIdx.y;
  const int cb = blockIdx.x / nt;
  const int tile = blockIdx.x - cb * nt;
  const int s0 = cb * TCH;
  const int tilelen = (C + nt - 1) / nt;
  const int t0 = tile * tilelen;
  const int t1 = min(t0 + tilelen, C);
  const int DLL = L - C;
  const long long ch = (long long)(b * K + k) * S + s0 + j0;

  // The dot of every piece, combined. Loaded in parallel and summed from shared: a loop of nc dependent global loads
  // by the 16 threads of one half-warp would take half of this kernel's time at nc = 32.
  const long long dbase = (((long long)(b * K + k) * S + s0) * (N - 1) + c);
  // PDL: the two reads `dots` does not feed - this thread's `lse` slot set and its `dlse` scalar - are taken here,
  // AHEAD of the wait, so that the window the consumer spends resident while `awm_dot` drains carries real memory
  // traffic and not only index arithmetic. Both are inputs of the call: neither can be written by the producer.
  float nl[NCH];
#pragma unroll
  for (int jj = 0; jj < NCH; ++jj) nl[jj] = -lse[(ch + jj) * N + (c + 1)] * LOG2E;
  float dl0 = 0.f;
  if (tid < TCH) dl0 = dlse[((long long)(b * K + k) * S + s0 + tid) * N + (c + 1)];
  // `dots` is the producer's and this is its first read: everything above is per-block work, so the wait sits here.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  for (int i = tid; i < nc * TCH; i += NTHR)
    s_dot[i] = dots[dbase + (long long)(i / TCH) * dstride + (long long)(i % TCH) * (N - 1)];
  __syncthreads();
  if (tid < TCH) {
    float tot = 0.f;
    for (int r = 0; r < nc; ++r) tot += s_dot[r * TCH + tid];
    s_c[tid] = dl0 - tot;
  }
  __syncthreads();

  float cj[NCH];
#pragma unroll
  for (int jj = 0; jj < NCH; ++jj) cj[jj] = s_c[j0 + jj];
  const long long keybase = ((long long)b * L + (long long)c * C) * K + k;
  const long long dkbase = ch * DLL + (long long)c * C;
  const unsigned char* mp = mask ? mask + (long long)b * L + (long long)c * C : nullptr;

  for (int rr = t0 + rg; rr < t1; rr += NROWG * UNROLL) {
    bool use[UNROLL];
    int tl[UNROLL];
    bool m_all = mp != nullptr;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int t = rr + u * NROWG;
      tl[u] = min(t, t1 - 1);
      const bool m = mp != nullptr && t < t1 && mp[tl[u]];
      use[u] = (t < t1) && !m;
      m_all &= m || t >= t1;
    }
    if (mp != nullptr && __all_sync(0xffffffffu, m_all)) continue;
    ET xv[UNROLL][NCH];
    ET gv[UNROLL][NCH];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      ldpart(key + (keybase + (long long)tl[u] * K) * S + s0 + j0, xv[u]);
      const ET* gp = dkkey + dkbase + (long long)tl[u];
#pragma unroll
      for (int jj = 0; jj < NCH; ++jj) gv[u][jj] = ldcg1(gp + (long long)jj * DLL);
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int t = rr + u * NROWG;
      if (use[u]) {  // a masked step is zero in dkey and dkey arrives zeroed, so it is not written at all
        ET dv[NCH];
#pragma unroll
        for (int jj = 0; jj < NCH; ++jj) {
          const float g = Cvt<ET>::f(gv[u][jj]);
          const float xf = Cvt<ET>::f(xv[u][jj]);
          dv[jj] = Cvt<ET>::e(ex2f(fmaf(xf, LOG2E, nl[jj])) * (g + cj[jj]));
        }
        stpart(dkey + (keybase + (long long)t * K) * S + s0 + j0, dv);
      }
    }
  }
}

template <typename ET, int CTA = CTAS>
void launch_fast(const at::Tensor& key, const unsigned char* maskp, const at::Tensor& lse,
                 const at::Tensor& dlse, const at::Tensor& dkkey, at::Tensor& dkey, at::Tensor& dots,
                 int B, int L, int K, int S, int C, int N, int nc, int nt, cudaStream_t stream) {
  const int cbd = S / TCH;   // 16 channels a CTA a step
  const int slice = (C + nc - 1) / nc;
  const long long dstride = (long long)B * K * S * (N - 1);
  const auto* kp = reinterpret_cast<const ET*>(key.data_ptr());
  const auto* lp = reinterpret_cast<const float*>(lse.data_ptr());
  const auto* dp = reinterpret_cast<const float*>(dlse.data_ptr());
  const auto* gp = reinterpret_cast<const ET*>(dkkey.data_ptr());
  auto* op = reinterpret_cast<ET*>(dkey.data_ptr());
  auto* sp = reinterpret_cast<float*>(dots.data_ptr());
  awm_dot<ET, CTA><<<dim3(cbd * nc, N - 1, B * K), NTHR, 0, stream>>>(kp, maskp, lp, gp, sp, L, K, S, C, N, nc,
                                                                slice, dstride);
  cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess, "awm_dot launch failed: ", cudaGetErrorString(status));
  dim3 wgrid(cbd * nt, N - 1, B * K);
  // PDL: the second kernel is launched with programmatic stream serialisation, so its blocks are resident and
  // PDL: the second kernel is launched with programmatic stream serialization, so its blocks are resident and run
  // everything the producer does not feed (in awm_write the lse/dlse reads are hoisted above the wait for that
  // purpose) while awm_dot drains. The wait sits at awm_write's first read of `dots`, so no load the producer feeds
  // is issued early. What is removed is a fixed serialization: the consumer's launch and ramp are not behind the
  // producer's retirement, only behind every producer CTA having signalled at awm_dot's entry. This saves 0.2-0.6 us
  // on five of the six training backward shapes (about 0.4% of the geomean over all shapes). Where the producer is
  // memory-saturated the consumer's resident blocks compete with its most productive phase.
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = wgrid;
  cfg.blockDim = dim3(NTHR, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute pdl_attr[1];
  pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl_attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = pdl_attr;
  cfg.numAttrs = 1;
  status = cudaLaunchKernelEx(&cfg, awm_write<ET, CTA>, kp, maskp, lp, dp, gp, sp, op, L, K, S, C, N, nc, nt, dstride);
  TORCH_CHECK(status == cudaSuccess, "awm_write launch failed: ", cudaGetErrorString(status));
}

// How finely a chunk is cut along time for a pass: aim at ~1024 CTAs, or ~2048 where the rows are batched
// (tiles >= 256), and keep at least one whole turn of the loop per piece. A CTA walks C/nc steps of a chunk, and
// tiles = B*K*(S/TCH)*(N-1) is 256 at the batch-4 shapes against 64 and 192 at the batch-1 ones, so a 1024-CTA target
// would cut the batched shapes too coarsely (nc = 4, pieces of 1024 steps); 2048 halves those pieces and is 5-6%
// faster there (4x256 b4 l8192: 35.2 -> 33.0 us, 8x128: 28.5 -> 27.0) and changes nothing elsewhere. For the batch-1
// shapes 1024 is the best target (4x256 b1 l8192: 22.5 / 16.7 / 14.4 / 16.6 us at nc = 4 / 8 / 16 / 32; 4x256 b1
// l16384: 26.4 / 22.0 / 22.6 / 29.2 us).
// The target is not a cap: the loop takes the first power of two that reaches it, so tiles in (256, 1024] lands at
// 2048-4096 CTAs, and tiles in [1024, 2048) also lands above it (B = 8, K = 4, S = 256, N = 4 gives tiles = 1536 and
// nc = 2, i.e. 3072 CTAs), never beyond 4096, where both sweeps show a cliff (44.2 us at 4x256 b4, 29.5 at 8x128).
// Shapes with tiles of 512-1536 therefore run at 2048-3072 CTAs, a range that is not measured.
// The order of the grid axes is a second lever: making the channel block the fastest axis of blockIdx.x instead of
// the piece buys 8.6% / 8.1% on the two batched shapes at a 1024-CTA target and 0.3% / 0.0% at the 2048 target, so
// the two are substitutes. At a 1024 target the reordering costs 8.7% on 8x128 b1 l8192, so it would need a guard
// on B.
int pick_n(int tiles, int C, int turn) {
  const int target = (tiles >= 256) ? 2048 : 1024;
  int n = 1;
  while (n < NCAP && (long long)tiles * n < target && (long long)n * turn <= C) n *= 2;
  return n;
}

static void awm_bwd_run(const at::Tensor& key, const at::Tensor& key_mask, const at::Tensor& lse,
                        const at::Tensor& dkkey, const at::Tensor& dlse, at::Tensor& dkey, int64_t chunk_size) {
  const int B = (int)key.size(0), L = (int)key.size(1), K = (int)key.size(2), S = (int)key.size(3);
  const int C = (int)chunk_size;
  const int N = (int)lse.size(3);
  if (N <= 1) return;  // nothing of the last chunk feeds the backward pass: dkey stays as the caller zeroed it
  const unsigned char* maskp = (key_mask.defined() && key_mask.numel() > 0)
                                   ? reinterpret_cast<const unsigned char*>(key_mask.data_ptr())
                                   : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // The loop geometry the kernel derives from sizeof(ET), on the host: one whole turn of the loop per piece, so
  // TURN is 128 at a 2-byte element (NCH = 8) and 64 at a 4-byte one (NCH = 4). It mirrors the kernel's own
  // NROWG * UNROLL, and it is read off key.element_size() while the instantiation below is chosen by scalar_type():
  // the two agree for the three dtypes this path takes (bf16, fp16, fp32), which is the set awm_bwd_try admits.
  const int nch_t = 16 / (int)key.element_size();
  const int TURN = (NTHR / (TCH / nch_t)) * UNROLL;
  const int nc = pick_n((S / TCH) * (N - 1) * B * K, C, TURN);
  auto dots = at::empty({(long long)nc * B * K * S * (N - 1)}, key.options().dtype(at::kFloat));

  // The kernels are instantiated with two register targets (minimum CTAs per SM in __launch_bounds__): 8 (64
  // registers: 8 x 128 threads x 64 registers is the whole register file) and 9 (56 registers). The 9th CTA per SM
  // is occupied only when the grid exceeds one full wave. Below one wave the 56-register target cannot add an
  // occupancy class - 8 CTAs at 64 registers already fill the SM - so there it can only cost ILP, and it is withheld.
  // One wave at 8 CTAs per SM is read off the device, not a literal: on a part with more SMs a literal would withhold
  // the target where a wave is still partly full, and on one with fewer it would impose it on a grid that already
  // fills a wave. On the training shapes the grids ((S/TCH)*nc*(N-1)*B*K) are 1024, 1024, 1536, 1536, 2048, 2048, so
  // on a 148-SM part the four at 1536 and 2048 take the 9-CTA instantiation and the two at 1024 do not.
  const long long grid = (long long)(S / TCH) * (long long)nc * (long long)(N - 1) * (long long)B * (long long)K;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const bool wide = grid >= (long long)sms * 8;
#define AWM_BWD_GO(TY)                                                                          \
  do {                                                                                          \
    if (wide)                                                                                   \
      launch_fast<TY, 9>(key, maskp, lse, dlse, dkkey, dkey, dots, B, L, K, S, C, N, nc, nc, stream); \
    else                                                                                        \
      launch_fast<TY, 8>(key, maskp, lse, dlse, dkkey, dkey, dots, B, L, K, S, C, N, nc, nc, stream); \
  } while (0)
  if (key.scalar_type() == at::kBFloat16)
    AWM_BWD_GO(nv_bfloat16);
  else if (key.scalar_type() == at::kHalf)
    AWM_BWD_GO(__half);
  else
    AWM_BWD_GO(float);
#undef AWM_BWD_GO
}

}  // namespace

// The guard the dispatch in the binding asks first: true when this file wrote dkey, false when it declines the shape.
// A channel count that is not a multiple of 16 is served by the fused one-pass bwd_kernel in awm_chunk_softmax.cu
// (which also serves fp32 there). fp32 is not declined: the two-pass geometry is derived from sizeof(ET) (NCH_OF,
// ldcg1, nch_t), so a 4-byte key takes this path at the same 16 B of key per thread.
bool awm_bwd_try(const at::Tensor& key, const at::Tensor& key_mask, const at::Tensor& lse, const at::Tensor& dkkey,
                 const at::Tensor& dlse, at::Tensor& dkey, int64_t chunk_size) {
  if (key.dim() != 4 || chunk_size <= 0) return false;
  if (lse.size(3) < 2) return false;
  if (key.size(3) % TCH != 0) return false;  // no dtype clause: the geometry follows sizeof(ET), so fp32 is taken
  awm_bwd_run(key, key_mask, lse, dkkey, dlse, dkey, chunk_size);
  return true;
}
