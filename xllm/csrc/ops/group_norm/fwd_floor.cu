// A shape-general forward kernel for the group RMS / layer norm, registered with the dispatcher as "fwd_floor" at
// priority 1: any kernel specialised to a shape class wins over it and this one only runs where nobody more specific
// claims the call -- including a specialist that claims the shape but declines it at run time (nodes return false and
// the dispatcher tries the next one).
//
// xllm's forward launches one 32-thread block per (row, group): at 64 groups of 64 channels that is 16.8 M blocks and
// the GPU spends its time launching them. This kernel instead treats the tensor as one flat array of rows*G groups,
// each a contiguous run of channels, and walks it with a fixed grid:
//
//   * a group is TPG consecutive lanes of a warp, TPG = the smallest power of two >= cpg/8 capped at 32 (so cpg 24 -> 4
//     lanes of which 3 load, 40 -> 8 of which 5, 64 -> 8, 128 -> 16, 1024 -> 32 in 4 chunks of 16 B a lane, 1536 -> 32
//     in 6), and a warp covers 32/TPG groups;
//   * one 16-byte load a lane a chunk, held in registers, streaming (__ldcs) since x is read once; the number of chunks
//     is a template parameter (1, 2, 4 or 8) so an instantiation never carries registers it does not use;
//   * the statistic is a shuffle reduction inside those TPG lanes, never shared memory, never __syncthreads;
//   * every lane of a warp stays in the loop until the whole warp's slots are past the end (only the loads, the
//     stores and the statistics' writes are guarded), so no shuffle ever names a lane that has left it;
//   * for the layer norm the mean comes first and the variance is taken from the centered values held in the
//     registers, so rstd matches the two-pass definition to fp32 rounding rather than a cancellation-prone one-pass formula;
//   * y is written 16 B a lane with the streaming store hint (__stcs), and rstd / mean are written fp32 [rows, G]
//     as the interface says.
//
// Shapes this node does not take (not 2 bytes, cpg not a multiple of 8, cpg > 2048, under 16 rows, groups that do not
// tile the row, a layer norm with gamma but an empty beta, unaligned or non-matching dtypes) are declined, and xllm's
// own kernels -- the correct path for every shape xllm can call -- take them.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "dispatch.h"

namespace {

constexpr int kVec = 8;        // elements of a 16-byte load, 2-byte dtypes
constexpr int kMaxChunks = 8;  // cpg <= kVec * 32 * 8 = 2048
constexpr int kThreads = 128;  // 4 warps a block: more, smaller CTAs in flight

__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ void from_f32(float v, __nv_bfloat16& o) { o = __float2bfloat16_rn(v); }
__device__ __forceinline__ void from_f32(float v, __half& o) { o = __float2half_rn(v); }

// Every loop over chunks is a full unroll of the constant kChunks with a runtime guard, so the register array is
// indexed by constants and stays in registers (a runtime index would put it in local memory).
template <typename T, bool kLayer, bool kAffine, int kChunks>
__global__ void __launch_bounds__(kThreads) fwd_floor_kernel(
    const int64_t row_groups, const int groups, const int cpg, const float eps, const int tpg,
    const int chunks, const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta,
    T* __restrict__ y, float* __restrict__ rstd, float* __restrict__ mean_out) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int groups_per_warp = 32 / tpg;
  const int slot = lane / tpg;   // which group of the warp this lane works on
  const int sub = lane % tpg;    // this lane's place inside its group
  const int64_t stride = static_cast<int64_t>(gridDim.x) * warps * groups_per_warp;
  const float inv_cpg = 1.0f / cpg;

  // The loop is bounded by this warp's first slot, so every lane of the warp runs the same number of iterations.
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * warps * groups_per_warp + warp * groups_per_warp;
       base < row_groups; base += stride) {
    const int64_t t = base + slot;
    const bool active = t < row_groups;  // this warp's last iteration may be short: guard the memory, not the shuffles
    const T* xt = x + (active ? t * cpg : 0);
    uint4 reg[kChunks];

    float sum = 0.0f;
    float sumsq = 0.0f;
#pragma unroll
    for (int q = 0; q < kChunks; ++q) {
      if (q >= chunks) break;
      const int col = (q * tpg + sub) * kVec;
      if (active && col < cpg) {
        reg[q] = __ldcs(reinterpret_cast<const uint4*>(xt + col));
        const T* p = reinterpret_cast<const T*>(&reg[q]);
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          const float v = to_f32(p[k]);
          sum += v;
          sumsq += v * v;
        }
      } else {
        reg[q] = make_uint4(0, 0, 0, 0);
      }
    }

    float mu = 0.0f;
    float var;
    if (kLayer) {
      float s = sum;
      for (int off = 1; off < tpg; off <<= 1) s += __shfl_xor_sync(0xffffffffu, s, off, tpg);
      mu = s * inv_cpg;
      // The variance from the centered values held in the registers: exact, and it costs no extra traffic.
      float vs = 0.0f;
#pragma unroll
      for (int q = 0; q < kChunks; ++q) {
        if (q >= chunks) break;
        const int col = (q * tpg + sub) * kVec;
        if (active && col < cpg) {
          const T* p = reinterpret_cast<const T*>(&reg[q]);
#pragma unroll
          for (int k = 0; k < kVec; ++k) {
            const float d = to_f32(p[k]) - mu;
            vs += d * d;
          }
        }
      }
      for (int off = 1; off < tpg; off <<= 1) vs += __shfl_xor_sync(0xffffffffu, vs, off, tpg);
      var = vs;
    } else {
      float ss = sumsq;
      for (int off = 1; off < tpg; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off, tpg);
      var = ss;
    }

    const float rinv = rsqrtf(var * inv_cpg + eps);
    if (active && sub == 0) {
      rstd[t] = rinv;
      if (kLayer) mean_out[t] = mu;
    }

    T* yt = y + (active ? t * cpg : 0);
    const int64_t gi = static_cast<int64_t>(t % groups) * cpg;
    const T* gt = kAffine ? gamma + gi : nullptr;
    const T* bt = (kLayer && kAffine) ? beta + gi : nullptr;
#pragma unroll
    for (int q = 0; q < kChunks; ++q) {
      if (q >= chunks) break;
      const int col = (q * tpg + sub) * kVec;
      if (active && col < cpg) {
        const T* p = reinterpret_cast<const T*>(&reg[q]);
        uint4 o4;
        T* op = reinterpret_cast<T*>(&o4);
        if (kAffine) {
          const uint4 g4 = __ldg(reinterpret_cast<const uint4*>(gt + col));
          const uint4 b4 = kLayer ? __ldg(reinterpret_cast<const uint4*>(bt + col)) : make_uint4(0, 0, 0, 0);
          const T* gp = reinterpret_cast<const T*>(&g4);
          const T* bp = reinterpret_cast<const T*>(&b4);
#pragma unroll
          for (int k = 0; k < kVec; ++k) {
            float v = (to_f32(p[k]) - mu) * rinv * to_f32(gp[k]);
            if (kLayer) v += to_f32(bp[k]);
            from_f32(v, op[k]);
          }
        } else {
#pragma unroll
          for (int k = 0; k < kVec; ++k) from_f32((to_f32(p[k]) - mu) * rinv, op[k]);
        }
        __stcs(reinterpret_cast<uint4*>(yt + col), o4);
      }
    }
  }
}

template <typename T, bool kLayer, bool kAffine>
bool launch(const gn::Shape& s, const gn::Call& c, const int chunks) {
  const int cpg = static_cast<int>(s.cpg);
  const int chunks_of_group = cpg / kVec;
  int tpg = 1;
  while (tpg < 32 && tpg < chunks_of_group) tpg <<= 1;  // the smallest power of two >= cpg/8, capped at 32
  const int groups_per_warp = 32 / tpg;
  const int per_block = (kThreads / 32) * groups_per_warp;
  const int64_t row_groups = s.rows * s.groups;
  const int64_t need = (row_groups + per_block - 1) / per_block;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int cap = sms * 16;  // resident CTA target; read from the device, not assumed
  const int grid = static_cast<int>(need < cap ? need : cap);

  auto stream = at::cuda::getCurrentCUDAStream();
  // reinterpret_cast, not data_ptr<T>(): libtorch instantiates data_ptr only for c10::Half, and __half is the same
  // 2 bytes under another name.
  const T* gamma = kAffine ? reinterpret_cast<const T*>(c.gamma->data_ptr()) : nullptr;
  const T* beta = (kLayer && kAffine) ? reinterpret_cast<const T*>(c.beta->data_ptr()) : nullptr;
  const T* x = reinterpret_cast<const T*>(c.x_or_y->data_ptr());
  T* y = reinterpret_cast<T*>(c.y->data_ptr());
  const int g = static_cast<int>(s.groups);
  const float eps = static_cast<float>(c.eps);
  float* rstd = c.rstd_out->data_ptr<float>();
  float* mean = kLayer ? c.mean_out->data_ptr<float>() : nullptr;
  if (chunks == 1) {
    fwd_floor_kernel<T, kLayer, kAffine, 1><<<grid, kThreads, 0, stream>>>(row_groups, g, cpg, eps, tpg, chunks, x,
                                                                          gamma, beta, y, rstd, mean);
  } else if (chunks == 2) {
    fwd_floor_kernel<T, kLayer, kAffine, 2><<<grid, kThreads, 0, stream>>>(row_groups, g, cpg, eps, tpg, chunks, x,
                                                                          gamma, beta, y, rstd, mean);
  } else if (chunks <= 4) {
    fwd_floor_kernel<T, kLayer, kAffine, 4><<<grid, kThreads, 0, stream>>>(row_groups, g, cpg, eps, tpg, chunks, x,
                                                                          gamma, beta, y, rstd, mean);
  } else {
    fwd_floor_kernel<T, kLayer, kAffine, kMaxChunks><<<grid, kThreads, 0, stream>>>(
        row_groups, g, cpg, eps, tpg, chunks, x, gamma, beta, y, rstd, mean);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

bool aligned16(const at::Tensor& t) {
  return t.numel() == 0 || reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

bool floor_contiguous(const gn::Shape& s, const gn::Call& c) {
  if (!c.x_or_y->is_contiguous() || !c.y->is_contiguous() || !c.rstd_out->is_contiguous()) return false;
  if (!aligned16(*c.x_or_y) || !aligned16(*c.y)) return false;
  // The outputs the kernel writes are a flat rows*channels for y and rows*groups for the statistics: a short output
  // would be written past its end, so its size is checked here.
  if (c.y->numel() != s.rows * s.channels) return false;
  if (c.rstd_out->numel() != s.rows * s.groups) return false;
  if (s.norm == gn::kLayer && c.mean_out->numel() != s.rows * s.groups) return false;
  if (c.gamma->numel() > 0) {
    if (c.gamma->numel() != s.channels) return false;
    if (!c.gamma->is_contiguous() || c.gamma->scalar_type() != s.dtype || !aligned16(*c.gamma)) return false;
  }
  if (c.beta->numel() > 0) {
    if (c.beta->numel() != s.channels) return false;
    if (!c.beta->is_contiguous() || c.beta->scalar_type() != s.dtype || !aligned16(*c.beta)) return false;
  }
  if (c.mean_out->numel() > 1 && !c.mean_out->is_contiguous()) return false;
  return true;
}

// Exactly as wide as the kernel is correct: a shape this claims and gets wrong is a wrong run(), so anything unusual
// goes to xllm's kernels instead (which carry the interface's TORCH_CHECKs).
bool floor_handles(const gn::Shape& s) {
  if (s.mode != gn::kFwd) return false;
  if (s.groups < 1 || s.cpg < kVec || s.cpg > kVec * 32 * kMaxChunks) return false;
  if (s.cpg % kVec != 0 || s.channels % kVec != 0) return false;
  if (s.groups * s.cpg != s.channels) return false;  // groups must tile the row, as xllm's kernels require
  if (s.rows < 16) return false;
  if (s.dtype != at::kBFloat16 && s.dtype != at::kHalf) return false;
  if (s.norm == gn::kLayer && s.affine && !s.has_beta) return false;  // layer norm with gamma but no beta
  return true;
}

bool floor_run(const gn::Shape& s, const gn::Call& c) {
  if (!floor_contiguous(s, c)) return false;  // decline: the dispatcher tries the next node, then xllm's kernels
  const int cpg = static_cast<int>(s.cpg);
  int tpg = 1;
  while (tpg < 32 && tpg < cpg / kVec) tpg <<= 1;
  const int chunks = (cpg + kVec * tpg - 1) / (kVec * tpg);
  if (chunks > kMaxChunks) return false;
  const bool layer = s.norm == gn::kLayer;
  const bool affine = s.affine;
  if (s.dtype == at::kBFloat16) {
    if (layer) return affine ? launch<__nv_bfloat16, true, true>(s, c, chunks) : launch<__nv_bfloat16, true, false>(s, c, chunks);
    return affine ? launch<__nv_bfloat16, false, true>(s, c, chunks) : launch<__nv_bfloat16, false, false>(s, c, chunks);
  }
  if (layer) return affine ? launch<__half, true, true>(s, c, chunks) : launch<__half, true, false>(s, c, chunks);
  return affine ? launch<__half, false, true>(s, c, chunks) : launch<__half, false, false>(s, c, chunks);
}

const bool w11_fwd_floor_registered = gn::register_fwd("fwd_floor", /*prio=*/1, floor_handles, floor_run);

}  // namespace
