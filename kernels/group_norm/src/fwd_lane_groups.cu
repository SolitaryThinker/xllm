// Group-norm forward pass (RMS norm and layer norm) with lane groups: one warp covers W = 32/P groups of one row, a
// group is P lanes of one 16-byte chunk each, so a group of 24..128 channels is 3..16 16-byte vectors over as many
// lanes, and the statistics stay in registers and go through shfl_xor only (no shared memory, no __syncthreads). The lane's
// channels are fixed while it walks its rows, so gamma and beta stay in registers, x is read with __ldcs and y
// written with __stcs. Two rows per iteration for load-level parallelism.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "fwd_lane_groups.h"

namespace xllm {
namespace ops {
namespace {

template <typename T>
struct Conv;

template <>
struct Conv<__nv_bfloat16> {
  static __device__ __forceinline__ float f(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 t(float v) { return __float2bfloat16(v); }
};

template <>
struct Conv<__half> {
  static __device__ __forceinline__ float f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half t(float v) { return __float2half(v); }
};

template <>
struct Conv<float> {
  static __device__ __forceinline__ float f(float v) { return v; }
  static __device__ __forceinline__ float t(float v) { return v; }
};

// Sums `a` and `b` over the P lanes of every group of a warp, leaving the group's total in each lane.
template <int P>
__device__ __forceinline__ void GroupReduce2(float& a, float& b) {
#pragma unroll
  for (int m = 1; m < P; m <<= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, m);
    b += __shfl_xor_sync(0xffffffffu, b, m);
  }
}

template <typename T, int EPB, int P, bool LAYER, bool AFFINE, int BLOCK>
__global__ __launch_bounds__(BLOCK) void GNFwdKernel(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t rows, int64_t inner, int64_t groups,
    int64_t nwarps, int nspans, int nrowtiles, int k, float eps, float inv_cpg) {
  using C = Conv<T>;
  constexpr int W = 32 / P;  // groups a warp covers at once
  union Vec {
    uint4 u;
    T t[EPB];
  };

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t gw = static_cast<int64_t>(blockIdx.x) * (BLOCK / 32) + warp;
  if (gw >= nwarps) return;
  const int span = static_cast<int>(gw % nspans);
  const int rtile = static_cast<int>(gw / nspans);
  const int ga = lane / P;   // group inside the span
  const int lpg = lane % P;  // lane inside its group
  const bool lane_ok = lpg < k;
  const int64_t cpg = static_cast<int64_t>(k) * EPB;
  const int64_t c = static_cast<int64_t>(span) * W * cpg + static_cast<int64_t>(ga) * cpg + static_cast<int64_t>(lpg) * EPB;
  const int64_t g = static_cast<int64_t>(span) * W + ga;

  float gv[EPB];
  float bv[EPB];
  if constexpr (AFFINE) {
    if (lane_ok) {
      Vec v;
      v.u = __ldg(reinterpret_cast<const uint4*>(gamma + c));
#pragma unroll
      for (int i = 0; i < EPB; ++i) gv[i] = C::f(v.t[i]);
      if constexpr (LAYER) {
        Vec w;
        w.u = __ldg(reinterpret_cast<const uint4*>(beta + c));
#pragma unroll
        for (int i = 0; i < EPB; ++i) bv[i] = C::f(w.t[i]);
      }
    }
  }

  const int64_t rstride = static_cast<int64_t>(nrowtiles) * 2;
  for (int64_t r0 = static_cast<int64_t>(rtile) * 2; r0 < rows; r0 += rstride) {
    const bool two = (r0 + 1) < rows;
    Vec va;
    Vec vb;
    va.u = make_uint4(0, 0, 0, 0);
    vb.u = make_uint4(0, 0, 0, 0);
    if (lane_ok) {
      va.u = __ldcs(reinterpret_cast<const uint4*>(x + r0 * inner + c));
      if (two) vb.u = __ldcs(reinterpret_cast<const uint4*>(x + (r0 + 1) * inner + c));
    }
    float fa[EPB];
    float fb[EPB];
#pragma unroll
    for (int i = 0; i < EPB; ++i) {
      fa[i] = C::f(va.t[i]);
      fb[i] = C::f(vb.t[i]);
    }

    float mean0 = 0.f;
    float mean1 = 0.f;
    float rs0;
    float rs1;
    if constexpr (LAYER) {
      float s0 = 0.f;
      float s1 = 0.f;
#pragma unroll
      for (int i = 0; i < EPB; ++i) {
        s0 += fa[i];
        s1 += fb[i];
      }
      GroupReduce2<P>(s0, s1);
      mean0 = s0 * inv_cpg;
      mean1 = s1 * inv_cpg;
      // A lane past the group's chunks holds zeros: its deviation from the mean must not enter the variance.
      const float mk0 = lane_ok ? mean0 : 0.f;
      const float mk1 = (lane_ok && two) ? mean1 : 0.f;
      float v0 = 0.f;
      float v1 = 0.f;
#pragma unroll
      for (int i = 0; i < EPB; ++i) {
        const float d0 = fa[i] - mk0;
        const float d1 = fb[i] - mk1;
        v0 += d0 * d0;
        v1 += d1 * d1;
      }
      GroupReduce2<P>(v0, v1);
      rs0 = rsqrtf(v0 * inv_cpg + eps);
      rs1 = rsqrtf(v1 * inv_cpg + eps);
    } else {
      float s0 = 0.f;
      float s1 = 0.f;
#pragma unroll
      for (int i = 0; i < EPB; ++i) {
        s0 += fa[i] * fa[i];
        s1 += fb[i] * fb[i];
      }
      GroupReduce2<P>(s0, s1);
      rs0 = rsqrtf(s0 * inv_cpg + eps);
      rs1 = rsqrtf(s1 * inv_cpg + eps);
    }

    if (lpg == 0) {
      const int64_t base = r0 * groups + g;
      rstd_out[base] = rs0;
      if constexpr (LAYER) mean_out[base] = mean0;
      if (two) {
        rstd_out[base + groups] = rs1;
        if constexpr (LAYER) mean_out[base + groups] = mean1;
      }
    }

    Vec oa;
    Vec ob;
#pragma unroll
    for (int i = 0; i < EPB; ++i) {
      float v0 = (fa[i] - mean0) * rs0;
      float v1 = (fb[i] - mean1) * rs1;
      if constexpr (AFFINE) {
        v0 = v0 * gv[i];
        v1 = v1 * gv[i];
        if constexpr (LAYER) {
          v0 += bv[i];
          v1 += bv[i];
        }
      }
      oa.t[i] = C::t(v0);
      ob.t[i] = C::t(v1);
    }
    if (lane_ok) {
      __stcs(reinterpret_cast<uint4*>(y + r0 * inner + c), oa.u);
      if (two) __stcs(reinterpret_cast<uint4*>(y + (r0 + 1) * inner + c), ob.u);
    }
  }
}

#define GN_FWD_LAUNCH(T, EPB, P, LAYER, AFFINE)                                                                    \
  GNFwdKernel<T, EPB, P, LAYER, AFFINE, kBlock><<<blocks, kBlock, 0, stream>>>(                                     \
      reinterpret_cast<const T*>(x_ptr), reinterpret_cast<const T*>(gamma_ptr), reinterpret_cast<const T*>(beta_ptr), \
      reinterpret_cast<T*>(y_ptr), mean_ptr, rstd_ptr, rows, inner, groups, nwarps, nspans_i, nrowtiles_i, k_i,      \
      static_cast<float>(eps), inv_cpg)

#define GN_FWD_P(T, EPB, LAYER, AFFINE)                 \
  do {                                                  \
    switch (p) {                                        \
      case 1:                                           \
        GN_FWD_LAUNCH(T, EPB, 1, LAYER, AFFINE);        \
        break;                                          \
      case 2:                                           \
        GN_FWD_LAUNCH(T, EPB, 2, LAYER, AFFINE);        \
        break;                                          \
      case 4:                                           \
        GN_FWD_LAUNCH(T, EPB, 4, LAYER, AFFINE);        \
        break;                                          \
      case 8:                                           \
        GN_FWD_LAUNCH(T, EPB, 8, LAYER, AFFINE);        \
        break;                                          \
      case 16:                                          \
        GN_FWD_LAUNCH(T, EPB, 16, LAYER, AFFINE);       \
        break;                                          \
      default:                                          \
        GN_FWD_LAUNCH(T, EPB, 32, LAYER, AFFINE);       \
        break;                                          \
    }                                                   \
  } while (0)

#define GN_FWD_AFFINE(T, EPB, LAYER)          \
  do {                                        \
    if (affine) {                             \
      GN_FWD_P(T, EPB, LAYER, true);          \
    } else {                                  \
      GN_FWD_P(T, EPB, LAYER, false);         \
    }                                         \
  } while (0)

#define GN_FWD_TYPE(T, EPB)                   \
  do {                                        \
    if (layer_norm) {                         \
      GN_FWD_AFFINE(T, EPB, true);            \
    } else {                                  \
      GN_FWD_AFFINE(T, EPB, false);           \
    }                                         \
  } while (0)

constexpr int kBlock = 512;

bool GroupNormFwdLaneGroupsImpl(const at::Tensor& x_in, const at::Tensor& gamma_in, const at::Tensor& beta_in, int64_t groups,
                          double eps, bool layer_norm, at::Tensor& y_in, at::Tensor& mean_out, at::Tensor& rstd_out) {
  if (!x_in.is_cuda() || x_in.dim() != 3 || groups <= 0) return false;
  const int64_t inner = x_in.size(-1);
  const int64_t rows = x_in.size(0) * x_in.size(1);
  if (inner <= 0 || rows <= 0 || inner % groups != 0) return false;
  const int64_t cpg = inner / groups;
  const auto scalar_type = x_in.scalar_type();
  const int elem_bytes = (scalar_type == at::kFloat) ? 4 : (scalar_type == at::kBFloat16 || scalar_type == at::kHalf) ? 2 : 0;
  if (elem_bytes == 0) return false;
  constexpr int kVecBytes = 16;
  const int epb = kVecBytes / elem_bytes;  // elements of one 16-byte vector
  if (cpg % epb != 0) return false;
  const int64_t k = cpg / epb;
  if (k < 1 || k > 32) return false;
  int p = 1;
  while (p < k) p <<= 1;
  const int w = 32 / p;
  if (groups % w != 0) return false;
  const bool affine = gamma_in.numel() > 0;
  if (affine && gamma_in.numel() != inner) return false;
  if (layer_norm && affine && beta_in.numel() != inner) return false;
  if (y_in.numel() != rows * inner || rstd_out.numel() != rows * groups) return false;

  torch::Tensor x = x_in.contiguous();
  torch::Tensor y = y_in.contiguous();
  torch::Tensor gamma;
  torch::Tensor beta;
  if (affine) {
    gamma = gamma_in.contiguous();
    if (layer_norm && affine) beta = beta_in.contiguous();
  }
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();
  const void* gamma_ptr = affine ? gamma.data_ptr() : nullptr;
  const void* beta_ptr = (layer_norm && affine) ? beta.data_ptr() : nullptr;
  float* mean_ptr = layer_norm ? mean_out.data_ptr<float>() : nullptr;
  float* rstd_ptr = rstd_out.data_ptr<float>();

  constexpr int wpb = kBlock / 32;
  const int64_t nspans = groups / w;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t nrowtiles = std::max<int64_t>(1, static_cast<int64_t>(sms) * 2 * wpb / nspans);
  const int64_t maxtiles = (rows + 1) / 2;
  if (nrowtiles > maxtiles) nrowtiles = maxtiles;
  const int64_t nwarps = nspans * nrowtiles;
  const int blocks = static_cast<int>((nwarps + wpb - 1) / wpb);
  const int nspans_i = static_cast<int>(nspans);
  const int nrowtiles_i = static_cast<int>(nrowtiles);
  const int k_i = static_cast<int>(k);
  const float inv_cpg = 1.f / static_cast<float>(cpg);

  if (scalar_type == at::kBFloat16) {
    GN_FWD_TYPE(__nv_bfloat16, 8);
  } else if (scalar_type == at::kHalf) {
    GN_FWD_TYPE(__half, 8);
  } else {
    GN_FWD_TYPE(float, 4);
  }
  return true;
}

}  // namespace

bool GroupNormFwdLaneGroups(const at::Tensor& x, const at::Tensor& gamma_or_empty, const at::Tensor& beta_or_empty,
                      int64_t groups, double eps, bool layer_norm, at::Tensor& y, at::Tensor& mean_out,
                      at::Tensor& rstd_out) {
  return GroupNormFwdLaneGroupsImpl(x, gamma_or_empty, beta_or_empty, groups, eps, layer_norm, y, mean_out, rstd_out);
}

}  // namespace ops
}  // namespace xllm
