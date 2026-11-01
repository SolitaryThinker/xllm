// One fused launch of the group timestep decay normalization for one step: the group's moments, the decayed scan with
// the state carried in, and the normalization, in one kernel. The regime it is for is decode, where L == 1 and the
// call's time is launch and latency, not bytes: one launch, one warp per (row, group), the state in registers, no
// tensor allocated on the host, and every load that does not wait on the group reduction (prev_count/mean/var, gamma,
// beta) issued first so its latency overlaps the reduction.
//
//   y[b, 0, h]      = valid ? ((x - cummean) * cumrstd * gamma[h] + beta[h]) : 0
//   count[b]        = the final count (int64);  mean/var [B, K] the final state, in x's dtype
//   cummean/cumrstd = [B, 1, K] fp32, written when the mode is training (the backward pass takes them back)
//
// A padded step leaves the state as it is and writes y = 0; a bos step restarts it (count = mean = var = 0). The
// statistics are written for a padded step too, from the unchanged state, as the reference computes them.
// The bias correction is -expm1f(count * ln beta): accurate in the relative sense at count 1, where 1 - beta ** count
// would lose five digits, and 1 for every count past ~1e5.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

#include "decode_fwd.h"

namespace xllm {
namespace ops {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

// The y path: one ladder over the pivot-shifted pair (S, Q), the count's chain and the two reciprocals
// taken before the group reduction (they touch no data), the variance's reciprocal applied as a multiply, and
// rsqrt.approx.ftz: the argument is `fmaf(fmaxf(var, 0), r2c, epsf) >= epsf ~ 1e-5`, so neither it nor the result
// `1/sqrt(arg) <= ~316` can be denormal and the flush cannot fire; it is the same single MUFU.RSQ either way.

// A sum over the warp, in every lane (butterfly).
__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

// Two elements of T out of (and into) one 4- or 8-byte load, so that a lane covers two channels.
template <typename T, typename V>
__device__ __forceinline__ void Unpack2(V raw, float& a, float& b) {
  T pair[2];
  *reinterpret_cast<V*>(pair) = raw;
  a = static_cast<float>(pair[0]);
  b = static_cast<float>(pair[1]);
}

template <typename T, typename V>
__device__ __forceinline__ V Pack2(float a, float b) {
  T pair[2] = {static_cast<T>(a), static_cast<T>(b)};
  return *reinterpret_cast<V*>(pair);
}

// The state after one step, and the two normalization constants of that step.
struct Step {
  float cm;  // cummean
  float cr;  // cumrstd
  int64_t cnt;
  float mean;
  float var;
};

__device__ __forceinline__ Step Update(const float m_t, const float v_t, const bool is_pad, const bool is_bos,
                                       const float beta1f, const float beta2f, const float lnb1, const float lnb2,
                                       const float epsf, int64_t cnt, float mean, float var) {
  if (is_bos) {  // a bos step restarts the state
    cnt = 0;
    mean = 0.0f;
    var = 0.0f;
  }
  if (!is_pad) {  // a padded step leaves the state as it is
    mean = fmaf(1.0f - beta1f, m_t, beta1f * mean);
    var = fmaf(1.0f - beta2f, v_t, beta2f * var);
    cnt += 1;
  }
  const float d1 = -expm1f(static_cast<float>(cnt) * lnb1);
  const float d2 = -expm1f(static_cast<float>(cnt) * lnb2);
  Step out;
  out.cm = 0.0f;
  out.cr = rsqrtf(epsf);
  if (d1 > 0.0f && d2 > 0.0f) {
    out.cm = mean / d1;
    out.cr = rsqrtf(var / d2 + epsf);
  }
  out.cnt = cnt;
  out.mean = mean;
  out.var = var;
  return out;
}

// G == 64 (the Gekko models): one 4- or 8-byte load per lane for x, gamma and beta, and one store for y.
template <typename T, typename V>
__global__ __launch_bounds__(kWarpsPerBlock* kWarpSize) void TdnDecodeFwdG64Kernel(
    const T* __restrict__ x, const bool* __restrict__ bos, const bool* __restrict__ padm,
    const int64_t* __restrict__ prev_count, const T* __restrict__ prev_mean, const T* __restrict__ prev_var,
    const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y, int64_t* __restrict__ count_out,
    T* __restrict__ mean_out, T* __restrict__ var_out, float* __restrict__ cm_out, float* __restrict__ cr_out,
    const int64_t H, const int64_t K, const float beta1f, const float beta2f, const float lnb1, const float lnb2,
    const float epsf) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t k = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
  if (k >= K) {
    return;  // a whole warp takes this branch
  }
  const int64_t b = blockIdx.y;
  const int64_t h0 = k * 64 + 2 * lane;  // this lane's two channels

  // Everything that does not wait on the reduction is issued first, so its latency overlaps it.
  const V xraw = *reinterpret_cast<const V*>(x + b * H + h0);
  const V graw = *reinterpret_cast<const V*>(gamma + h0);
  const V braw = *reinterpret_cast<const V*>(beta + h0);
  const bool is_pad = padm != nullptr && padm[b];
  const bool is_bos = bos != nullptr && bos[b] && !is_pad;
  const int64_t cnt0 = prev_count != nullptr ? prev_count[b] : 0;
  const float mean0 = prev_mean != nullptr ? static_cast<float>(prev_mean[b * K + k]) : 0.0f;
  const float var0 = prev_var != nullptr ? static_cast<float>(prev_var[b * K + k]) : 0.0f;
  float x0, x1, g0, g1, sh0, sh1;
  Unpack2<T, V>(xraw, x0, x1);
  Unpack2<T, V>(graw, g0, g1);
  Unpack2<T, V>(braw, sh0, sh1);

  // Two passes over the group's 64 channels, both from registers: the mean, then the variance about it. No
  // sum-of-squares cancellation term, and the same moments the tiled forward computes.
  float s = 0.0f;
  if (!is_pad) {
    s = x0 + x1;
  }
  const float m_t = WarpSum(s) * (1.0f / 64.0f);
  float q = 0.0f;
  if (!is_pad) {
    const float d0 = x0 - m_t, d1 = x1 - m_t;
    q = fmaf(d0, d0, d1 * d1);
  }
  const float v_t = WarpSum(q) * (1.0f / 64.0f);

  const Step st = Update(m_t, v_t, is_pad, is_bos, beta1f, beta2f, lnb1, lnb2, epsf, cnt0, mean0, var0);

  V* yraw = reinterpret_cast<V*>(y + b * H + h0);
  if (is_pad) {
    *yraw = Pack2<T, V>(0.0f, 0.0f);
  } else {
    *yraw = Pack2<T, V>(fmaf((x0 - st.cm) * st.cr, g0, sh0), fmaf((x1 - st.cm) * st.cr, g1, sh1));
  }
  if (lane == 0) {
    if (cm_out != nullptr) {
      cm_out[b * K + k] = st.cm;
      cr_out[b * K + k] = st.cr;
    }
    count_out[b] = st.cnt;
    mean_out[b * K + k] = static_cast<T>(st.mean);
    var_out[b * K + k] = static_cast<T>(st.var);
  }
}

// Any other group width: this lane takes channels lane, lane + 32, ... of the group.
template <typename T>
__global__ __launch_bounds__(kWarpsPerBlock* kWarpSize) void TdnDecodeFwdKernel(
    const T* __restrict__ x, const bool* __restrict__ bos, const bool* __restrict__ padm,
    const int64_t* __restrict__ prev_count, const T* __restrict__ prev_mean, const T* __restrict__ prev_var,
    const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y, int64_t* __restrict__ count_out,
    T* __restrict__ mean_out, T* __restrict__ var_out, float* __restrict__ cm_out, float* __restrict__ cr_out,
    const int64_t H, const int64_t K, const int64_t G, const float beta1f, const float beta2f, const float lnb1,
    const float lnb2, const float epsf) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t k = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
  if (k >= K) {
    return;
  }
  const int64_t b = blockIdx.y;
  const int64_t h0 = k * G;
  const T* __restrict__ xr = x + b * H + h0;
  T* __restrict__ yr = y + b * H + h0;

  const bool is_pad = padm != nullptr && padm[b];
  const bool is_bos = bos != nullptr && bos[b] && !is_pad;
  const int64_t cnt0 = prev_count != nullptr ? prev_count[b] : 0;
  const float mean0 = prev_mean != nullptr ? static_cast<float>(prev_mean[b * K + k]) : 0.0f;
  const float var0 = prev_var != nullptr ? static_cast<float>(prev_var[b * K + k]) : 0.0f;
  float s = 0.0f;
  if (!is_pad) {
    for (int64_t i = lane; i < G; i += kWarpSize) {
      s += static_cast<float>(xr[i]);
    }
  }
  const float inv_g = 1.0f / static_cast<float>(G);
  const float m_t = WarpSum(s) * inv_g;
  float q = 0.0f;
  if (!is_pad) {
    for (int64_t i = lane; i < G; i += kWarpSize) {
      const float d = static_cast<float>(xr[i]) - m_t;
      q = fmaf(d, d, q);
    }
  }
  const float v_t = WarpSum(q) * inv_g;

  const Step st = Update(m_t, v_t, is_pad, is_bos, beta1f, beta2f, lnb1, lnb2, epsf, cnt0, mean0, var0);

  for (int64_t i = lane; i < G; i += kWarpSize) {
    if (is_pad) {
      yr[i] = static_cast<T>(0.0f);
    } else {
      const float xv = static_cast<float>(xr[i]);
      yr[i] = static_cast<T>(fmaf((xv - st.cm) * st.cr, static_cast<float>(gamma[h0 + i]),
                                  static_cast<float>(beta[h0 + i])));
    }
  }
  if (lane == 0) {
    if (cm_out != nullptr) {
      cm_out[b * K + k] = st.cm;
      cr_out[b * K + k] = st.cr;
    }
    count_out[b] = st.cnt;
    mean_out[b * K + k] = static_cast<T>(st.mean);
    var_out[b * K + k] = static_cast<T>(st.var);
  }
}

// G == 64 with 2-byte elements: 8 channels per lane through one 16-byte load, so a warp covers four groups and a row
// is a handful of CTAs: four times fewer memory instructions than the two-channel path, and a shorter reduction.
constexpr int kLanesPerGroup = 8;  // 8 lanes x 8 channels = one group of 64
constexpr int kGroupsPerWarp = 4;

template <typename T>
__device__ __forceinline__ void Unpack8(uint4 raw, float v[8]) {
  T pair[8];
  *reinterpret_cast<uint4*>(pair) = raw;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = static_cast<float>(pair[i]);
  }
}

template <typename T>
__device__ __forceinline__ uint4 Pack8(const float v[8]) {
  T pair[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    pair[i] = static_cast<T>(v[i]);
  }
  return *reinterpret_cast<uint4*>(pair);
}

template <typename T>
__global__ __launch_bounds__(kWarpsPerBlock* kWarpSize) void TdnDecodeFwdG64WideKernel(
    const T* __restrict__ x, const bool* __restrict__ bos, const bool* __restrict__ padm,
    const int64_t* __restrict__ prev_count, const T* __restrict__ prev_mean, const T* __restrict__ prev_var,
    const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ y, int64_t* __restrict__ count_out,
    T* __restrict__ mean_out, T* __restrict__ var_out, float* __restrict__ cm_out, float* __restrict__ cr_out,
    const int64_t H, const int64_t K, const float beta1f, const float beta2f, const float lnb1, const float lnb2,
    const float epsf) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t k = static_cast<int64_t>(blockIdx.x) * (kWarpsPerBlock * kGroupsPerWarp) +
                    (threadIdx.x >> 5) * kGroupsPerWarp + (lane / kLanesPerGroup);
  if (k >= K) {
    return;  // the whole eight-lane group takes this branch
  }
  const int64_t b = blockIdx.y;
  const int64_t h0 = k * 64 + (lane % kLanesPerGroup) * 8;

  const uint4 xraw = *reinterpret_cast<const uint4*>(x + b * H + h0);
  const uint4 graw = *reinterpret_cast<const uint4*>(gamma + h0);
  const uint4 braw = *reinterpret_cast<const uint4*>(beta + h0);
  const bool is_pad = padm != nullptr && padm[b];
  const bool is_bos = bos != nullptr && bos[b] && !is_pad;
  const int64_t cnt0 = prev_count != nullptr ? prev_count[b] : 0;
  const float mean0 = prev_mean != nullptr ? static_cast<float>(prev_mean[b * K + k]) : 0.0f;
  const float var0 = prev_var != nullptr ? static_cast<float>(prev_var[b * K + k]) : 0.0f;
  float xv[8], gv[8], sv[8];
  Unpack8<T>(xraw, xv);
  Unpack8<T>(graw, gv);
  Unpack8<T>(braw, sv);

  // The count's chain touches no data, so it is taken first: the two bias corrections, their reciprocals and
  // eps are all ready before the reduction publishes anything, and the two IEEE divisions leave the critical path.
  // The values are Update()'s, term for term; the reciprocal is 0 at count 0 or on a padded step, which reproduces
  // Update()'s two branches (cm = 0, cr = rsqrt(eps)) by multiply without a select on the path.
  int64_t cnt = is_bos ? 0 : cnt0;
  if (!is_pad) {
    cnt += 1;
  }
  const float cntf = static_cast<float>(cnt);
  const float d1c = -expm1f(cntf * lnb1);
  const float d2c = -expm1f(cntf * lnb2);
  const float r1c = d1c > 0.0f ? 1.0f / d1c : 0.0f;
  const float r2c = d2c > 0.0f ? 1.0f / d2c : 0.0f;

  // One ladder over the pivot-shifted pair (S, Q): the mean and the variance about it come out of one reduction of
  // three rounds instead of two of three, with Q accumulated while S's shuffles are in flight. The shift is the
  // group's first channel, which is what keeps Q/64 - (S/64)**2 from cancelling at a large offset; it is the form
  // the tiled forward's moments use.
  const unsigned gmask = 0xffffffffu;
  const float pivot = is_pad ? 0.0f : __shfl_sync(gmask, xv[0], lane & ~(kLanesPerGroup - 1));
  float sd = 0.0f, qd = 0.0f;
  if (!is_pad) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float d = xv[i] - pivot;
      sd += d;
      qd = fmaf(d, d, qd);
    }
  }
#pragma unroll
  for (int off = 1; off < kLanesPerGroup; off <<= 1) {
    sd += __shfl_xor_sync(gmask, sd, off);
    qd += __shfl_xor_sync(gmask, qd, off);
  }
  const float sd64 = sd * (1.0f / 64.0f);
  const float m_t = pivot + sd64;
  const float v_t = fmaf(-sd64, sd64, qd * (1.0f / 64.0f));

  Step st;
  st.cnt = cnt;
  st.mean = is_bos ? 0.0f : mean0;
  st.var = is_bos ? 0.0f : var0;
  if (!is_pad) {
    st.mean = fmaf(1.0f - beta1f, m_t, beta1f * st.mean);
    st.var = fmaf(1.0f - beta2f, v_t, beta2f * st.var);
  }
  st.cm = st.mean * r1c;
  {
    float r = 0.0f;
    const float arg = fmaf(fmaxf(st.var, 0.0f), r2c, epsf);
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(arg));
    st.cr = r;
  }
  const float ncr = -st.cm * st.cr;

  float yv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    yv[i] = is_pad ? 0.0f : fmaf(fmaf(xv[i], st.cr, ncr), gv[i], sv[i]);
  }
  *reinterpret_cast<uint4*>(y + b * H + h0) = Pack8<T>(yv);
  if (lane % kLanesPerGroup == 0) {
    if (cm_out != nullptr) {
      cm_out[b * K + k] = st.cm;
      cr_out[b * K + k] = st.cr;
    }
    count_out[b] = st.cnt;
    mean_out[b * K + k] = static_cast<T>(st.mean);
    var_out[b * K + k] = static_cast<T>(st.var);
  }
}

template <typename T>
bool Aligned(const at::Tensor& t, const size_t bytes) {
  return reinterpret_cast<uintptr_t>(t.data_ptr<T>()) % bytes == 0;
}

// path: 2 = G == 64 through 16-byte loads (2-byte elements), 1 = G == 64 through 4/8-byte loads, 0 = strided.
template <typename T>
void LaunchDecode(const at::Tensor& x, const bool* bos, const bool* pad, const int64_t* prev_count, const T* prev_mean,
                  const T* prev_var, const T* gamma, const T* beta, T* y, int64_t* count_out, T* mean_out, T* var_out,
                  float* cm_out, float* cr_out, int64_t B, int64_t H, int64_t K, int path, float beta1f,
                  float beta2f, float lnb1, float lnb2, float epsf, cudaStream_t stream) {
  const dim3 block(kWarpsPerBlock * kWarpSize);
  if (path == 2) {
    if constexpr (sizeof(T) == 2) {
      const int64_t per_cta = kWarpsPerBlock * kGroupsPerWarp;
      const dim3 blocks(static_cast<unsigned>((K + per_cta - 1) / per_cta), static_cast<unsigned>(B));
      TdnDecodeFwdG64WideKernel<T><<<blocks, block, 0, stream>>>(
          x.data_ptr<T>(), bos, pad, prev_count, prev_mean, prev_var, gamma, beta, y, count_out, mean_out, var_out,
          cm_out, cr_out, H, K, beta1f, beta2f, lnb1, lnb2, epsf);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return;
  }
  const dim3 blocks(static_cast<unsigned>((K + kWarpsPerBlock - 1) / kWarpsPerBlock), static_cast<unsigned>(B));
  if (path == 1) {
    if constexpr (sizeof(T) == 2) {
      TdnDecodeFwdG64Kernel<T, uint32_t><<<blocks, block, 0, stream>>>(x.data_ptr<T>(), bos, pad, prev_count, prev_mean,
                                                                     prev_var, gamma, beta, y, count_out, mean_out,
                                                                     var_out, cm_out, cr_out, H, K, beta1f, beta2f,
                                                                     lnb1, lnb2, epsf);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    } else {
      TdnDecodeFwdG64Kernel<T, float2><<<blocks, block, 0, stream>>>(x.data_ptr<T>(), bos, pad, prev_count, prev_mean,
                                                                   prev_var, gamma, beta, y, count_out, mean_out,
                                                                   var_out, cm_out, cr_out, H, K, beta1f, beta2f,
                                                                   lnb1, lnb2, epsf);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  TdnDecodeFwdKernel<T><<<blocks, block, 0, stream>>>(x.data_ptr<T>(), bos, pad, prev_count, prev_mean, prev_var, gamma,
                                                      beta, y, count_out, mean_out, var_out, cm_out, cr_out, H, K,
                                                      H / K, beta1f, beta2f, lnb1, lnb2, epsf);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Whether a state tensor is one this kernel may index as a flat [.., K] buffer.
bool StateOk(const at::Tensor& t, const int64_t numel, const at::ScalarType dtype) {
  return t.numel() == 0 || (t.numel() == numel && t.scalar_type() == dtype && t.is_contiguous());
}

}  // namespace

bool GroupTimestepDecayNormDecodeCUDAFwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask, const torch::Tensor& prev_count,
    const torch::Tensor& prev_mean, const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups, double beta1,
    double beta2, double eps, torch::Tensor y_out, torch::Tensor count_out, torch::Tensor mean_out,
    torch::Tensor var_out, torch::Tensor cummean_out, torch::Tensor cumrstd_out, const bool write_stats) {
  if (x.dim() != 3 || x.size(1) != 1) {
    return false;  // decode only: one step
  }
  const int64_t B = x.size(0), H = x.size(2);
  const int64_t K = num_groups;
  if (B <= 0 || B > 65535 || H <= 0 || K <= 0 || H % K != 0) {
    return false;  // gridDim.y is B
  }
  const auto dtype = x.scalar_type();
  if (dtype != at::kFloat && dtype != at::kHalf && dtype != at::kBFloat16) {
    return false;  // the vectorized path reads two channels as one 4- or 8-byte word
  }
  if (!x.is_contiguous() || !y_out.is_contiguous() || !gamma.is_contiguous() || !beta.is_contiguous() ||
      !mean_out.is_contiguous() || !var_out.is_contiguous() || !count_out.is_contiguous()) {
    return false;
  }
  if (gamma.numel() != H || beta.numel() != H || gamma.scalar_type() != dtype || beta.scalar_type() != dtype ||
      y_out.scalar_type() != dtype || y_out.numel() != x.numel() || count_out.numel() != B ||
      count_out.scalar_type() != at::kLong || mean_out.scalar_type() != dtype || mean_out.numel() != B * K ||
      var_out.scalar_type() != dtype || var_out.numel() != B * K) {
    return false;
  }
  if (!StateOk(prev_count, B, at::kLong) || !StateOk(prev_mean, B * K, dtype) || !StateOk(prev_var, B * K, dtype)) {
    return false;
  }
  if ((bos_mask.has_value() && bos_mask->numel() > 0 &&
       (bos_mask->scalar_type() != at::kBool || !bos_mask->is_contiguous() || bos_mask->numel() != B)) ||
      (padding_mask.has_value() && padding_mask->numel() > 0 &&
       (padding_mask->scalar_type() != at::kBool || !padding_mask->is_contiguous() ||
        padding_mask->numel() != B))) {
    return false;
  }
  if (write_stats && (cummean_out.scalar_type() != at::kFloat || cumrstd_out.scalar_type() != at::kFloat ||
                      !cummean_out.is_contiguous() || !cumrstd_out.is_contiguous() ||
                      cummean_out.numel() != B * K || cumrstd_out.numel() != B * K)) {
    return false;
  }
  const bool has_prev_count = prev_count.numel() == B;
  const bool has_prev_mean = prev_mean.numel() == B * K;
  const bool has_prev_var = prev_var.numel() == B * K;
  const bool* bos = bos_mask.has_value() && bos_mask->numel() > 0 ? bos_mask->data_ptr<bool>() : nullptr;
  const bool* pad = padding_mask.has_value() && padding_mask->numel() > 0 ? padding_mask->data_ptr<bool>() : nullptr;
  const int64_t* pcount = has_prev_count ? prev_count.data_ptr<int64_t>() : nullptr;
  const float beta1f = static_cast<float>(beta1);
  const float beta2f = static_cast<float>(beta2);
  const float epsf = static_cast<float>(eps);
  const float lnb1 = static_cast<float>(std::log(beta1));  // computed in double on the host, then rounded to float
  const float lnb2 = static_cast<float>(std::log(beta2));
  float* cm = write_stats ? cummean_out.data_ptr<float>() : nullptr;
  float* cr = write_stats ? cumrstd_out.data_ptr<float>() : nullptr;
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "tdn_decode_fwd", [&] {
    // G == 64 through wide loads when the four buffers allow it, through 4/8-byte loads otherwise, strided else.
    const size_t word = sizeof(scalar_t) == 2 ? 4 : 8;
    int path = 0;
    if (H == 64 * K && Aligned<scalar_t>(x, word) && Aligned<scalar_t>(y_out, word) &&
        Aligned<scalar_t>(gamma, word) && Aligned<scalar_t>(beta, word)) {
      path = 1;
      if (sizeof(scalar_t) == 2 && K % kGroupsPerWarp == 0 && Aligned<scalar_t>(x, 16) &&
          Aligned<scalar_t>(y_out, 16) && Aligned<scalar_t>(gamma, 16) && Aligned<scalar_t>(beta, 16)) {
        path = 2;  // eight channels per lane
      }
    }
    const scalar_t* pm = has_prev_mean ? prev_mean.data_ptr<scalar_t>() : nullptr;
    const scalar_t* pv = has_prev_var ? prev_var.data_ptr<scalar_t>() : nullptr;
    LaunchDecode<scalar_t>(x, bos, pad, pcount, pm, pv, gamma.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(),
                           y_out.data_ptr<scalar_t>(), count_out.data_ptr<int64_t>(), mean_out.data_ptr<scalar_t>(),
                           var_out.data_ptr<scalar_t>(), cm, cr, B, H, K, path, beta1f, beta2f, lnb1, lnb2,
                           epsf, stream);
  });
  return true;
}

}  // namespace ops
}  // namespace xllm
