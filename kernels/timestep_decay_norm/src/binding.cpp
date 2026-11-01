// The entry point: one call of xllm's group timestep decay normalization, chosen by mode. Called positionally with the
// inputs and then the outputs (destination passing): y, count, mean, var, cummean, cumrstd, dx, dprev_mean, dprev_var,
// dgamma, dbeta. An output a mode does not produce is a one-element placeholder that stays as given; an input a mode
// does not take is an empty tensor.
//
//   mode 0 (inference forward): y, count, mean, var
//   mode 1 (training forward):  y, count, mean, var, and the statistics the backward pass takes, [B, L, K] fp32
//   mode 2 (backward):          dx, dprev_mean, dprev_var, dgamma, dbeta; from y when memory_efficient
#include <torch/extension.h>

#include "bwd_fused.h"
#include "bwd_general.h"
#include "decode_fwd.h"
#include "fwd_general.h"
#include "fwd_persistent.h"

namespace {

constexpr int64_t kInfer = 0;
constexpr int64_t kTrain = 1;

c10::optional<torch::Tensor> optional(const at::Tensor& t) {
  return t.numel() > 0 ? c10::optional<torch::Tensor>(t) : c10::nullopt;
}

}  // namespace

void run(int64_t mode, int64_t num_groups, double beta1, double beta2, double eps, bool memory_efficient,
         const at::Tensor& x_or_y, const at::Tensor& bos_mask, const at::Tensor& padding_mask,
         const at::Tensor& prev_count, const at::Tensor& prev_mean, const at::Tensor& prev_var,
         const at::Tensor& gamma, const at::Tensor& beta, const at::Tensor& dy, const at::Tensor& dmean,
         const at::Tensor& dvar, const at::Tensor& cummean, const at::Tensor& cumrstd, at::Tensor y, at::Tensor count,
         at::Tensor mean, at::Tensor var, at::Tensor cummean_out, at::Tensor cumrstd_out, at::Tensor dx,
         at::Tensor dprev_mean, at::Tensor dprev_var, at::Tensor dgamma, at::Tensor dbeta) {
  using namespace xllm::ops;
  TORCH_CHECK(x_or_y.is_cuda() && x_or_y.dim() == 3, "x_or_y: [B, L, H] on the GPU");
  const auto bos = optional(bos_mask);
  const auto pad = optional(padding_mask);
  if (mode == kInfer || mode == kTrain) {
    // A single step (decode) is one fused launch: one warp per (row, group), the state carried in registers.
    if (GroupTimestepDecayNormDecodeCUDAFwd(x_or_y, bos, prev_count, prev_mean, prev_var, gamma, beta, pad, num_groups,
                                            beta1, beta2, eps, y, count, mean, var, cummean_out, cumrstd_out,
                                            mode == kTrain)) {
      return;
    }
    // One persistent launch for the forward pass: L >= 2, groups of 64 channels, 16-bit elements. It returns false
    // when the shape is not its, and the general path below takes it.
    if (TdnFwdPersistent(x_or_y, bos, pad, prev_count, prev_mean, prev_var, gamma, beta, num_groups, beta1, beta2, eps, y,
                    count, mean, var,
                    mode == kTrain ? c10::optional<torch::Tensor>(cummean_out) : c10::nullopt,
                    mode == kTrain ? c10::optional<torch::Tensor>(cumrstd_out) : c10::nullopt)) {
      return;
    }
    // Every other forward (L >= 2, any shape, any dtype): the two-phase scan, which writes the saved statistics
    // straight as [B, L, K] fp32.
    TdnFwdGeneral(x_or_y, bos, prev_count, prev_mean, prev_var, gamma, beta, pad, num_groups, beta1, beta2, eps,
                  mode == kTrain, y, count, mean, var, cummean_out, cumrstd_out);
    return;
  }
  // mode 2: the fused pass first, and behind it the generic path for any shape the fused entry declines. The fused
  // entry returns false before it writes anything, so the fall-through is safe and the predicate lives in the entry.
  if (GroupTimestepDecayNormBwdFused(dy, dmean, dvar, x_or_y, prev_count, bos, cummean, cumrstd, gamma, beta, pad,
                                     num_groups, beta1, beta2, memory_efficient, dx, dprev_mean, dprev_var, dgamma,
                                     dbeta)) {
    return;
  }
  GroupTimestepDecayNormBwdGeneralCUDA(dy, dmean, dvar, x_or_y, prev_count, bos, pad, cummean, cumrstd, gamma, beta,
                                       num_groups, beta1, beta2, eps, memory_efficient, dx, dprev_mean, dprev_var,
                                       dgamma, dbeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
