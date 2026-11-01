#ifdef XLLM_BLACKWELL
#include "timestep_decay_norm/native.h"
#endif
#include "ops/timestep_decay_norm_cub.h"

namespace xllm {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepDecayNormCubFwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& prev_count, const torch::Tensor& prev_mean,
    const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask,
    int64_t num_groups, double beta1, double beta2, double eps) {
#ifdef XLLM_BLACKWELL
  if (blackwell::Supported(x, num_groups) && beta1 > 0 && beta1 < 1 && beta2 > 0 && beta2 < 1) return blackwell::Forward(x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, num_groups, beta1, beta2, eps, true);
#endif

  return GroupTimestepDecayNormCubCUDAFwd(
      x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, padding_mask,
      num_groups, beta1, beta2, eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepDecayNormCubBwd(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x,
    const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    double beta1, double beta2) {
#ifdef XLLM_BLACKWELL
  if (blackwell::Supported(x, num_groups) && beta1 > 0 && beta1 < 1 && beta2 > 0 && beta2 < 1) return blackwell::Backward(y_grad, mean_grad, var_grad, x, prev_count, bos_mask, cummean, cumrstd, gamma, at::zeros_like(gamma), padding_mask, num_groups, beta1, beta2, 1e-5, false, true);
#endif

  return GroupTimestepDecayNormCubCUDABwd(
      y_grad, mean_grad, var_grad, x, prev_count, bos_mask, cummean, cumrstd,
      gamma, padding_mask, num_groups, beta1, beta2);
}

void DefineTimestepDecayNormCubOp(py::module& m) {
  m.def("group_timestep_decay_norm_cub_fwd", &GroupTimestepDecayNormCubFwd,
        "GroupTimestepDecayNormCub forward")
      .def("group_timestep_decay_norm_cub_bwd", &GroupTimestepDecayNormCubBwd,
           "GroupTimestepDecayNormCub backward");
}

}  // namespace ops
}  // namespace xllm
