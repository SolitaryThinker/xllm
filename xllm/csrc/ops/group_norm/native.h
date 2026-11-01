// Native group-norm signatures backed by the fused dispatcher.
#pragma once
#include <torch/extension.h>
namespace xllm::ops::blackwell {
bool Supported(const at::Tensor& x);
torch::Tensor GroupRMSNormBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                              int64_t num_channels, int64_t num_groups, const torch::Tensor& rstd,
                              bool memory_efficient);
std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormBwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, const torch::Tensor& gamma, bool memory_efficient);
std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwd(const torch::Tensor& x,
                                                         int64_t num_channels, int64_t num_groups,
                                                         double eps);
std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwdAffine(const torch::Tensor& x,
                                                               int64_t num_channels,
                                                               int64_t num_groups,
                                                               const torch::Tensor& gamma,
                                                               double eps);
torch::Tensor GroupLayerNormBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                                int64_t num_channels, int64_t num_groups, const torch::Tensor& mean,
                                const torch::Tensor& rstd, bool memory_efficient);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormBwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd, const torch::Tensor& gamma,
    const torch::Tensor& beta, bool memory_efficient);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwd(const torch::Tensor& x,
                                                                          int64_t num_channels,
                                                                          int64_t num_groups,
                                                                          double eps);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, const torch::Tensor& gamma,
    const torch::Tensor& beta, double eps);
}  // namespace xllm::ops::blackwell
