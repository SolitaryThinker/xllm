#pragma once

#include <torch/torch.h>

#include <tuple>

namespace xllm {
namespace ops {

// Group layer norm over the last dimension of x [B, S, C], split into num_groups groups of C / num_groups channels.
// mean and rstd are [B * S, num_groups] in fp32. Every output is allocated by the caller.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormCUDAFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& gamma, const torch::Tensor& beta, double eps,
    torch::Tensor y_out, torch::Tensor mean_out, torch::Tensor rstd_out);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormCUDAFwd(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, double eps,
    torch::Tensor y_out, torch::Tensor mean_out, torch::Tensor rstd_out);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormCUDABwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
    bool memory_efficient, torch::Tensor x_grad_out, torch::Tensor gamma_grad_out, torch::Tensor beta_grad_out);
torch::Tensor GroupLayerNormCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd, bool memory_efficient, torch::Tensor x_grad_out);

}  // namespace ops
}  // namespace xllm
