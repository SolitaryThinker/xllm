#pragma once

#include <torch/torch.h>

#include <c10/util/Optional.h>

namespace xllm {
namespace ops {

// The backward pass of group timestep decay normalization, one fused kernel:
//
//   dy [B, L, H], dmean/dvar [B, K] (the gradient of the final state), x (or y when memory_efficient),
//   bos_mask/padding_mask (optional [B, L] bool), prev_count [B] int64, cummean/cumrstd [B, L, K] fp32
//   (the statistics mode 1 saved), gamma/beta [H].
//
// Writes dx [B, L, H], dprev_mean/dprev_var [B, K], dgamma/dbeta [H].
//
// The adjoint of the state along the sequence is the forward's affine recurrence read backwards: per (row,
// group, statistic) R_{t} = alpha_{t+1} * R_{t+1} + C_t, with alpha = 1 at a padded step, 0 at a bos reset and
// beta otherwise, and C_t the gradient the step injects into the state. Tiles of time therefore compose
// associatively and are scanned with a decoupled look-back from the last tile to the first.
//
// Returns true when it took the shape and wrote every output. Returns false, having written nothing, when the
// shape or dtype is outside what it covers (a group wider than 256 channels, a dtype other than fp32/fp16/bf16,
// statistics not [B, L, K] fp32): the caller then falls through to another path with its outputs untouched.
bool GroupTimestepDecayNormBwdFused(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad, const torch::Tensor& var_grad,
    const torch::Tensor& x_or_y, const torch::Tensor& prev_count,
    const c10::optional<torch::Tensor>& bos_mask, const torch::Tensor& cummean,
    const torch::Tensor& cumrstd, const torch::Tensor& gamma, const torch::Tensor& beta,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups, double beta1, double beta2,
    bool memory_efficient, torch::Tensor x_grad_out, torch::Tensor prev_mean_grad_out,
    torch::Tensor prev_var_grad_out, torch::Tensor gamma_grad_out, torch::Tensor beta_grad_out);

}  // namespace ops
}  // namespace xllm
