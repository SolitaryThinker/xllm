#pragma once

#include <torch/extension.h>

#include <c10/util/Optional.h>

namespace xllm {
namespace ops {

// The backward pass of xllm's group timestep decay normalization, in a few memory-bound launches: the per-step
// group adjoints, a parallel reverse scan of the state's adjoints along the sequence, and the element pass that
// writes dx. Correct for every shape xllm can call (any group width, any length, masks, carried state, nonzero
// gradients of the final state, fp32/fp16/bf16), dense (x given) and memory-efficient (x recovered from y).
void GroupTimestepDecayNormBwdGeneralCUDA(
    const torch::Tensor& dy, const torch::Tensor& dmean, const torch::Tensor& dvar, const torch::Tensor& x_or_y,
    const torch::Tensor& prev_count, const c10::optional<torch::Tensor>& bos_mask,
    const c10::optional<torch::Tensor>& padding_mask, const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& beta, int64_t num_groups, double beta1, double beta2, double eps,
    bool memory_efficient, torch::Tensor dx, torch::Tensor dprev_mean, torch::Tensor dprev_var, torch::Tensor dgamma,
    torch::Tensor dbeta);

}  // namespace ops
}  // namespace xllm
