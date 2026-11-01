// The persistent single-pass forward pass (modes 0 and 1), groups of 64 channels, 2-byte elements.
#pragma once

#include <torch/extension.h>

#include <c10/util/Optional.h>

namespace xllm {
namespace ops {

// One launch for the forward pass (mode 0 and mode 1) of a group timestep decay normalization of groups of 64
// channels with bf16/fp16 elements and L >= 2. Returns false for any other shape, and the caller then runs its own
// backend unchanged. `cummean_out`/`cumrstd_out` are the [B, L, K] fp32 statistics (mode 1 only).
bool TdnFwdPersistent(const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask,
                 const c10::optional<torch::Tensor>& padding_mask, const torch::Tensor& prev_count,
                 const torch::Tensor& prev_mean, const torch::Tensor& prev_var, const torch::Tensor& gamma,
                 const torch::Tensor& beta, int64_t num_groups, double beta1, double beta2, double eps,
                 torch::Tensor y, torch::Tensor count_out, torch::Tensor mean_out, torch::Tensor var_out,
                 const c10::optional<torch::Tensor>& cummean_out,
                 const c10::optional<torch::Tensor>& cumrstd_out);

}  // namespace ops
}  // namespace xllm
