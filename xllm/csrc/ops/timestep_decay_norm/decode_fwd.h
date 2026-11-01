#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

namespace xllm {
namespace ops {

// One fused launch for a single step (decode): the group moments, the decayed scan with the state carried in, and the
// normalization, writing y, the final count/mean/var and (training) the [B, 1, K] fp32 statistics. Returns false when
// this path does not apply (not L == 1, a dtype or a layout it does not take, a shape out of range), in which case
// nothing was launched and the caller must use another path.
bool GroupTimestepDecayNormDecodeCUDAFwd(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& bos_mask, const torch::Tensor& prev_count,
    const torch::Tensor& prev_mean, const torch::Tensor& prev_var, const torch::Tensor& gamma,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups, double beta1,
    double beta2, double eps, torch::Tensor y_out, torch::Tensor count_out, torch::Tensor mean_out,
    torch::Tensor var_out, torch::Tensor cummean_out, torch::Tensor cumrstd_out, const bool write_stats);

}  // namespace ops
}  // namespace xllm
