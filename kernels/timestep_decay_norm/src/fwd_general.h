#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

namespace xllm {
namespace ops {

// Group timestep decay normalization, forward (mode 0/1), for every shape xllm can call: any number of groups of any
// width, bf16/fp16/fp32, bos and padding masks, fresh and carried state. Two launches at most: one that reduces each
// (row, group) x time-tile to its affine aggregate, and one that composes the aggregates of the earlier tiles, scans
// the tile and normalizes (x read once in each; y written once, and the statistics [B, L, K] fp32 when save_stats).
// With a single time tile per (row, group) the first launch is not needed and the same kernel does both passes.
void TdnFwdGeneral(const at::Tensor& x, const c10::optional<at::Tensor>& bos, const at::Tensor& prev_count,
                   const at::Tensor& prev_mean, const at::Tensor& prev_var, const at::Tensor& gamma,
                   const at::Tensor& beta, const c10::optional<at::Tensor>& pad, int64_t num_groups, double beta1,
                   double beta2, double eps, bool save_stats, at::Tensor y, at::Tensor count, at::Tensor mean,
                   at::Tensor var, at::Tensor cummean_out, at::Tensor cumrstd_out);

}  // namespace ops
}  // namespace xllm
