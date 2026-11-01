// The wide class: a group of cpg >= 512 channels (vpg = cpg / 8 >= 64 sixteen-byte vectors). These are xllm's
// residual-stream norms (1, 2, 4 and 1 groups of 1536, 1280, 1024 and 4096 channels). One CTA covers a whole group:
// W = ceil(vpg / 32) warps, a lane is one 16-byte vector (8 channels), and the group's sums are reduced across the CTA
// with two barriers a row. Statistics fp32 [rows, G]; the backward pass
// is one pass over x_or_y and dy that writes dx and accumulates dgamma/dbeta per lane in registers, combined across
// CTAs by one atomicAdd a channel at the end (fp32 workspace, cast to the output dtype by one small kernel).
#pragma once

#include <cstdint>

#include <ATen/ATen.h>

namespace xllm {
namespace ops {

// mode 0 forward, 1 backward; norm 0 RMS, 1 layer. Answers false, and touches nothing, for every shape whose group
// width is not this class.
bool GroupNormWideRun(int64_t mode, int64_t norm, int64_t num_groups, double eps, bool memory_efficient,
                      const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean, const at::Tensor& rstd,
                      const at::Tensor& gamma, const at::Tensor& beta, const at::Tensor& y, const at::Tensor& mean_out,
                      const at::Tensor& rstd_out, const at::Tensor& dx, const at::Tensor& dgamma,
                      const at::Tensor& dbeta);

}  // namespace ops
}  // namespace xllm
