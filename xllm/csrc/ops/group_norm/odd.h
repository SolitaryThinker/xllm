// The group norms for group widths whose 16-byte vector count is NOT a power of two: cpg = 24, 40, 80, 56, 72,
// 88, 96, ... (vpg = cpg / 8 lanes per group, vpg in 3..32 not a power of two). These are xllm's ortho residual
// norms (64 residual heads of 24, 40 and 64 channels; 32 heads of 80) and similar widths. A kernel that maps a group
// onto a power-of-two sub-warp cannot serve them, and xllm's own kernels serve them at ~0.5 TB/s.
//
// Mapping: one lane holds 8 channels (16 B for bf16/fp16, 32 B for fp32 = two 128-bit accesses); a group of vpg
// lanes sits inside one warp, gpw = 32 / vpg groups per warp (lanes past gpw * vpg idle); the group's sums are
// reduced with shuffles having `base_lane = gidx * vpg` as the group's lane base and a shift loop for the
// non-power-of-two count. Nothing goes through shared memory and no __syncthreads is taken in the data path.
//
// Forward: x -> y, rstd (and mean), one row tile per iteration with a one-deep prefetch of the next row.
// Backward: dy, x_or_y -> dx; with an affine transform also dgamma (and dbeta) accumulated per lane in fp32 and
// combined across CTAs by atomics into an fp32 workspace, cast to the output dtype by one small kernel.
#pragma once

#include <cstdint>

#include <ATen/ATen.h>

namespace xllm {
namespace ops {

// mode 0 forward, 1 backward; norm 0 RMS, 1 layer. Answers false, and touches nothing, for every shape whose
// group width is not this class (its owner's kernel takes it).
bool GroupNormOddRun(int64_t mode, int64_t norm, int64_t num_groups, double eps, bool memory_efficient,
                     const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean,
                     const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
                     const at::Tensor& y, const at::Tensor& mean_out, const at::Tensor& rstd_out,
                     const at::Tensor& dx, const at::Tensor& dgamma, const at::Tensor& dbeta);

}  // namespace ops
}  // namespace xllm
