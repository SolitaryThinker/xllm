// Group RMS norm / group layer norm forward whose mapping is chosen by channels per group. It takes only the shape
// classes where it is fastest (wide groups of more than 256 channels, and the 40/80-channel widths) and returns false
// otherwise, leaving everything else to the other forward kernels.
#pragma once

#include <ATen/ATen.h>
#include <cstdint>

namespace gn {
// Returns true when it wrote y/mean/rstd, false when the shape is not one this kernel claims (the caller takes another
// path). norm: 0 RMS, 1 layer norm.
bool GroupNormFwdByWidth(int64_t norm, const at::Tensor& x, int64_t num_groups, double eps, const at::Tensor& gamma,
                     const at::Tensor& beta, at::Tensor& y, at::Tensor& mean_out, at::Tensor& rstd_out);
}  // namespace gn
