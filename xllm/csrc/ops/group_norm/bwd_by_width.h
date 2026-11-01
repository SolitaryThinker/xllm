// The backward pass of xllm's group RMS norm and group layer norm in one pass over the data (dx, dgamma and
// dbeta from the same load).
#pragma once

#include <ATen/ATen.h>
#include <cstdint>

namespace gn {
// Returns true when it wrote dx (and dgamma/dbeta when affine), false when the shape is not one this kernel covers;
// the caller then takes another path. norm: 0 RMS, 1 layer norm.
bool GroupNormBwdByWidth(int64_t norm, const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean,
                     const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta, bool memory_efficient,
                     at::Tensor& dx, at::Tensor& dgamma, at::Tensor& dbeta);
}  // namespace gn
