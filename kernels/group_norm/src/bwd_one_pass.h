// The one-pass backward kernel (bwd_one_pass.cu): dx, dgamma and dbeta from one read of x_or_y and dy.
//
// Returns false for any call it does not handle (a dtype or a layout it does not vectorize), so the caller can use
// another kernel instead.
#pragma once

#include <torch/torch.h>

namespace xllm {
namespace ops {

bool GroupNormBwdOnePass(const torch::Tensor& dy, const torch::Tensor& x_or_y, const torch::Tensor& mean,
                        const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
                        int64_t channels, int64_t num_groups, bool layer, bool memory_efficient, torch::Tensor dx,
                        torch::Tensor dgamma, torch::Tensor dbeta);

}  // namespace ops
}  // namespace xllm
