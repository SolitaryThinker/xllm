// The many-group class of xllm's group norms. A group is 1..32 sixteen-byte vectors
// (cpg * elemsize % 16 == 0, V = cpg * elemsize / 16 in [1, 32], so cpg <= 256 bf16 / 128 fp32).
// Lane map: a group takes V lanes of a warp -- not pow2ceil(V) -- so that groups of 3, 5 or 10 vectors
// (cpg 24, 40, 80 bf16) keep every lane busy. On the two paths that carry no affine transform and have an
// even V (the forward and the flat backward), a group instead takes V/2 lanes of TWO 16-byte chunks each,
// lane `lid` holding chunks `lid` and `lid + V/2`: that halves the lanes a group occupies, so a warp covers
// twice as many whole groups and its statistics land in one 32 B sector. Either way gw = 32 / W groups
// share a warp and the group's statistics never leave it: a segmented shfl tree, no shared memory and no
// __syncthreads in the data path (the two backward paths: flat when there is no affine transform, fixed
// column slot per warp when there is, so dgamma/dbeta stay in registers; only the small dgamma/dbeta fold
// kernel that ends the affine backward uses shared memory).
#pragma once

#include <torch/extension.h>

namespace xllm {
namespace ops {
namespace dense_lanes {

// True when these kernels handle the call (contiguous, cuda, fp32/fp16/bf16, V in [1, 32]).
bool Eligible(const torch::Tensor& x_or_y, int64_t num_groups);

void Fwd(const torch::Tensor& x, int64_t num_groups, bool layer, const torch::Tensor& gamma,
         const torch::Tensor& beta, double eps, torch::Tensor y, torch::Tensor mean_out, torch::Tensor rstd_out);

void Bwd(const torch::Tensor& x_or_y, const torch::Tensor& dy, int64_t num_groups, bool layer, bool memory_efficient,
         const torch::Tensor& mean, const torch::Tensor& rstd, const torch::Tensor& gamma, const torch::Tensor& beta,
         torch::Tensor dx, torch::Tensor dgamma, torch::Tensor dbeta);

}  // namespace dense_lanes
}  // namespace ops
}  // namespace xllm
