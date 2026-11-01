#pragma once

#include <torch/torch.h>

namespace xllm {
namespace ops {

// Lane-group path of the group-norm forward pass (RMS norm or layer norm, with or without the affine transform).
// x is [B, S, C]; the last dimension is split into `groups` groups of C / groups channels. Writes y [B, S, C],
// rstd [B*S, groups] fp32 and, for the layer norm, mean [B*S, groups] fp32. gamma_or_empty has C elements (bf16,
// fp16 or fp32, as x) or is empty for no affine transform; beta_or_empty likewise (layer norm only).
// Returns false when the shape is not handled here, and the caller takes another path.
bool GroupNormFwdLaneGroups(const at::Tensor& x, const at::Tensor& gamma_or_empty, const at::Tensor& beta_or_empty,
                      int64_t groups, double eps, bool layer_norm, at::Tensor& y, at::Tensor& mean_out,
                      at::Tensor& rstd_out);

}  // namespace ops
}  // namespace xllm
