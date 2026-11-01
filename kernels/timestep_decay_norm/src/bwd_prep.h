// The torch-facing declaration of the per-(row, step) prep (see bwd_prep.cuh for what it computes).
#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

namespace xllm {
namespace ops {

// out: fp32 float2 [B, L] (1/(1 - beta1**count), 1/(1 - beta2**count)), already allocated by the caller;
// bos_mask/padding_mask: bool [B, L] or a tensor with no elements. One launch, no allocation inside.
void GroupTimestepDecayNormBwdPrep(const c10::optional<torch::Tensor>& bos_mask,
                                   const c10::optional<torch::Tensor>& padding_mask, const torch::Tensor& prev_count,
                                   torch::Tensor& out, int64_t B, int64_t L, double beta1, double beta2);

}  // namespace ops
}  // namespace xllm
