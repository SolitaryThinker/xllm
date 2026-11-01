// The host side of the per-(row, step) prep: one launch, and no allocation (the destination is the caller's).
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>

#include "bwd_prep.cuh"
#include "bwd_prep.h"

namespace xllm {
namespace ops {

void GroupTimestepDecayNormBwdPrep(const c10::optional<torch::Tensor>& bos_mask,
                                   const c10::optional<torch::Tensor>& padding_mask, const torch::Tensor& prev_count,
                                   torch::Tensor& out, int64_t B, int64_t L, double beta1, double beta2) {
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.dim() == 3 &&
                  out.size(2) == 2,
              "prep: out must be fp32 [B, L, 2] contiguous");
  TORCH_CHECK(out.size(0) == B && out.size(1) == L, "prep: out must be [B, L, 2]");
  TORCH_CHECK(L >= 1 && B >= 1, "prep: B and L at least 1");
  TORCH_CHECK(B <= 65535, "prep: B must fit gridDim.y");
  // xllm hands these in as views of a cache or of a packed batch: take a contiguous handle of each (the same tensor
  // when it already is one) and check they live where out does.
  const auto dev = out.device();
  TORCH_CHECK(prev_count.is_cuda() && prev_count.scalar_type() == at::kLong && prev_count.numel() == B &&
                  prev_count.device() == dev,
              "prep: prev_count must be int64 [B] on out's device");
  const bool has_bos = bos_mask.has_value() && bos_mask->numel() > 0;
  const bool has_pad = padding_mask.has_value() && padding_mask->numel() > 0;
  TORCH_CHECK(!has_bos || (bos_mask->is_cuda() && bos_mask->scalar_type() == at::kBool && bos_mask->numel() == B * L &&
                            bos_mask->device() == dev),
              "prep: bos_mask must be bool [B, L] on out's device");
  TORCH_CHECK(!has_pad || (padding_mask->is_cuda() && padding_mask->scalar_type() == at::kBool &&
                            padding_mask->numel() == B * L && padding_mask->device() == dev),
              "prep: padding_mask must be bool [B, L] on out's device");
  // Kept alive to the end of the launch, so a copy made here outlives the kernel that reads it.
  const at::Tensor pc = *prev_count.expect_contiguous();
  const at::Tensor bos_t = has_bos ? *bos_mask->expect_contiguous() : at::Tensor();
  const at::Tensor pad_t = has_pad ? *padding_mask->expect_contiguous() : at::Tensor();
  const bool* bos = has_bos ? bos_t.data_ptr<bool>() : nullptr;
  const bool* pad = has_pad ? pad_t.data_ptr<bool>() : nullptr;

  const c10::cuda::CUDAGuard guard(out.device());
  const dim3 grid((unsigned)((L + prep::kTile - 1) / prep::kTile), (unsigned)B);
  const float lb1 = (float)std::log(beta1);
  const float lb2 = (float)std::log(beta2);
  const int max_count = 1 << 23;   // past this 1 - beta**count is 1 in fp32
  auto* dst = reinterpret_cast<float2*>(out.data_ptr<float>());
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int Li = (int)L;

  const int64_t* pcp = pc.data_ptr<int64_t>();
  if (has_bos) {
    if (has_pad) {
      prep::prep_kernel<true, true><<<grid, prep::kThreads, 0, stream>>>(bos, pad, pcp, dst, Li, lb1, lb2, max_count);
    } else {
      prep::prep_kernel<true, false><<<grid, prep::kThreads, 0, stream>>>(bos, pad, pcp, dst, Li, lb1, lb2, max_count);
    }
  } else {
    if (has_pad) {
      prep::prep_kernel<false, true><<<grid, prep::kThreads, 0, stream>>>(bos, pad, pcp, dst, Li, lb1, lb2, max_count);
    } else {
      prep::prep_kernel<false, false><<<grid, prep::kThreads, 0, stream>>>(bos, pad, pcp, dst, Li, lb1, lb2, max_count);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ops
}  // namespace xllm
