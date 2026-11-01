// Native TDN entry points, preserving cub [B,K,L] and chunkwise [B,L,K] saved-statistic layouts.
#pragma once
#include <torch/extension.h>
namespace xllm::ops::blackwell {
bool Supported(const at::Tensor& x, int64_t groups);
using ForwardResult =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
using BackwardResult = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
ForwardResult Forward(const at::Tensor& x, const c10::optional<at::Tensor>& bos,
                      const at::Tensor& pc, const at::Tensor& pm, const at::Tensor& pv,
                      const at::Tensor& gamma, const at::Tensor& beta,
                      const c10::optional<at::Tensor>& pad, int64_t groups, double b1, double b2,
                      double eps, bool cub);
BackwardResult Backward(const at::Tensor& dy, const at::Tensor& dm, const at::Tensor& dv,
                        const at::Tensor& x, const at::Tensor& pc,
                        const c10::optional<at::Tensor>& bos, const at::Tensor& cm,
                        const at::Tensor& cr, const at::Tensor& gamma, const at::Tensor& beta,
                        const c10::optional<at::Tensor>& pad, int64_t groups, double b1, double b2,
                        double eps, bool me, bool cub);
}  // namespace xllm::ops::blackwell
