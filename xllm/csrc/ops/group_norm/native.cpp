// Allocate the native op outputs and route one group-norm call through the fused dispatcher.
#include "native.h"

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
void GroupNormRun(int64_t mode, int64_t norm, int64_t num_groups, double eps, bool memory_efficient,
                  const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean,
                  const at::Tensor& rstd, const at::Tensor& gamma, const at::Tensor& beta,
                  at::Tensor y, at::Tensor mean_out, at::Tensor rstd_out, at::Tensor dx,
                  at::Tensor dgamma, at::Tensor dbeta);

namespace xllm::ops::blackwell {
bool Supported(const at::Tensor& x) {
  return !at::globalContext().deterministicAlgorithms() && x.is_cuda() &&
         at::cuda::getDeviceProperties(x.get_device())->major >= 10 &&
         (x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
          x.scalar_type() == at::kBFloat16) &&
         x.numel() > 0;
}
torch::Tensor GroupRMSNormBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                              int64_t num_channels, int64_t num_groups, const torch::Tensor& rstd,
                              bool memory_efficient) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      x.dim() == 3 && num_groups > 0 && num_channels == x.size(2) && num_channels % num_groups == 0,
      "invalid group-norm shape");
  auto xc = x.contiguous();
  auto empty = at::empty({0}, x.options());
  auto y = empty;

  auto mu = empty;
  auto rs = empty;
  auto dx = at::empty_like(xc);
  auto dg = empty;
  auto db = empty;
  GroupNormRun(1, 0, num_groups, 0.0, memory_efficient, xc, y_grad.contiguous(), empty,
               rstd.contiguous(), empty, empty, y, mu, rs, dx, dg, db);
  return dx;
}
std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormBwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& rstd, const torch::Tensor& gamma, bool memory_efficient) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      x.dim() == 3 && num_groups > 0 && num_channels == x.size(2) && num_channels % num_groups == 0,
      "invalid group-norm shape");
  auto xc = x.contiguous();
  auto empty = at::empty({0}, x.options());
  auto y = empty;

  auto mu = empty;
  auto rs = empty;
  auto dx = at::empty_like(xc);
  auto dg = at::empty_like(gamma);
  auto db = empty;
  GroupNormRun(1, 0, num_groups, 0.0, memory_efficient, xc, y_grad.contiguous(), empty,
               rstd.contiguous(), gamma.contiguous(), empty, y, mu, rs, dx, dg, db);
  return std::make_tuple(dx, dg);
}
std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwd(const torch::Tensor& x,
                                                         int64_t num_channels, int64_t num_groups,
                                                         double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      x.dim() == 3 && num_groups > 0 && num_channels == x.size(2) && num_channels % num_groups == 0,
      "invalid group-norm shape");
  auto xc = x.contiguous();
  auto empty = at::empty({0}, x.options());
  auto y = at::empty_like(xc);
  auto stats = at::empty({x.size(0) * x.size(1), num_groups}, x.options().dtype(at::kFloat));
  auto mu = empty;
  auto rs = stats;
  auto dx = empty;
  auto dg = empty;
  auto db = empty;
  GroupNormRun(0, 0, num_groups, eps, false, xc, empty, empty, empty, empty, empty, y, mu, rs, dx,
               dg, db);
  return std::make_tuple(y, rs);
}
std::tuple<torch::Tensor, torch::Tensor> GroupRMSNormFwdAffine(const torch::Tensor& x,
                                                               int64_t num_channels,
                                                               int64_t num_groups,
                                                               const torch::Tensor& gamma,
                                                               double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      x.dim() == 3 && num_groups > 0 && num_channels == x.size(2) && num_channels % num_groups == 0,
      "invalid group-norm shape");
  auto xc = x.contiguous();
  auto empty = at::empty({0}, x.options());
  auto y = at::empty_like(xc);
  auto stats = at::empty({x.size(0) * x.size(1), num_groups}, x.options().dtype(at::kFloat));
  auto mu = empty;
  auto rs = stats;
  auto dx = empty;
  auto dg = empty;
  auto db = empty;
  GroupNormRun(0, 0, num_groups, eps, false, xc, empty, empty, empty, gamma.contiguous(), empty, y,
               mu, rs, dx, dg, db);
  return std::make_tuple(y, rs);
}
torch::Tensor GroupLayerNormBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                                int64_t num_channels, int64_t num_groups, const torch::Tensor& mean,
                                const torch::Tensor& rstd, bool memory_efficient) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      x.dim() == 3 && num_groups > 0 && num_channels == x.size(2) && num_channels % num_groups == 0,
      "invalid group-norm shape");
  auto xc = x.contiguous();
  auto empty = at::empty({0}, x.options());
  auto y = empty;

  auto mu = empty;
  auto rs = empty;
  auto dx = at::empty_like(xc);
  auto dg = empty;
  auto db = empty;
  GroupNormRun(1, 1, num_groups, 0.0, memory_efficient, xc, y_grad.contiguous(), mean.contiguous(),
               rstd.contiguous(), empty, empty, y, mu, rs, dx, dg, db);
  return dx;
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormBwdAffine(
    const torch::Tensor& y_grad, const torch::Tensor& x, int64_t num_channels, int64_t num_groups,
    const torch::Tensor& mean, const torch::Tensor& rstd, const torch::Tensor& gamma,
    const torch::Tensor& beta, bool memory_efficient) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      x.dim() == 3 && num_groups > 0 && num_channels == x.size(2) && num_channels % num_groups == 0,
      "invalid group-norm shape");
  auto xc = x.contiguous();
  auto empty = at::empty({0}, x.options());
  auto y = empty;

  auto mu = empty;
  auto rs = empty;
  auto dx = at::empty_like(xc);
  auto dg = at::empty_like(gamma);
  auto db = at::empty_like(beta);
  GroupNormRun(1, 1, num_groups, 0.0, memory_efficient, xc, y_grad.contiguous(), mean.contiguous(),
               rstd.contiguous(), gamma.contiguous(), beta.contiguous(), y, mu, rs, dx, dg, db);
  return std::make_tuple(dx, dg, db);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwd(const torch::Tensor& x,
                                                                          int64_t num_channels,
                                                                          int64_t num_groups,
                                                                          double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      x.dim() == 3 && num_groups > 0 && num_channels == x.size(2) && num_channels % num_groups == 0,
      "invalid group-norm shape");
  auto xc = x.contiguous();
  auto empty = at::empty({0}, x.options());
  auto y = at::empty_like(xc);
  auto stats = at::empty({x.size(0) * x.size(1), num_groups}, x.options().dtype(at::kFloat));
  auto mu = at::empty_like(stats);
  auto rs = stats;
  auto dx = empty;
  auto dg = empty;
  auto db = empty;
  GroupNormRun(0, 1, num_groups, eps, false, xc, empty, empty, empty, empty, empty, y, mu, rs, dx,
               dg, db);
  return std::make_tuple(y, mu, rs);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupLayerNormFwdAffine(
    const torch::Tensor& x, int64_t num_channels, int64_t num_groups, const torch::Tensor& gamma,
    const torch::Tensor& beta, double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      x.dim() == 3 && num_groups > 0 && num_channels == x.size(2) && num_channels % num_groups == 0,
      "invalid group-norm shape");
  auto xc = x.contiguous();
  auto empty = at::empty({0}, x.options());
  auto y = at::empty_like(xc);
  auto stats = at::empty({x.size(0) * x.size(1), num_groups}, x.options().dtype(at::kFloat));
  auto mu = at::empty_like(stats);
  auto rs = stats;
  auto dx = empty;
  auto dg = empty;
  auto db = empty;
  GroupNormRun(0, 1, num_groups, eps, false, xc, empty, empty, empty, gamma.contiguous(),
               beta.contiguous(), y, mu, rs, dx, dg, db);
  return std::make_tuple(y, mu, rs);
}
}  // namespace xllm::ops::blackwell
