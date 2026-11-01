// Allocate native TDN outputs and convert saved statistics only at the cub API boundary.
#include "native.h"

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
void TimestepDecayNormRun(int64_t mode, int64_t num_groups, double beta1, double beta2, double eps,
                          bool memory_efficient, const at::Tensor& x_or_y,
                          const at::Tensor& bos_mask, const at::Tensor& padding_mask,
                          const at::Tensor& prev_count, const at::Tensor& prev_mean,
                          const at::Tensor& prev_var, const at::Tensor& gamma,
                          const at::Tensor& beta, const at::Tensor& dy, const at::Tensor& dmean,
                          const at::Tensor& dvar, const at::Tensor& cummean,
                          const at::Tensor& cumrstd, at::Tensor y, at::Tensor count,
                          at::Tensor mean, at::Tensor var, at::Tensor cummean_out,
                          at::Tensor cumrstd_out, at::Tensor dx, at::Tensor dprev_mean,
                          at::Tensor dprev_var, at::Tensor dgamma, at::Tensor dbeta);

namespace xllm::ops::blackwell {
namespace {

void CheckOperand(const at::Tensor& tensor, const at::Tensor& x, at::IntArrayRef shape,
                  at::ScalarType dtype, const char* name) {
  TORCH_CHECK(
      tensor.device() == x.device() && tensor.scalar_type() == dtype && tensor.sizes() == shape,
      name, ": wrong device, dtype or shape");
}

void CheckCommon(const at::Tensor& x, const at::Tensor& count, const c10::optional<at::Tensor>& bos,
                 const c10::optional<at::Tensor>& pad, const at::Tensor& gamma,
                 const at::Tensor& beta, double eps) {
  TORCH_CHECK(eps > 0.0, "eps must be positive");
  CheckOperand(count, x, {x.size(0)}, at::kLong, "prev_count");
  CheckOperand(gamma, x, {x.size(2)}, x.scalar_type(), "gamma");
  CheckOperand(beta, x, {x.size(2)}, x.scalar_type(), "beta");
  if (bos.has_value()) CheckOperand(*bos, x, {x.size(0), x.size(1)}, at::kBool, "bos_mask");
  if (pad.has_value()) CheckOperand(*pad, x, {x.size(0), x.size(1)}, at::kBool, "padding_mask");
}

}  // namespace
bool Supported(const at::Tensor& x, int64_t groups) {
  return !at::globalContext().deterministicAlgorithms() && x.is_cuda() && x.dim() == 3 &&
         x.numel() > 0 && groups > 0 && x.size(2) % groups == 0 && x.size(2) / groups <= 256 &&
         (x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
          x.scalar_type() == at::kBFloat16) &&
         at::cuda::getDeviceProperties(x.get_device())->major >= 10;
}
ForwardResult Forward(const at::Tensor& x, const c10::optional<at::Tensor>& bos,
                      const at::Tensor& pc, const at::Tensor& pm, const at::Tensor& pv,
                      const at::Tensor& gamma, const at::Tensor& beta,
                      const c10::optional<at::Tensor>& pad, int64_t groups, double b1, double b2,
                      double eps, bool cub) {
  const c10::cuda::CUDAGuard guard(x.device());
  CheckCommon(x, pc, bos, pad, gamma, beta, eps);
  auto empty = at::empty({0}, x.options());
  CheckOperand(pm, x, {x.size(0), groups}, x.scalar_type(), "prev_mean");
  CheckOperand(pv, x, {x.size(0), groups}, x.scalar_type(), "prev_var");
  auto y = at::empty_like(x, x.options(), at::MemoryFormat::Contiguous);
  auto count = at::empty_like(pc);
  auto mean = at::empty_like(pm);
  auto var = at::empty_like(pv);
  auto cm = at::empty({x.size(0), x.size(1), groups}, x.options().dtype(at::kFloat));
  auto cr = at::empty_like(cm);
  TimestepDecayNormRun(1, groups, b1, b2, eps, false, x.contiguous(),
                       bos.value_or(empty).contiguous(), pad.value_or(empty).contiguous(),
                       pc.contiguous(), pm.contiguous(), pv.contiguous(), gamma.contiguous(),
                       beta.contiguous(), empty, empty, empty, empty, empty, y, count, mean, var,
                       cm, cr, empty, empty, empty, empty, empty);
  if (cub) {
    cm = cm.transpose(1, 2).contiguous();
    cr = cr.transpose(1, 2).contiguous();
  }
  return {y, count, mean, var, cm, cr};
}
BackwardResult Backward(const at::Tensor& dy, const at::Tensor& dm, const at::Tensor& dv,
                        const at::Tensor& x, const at::Tensor& pc,
                        const c10::optional<at::Tensor>& bos, const at::Tensor& cm,
                        const at::Tensor& cr, const at::Tensor& gamma, const at::Tensor& beta,
                        const c10::optional<at::Tensor>& pad, int64_t groups, double b1, double b2,
                        double eps, bool me, bool cub) {
  const c10::cuda::CUDAGuard guard(x.device());
  CheckCommon(x, pc, bos, pad, gamma, beta, eps);
  auto empty = at::empty({0}, x.options());
  CheckOperand(dy, x, x.sizes(), x.scalar_type(), "y_grad");
  CheckOperand(dm, x, {x.size(0), groups}, x.scalar_type(), "mean_grad");
  CheckOperand(dv, x, {x.size(0), groups}, x.scalar_type(), "var_grad");
  const std::vector<int64_t> shape = cub ? std::vector<int64_t>{x.size(0), groups, x.size(1)}
                                         : std::vector<int64_t>{x.size(0), x.size(1), groups};
  CheckOperand(cm, x, shape, at::kFloat, "cummean");
  CheckOperand(cr, x, shape, at::kFloat, "cumrstd");
  auto dx = at::empty_like(x, x.options(), at::MemoryFormat::Contiguous);
  auto dpm = at::empty_like(dm);
  auto dpv = at::empty_like(dv);
  auto dg = at::empty_like(gamma);
  auto db = at::empty_like(gamma);
  auto stat_mean = cub ? cm.transpose(1, 2).contiguous() : cm.contiguous();
  auto stat_rstd = cub ? cr.transpose(1, 2).contiguous() : cr.contiguous();
  TimestepDecayNormRun(2, groups, b1, b2, eps, me, x.contiguous(), bos.value_or(empty).contiguous(),
                       pad.value_or(empty).contiguous(), pc.contiguous(), empty, empty,
                       gamma.contiguous(), beta.contiguous(), dy.contiguous(), dm.contiguous(),
                       dv.contiguous(), stat_mean, stat_rstd, empty, empty, empty, empty, empty,
                       empty, dx, dpm, dpv, dg, db);
  return {dx, dpm, dpv, dg, db};
}
}  // namespace xllm::ops::blackwell
