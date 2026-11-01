// The Python entry point: one call of one of the eight group-norm operations (forward or backward, RMS norm or layer
// norm, with or without the affine transform), chosen by `mode`, `norm` and the tensors given.
//
// The arguments are, in order: the inputs (mode, norm, num_groups, eps, memory_efficient, x_or_y, dy, mean, rstd,
// gamma, beta), then the outputs, which are written in place: y, mean_out, rstd_out, dx, dgamma, dbeta. An output the
// operation does not produce is a one-element placeholder and stays as it was given. An input it does not take is an
// empty tensor.
//
// This file only describes the call: gn::forward / gn::backward (dispatch.cu) route it to the implementation that
// handles its shape, and to xllm's own kernels when none does. An implementation is added by registering it with the
// dispatcher (see the node_*.cu files), not by adding a branch here.
#include <torch/extension.h>

#include "dispatch.h"
#include <c10/cuda/CUDAGuard.h>

void GroupNormRun(int64_t mode, int64_t norm, int64_t num_groups, double eps, bool memory_efficient,
         const at::Tensor& x_or_y, const at::Tensor& dy, const at::Tensor& mean, const at::Tensor& rstd,
         const at::Tensor& gamma, const at::Tensor& beta,
         at::Tensor y, at::Tensor mean_out, at::Tensor rstd_out, at::Tensor dx, at::Tensor dgamma, at::Tensor dbeta) {
  TORCH_CHECK(x_or_y.is_cuda() && x_or_y.dim() == 3, "x_or_y: [B, S, C] on the GPU");
  TORCH_CHECK(mode == gn::kFwd || mode == gn::kBwd, "mode: 0 (forward) or 1 (backward)");
  const c10::cuda::CUDAGuard guard(x_or_y.device());
  const int64_t channels = x_or_y.size(-1);

  TORCH_CHECK(norm == gn::kRMS || norm == gn::kLayer, "norm: 0 (RMS) or 1 (layer)");
  TORCH_CHECK(channels > 0 && num_groups > 0 && channels % num_groups == 0, "channels must be divisible by positive num_groups");
  auto check = [&](const at::Tensor& t, int64_t size, at::ScalarType dtype) {
    TORCH_CHECK(t.device() == x_or_y.device() && t.scalar_type() == dtype && t.numel() == size,
                "group-norm operand has the wrong device, dtype or size");
  };
  if (gamma.numel()) check(gamma, channels, x_or_y.scalar_type());
  if (beta.numel()) check(beta, channels, x_or_y.scalar_type());
  if (mode == gn::kBwd) {
    check(dy, x_or_y.numel(), x_or_y.scalar_type());
    check(rstd, x_or_y.numel() / channels * num_groups, at::kFloat);
    if (norm == gn::kLayer) check(mean, rstd.numel(), at::kFloat);
  }
  gn::Shape s;
  s.mode = mode;
  s.norm = norm;
  s.groups = num_groups;
  s.channels = channels;
  s.cpg = num_groups > 0 ? channels / num_groups : channels;
  s.rows = x_or_y.numel() / channels;
  s.dtype = x_or_y.scalar_type();
  s.affine = gamma.numel() > 0;
  s.has_beta = beta.numel() > 0;
  s.memory_efficient = memory_efficient;

  gn::Call c;
  c.x_or_y = &x_or_y;
  c.dy = &dy;
  c.mean = &mean;
  c.rstd = &rstd;
  c.gamma = &gamma;
  c.beta = &beta;
  c.y = &y;
  c.mean_out = &mean_out;
  c.rstd_out = &rstd_out;
  c.dx = &dx;
  c.dgamma = &dgamma;
  c.dbeta = &dbeta;
  c.eps = eps;

  if (mode == gn::kFwd) {
    gn::forward(s, c);
  } else {
    gn::backward(s, c);
  }
}
