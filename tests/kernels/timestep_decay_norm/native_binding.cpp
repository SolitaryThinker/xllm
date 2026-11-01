// Bind the production implementations for the native integration tests.
#include "xllm/csrc/ops/timestep_decay_norm/binding.cpp"
namespace xllm::ops { void DefineTimestepNormOp(py::module&);void DefineTimestepDecayNormCubOp(py::module&);
}
#include "xllm/csrc/ops/timestep_norm.h"
#include "xllm/csrc/ops/timestep_decay_norm_cub.h"
void OriginalTdnRun(int64_t mode, int64_t num_groups, double beta1, double beta2, double eps, bool memory_efficient,
         const at::Tensor& x_or_y, const at::Tensor& bos_mask, const at::Tensor& padding_mask,
         const at::Tensor& prev_count, const at::Tensor& prev_mean, const at::Tensor& prev_var,
         const at::Tensor& gamma, const at::Tensor& beta, const at::Tensor& dy, const at::Tensor& dmean,
         const at::Tensor& dvar, const at::Tensor& cummean, const at::Tensor& cumrstd, at::Tensor y, at::Tensor count,
         at::Tensor mean, at::Tensor var, at::Tensor cummean_out, at::Tensor cumrstd_out, at::Tensor dx,
         at::Tensor dprev_mean, at::Tensor dprev_var, at::Tensor dgamma, at::Tensor dbeta) {
 using namespace xllm::ops;
 if (mode < 2) {
   auto out = GroupTimestepDecayNormCubCUDAFwd(x_or_y, optional(bos_mask), prev_count, prev_mean, prev_var, gamma, beta,
                                              optional(padding_mask), num_groups, beta1, beta2, eps);
   y.copy_(std::get<0>(out)); count.copy_(std::get<1>(out)); mean.copy_(std::get<2>(out)); var.copy_(std::get<3>(out));
   if (mode == 1) { cummean_out.copy_(std::get<4>(out).transpose(1,2)); cumrstd_out.copy_(std::get<5>(out).transpose(1,2));
}
 } else {
   auto out = memory_efficient ? GroupTimestepDecayNormCUDABwd(dy, dmean, dvar, x_or_y, prev_count, optional(bos_mask), cummean, cumrstd, gamma, beta, optional(padding_mask), num_groups, beta1, beta2, eps, true)
       : GroupTimestepDecayNormCubCUDABwd(dy, dmean, dvar, x_or_y, prev_count, optional(bos_mask), cummean.transpose(1,2).contiguous(), cumrstd.transpose(1,2).contiguous(), gamma, optional(padding_mask), num_groups, beta1, beta2);
   dx.copy_(std::get<0>(out)); dprev_mean.copy_(std::get<1>(out)); dprev_var.copy_(std::get<2>(out));
   dgamma.copy_(std::get<3>(out)); dbeta.copy_(std::get<4>(out));
 }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &TimestepDecayNormRun);
  m.def("baseline_run", &OriginalTdnRun);
  m.def("baseline_CubCUDAFwd", &xllm::ops::GroupTimestepDecayNormCubCUDAFwd);
  m.def("baseline_CubCUDABwd", &xllm::ops::GroupTimestepDecayNormCubCUDABwd);
  m.def("baseline_CUDAFwd", &xllm::ops::GroupTimestepDecayNormCUDAFwd);
  m.def("baseline_CUDABwd", &xllm::ops::GroupTimestepDecayNormCUDABwd);
  auto ops=m.def_submodule("ops");
  xllm::ops::DefineTimestepNormOp(ops);xllm::ops::DefineTimestepDecayNormCubOp(ops);
}
