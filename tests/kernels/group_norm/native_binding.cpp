// Bind the production implementations for the native integration tests.
#include "xllm/csrc/ops/group_norm/binding.cpp"
namespace xllm::ops { void DefineGroupRMSNormOp(py::module&);void DefineGroupLayerNormOp(py::module&);
}
#include "xllm/csrc/ops/group_rms_norm.h"
#include "xllm/csrc/ops/group_layer_norm.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &GroupNormRun);
  m.def("baseline_group_rms_norm_fwd", &xllm::ops::GroupRMSNormCUDAFwd);
  m.def("baseline_group_rms_norm_fwd_affine", &xllm::ops::GroupRMSNormCUDAFwdAffine);
  m.def("baseline_group_rms_norm_bwd", &xllm::ops::GroupRMSNormCUDABwd);
  m.def("baseline_group_rms_norm_bwd_affine", &xllm::ops::GroupRMSNormCUDABwdAffine);
  m.def("baseline_group_layer_norm_fwd", &xllm::ops::GroupLayerNormCUDAFwd);
  m.def("baseline_group_layer_norm_fwd_affine", &xllm::ops::GroupLayerNormCUDAFwdAffine);
  m.def("baseline_group_layer_norm_bwd", &xllm::ops::GroupLayerNormCUDABwd);
  m.def("baseline_group_layer_norm_bwd_affine", &xllm::ops::GroupLayerNormCUDABwdAffine);
  auto ops=m.def_submodule("ops");
  xllm::ops::DefineGroupRMSNormOp(ops);xllm::ops::DefineGroupLayerNormOp(ops);
}
