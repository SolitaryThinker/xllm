// Bind the production implementations for the native integration tests.
#include "xllm/csrc/ops/awm_softmax/awm_softmax_binding.cpp"
namespace xllm::ops { void DefineChunkSoftmaxOp(py::module&);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &ChunkSoftmaxCUDA);
  auto ops=m.def_submodule("ops");
  xllm::ops::DefineChunkSoftmaxOp(ops);
}
