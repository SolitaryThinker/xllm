// Bind the production implementations for the native integration tests.
#include "xllm/csrc/ops/rope/binding.cpp"
namespace xllm::ops { void DefineRotaryEmbeddingOp(py::module&);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &RotaryEmbeddingCUDA);
  auto ops=m.def_submodule("ops");
  xllm::ops::DefineRotaryEmbeddingOp(ops);
}
