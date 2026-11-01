"""Build the production CUDA sources and public op bindings for integration tests."""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "xllm/csrc/ops/awm_softmax"

def build():
    import os
    if os.environ.get("XLLM_TEST_INSTALLED") == "1":
        import importlib
        return importlib.import_module("xllm_extension")
    sources = [str(Path(__file__).with_name("native_binding.cpp"))]
    sources += [str(p) for p in sorted(SRC.glob("*.cu"))]
    sources += [str(p) for p in sorted(SRC.glob("*.cpp")) if p.name not in ("binding.cpp", "awm_softmax_binding.cpp")]
    sources += [str(ROOT / "xllm/csrc/ops" / p) for p in []]
    major, minor = torch.cuda.get_device_capability()
    return load("awm_softmax_native_ext", sources, extra_include_paths=[str(ROOT), str(ROOT / "xllm/csrc")],
                extra_cflags=["-O3", "-std=c++20", "-DXLLM_BLACKWELL"],
                extra_cuda_cflags=["-O3", "-std=c++20", "-DXLLM_BLACKWELL", "--expt-relaxed-constexpr",
                                  "--expt-extended-lambda", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"])
