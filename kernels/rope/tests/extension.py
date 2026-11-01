"""Builds the extension from ../src with torch.utils.cpp_extension.load (JIT)."""

import os

import torch
from torch.utils.cpp_extension import load

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


def supported():
    """None when the GPU can run the kernel, else the reason it cannot."""
    if not torch.cuda.is_available():
        return "CUDA is not available"
    cap = torch.cuda.get_device_capability()
    if cap < (10, 0):
        return f"compute capability {cap[0]}.{cap[1]} is below 10.0"
    return None


def build(verbose=False):
    """Compiles binding.cpp and rope.cu for the current GPU and returns the module, whose `run` is the entry point."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    return load(
        name="rope_ext",
        sources=[os.path.join(SRC, "binding.cpp"), os.path.join(SRC, "rope.cu")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
                           f"-gencode=arch=compute_{arch},code=sm_{arch}"],
        verbose=verbose,
    )
