"""Builds the extension from ../src with torch.utils.cpp_extension.load (JIT) for the GPU in use.

The flags are the ones the README lists, plus the GPU's own target with its arch-specific features
(sm_100a on compute capability 10.0, sm_103a on 10.3).
"""

import os

import torch
from torch.utils.cpp_extension import load

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
SOURCES = ["awm_softmax_binding.cpp", "awm_chunk_softmax.cu", "awm_two_pass_bwd.cu"]


def gpu_supported():
    """None when a CUDA GPU of compute capability 10.0 or later is present, else the reason it is not."""
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if torch.cuda.get_device_capability() < (10, 0):
        return "compute capability 10.0 or later is required (clusters, PDL, sm_100a)"
    return None


def build(name="awm_softmax_ext", verbose=False):
    """The built extension module; its `run` is the entry point."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    cuda_flags = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
                  f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    return load(name=name, sources=[os.path.join(SRC, s) for s in SOURCES], extra_include_paths=[SRC],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=cuda_flags, verbose=verbose)
