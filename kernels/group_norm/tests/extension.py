"""Builds the group-norm extension from ../src with torch.utils.cpp_extension.load (just-in-time)."""

import os
from pathlib import Path

import torch
from torch.utils import cpp_extension

SRC = Path(__file__).resolve().parents[1] / "src"
MODULE = "group_norm_ext"


def cuda_available():
    """True when a GPU of compute capability 10.0 or higher is visible."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def flags():
    """(C++ flags, CUDA flags): the build flags of the README plus the GPU's own feature-specific target (sm_XXXa)."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    cuda = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
            f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    return ["-O3", "-std=c++20"], cuda


def build(build_directory=None):
    """Compiles every source of ../src and returns the module, whose entry point is `run`."""
    sources = [str(SRC / "binding.cpp")] + [str(p) for p in sorted(SRC.glob("*.cu"))]
    cflags, cuda_cflags = flags()
    return cpp_extension.load(name=MODULE, sources=sources, extra_cflags=cflags, extra_cuda_cflags=cuda_cflags,
                              build_directory=build_directory or os.environ.get("GROUP_NORM_BUILD_DIR"),
                              verbose=False)
