"""Builds the extension from ../src with torch.utils.cpp_extension.load (JIT) and returns the module."""

import glob
import os

import torch
from torch.utils.cpp_extension import load

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

CFLAGS = ["-O3", "-std=c++20"]
CUDA_CFLAGS = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo"]


def supported():
    """Whether this machine has a GPU the extension builds for: compute capability 10.0 or newer."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def load_extension(verbose=False):
    """Compiles (or reuses the cached build of) the sources for the current GPU and imports them. The build goes to
    $TORCH_EXTENSIONS_DIR when it is set."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a" if major >= 9 else f"{major}{minor}"
    sources = sorted(glob.glob(os.path.join(SRC, "*.cpp")) + glob.glob(os.path.join(SRC, "*.cu")))
    return load(
        name="timestep_decay_norm_ext",
        sources=sources,
        extra_include_paths=[SRC],
        extra_cflags=CFLAGS,
        extra_cuda_cflags=[f"-gencode=arch=compute_{arch},code=sm_{arch}"] + CUDA_CFLAGS,
        verbose=verbose,
    )
