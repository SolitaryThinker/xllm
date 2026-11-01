"""Build production CUDA sources for the kernel regression tests."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from tests.kernels.group_norm import native_build


def cuda_available():
    """Whether a Blackwell CUDA GPU is visible."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def build(build_directory=None):
    """Build the production group-norm sources and native bindings."""
    return native_build.build()
