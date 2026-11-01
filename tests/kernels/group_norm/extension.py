"""Build production CUDA sources for the kernel regression tests."""

import torch
import native_build


def cuda_available():
    """Whether a Blackwell CUDA GPU is visible."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def build(build_directory=None):
    """Build the production group-norm sources and native bindings."""
    return native_build.build()
