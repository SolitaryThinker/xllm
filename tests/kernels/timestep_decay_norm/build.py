"""Build production CUDA sources for the kernel regression tests."""

import torch
import native_build


def supported():
    """Whether a Blackwell CUDA GPU is visible."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def load_extension(verbose=False):
    """Build the production timestep-decay-norm sources and native bindings."""
    return native_build.build()
