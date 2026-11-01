"""Build production CUDA sources for the kernel regression tests."""

import torch
import native_build


def supported():
    """Return None when a Blackwell GPU is visible, otherwise the skip reason."""
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if torch.cuda.get_device_capability() < (10, 0):
        return "compute capability 10.0 or later is required"
    return None


def build(verbose=False):
    """Build production CUDA sources and native bindings."""
    return native_build.build()
