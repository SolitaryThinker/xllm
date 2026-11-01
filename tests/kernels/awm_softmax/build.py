"""Build production CUDA sources for the kernel regression tests."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from tests.kernels.awm_softmax import native_build


def gpu_supported():
    """Return None when a Blackwell GPU is visible, otherwise the skip reason."""
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if torch.cuda.get_device_capability() < (10, 0):
        return "compute capability 10.0 or later is required"
    return None


def build(name="awm_softmax_ext", verbose=False):
    """Build production CUDA sources and native bindings."""
    return native_build.build()
