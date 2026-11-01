"""Check architecture selection without compiling or requiring a GPU."""
from pathlib import Path
import runpy
import pytest


@pytest.mark.parametrize("targets,enabled", [("8.0", False), ("8.0;10.3a", False),
                                             ("10.3a", True), ("10.0+PTX", True)])
def test_architecture_source_selection(monkeypatch, targets, enabled):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", targets)
    setup = runpy.run_path(str(root / "setup.py"), run_name="kernel_build_probe")
    assert setup["_build_blackwell"] is enabled
    sources = {str(Path(p).relative_to(root)) for p in setup["CSRCS"]}
    optimized = {str(p.relative_to(root)) for pattern in ("*/*.cu", "*/*.cpp")
                 for p in (root / "xllm/csrc/ops").glob(pattern)}
    assert optimized
    assert (optimized <= sources) if enabled else not (optimized & sources)
