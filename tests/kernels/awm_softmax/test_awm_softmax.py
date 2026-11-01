"""Checks the chunked softmax and logsumexp kernels against the fp64 reference of reference.py.

Every case of cases.py (the main shapes of xllm's Gekko models and the sweep of other shapes, dtypes and masks) is run
in its mode (forward, backward, decode) on inputs drawn from a fixed generator, and every output is compared with the
reference at the case's tolerances: no NaN or Inf, not all zeros where the reference is not, and at least `match` of
the elements the reference defines within atol + rtol * |ref|.

The extension is built from xllm/csrc on first use. The tests skip without a CUDA GPU of compute capability 10.0 or later.

    python -m pytest tests/test_awm_softmax.py
    python tests/test_awm_softmax.py            # without pytest; prints one line per case
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import math
import os
import sys

import torch

from tests.kernels.awm_softmax import build
from tests.kernels.awm_softmax import cases
from tests.kernels.awm_softmax import reference

_EXT = None


def extension():
    """The extension, built once per process."""
    global _EXT
    if _EXT is None:
        _EXT = build.build()
    return _EXT


def compare(outs, refs, case):
    """One row per output: (name, passed, matched ratio, max abs error, note)."""
    rows = []
    for name, out, ref in zip(cases.OUTPUTS, outs, refs):
        if out.shape != ref.shape or out.dtype != ref.dtype:
            rows.append((name, False, 0.0, math.inf, f"{tuple(out.shape)} {out.dtype}, expected "
                                                       f"{tuple(ref.shape)} {ref.dtype}"))
            continue
        x, y = out.to(ref.device, torch.float32), ref.float()
        if not torch.isfinite(x).all():
            rows.append((name, False, 0.0, math.inf, "NaN or Inf"))
            continue
        defined = ~torch.isnan(y)
        if not defined.any():
            rows.append((name, False, 0.0, math.inf, "the reference defines no element"))
            continue
        if not defined.all():
            x, y = x[defined], y[defined]
        if y.abs().sum() > 0 and x.abs().sum() == 0:
            rows.append((name, False, 0.0, math.inf, "all zeros"))
            continue
        err = (x - y).abs()
        own = cases.tolerance(case, name)
        # A count, not a float32 mean: over 1e8 elements a mean of all ones can round below 1.
        ratio = (err <= own["atol"] + own["rtol"] * y.abs()).sum().item() / err.numel()
        rows.append((name, ratio >= own["match"], ratio, err.max().item(), ""))
    return rows


def check(name):
    """Runs case `name` and returns the comparison rows of its four outputs."""
    inputs = cases.make_inputs(name)
    outs = cases.make_outputs(inputs)
    extension().run(*inputs, *outs)
    torch.cuda.synchronize()
    refs = reference.run(*inputs)
    return compare(outs, refs, cases.CASES[name])


def _failures(name):
    return [f"{n}: match {m:.6f}, max err {e:.3g} {note}".strip() for n, ok, m, e, note in check(name) if not ok]


try:
    import pytest
except ImportError:  # run as a script
    pytest = None

if pytest is not None:

    @pytest.fixture(scope="module", autouse=True)
    def _gpu():
        reason = build.gpu_supported()
        if reason:
            pytest.skip(reason)

    @pytest.mark.parametrize("name", cases.SHAPE_NAMES)
    def test_shape(name):
        assert not _failures(name)

    @pytest.mark.parametrize("name", cases.SWEEP_NAMES)
    def test_sweep(name):
        assert not _failures(name)


def main():
    reason = build.gpu_supported()
    if reason:
        print(f"skipped: {reason}")
        return 0
    failed = 0
    for name in cases.SHAPE_NAMES + cases.SWEEP_NAMES:
        bad = _failures(name)
        failed += bool(bad)
        print(f"{'FAIL' if bad else 'ok  '} {name}" + "".join(f"\n     {b}" for b in bad), flush=True)
    total = len(cases.SHAPE_NAMES) + len(cases.SWEEP_NAMES)
    print(f"{total - failed} of {total} cases correct")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
