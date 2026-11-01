"""Correctness of the rotary position embedding kernel against the fp64 reference.

Builds src/ for the current GPU and runs every case of cases.CASES (the shapes xllm's models produce) and of
cases.sweep_cases() (a fixed sweep over batches, lengths, heads, widths, layouts, table offsets and dtypes), forward
and backward. An output passes when at least `match` of its elements satisfy |out - ref| <= atol + rtol * |ref|, it
holds no NaN or Inf, and it is not all zeros where the reference is not.

    python -m pytest tests/test_rope.py
    python tests/test_rope.py            # without pytest
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extension  # noqa: E402
import cases  # noqa: E402
import reference  # noqa: E402

try:
    import pytest
except ImportError:  # the file also runs as a plain script
    pytest = None

SWEEP = cases.sweep_cases()


def compare(out, ref, tol):
    """(passed, matched fraction, max abs error, note) of one output against its reference."""
    if out.shape != ref.shape or out.dtype != ref.dtype:
        return False, 0.0, math.inf, f"{tuple(out.shape)} {out.dtype}, expected {tuple(ref.shape)} {ref.dtype}"
    x, y = out.float(), ref.float()
    if not torch.isfinite(x).all():
        return False, 0.0, math.inf, "NaN or Inf"
    if y.abs().sum() > 0 and x.abs().sum() == 0:
        return False, 0.0, math.inf, "all zeros"
    err = (x - y).abs()
    # A count, not a float32 mean: over 1e8 elements a mean of all ones can round below 1.
    ratio = (err <= tol["atol"] + tol["rtol"] * y.abs()).sum().item() / err.numel()
    return ratio >= tol["match"], ratio, err.max().item(), ""


def check(ext, case, key):
    """Runs one case through the kernel and the reference; returns the failure messages (empty when it passes)."""
    inputs = cases.make_inputs(case, key, "cuda")
    outs = cases.make_outputs(inputs)
    ext.run(*inputs, *outs)
    torch.cuda.synchronize()
    refs = reference.reference(*inputs)
    failures = []
    for name, out, ref in zip(("out1", "out2"), outs, refs):
        ok, ratio, err, note = compare(out, ref, case)
        if not ok:
            failures.append(f"{name}: match {ratio:.6f}, max err {err:.3g} {note}".strip())
    return failures


if pytest is not None:

    @pytest.fixture(scope="module")
    def ext():
        reason = extension.supported()
        if reason:
            pytest.skip(reason)
        return extension.build()

    @pytest.mark.parametrize("name", list(cases.CASES))
    def test_shape(ext, name):
        assert not check(ext, cases.CASES[name], key=1234)

    @pytest.mark.parametrize("name", list(SWEEP))
    def test_sweep(ext, name):
        index = list(SWEEP).index(name)
        assert not check(ext, SWEEP[name], key=cases.SWEEP_KEY + index)


def main():
    reason = extension.supported()
    if reason:
        print(f"skipped: {reason}")
        return 0
    ext = extension.build()
    bad = 0
    for name, case in cases.CASES.items():
        failures = check(ext, case, key=1234)
        bad += bool(failures)
        print(("FAIL " if failures else "ok   ") + name, *failures)
    for i, (name, case) in enumerate(SWEEP.items()):
        failures = check(ext, case, key=cases.SWEEP_KEY + i)
        bad += bool(failures)
        print(("FAIL " if failures else "ok   ") + name, *failures)
    total = len(cases.CASES) + len(SWEEP)
    print(f"{total - bad} of {total} cases correct")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
