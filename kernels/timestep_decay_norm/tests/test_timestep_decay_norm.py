"""Checks the timestep decay normalization extension against the fp64 reference.

The extension is built from ../src (JIT, for the GPU in the machine). Every case of cases.shapes() (the model calls:
training forward, backward and memory-efficient backward, prefill, prompt tail and decode) and of cases.sweep()
(edge shapes and random inputs: lengths 1 to 4097, 1 to 256 groups of 32 to 256 channels, bos and padding masks,
fresh and carried state, nonzero gradients of the final state, fp32, fp16 and bf16) is run through `run` in its mode
and every output is compared with reference.run under the case's tolerances.

    python -m pytest tests/test_timestep_decay_norm.py          # from this directory's parent
    python tests/test_timestep_decay_norm.py [case-name ...]     # without pytest

The tests are skipped where there is no CUDA GPU of compute capability 10.0 or newer.
"""

import sys

import torch

import build
import cases
import reference

try:
    import pytest
except ImportError:  # the file also runs as a plain script, through main() below

    class _Mark:
        @staticmethod
        def parametrize(_argname, _values):
            return lambda fn: fn

        @staticmethod
        def skipif(_condition, reason=""):
            return lambda fn: fn

    class pytest:  # noqa: N801 - stands in for the module
        mark = _Mark()

        @staticmethod
        def fixture(**_kwargs):
            return lambda fn: fn


SHAPES = cases.shapes()
SWEEP = cases.sweep()

pytestmark = pytest.mark.skipif(not build.supported(), reason="needs a CUDA GPU of compute capability 10.0 or newer")


@pytest.fixture(scope="module")
def ext():  # noqa: D103
    return build.load_extension()


def run_case(ext, case):
    """Runs one case; returns the comparison rows of its outputs (see cases.compare)."""
    inputs = cases.make_inputs(case)
    outputs = cases.make_outputs(inputs)
    ext.run(*inputs, *outputs)
    torch.cuda.synchronize()
    expected = reference.run(*inputs)
    return cases.compare(outputs, expected, case)


def failures(rows):
    return "; ".join(f"{name}: match {ratio:.6f}, max err {err:.3g} {note}".strip()
                     for name, ok, ratio, err, note in rows if not ok)


@pytest.mark.parametrize("name", list(SHAPES))
def test_shape(ext, name):
    bad = failures(run_case(ext, SHAPES[name]))
    assert not bad, bad


@pytest.mark.parametrize("name", list(SWEEP))
def test_sweep(ext, name):
    bad = failures(run_case(ext, SWEEP[name]))
    assert not bad, bad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_reference_forms_agree():
    """The closed-form reference and the step-by-step recurrence compute the same forward pass."""
    case = cases.make_case("forms", cases.INFER, 2, 300, 4, 64, 1 / 16, 0.05, True, False, False, "float32", 7)
    inputs = cases.make_inputs(case)
    (_, K, b1, b2, eps, _, x, bos, pad, count, pm, pv, gamma, beta, *_) = inputs
    args = (x.double(), bos, count, pm.double(), pv.double(), gamma.double(), beta.double(), pad, K, b1, b2, eps)
    with torch.no_grad():
        closed = reference.forward(*args)
        loop = reference.forward_loop(*args)
    for a, b in zip(closed, loop):
        torch.testing.assert_close(a.double(), b.double(), rtol=1e-6, atol=1e-6)


def main(argv):
    if not build.supported():
        print("skipped: needs a CUDA GPU of compute capability 10.0 or newer")
        return 0
    ext_module = build.load_extension()
    table = {**SHAPES, **SWEEP}
    names = argv or list(table)
    failed = 0
    for name in names:
        bad = failures(run_case(ext_module, table[name]))
        print(("FAIL " if bad else "ok   ") + name + (f": {bad}" if bad else ""), flush=True)
        failed += bool(bad)
    print(f"{len(names) - failed} of {len(names)} cases correct")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
