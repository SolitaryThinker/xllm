"""Checks the group RMS norm and group layer norm kernels against an fp64 reference.

Every case of cases.shape_cases() (the shapes of xllm's example models) and of cases.sweep() (shapes beyond them: odd
group counts and widths, tiny and ragged row counts, fp32 / fp16 / bf16, with and without the affine transform and
memory_efficient) is run forward or backward, and each of the six outputs is compared with reference.run under the
case's tolerances. A call must also leave the outputs it does not produce as they were given.

    python -m pytest tests/test_group_norm.py       (or: python tests/test_group_norm.py [name-substring ...])

The tests skip when no GPU of compute capability 10.0 or higher is available.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import sys
from pathlib import Path

import pytest
import torch

from tests.kernels.group_norm import cases
from tests.kernels.group_norm import extension
from tests.kernels.group_norm import reference

pytestmark = pytest.mark.skipif(not extension.cuda_available(), reason="needs a CUDA GPU of compute capability 10.0+")

ALL = cases.all_cases()


@pytest.fixture(scope="session")
def module():
    """The extension, built once per session."""
    return extension.build()


def check(module, case, index):
    """Runs one case and returns the failing rows of the comparison (an empty list when it passes)."""
    inputs = cases.make_inputs(case, 1000 + index, "cuda")
    outputs = cases.make_outputs(inputs)
    module.run(*inputs, *outputs)
    torch.cuda.synchronize()
    expected = reference.run(*inputs)
    return [row for row in cases.compare(outputs, expected, case) if not row[1]]


def describe(rows):
    return "; ".join(f"{name}: matched {ratio:.6f}, max error {err:.3g} {note}".strip() for name, _, ratio, err, note in rows)


@pytest.mark.parametrize("index", range(len(ALL)), ids=[c["name"] for c in ALL])
def test_case(module, index):
    """The six outputs of one call agree with the fp64 reference."""
    failing = check(module, ALL[index], index)
    assert not failing, describe(failing)


if __name__ == "__main__":
    if not extension.cuda_available():
        print("skipped: needs a CUDA GPU of compute capability 10.0+")
        sys.exit(0)
    ext = extension.build()
    wanted = sys.argv[1:]
    failed = total = 0
    for i, c in enumerate(ALL):
        if wanted and not any(w in c["name"] for w in wanted):
            continue
        total += 1
        try:
            failing = check(ext, c, i)
        except Exception as e:  # noqa: BLE001 - a case the kernels refuse is reported as a failure
            failing = [("call", False, 0.0, float("inf"), repr(e))]
        failed += bool(failing)
        print(f"{'FAIL' if failing else 'ok  '} {c['name']} {describe(failing)}", flush=True)
    print(f"{total - failed} of {total} cases passed")
    sys.exit(1 if failed else 0)
