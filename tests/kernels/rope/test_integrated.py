"""Exercise the public Python rotation helpers, including autograd and strided views."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import importlib.util
from pathlib import Path
import sys
import pytest
import torch
from tests.kernels.rope import native_build


@pytest.fixture(scope="module")
def module():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires a Blackwell GPU")
    ext = native_build.build()
    previous = sys.modules.get("xllm_extension")
    sys.modules["xllm_extension"] = ext
    path = Path(__file__).resolve().parents[3] / "xllm/modules/rotary_positional_embedding.py"
    spec = importlib.util.spec_from_file_location("rotary_integration", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    yield result
    if previous is None:
        sys.modules.pop("xllm_extension", None)
    else:
        sys.modules["xllm_extension"] = previous


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_public_autograd(module, pair, conjugate, dtype):
    x = torch.randn(2, 7, 4, 128, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(2, 7, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    freq = torch.polar(torch.ones(12, 32, device="cuda"), torch.randn(12, 32, device="cuda"))[5:]
    table = freq.conj() if conjugate else freq
    def reference(t):
        z = torch.view_as_complex(t.double().reshape(*t.shape[:-1], 32, 2))
        return torch.view_as_real(z * table.to(torch.complex128).unsqueeze(1)).flatten(-2)
    view = x[..., :64]
    y = module.apply_rotary_embeddings(view, k, freq, conjugate) if pair else (module.apply_rotary_embedding(view, freq, conjugate),)
    inputs = [x, k] if pair else [x]
    ref = [reference(view), reference(k)] if pair else [reference(view)]
    dys = [torch.randn_like(t) for t in y]
    actual_grads = torch.autograd.grad(y, inputs, dys)
    ref_grads = torch.autograd.grad(ref, inputs, [d.double() for d in dys])
    tol = dict(atol=0.04, rtol=0.01) if dtype == torch.bfloat16 else dict(atol=2e-6, rtol=2e-6)
    for actual, expected in zip(y, ref):
        torch.testing.assert_close(actual.float(), expected.float(), **tol)
    for actual, expected in zip(actual_grads, ref_grads):
        torch.testing.assert_close(actual.float(), expected.float(), **tol)


def test_sum_gradient_and_table_gradient(module):
    x = torch.randn(1, 5, 2, 32, device="cuda", requires_grad=True)
    f = torch.polar(torch.ones(5, 16, device="cuda"), torch.randn(5, 16, device="cuda"))
    module.apply_rotary_embedding(x, f, False).sum().backward()
    assert x.grad.isfinite().all()
    x.grad = None
    f.requires_grad_()
    module.apply_rotary_embedding(x, f, False).sum().backward()
    assert x.grad.isfinite().all() and f.grad.isfinite().all()


def test_cpu_fallback_without_extension(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "xllm/modules/rotary_positional_embedding.py"
    spec = importlib.util.spec_from_file_location("rotary_cpu", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    monkeypatch.setitem(sys.modules, "xllm_extension", None)
    x = torch.randn(1, 3, 2, 32, requires_grad=True)
    table = torch.ones(3, 16, dtype=torch.complex64)
    y = result.apply_rotary_embedding(x, table, False)
    torch.testing.assert_close(y, x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("variant", ["mixed_dtype", "complex128", "broadcast_table", "different_batch"])
def test_original_broadcast_and_dtype_support(module, variant):
    x = torch.randn(1, 3, 2, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2 if variant == "different_batch" else 1, 3, 1, 32, device="cuda",
                    dtype=torch.float32 if variant == "mixed_dtype" else x.dtype)
    length = 1 if variant == "broadcast_table" else 3
    table = torch.polar(torch.ones(length, 16, device="cuda"), torch.randn(length, 16, device="cuda"))
    if variant == "complex128":
        table = table.to(torch.complex128)
    actual = module.apply_rotary_embeddings(x, k, table, False)
    for got, original in zip(actual, [x, k]):
        z = torch.view_as_complex(original.float().reshape(*original.shape[:-1], 16, 2))
        expected = torch.view_as_real(z * table.unsqueeze(1)).flatten(3).type_as(original)
        assert torch.equal(got, expected)


def test_negative_view_preserves_values_and_gradients(module):
    """Logical negative views must not be read as their underlying positive storage."""
    x = torch.randn(1, 7, 2, 32, device="cuda", requires_grad=True)
    negative = x._neg_view()
    table = torch.polar(torch.ones(7, 16, device="cuda"), torch.randn(7, 16, device="cuda"))
    actual = module.apply_rotary_embedding(negative, table, False)
    z = torch.view_as_complex(negative.float().reshape(1, 7, 2, 16, 2))
    expected = torch.view_as_real(z * table.unsqueeze(1)).flatten(3)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    grad = torch.randn_like(actual)
    got = torch.autograd.grad(actual, x, grad, retain_graph=True)[0]
    ref = torch.autograd.grad(expected, x, grad)[0]
    torch.testing.assert_close(got, ref, atol=0, rtol=0)
