"""Check the native op signatures and all gradients against independent fp64 autograd."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
from tests.kernels.group_norm import native_build


@pytest.fixture(scope="module")
def ext():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires a Blackwell GPU")
    return native_build.build()


@pytest.mark.parametrize("groups,width", [(1, 1536), (64, 64), (40, 24), (3, 37)])
@pytest.mark.parametrize("layer", [False, True])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("memory_efficient", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_autograd(ext, groups, width, layer, affine, memory_efficient, dtype):
    torch.manual_seed(728)
    x = torch.randn(2, 7, groups * width, device="cuda", dtype=dtype)
    gamma = (0.5 + torch.rand(groups * width, device="cuda", dtype=dtype))
    beta = torch.randn_like(gamma) * 0.1
    dy = torch.randn_like(x)
    prefix = "group_layer_norm" if layer else "group_rms_norm"
    fwd = getattr(ext.ops, prefix + ("_fwd_affine" if affine else "_fwd"))
    args = [x, x.size(-1), groups] + ([gamma, beta] if layer and affine else [gamma] if affine else []) + [1e-5]
    result = fwd(*args)
    y = result[0]
    xx = x.double().requires_grad_()
    gg, bb = gamma.double().requires_grad_(), beta.double().requires_grad_()
    z = xx.reshape(2, 7, groups, width)
    z = z - z.mean(-1, keepdim=True) if layer else z
    ref = (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5)).reshape_as(xx)
    if affine:
        ref = ref * gg + (bb if layer else 0)
    parameters = [xx] + ([gg, bb] if layer and affine else [gg] if affine else [])
    grads = torch.autograd.grad(ref, parameters, dy.double())
    bwd = getattr(ext.ops, prefix + ("_bwd_affine" if affine else "_bwd"))
    inputs = [dy, y if memory_efficient else x, x.size(-1), groups, *result[1:]]
    inputs += [gamma, beta] if layer and affine else [gamma] if affine else []
    actual = bwd(*inputs, memory_efficient)
    actual = actual if isinstance(actual, tuple) else (actual,)
    tol = dict(atol=0.07, rtol=0.04) if dtype == torch.bfloat16 else dict(atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(y.float(), ref.detach().float(), **tol)
    for got, expected in zip(actual, grads):
        torch.testing.assert_close(got.float(), expected.float(), **tol)


@pytest.mark.parametrize("groups,width", [(4, 32), (64, 64), (3, 37)])
def test_layer_norm_large_offset(ext, groups, width):
    """Centered variance must stay accurate when mean squared is much larger than variance."""
    x = 1000 + 0.01 * torch.randn(2, 11, groups * width, device="cuda")
    y, mean, rstd = ext.ops.group_layer_norm_fwd(x, x.size(-1), groups, 1e-5)
    z = x.double().reshape(2, 11, groups, width)
    mu = z.mean(-1, keepdim=True)
    inv = torch.rsqrt((z - mu).square().mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(mean.double().reshape_as(mu), mu, atol=2e-4, rtol=0)
    torch.testing.assert_close(rstd.double().reshape_as(inv), inv, atol=0.02, rtol=0.001)
    torch.testing.assert_close(y.double(), ((z - mu) * inv).reshape_as(x), atol=0.03, rtol=0.02)


def test_deterministic_algorithms(ext):
    x = torch.randn(2, 17, 64 * 64, device="cuda", dtype=torch.bfloat16)
    gamma, dy = torch.ones(x.size(-1), device="cuda", dtype=x.dtype), torch.randn_like(x)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        y, rs = ext.ops.group_rms_norm_fwd_affine(x, x.size(-1), 64, gamma, 1e-5)
        a = ext.ops.group_rms_norm_bwd_affine(dy, x, x.size(-1), 64, rs, gamma, False)
        b = ext.ops.group_rms_norm_bwd_affine(dy, x, x.size(-1), 64, rs, gamma, False)
        for first, second in zip(a, b):
            assert torch.equal(first, second)
    finally:
        torch.use_deterministic_algorithms(previous)
