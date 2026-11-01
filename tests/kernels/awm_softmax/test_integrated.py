"""Test native working-memory primitives with actual float32 incoming gradients."""
import pytest
import torch
import native_build


@pytest.fixture(scope="module")
def ext():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires a Blackwell GPU")
    return native_build.build()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunk", [64, 256, 1024])
@pytest.mark.parametrize("masked", [False, True])
def test_float32_gradients(ext, dtype, chunk, masked):
    b, n, k, s = 2, 3, 2, 32
    key = torch.randn(b, n * chunk, k, s, device="cuda", dtype=dtype)
    mask = torch.rand(b, n, chunk, device="cuda") < 0.6
    mask[:, :, -1] = False
    mask = mask.reshape(b, n * chunk) if masked else torch.empty(0, device="cuda", dtype=torch.bool)
    empty = key.new_empty(0)
    probs = key.new_empty(b, k, s, n * chunk)
    lse = torch.full((b, k, s, n), -torch.inf, device="cuda", dtype=torch.float32)
    ext.ops.chunk_softmax(0, chunk, key, mask, empty, empty, empty, empty, probs, lse, empty, empty)
    xx = key.double().requires_grad_()
    z = xx.permute(0, 2, 3, 1).reshape(b, k, s, n, chunk)
    if masked:
        z = z.masked_fill(mask.reshape(b, 1, 1, n, chunk), -torch.inf)
    p = z.softmax(-1)
    log = z[..., :-1, :].logsumexp(-1)
    g = torch.randn(b, k, s, (n - 1) * chunk, device="cuda", dtype=torch.float32)
    dlse = torch.randn_like(lse)
    expected = torch.autograd.grad((p[..., :-1, :], log), xx, (g.double().reshape(b, k, s, n-1, chunk), dlse[..., 1:].double()))[0]
    actual = torch.zeros_like(key)
    ext.ops.chunk_softmax(1, chunk, key, mask, empty, lse, g, dlse, empty, empty, empty, actual)
    tol = dict(atol=5e-4, rtol=0.02) if dtype == torch.bfloat16 else dict(atol=1e-5, rtol=0.003)
    torch.testing.assert_close(actual.float(), expected.float(), **tol)
    torch.testing.assert_close(probs.float(), p.reshape_as(probs).float(), **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_previous", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
def test_working_memory_forward_and_autograd(ext, dtype, with_previous, segmented):
    """Run the production working-memory module with its native route enabled and disabled."""
    import ast
    import importlib.util
    import math
    from pathlib import Path
    import sys
    from typing import Optional, Tuple
    import torch.nn.functional as F
    from einops import rearrange, repeat
    root = Path(__file__).resolve().parents[3]
    sys.modules["xllm_extension"] = ext
    spec = importlib.util.spec_from_file_location("chunk_softmax_integration", root / "xllm/modules/fused_ops/chunk_softmax.py")
    chunk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chunk)
    utils_spec = importlib.util.spec_from_file_location("awm_utils_integration", root / "xllm/modules/fused_ops/utils.py")
    utils = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils)
    source = root / "xllm/modules/fused_ops/adaptive_working_memory.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = dict(torch=torch, Tensor=torch.Tensor, FunctionCtx=torch.autograd.function.FunctionCtx,
                     Optional=Optional, Tuple=Tuple, math=math, F=F, rearrange=rearrange, repeat=repeat,
                     chunk_softmax=chunk, logsumexp_backward=utils.logsumexp_backward,
                     get_active_health_collector=lambda: None)
    exec(compile(tree, str(source), "exec"), namespace)
    torch.manual_seed(177)
    b, h, k, c, s, v, n = 1, 2, 1, 256, 32, 8, 3
    args = [torch.randn(b, heads, n*c, dim, device="cuda", dtype=dtype) * 0.1
            for heads, dim in [(h, s), (k, s), (k, s), (k, v)]]
    previous = [torch.randn(b, k, s, v, device="cuda", dtype=dtype) * 0.01,
                torch.randn(b, k, s, device="cuda", dtype=torch.float32)]
    previous += [torch.randn(b, k, c, dim, device="cuda", dtype=dtype) * 0.1 for dim in [s, s, v]]
    if not with_previous:
        previous = [None] * 5
    segments = torch.arange((n + int(with_previous))*c, device="cuda").reshape(b, -1) // 73 if segmented else None
    prev_count = torch.zeros(b, dtype=torch.int64, device="cuda") if segmented and with_previous else None
    saved = chunk.available
    runs = []
    dys = None
    for enabled in [True, False]:
        chunk.available = saved if enabled else lambda key: False
        tensors = [t.detach().clone().requires_grad_() for t in args]
        state = [None if t is None else t.detach().clone().requires_grad_() for t in previous]
        result = namespace["adaptive_working_memory"](*tensors, c, *state, segments, prev_count, False, 1e-6, None)
        outputs = [result[i] for i in [0, 2, 3]]
        if dys is None:
            dys = [torch.randn_like(t) for t in outputs]
        gradients = torch.autograd.grad(outputs, tensors + [t for t in state if t is not None], dys)
        runs.append([t.detach() for t in outputs] + list(gradients))
    chunk.available = saved
    tol = dict(atol=0.03, rtol=0.06) if dtype == torch.bfloat16 else dict(atol=2e-5, rtol=2e-4)
    for actual, expected in zip(*runs):
        torch.testing.assert_close(actual, expected, **tol)
