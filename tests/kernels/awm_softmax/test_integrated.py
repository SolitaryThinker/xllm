"""Test native working-memory primitives with actual float32 incoming gradients."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
from tests.kernels.awm_softmax import native_build


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
def test_working_memory_forward_and_autograd(ext, dtype, with_previous, segmented, monkeypatch):
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
    monkeypatch.setitem(sys.modules, "xllm_extension", ext)
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


@pytest.mark.parametrize("invalid", ["mode", "chunk", "mask_dtype", "mask_shape", "output_shape", "lse_dtype", "prev_dtype"])
def test_invalid_primitive_metadata(ext, invalid):
    """The public destination-passing op rejects incompatible operands before launch."""
    key = torch.randn(1, 512, 2, 32, device="cuda", dtype=torch.bfloat16)
    empty = key.new_empty(0)
    args = [0, 256, key, empty, empty, empty, empty, empty,
            key.new_empty(1, 2, 32, 512), torch.empty(1, 2, 32, 2, device="cuda"), empty, empty]
    if invalid == "mode": args[0] = 9
    elif invalid == "chunk": args[1] = 0
    elif invalid == "mask_dtype": args[3] = torch.zeros(1, 512, device="cuda", dtype=torch.float32)
    elif invalid == "mask_shape": args[3] = torch.zeros(512, device="cuda", dtype=torch.bool)
    elif invalid == "output_shape": args[8] = args[8].flatten()
    elif invalid == "lse_dtype": args[9] = args[9].double()
    else: args[4] = torch.zeros(1, 256, 2, 32, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        ext.ops.chunk_softmax(*args)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_storage_offset_alignment(ext, dtype):
    """Contiguous views can have pointers unsuitable for vector loads and stores."""
    shape = (1, 512, 2, 32)
    key = torch.randn(1 + 512*2*32, device="cuda", dtype=dtype)[1:].view(shape)
    empty = key.new_empty(0)
    probs = key.new_empty(1, 2, 32, 512)
    lse = torch.full((1, 2, 32, 2), -torch.inf, device="cuda")
    ext.ops.chunk_softmax(0, 256, key, empty, empty, empty, empty, empty, probs, lse, empty, empty)
    z = key.double().permute(0, 2, 3, 1).reshape(1, 2, 32, 2, 256)
    reference = z.softmax(-1)
    torch.testing.assert_close(probs.double(), reference.reshape_as(probs), atol=1e-3, rtol=0.01)
    grad = torch.randn(1, 2, 32, 256, device="cuda")
    dlse = torch.randn_like(lse)
    dkey = torch.zeros(1 + key.numel(), device="cuda", dtype=dtype)[1:].view(shape)
    ext.ops.chunk_softmax(1, 256, key, empty, empty, lse, grad, dlse, empty, empty, empty, dkey)
    expected = torch.zeros_like(z)
    prob = reference[..., 0, :]
    g = grad.double()
    expected[..., 0, :] = prob * (g - (g*prob).sum(-1, keepdim=True) + dlse[..., 1:].double())
    expected = expected.reshape(1, 2, 32, 512).permute(0, 3, 1, 2)
    torch.testing.assert_close(dkey.double(), expected, atol=1e-3, rtol=0.03)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_decode_update_preserves_key_autograd(ext, monkeypatch, grad_enabled):
    """The inference primitive must not detach differentiable update_memory calls."""
    import ast
    import importlib.util
    import math
    from pathlib import Path
    import sys
    from typing import Optional
    import torch.nn.functional as F
    from einops import rearrange, repeat
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setitem(sys.modules, "xllm_extension", ext)
    spec = importlib.util.spec_from_file_location("decode_update_chunk", root / "xllm/modules/fused_ops/chunk_softmax.py")
    chunk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chunk)
    source = root / "xllm/modules/adaptive_working_memory.py"
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace = dict(torch=torch, Tensor=torch.Tensor, nn=torch.nn, Optional=Optional,
                     math=math, F=F, rearrange=rearrange, repeat=repeat, chunk_softmax=chunk)
    exec(compile(tree, str(source), "exec"), namespace)
    module = namespace["AdaptiveWorkingMemory"](None, 2, 1, 32, 32, 64, False, 1e-6)
    q = torch.randn(1, 1, 128, 32, device="cuda") * .1
    key = torch.randn_like(q).requires_grad_()
    value = torch.randn_like(q)
    memory = torch.randn(1, 1, 32, 32, device="cuda") * .01
    logs = torch.zeros(1, 1, 32, device="cuda")
    calls = []
    decode = chunk.decode
    def counted(t):
        calls.append(t.shape)
        return decode(t)
    monkeypatch.setattr(chunk, "decode", counted)
    with torch.set_grad_enabled(grad_enabled):
        actual = module.update_memory(q, key, value, memory, logs, None, None, None)
        monkeypatch.setattr(chunk, "available", lambda key: False)
        expected = module.update_memory(q, key, value, memory, logs, None, None, None)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        if grad_enabled:
            grad = torch.autograd.grad([x.sum() for x in actual], key, retain_graph=True)[0]
            reference = torch.autograd.grad([x.sum() for x in expected], key)[0]
            assert grad.count_nonzero() > 0
            torch.testing.assert_close(grad, reference, atol=0, rtol=0)
    assert len(calls) == (0 if grad_enabled else 1)


def test_logical_negative_views(ext, monkeypatch):
    """Python helpers resolve logical values before handing raw pointers to CUDA."""
    import importlib.util
    from pathlib import Path
    import sys
    monkeypatch.setitem(sys.modules, "xllm_extension", ext)
    source = Path(__file__).resolve().parents[3] / "xllm/modules/fused_ops/chunk_softmax.py"
    spec = importlib.util.spec_from_file_location("negative_view_chunk", source)
    chunk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chunk)
    key = torch.randn(1, 1, 512, 32, device="cuda")._neg_view()
    probs, logs, _ = chunk.forward(key, 256)
    reference = key.permute(0, 1, 3, 2).reshape(1, 1, 32, 2, 256)
    torch.testing.assert_close(probs, reference.softmax(-1).reshape_as(probs), atol=1e-6, rtol=1e-5)
    decoded, _ = chunk.decode(key[:, :, :256])
    torch.testing.assert_close(decoded, reference[..., 0, :].softmax(-1), atol=1e-6, rtol=1e-5)
    grad = torch.randn(1, 1, 32, 256, device="cuda")._neg_view()
    dlse = torch.randn_like(logs)._neg_view()
    result = chunk.backward(key, 256, None, logs, grad, dlse)
    z = key.detach().double().requires_grad_()
    zz = z.permute(0, 1, 3, 2).reshape(1, 1, 32, 2, 256)
    expected = torch.autograd.grad((zz[..., 0, :].softmax(-1), zz[..., 0, :].logsumexp(-1)), z,
                                   (grad.double(), dlse[..., 1].double()))[0]
    torch.testing.assert_close(result.double(), expected, atol=1e-6, rtol=1e-5)
