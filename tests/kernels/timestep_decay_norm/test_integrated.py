"""Check native TDN layouts and gradients against the fp64 step-by-step recurrence."""
import pytest
import torch
import reference
import native_build


@pytest.fixture(scope="module")
def ext():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires a Blackwell GPU")
    return native_build.build()


@pytest.mark.parametrize("cub", [False, True])
@pytest.mark.parametrize("length", [1, 7, 129])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("width", [24, 64])
def test_native_recurrence_and_gradients(ext, cub, length, masked, width):
    torch.manual_seed(712)
    b, k = 2, 3
    x = torch.randn(b, length, k * width, device="cuda")
    count = torch.tensor([0, 23], device="cuda", dtype=torch.int64)
    pm = torch.randn(b, k, device="cuda") * 0.01
    pv = torch.rand(b, k, device="cuda") * 0.02
    gamma = torch.rand(k * width, device="cuda") + 0.5
    beta = torch.randn_like(gamma) * 0.1
    bos = torch.rand(b, length, device="cuda") < 0.2 if masked else None
    pad = torch.rand(b, length, device="cuda") < 0.2 if masked else None
    fwd = ext.ops.group_timestep_decay_norm_cub_fwd if cub else ext.ops.group_timestep_decay_norm_fwd
    actual = fwd(x, bos, count, pm, pv, gamma, beta, pad, k, 0.9, 0.99, 1e-5)
    params = [t.double().requires_grad_() for t in (x, pm, pv, gamma, beta)]
    xx, mm, vv, gg, bb = params
    ref = reference.forward_loop(xx, bos, count, mm, vv, gg, bb, pad, k, 0.9, 0.99, 1e-5)
    for i, (got, expected) in enumerate(zip(actual, ref)):
        if cub and i >= 4:
            got = got.transpose(1, 2)
        torch.testing.assert_close(got.double(), expected.double(), atol=3e-4, rtol=3e-4)
    dy, dm, dv = torch.randn_like(x), torch.randn_like(pm), torch.randn_like(pv)
    expected = torch.autograd.grad((ref[0], ref[2], ref[3]), params, (dy.double(), dm.double(), dv.double()))
    bwd = ext.ops.group_timestep_decay_norm_cub_bwd if cub else ext.ops.group_timestep_decay_norm_bwd
    args = [dy, dm, dv, x, count, bos, actual[4], actual[5], gamma]
    args += [pad, k, 0.9, 0.99] if cub else [beta, pad, k, 0.9, 0.99, 1e-5, False]
    got = bwd(*args)
    for a, e in zip(got, expected):
        torch.testing.assert_close(a.double(), e.double(), atol=2e-3, rtol=2e-3)
    if not cub:
        args[3] = actual[0]
        args[-1] = True
        for a, e in zip(bwd(*args), expected):
            torch.testing.assert_close(a.double(), e.double(), atol=3e-3, rtol=3e-3)


def _persistent_inputs(device="cuda", length=1025):
    x = torch.randn(2, length, 24 * 64, device=device, dtype=torch.bfloat16)
    pc = torch.zeros(2, device=device, dtype=torch.int64)
    pm = torch.zeros(2, 24, device=device, dtype=x.dtype)
    pv = torch.zeros_like(pm)
    gamma, beta = torch.ones(24 * 64, device=device, dtype=x.dtype), torch.zeros(24 * 64, device=device, dtype=x.dtype)
    return (x, None, pc, pm, pv, gamma, beta, None, 24, 0.999, 0.9999, 1e-5)


def test_multiple_devices_default_stream(ext):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible GPUs")
    # Default streams have the same raw handle on different devices. Cached workspaces must include the device.
    inputs = [_persistent_inputs(f"cuda:{i}") for i in [0, 1]]
    for _ in range(3):
        for i in [0, 1]:
            out = ext.ops.group_timestep_decay_norm_fwd(*inputs[i])
            assert out[0].device.index == i
            assert torch.isfinite(out[0]).all()
            torch.testing.assert_close(out[1], torch.full_like(out[1], inputs[i][0].size(1)))


def test_concurrent_streams_and_graph_replay(ext):
    inputs = [_persistent_inputs(length=1025) for _ in range(2)]
    expected = [ext.ops.group_timestep_decay_norm_fwd(*args) for args in inputs]
    streams = [torch.cuda.Stream() for _ in range(2)]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(3):
        outputs = []
        for stream, args in zip(streams, inputs):
            with torch.cuda.stream(stream):
                outputs.append(ext.ops.group_timestep_decay_norm_fwd(*args))
        torch.cuda.synchronize()
        for got, ref in zip(outputs, expected):
            for a, b in zip(got, ref):
                torch.testing.assert_close(a, b, atol=0.02, rtol=0.02)
    args = _persistent_inputs(length=257)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            ext.ops.group_timestep_decay_norm_fwd(*args)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ext.ops.group_timestep_decay_norm_fwd(*args)
    for _ in range(3):
        args[0].add_(0.05)
        graph.replay()
        ref = ext.ops.group_timestep_decay_norm_fwd(*args)
        for a, b in zip(captured, ref):
            torch.testing.assert_close(a, b, atol=0.04, rtol=0.02)
