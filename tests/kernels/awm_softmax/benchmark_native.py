"""Time the integrated public interfaces against their original implementations."""
import argparse, importlib.util, json, math, os, statistics, sys
from pathlib import Path
import torch
family = 'awm_softmax'
root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('awm_softmax-latency.json'))
options = parser.parse_args()
os.environ.pop('XLLM_TEST_INSTALLED', None)
sys.path[:0] = [str(root), str(root / 'tests/kernels' / family)]
import cases, native_build
native = native_build.build()
import xllm_extension
ops = xllm_extension.ops

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
current = module('chunk_current', root / 'xllm/modules/fused_ops/chunk_softmax.py')
flush = torch.empty(512 << 20, device='cuda', dtype=torch.uint8)
spin = 4000000

def time_call(fn):
    for _ in range(5):
        fn()
    values = []
    for _ in range(21):
        flush.zero_()
        torch.cuda._sleep(spin)
        start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)
empty = time_call(lambda: None)
work = [(name, cases.CASES[name]) for name in cases.SHAPE_NAMES]
results = []
for i, (name, case) in enumerate(work):
    inp = cases.make_inputs(name)
    mode, C, key, mask, prev, lse, grad, dlse = inp
    B = key.size(0)
    if mode == 2:
        new = lambda: current.decode(key)

        def old():
            z = key.float().transpose(2, 3).contiguous()
            return (z.softmax(-1).to(key.dtype), z.logsumexp(-1))
    else:
        B, L, K, S = key.shape
        N = L // C
        k = key.permute(0, 2, 1, 3)
        m = None if mask.numel() == 0 else mask
        prev_k = None if prev.numel() == 0 else prev.permute(0, 2, 1, 3)

        def fp32_key():
            z = key.permute(0, 2, 3, 1).float().contiguous()
            if m is not None:
                z.masked_fill_(m.reshape(B, 1, 1, L), -torch.inf)
            return z.reshape(B, K, S, N, C)
        if mode == 0:
            new = lambda: current.forward(k, C, m, prev_k)

            def old():
                z = fp32_key()
                p = z.softmax(-1).to(key.dtype).reshape(B, K, S, L)
                logs = torch.empty(B, K, S, N, device=key.device, dtype=torch.float32)
                torch.logsumexp(z[..., :-1, :], -1, out=logs[..., 1:])
                if prev.numel():
                    zp = prev.permute(0, 2, 3, 1).float().contiguous()
                    pp = zp.softmax(-1).to(key.dtype)
                    torch.logsumexp(zp, -1, out=logs[..., 0])
                else:
                    logs[..., 0] = -torch.inf
                    pp = None
                return (p, logs, pp)
        else:
            grad = grad.float()
            new = lambda: current.backward(k, C, m, lse, grad, dlse)

            def old():
                z = fp32_key()
                p = z.softmax(-1)
                d = torch.zeros_like(key)
                dst = d.permute(0, 2, 1, 3)
                for j in range(N - 1):
                    dst[:, :, j * C:(j + 1) * C].copy_(torch.ops.aten._softmax_backward_data(grad[..., j * C:(j + 1) * C].float(), p[..., j, :], -1, torch.float32).to(key.dtype).transpose(2, 3))
                delta = (dlse[..., 1:].unsqueeze(-1) * (z[..., :-1, :] - lse[..., 1:].unsqueeze(-1)).exp()).to(key.dtype)
                dst[:, :, :L - C].add_(delta.reshape(B, K, S, L - C).transpose(2, 3))
                if m is not None:
                    dst.masked_fill_(m.reshape(B, 1, L, 1), 0)
                return d
    ms_new = time_call(new)
    ms_old = time_call(old)
    row = dict(case=name, native_ms=ms_new, baseline_ms=ms_old, speedup=ms_old / ms_new)
    results.append(row)
    print(json.dumps(row), flush=True)
    options.output.write_text(json.dumps(dict(family=family, empty_event_ms=empty, results=results), indent=2))
