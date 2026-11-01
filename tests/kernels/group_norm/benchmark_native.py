"""Time the integrated public interfaces against their original implementations."""
import argparse, importlib.util, json, math, os, statistics, sys
from pathlib import Path
import torch
family = 'group_norm'
root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('group_norm-latency.json'))
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
work = [(c['name'], c) for c in cases.shape_cases()]
results = []
for i, (name, case) in enumerate(work):
    inp = cases.make_inputs(case, 1000 + i, 'cuda')
    mode, norm, G, eps, me, x, dy, mu, rs, g, b = inp
    opname = 'group_' + ('rms' if norm == 0 else 'layer') + '_norm_' + ('fwd' if mode == 0 else 'bwd') + ('_affine' if g.numel() else '')
    args = [x, x.size(-1), G] if mode == 0 else [dy, x, x.size(-1), G, *([mu, rs] if norm else [rs])]
    if g.numel():
        args += [g, b] if norm else [g]
    args += [eps] if mode == 0 else [me]
    new = lambda: getattr(ops, opname)(*args)
    old = lambda: getattr(native, 'baseline_' + opname)(*args)
    ms_new = time_call(new)
    ms_old = time_call(old)
    row = dict(case=name, native_ms=ms_new, baseline_ms=ms_old, speedup=ms_old / ms_new)
    results.append(row)
    print(json.dumps(row), flush=True)
    options.output.write_text(json.dumps(dict(family=family, empty_event_ms=empty, results=results), indent=2))
