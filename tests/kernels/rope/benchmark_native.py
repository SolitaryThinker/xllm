"""Time the integrated public interfaces against their original implementations."""
import argparse, importlib.util, json, math, os, statistics, sys
from pathlib import Path
import torch
family = 'rope'
root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('rope-latency.json'))
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
current = module('rotary_current', root / 'xllm/modules/rotary_positional_embedding.py')
import types

def original_rotation(x, table, backward):
    if backward:
        table = table.conj()
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(z * table.unsqueeze(1)).flatten(3).type_as(x)
previous = types.SimpleNamespace(apply_rotary_embedding=original_rotation, apply_rotary_embeddings=lambda q, k, f, b: (original_rotation(q, f, b), original_rotation(k, f, b)))
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
work = list(cases.CASES.items())
results = []
for i, (name, case) in enumerate(work):
    inp = cases.make_inputs(case, 1234, 'cuda')
    conj, x, x2, table = inp
    if x2.numel():
        new = lambda: current.apply_rotary_embeddings(x, x2, table, conj)
        old = lambda: previous.apply_rotary_embeddings(x, x2, table, conj)
    else:
        new = lambda: current.apply_rotary_embedding(x, table, conj)
        old = lambda: previous.apply_rotary_embedding(x, table, conj)
    ms_new = time_call(new)
    ms_old = time_call(old)
    row = dict(case=name, native_ms=ms_new, baseline_ms=ms_old, speedup=ms_old / ms_new)
    results.append(row)
    print(json.dumps(row), flush=True)
    options.output.write_text(json.dumps(dict(family=family, empty_event_ms=empty, results=results), indent=2))
