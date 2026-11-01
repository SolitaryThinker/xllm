"""Time the integrated public interfaces against their original implementations."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse, importlib.util, json, math, os, statistics, sys
from pathlib import Path
import torch
family = 'timestep_decay_norm'
root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('timestep_decay_norm-latency.json'))
options = parser.parse_args()
os.environ.pop('XLLM_TEST_INSTALLED', None)
from tests.kernels.timestep_decay_norm import cases, native_build
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
work = list(cases.shapes().items())
results = []
for i, (name, case) in enumerate(work):
    inp = cases.make_inputs(case)
    mode, G, b1, b2, eps, me, x, bos, pad, pc, pm, pv, g, b, dy, dm, dv, cm, cr = inp
    bos = None if bos.numel() == 0 else bos
    pad = None if pad.numel() == 0 else pad
    if mode < 2:
        args = [x, bos, pc, pm, pv, g, b, pad, G, b1, b2, eps]
        new = lambda: ops.group_timestep_decay_norm_cub_fwd(*args)
        old = lambda: native.baseline_CubCUDAFwd(*args)
    elif me:
        args = [dy, dm, dv, x, pc, bos, cm, cr, g, b, pad, G, b1, b2, eps, True]
        new = lambda: ops.group_timestep_decay_norm_bwd(*args)
        old = lambda: native.baseline_CUDABwd(*args)
    else:
        cm = cm.transpose(1, 2).contiguous()
        cr = cr.transpose(1, 2).contiguous()
        args = [dy, dm, dv, x, pc, bos, cm, cr, g, pad, G, b1, b2]
        new = lambda: ops.group_timestep_decay_norm_cub_bwd(*args)
        old = lambda: native.baseline_CubCUDABwd(*args)
    ms_new = time_call(new)
    ms_old = time_call(old)
    row = dict(case=name, native_ms=ms_new, baseline_ms=ms_old, speedup=ms_old / ms_new)
    results.append(row)
    print(json.dumps(row), flush=True)
    options.output.write_text(json.dumps(dict(family=family, empty_event_ms=empty, results=results), indent=2))
