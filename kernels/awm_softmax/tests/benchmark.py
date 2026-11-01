"""Times one call of the kernels per case of cases.py (main shapes only) with CUDA events.

Before every call the inputs are copied to fresh addresses, the destinations are zeroed and the L2 cache is flushed;
the time of a call is from event to event around the call alone. Prints the median over the repetitions in ms per case
and the geometric mean.

    python tests/benchmark.py [--reps 50] [--cases substring ...]
"""

import argparse
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build  # noqa: E402
import cases  # noqa: E402


def time_case(ext, name, reps, warmup=10):
    """Median milliseconds of one call of case `name`."""
    inputs = cases.make_inputs(name)
    outs = cases.make_outputs(inputs)
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.int8, device="cuda")
    times = []
    for i in range(warmup + reps):
        args = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
        dst = [torch.zeros_like(o) for o in outs]
        flush.zero_()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        ext.run(*args, *dst)
        stop.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(stop))
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--cases", nargs="*", default=[], help="only the cases whose name contains one of these")
    a = p.parse_args()
    reason = build.gpu_supported()
    if reason:
        print(f"skipped: {reason}")
        return 0
    ext = build.build()
    names = [n for n in cases.SHAPE_NAMES if not a.cases or any(s in n for s in a.cases)]
    width = max(len(n) for n in names)
    results = []
    for name in names:
        ms = time_case(ext, name, a.reps)
        results.append(ms)
        print(f"{name:<{width}} {ms:9.4f} ms", flush=True)
    print(f"geomean {math.exp(statistics.mean(math.log(t) for t in results)):.4f} ms over {len(results)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
