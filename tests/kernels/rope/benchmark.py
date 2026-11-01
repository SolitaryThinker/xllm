"""Times the rotary position embedding kernel on every case of cases.CASES and prints milliseconds per call.

Each call is timed with CUDA events after the L2 cache is flushed; the events are enqueued behind a short GPU spin so
that the host's launch latency does not count. A trial is the median of its calls, and the reported time is the mean of
the trials, minus the time of an empty region timed the same way (the cost of the two events). With --torch the same
cases are also timed on a PyTorch implementation of the operation (fp32 complex multiply, then a cast to x's dtype),
which is what the kernel replaces.

    python tests/benchmark.py [--case SUBSTRING ...] [--calls 50] [--trials 3] [--torch]
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import math
import os
import statistics
import sys

import torch

from tests.kernels.rope import cases
from tests.kernels.rope import extension


def torch_rotate(conj, x1, x2, freqs_cis, out1, out2):
    """PyTorch implementation: x viewed as complex fp32, multiplied by the table (conjugated for backward)."""
    f = freqs_cis.conj() if conj else freqs_cis
    for x, out in ((x1, out1), (x2, out2)):
        if x.numel():
            z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            out.copy_(torch.view_as_real(z * f.unsqueeze(1)).flatten(3))


def time_call(fn, flush, calls, trials):
    """Mean over trials of the median call time in ms."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(trials):
        times = []
        for _ in range(calls):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.zero_()
            torch.cuda._sleep(400_000)  # keeps the GPU busy while the host enqueues the call
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        means.append(statistics.median(times))
    return sum(means) / len(means)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", nargs="*", default=[], help="only the cases whose name contains one of these")
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--torch", action="store_true", help="also time the PyTorch implementation")
    a = p.parse_args()
    reason = extension.supported()
    if reason:
        print(f"skipped: {reason}")
        return 0
    ext = extension.build()
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")  # larger than the L2 cache
    empty = time_call(lambda: None, flush, a.calls, a.trials)
    print(f"event overhead (subtracted): {empty * 1e3:.2f} us")
    head = f"{'case':36s} {'kernel ms':>10s}" + (f" {'torch ms':>10s} {'speedup':>8s}" if a.torch else "")
    print(head)
    logs, logt = [], []
    for name, case in cases.CASES.items():
        if a.case and not any(s in name for s in a.case):
            continue
        inputs = cases.make_inputs(case, 1234, "cuda")
        outs = cases.make_outputs(inputs)
        ms = time_call(lambda: ext.run(*inputs, *outs), flush, a.calls, a.trials) - empty
        logs.append(math.log(ms))
        line = f"{name:36s} {ms:10.4f}"
        if a.torch:
            outs_t = cases.make_outputs(inputs)
            mt = time_call(lambda: torch_rotate(*inputs, *outs_t), flush, a.calls, a.trials) - empty
            logt.append(math.log(mt))
            line += f" {mt:10.4f} {mt / ms:7.2f}x"
        print(line)
    if logs:
        gm = math.exp(sum(logs) / len(logs))
        line = f"{'geomean':36s} {gm:10.4f}"
        if logt:
            line += f" {math.exp(sum(logt) / len(logt)):10.4f} {math.exp(sum(logt) / len(logt)) / gm:7.2f}x"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
