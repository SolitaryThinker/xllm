"""Times the group norm kernels case by case with CUDA events, flushing the L2 cache before every call.

    python tests/benchmark.py [name-substring ...] [--sweep] [--reps N]

Prints the median time of a call in milliseconds for each case of cases.shape_cases() (and of cases.sweep() with
--sweep). Setting GN_IMPL=off in the environment routes every call to xllm's own kernels instead, which gives the
baseline the fast kernels are compared with.
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases  # noqa: E402
import extension  # noqa: E402

FLUSH_BYTES = 512 << 20  # four times the L2 cache of a B300


def time_case(module, case, index, reps, warmup, flush):
    """Median milliseconds of one call of `case`."""
    inputs = cases.make_inputs(case, 1000 + index, "cuda")
    outputs = cases.make_outputs(inputs)
    for _ in range(warmup):
        module.run(*inputs, *outputs)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        flush.zero_()
        starts[i].record()
        module.run(*inputs, *outputs)
        stops[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in zip(starts, stops))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="run only the cases whose name contains one of these")
    parser.add_argument("--sweep", action="store_true", help="also time the sweep of cases.sweep()")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()
    if not extension.cuda_available():
        print("skipped: needs a CUDA GPU of compute capability 10.0+")
        return 0
    module = extension.build()
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    selected = cases.all_cases() if args.sweep else cases.shape_cases()
    times = []
    for i, case in enumerate(selected):
        if args.names and not any(n in case["name"] for n in args.names):
            continue
        ms = time_case(module, case, i, args.reps, 10, flush)
        times.append(ms)
        print(f"{case['name']:<48s} {ms:8.4f} ms", flush=True)
    if times:
        print(f"{'geomean':<48s} {statistics.geometric_mean(times):8.4f} ms over {len(times)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
