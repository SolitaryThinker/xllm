"""Times the timestep decay normalization extension on the model cases of cases.shapes() and prints ms per case.

Each timed call runs with a cold L2 cache: before it, a buffer of twice the L2 size is written, then the GPU is kept
busy by a short spin kernel while the host enqueues the call, so that the interval between the two CUDA events that
bracket the call holds the call's own GPU work (its kernels and the gaps between them) and not the host's launch
latency. The result of a case is the median over --reps calls.

    python tests/benchmark.py [--reps N] [--warmup N] [case-name ...]
"""

import argparse
import statistics
import sys

import torch

import build
import cases


def calibrate_spin():
    """GPU cycles that torch.cuda._sleep must spin for to last about one millisecond."""
    cycles = 10_000_000
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(cycles)
    torch.cuda.synchronize()
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    torch.cuda.synchronize()
    return int(cycles / start.elapsed_time(end))


def time_case(ext, case, reps, warmup, flush, spin):
    """Median time of one call of `run` in ms, with a cold L2 cache."""
    inputs = cases.make_inputs(case)
    outputs = cases.make_outputs(inputs)
    times = []
    for i in range(warmup + reps):
        flush.fill_(i & 1)
        torch.cuda._sleep(spin)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        ext.run(*inputs, *outputs)
        end.record()
        end.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    return statistics.median(times)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="cases to time (default: all of cases.shapes())")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)
    if not build.supported():
        print("skipped: needs a CUDA GPU of compute capability 10.0 or newer")
        return 0
    shapes = cases.shapes()
    names = args.names or list(shapes)
    ext = build.load_extension()
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.int8, device="cuda")
    spin = calibrate_spin()
    print(f"{torch.cuda.get_device_name(0)}, median of {args.reps} calls, cold L2")
    times = []
    for name in names:
        ms = time_case(ext, shapes[name], args.reps, args.warmup, flush, spin)
        times.append(ms)
        print(f"{name:<22} {ms:9.4f} ms", flush=True)
    geomean = statistics.geometric_mean(times)
    print(f"{'geomean':<22} {geomean:9.4f} ms over {len(times)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
