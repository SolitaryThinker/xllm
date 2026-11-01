# Timestep decay normalization

Both existing public backends dispatch to the new kernels on Blackwell for supported group widths (up to
256 channels) and bf16/fp16/fp32. Other devices, double precision, wider groups, boundary decay coefficients and
deterministic-algorithm mode retain their original implementations. The cub API keeps its [B,K,L] saved statistics;
the chunkwise API keeps [B,L,K]. The conversion cost is included in the native latency table.

The forward workspace is keyed by host thread, CUDA device and stream. Its persistent scan uses a cooperative
launch to enforce full-grid residency. Graph capture uses the general scan rather than reusing host-generated epoch
tags. Tests cover two devices sharing default-stream handle values, concurrent streams and changed-input graph replay.

The native backward tests compare all five gradients against the fp64 step-by-step recurrence, including nonzero
carried-state gradients, resets, padding and backward from y. Fresh-state bf16 state gradients can be around 1e5;
one bf16 rounding step there can be 128 or 512. The measured absolute state-gradient errors below must be read
alongside their relative errors and the stricter fp32 autograd tests, not as a claim of bitwise equivalence.

## Build and check

From the repository root, on a B300 with a CUDA toolchain that supports sm_103a:

```bash
TORCH_CUDA_ARCH_LIST=10.3a python setup.py build_ext --inplace
python -m pytest -q tests/kernels/timestep_decay_norm
XLLM_TEST_INSTALLED=1 python -m pytest -q tests/kernels/timestep_decay_norm/test_integrated.py
python tests/kernels/timestep_decay_norm/benchmark_native.py --output /tmp/timestep_decay_norm-latency.json
```

The regular setup includes the optimized sources when all selected CUDA targets are Blackwell or newer; mixed/older
architecture builds retain the original ops. Tests skip without a CUDA device of compute capability 10.0 or newer.
The regression extension compiles the production sources, rather than maintaining separate kernel copies.
Hardware validation is limited to NVIDIA B300 (sm_103a), PyTorch 2.16.0.dev20261009+cu134 and CUDA 13.4.

## Latency reproduction

The optimized public interfaces achieve a geometric mean speedup against original xllm
of **8.61x** across 27 model cases: **0.428274 ms** for the original
interfaces and **0.049730 ms** for the new interfaces. Every case below is faster than its original
counterpart; the smallest measured gain is 2.28x.

These are median CUDA-event latencies over 21 calls with 512 MiB of L2 flushing. The device is kept occupied while
the host queues each call. Native times include output allocation kernels and saved-statistic/layout conversions;
event overhead is retained in both native columns.

| Case | Original xllm ms | Optimized ms | Speedup |
| --- | ---: | ---: | ---: |
| train-0.9b-fwd | 0.767904 | 0.069056 | 11.12x |
| train-0.9b-bwd | 1.556480 | 0.122912 | 12.66x |
| train-0.9b-bwd-me | 11.165696 | 0.124800 | 89.47x |
| prefill-0.9b-b1 | 0.368608 | 0.046112 | 7.99x |
| prefill-0.9b-b8 | 2.423776 | 0.186944 | 12.97x |
| tail-0.9b-b8 | 0.544800 | 0.063488 | 8.58x |
| decode-0.9b-b1 | 0.016352 | 0.007168 | 2.28x |
| decode-0.9b-b8 | 0.016416 | 0.007168 | 2.29x |
| decode-0.9b-b64 | 0.022528 | 0.007168 | 3.14x |
| train-4b-fwd | 1.148928 | 0.094208 | 12.20x |
| train-4b-bwd | 2.144224 | 0.176064 | 12.18x |
| train-4b-bwd-me | 11.562016 | 0.188352 | 61.39x |
| prefill-4b-b1 | 0.566240 | 0.063488 | 8.92x |
| prefill-4b-b8 | 3.941440 | 0.296928 | 13.27x |
| tail-4b-b8 | 0.882656 | 0.087200 | 10.12x |
| decode-4b-b1 | 0.018464 | 0.007168 | 2.58x |
| decode-4b-b8 | 0.019456 | 0.007168 | 2.71x |
| decode-4b-b64 | 0.028608 | 0.007136 | 4.01x |
| train-7.1b-fwd | 1.708096 | 0.135072 | 12.65x |
| train-7.1b-bwd | 2.220128 | 0.258048 | 8.60x |
| train-7.1b-bwd-me | 12.192928 | 0.280576 | 43.46x |
| prefill-7.1b-b1 | 0.847872 | 0.079904 | 10.61x |
| prefill-7.1b-b8 | 6.191104 | 0.458752 | 13.50x |
| tail-7.1b-b8 | 1.381408 | 0.122944 | 11.24x |
| decode-7.1b-b1 | 0.020448 | 0.007136 | 2.87x |
| decode-7.1b-b8 | 0.020480 | 0.007104 | 2.88x |
| decode-7.1b-b64 | 0.035808 | 0.007168 | 5.00x |

The JSON companion records raw event latency and CUPTI kernel spans as separate measurements. Decode rows are
particularly sensitive to a few microseconds of overhead. Timings measure these operations, not full-model throughput.

## Numerical agreement

This is **tolerance-based**, not a bit-exact replacement. The table summarizes model-case errors against the fp64
reference rounded to each output dtype. `atol + rtol * abs(reference)` is the per-element criterion. Some regression
checks admit a small fraction of elements outside it; the exact thresholds and achieved fractions are recorded for
every output in the JSON companion. The additional small fp32 native/autograd tests require every element to pass.

| Output | Largest absolute error | Largest relative L2 error | Exactly equal tensors / tested tensors |
|---|---:|---:|---:|
| y | 0.0625 | 0.00023354463 | 3 / 21 |
| count | 0 | 0 | 15 / 15 |
| mean | 0.00048828125 | 0.00089909926 | 15 / 21 |
| var | 0.0078125 | 0.00094996285 | 11 / 21 |
| cummean | 1.4901161e-07 | 6.5326086e-07 | 0 / 3 |
| cumrstd | 3.5762787e-07 | 1.0294627e-07 | 0 / 3 |
| dx | 0.03125 | 0.00030561473 | 0 / 6 |
| dprev_mean | 128 | 0.0018182834 | 2 / 6 |
| dprev_var | 512 | 0.0024795709 | 3 / 6 |
| dgamma | 2 | 0.00035658711 | 2 / 6 |
| dbeta | 0.03125 | 4.3698723e-06 | 5 / 6 |

Repeated calls are not universally bitwise repeatable on the fast paths. Original paths are selected when deterministic algorithms are requested.

Detailed inputs and tolerances live in `tests/kernels/timestep_decay_norm/cases.py`; the measured per-output errors, match fractions and repeatability are in [timestep_decay_norm-measurements.json](timestep_decay_norm-measurements.json).

## Validation

The full native regression suite passes 142 tests. The complete `setup.py build_ext --inplace` build passes.
The installed extension's integration cases pass compute-sanitizer memcheck with zero errors.

## Combined model training

See [Gekko training comparison](gekko-training.md) for all four kernel families
trained together with nonzero AWM gradients and same-binary repeat controls.

See [TDN and AWM precision](precision.md) for original-versus-optimized accuracy
and measured FP32 performance tradeoffs.

## Original and optimized errors against unrounded FP64 math

The table reports the largest tensor relative L2 error across the model cases for
each output: `norm(actual - reference) / norm(reference)`. Both implementations
use identical inputs and are measured against independent, unrounded FP64 math.
These errors include output rounding; the earlier numerical-agreement table uses
a reference rounded to the output dtype and therefore measures a different error.
Input quantization and RoPE angle-table generation are outside this comparison.
Backward comparisons use each operation's specified inputs and saved statistics.

| Output | Original relative L2 error (%) | Optimized relative L2 error (%) |
|---|---:|---:|
| y | 0.16982417 | 0.16982417 |
| count | 0 | 0 |
| mean | 0.18097047 | 0.18097215 |
| var | 0.18576194 | 0.18576194 |
| cummean | 0.00084215033 | 6.5464219e-05 |
| cumrstd | 0.0016362352 | 9.7252459e-06 |
| dx | 0.16602079 | 0.1660208 |
| dprev_mean | 0.17584297 | 0.17584297 |
| dprev_var | 0.20886832 | 0.20886832 |
| dgamma | 0.16884993 | 0.16884993 |
| dbeta | 0.16593221 | 0.16593221 |

Per-case outputs, maximum absolute errors, and relative L2 errors are recorded in
[timestep_decay_norm-fp64-errors.json](timestep_decay_norm-fp64-errors.json).
