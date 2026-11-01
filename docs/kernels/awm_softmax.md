# Working-memory chunk softmax

The native chunk-softmax primitive is used by working-memory forward, recomputation, backward and decode
updates. Existing Python interfaces and masked-previous-chunk behavior are preserved. Previous-chunk masks use the
original implementation. Existing key tensors are passed to the backward primitive to avoid reconstructing them.

The two-pass backward supports float32 incoming probability gradients independently of the key dtype, as the real
working-memory backward produces. Native latency measurements use those float32 gradients. The tests compare complete working-memory outputs and gradients with the native route enabled and disabled, both
with and without previous state and packed segments.

## Build and check

From the repository root, on a B300 with a CUDA toolchain that supports sm_103a:

```bash
TORCH_CUDA_ARCH_LIST=10.3a python setup.py build_ext --inplace
python -m pytest -q tests/kernels/awm_softmax
XLLM_TEST_INSTALLED=1 python -m pytest -q tests/kernels/awm_softmax/test_integrated.py
python tests/kernels/awm_softmax/benchmark_native.py --output /tmp/awm_softmax-latency.json
```

The regular setup includes the optimized sources when all selected CUDA targets are Blackwell or newer; mixed/older
architecture builds retain the original ops. Tests skip without a CUDA device of compute capability 10.0 or newer.
The regression extension compiles the production sources, rather than maintaining separate kernel copies.
Hardware validation is limited to NVIDIA B300 (sm_103a), PyTorch 2.16.0.dev20261009+cu134 and CUDA 13.4.

## Latency reproduction

The optimized public interfaces achieve a geometric mean speedup against original xllm
of **8.55x** across 24 model cases: **0.267952 ms** for the original
interfaces and **0.031335 ms** for the new interfaces. Every case below is faster than its original
counterpart; the smallest measured gain is 5.28x.

These are median CUDA-event latencies over 21 calls with 512 MiB of L2 flushing. The device is kept occupied while
the host queues each call. Native times include output allocation kernels and saved-statistic/layout conversions;
event overhead is retained in both native columns.

| Case | Original xllm ms | Optimized ms | Speedup |
| --- | ---: | ---: | ---: |
| train_fwd_4x256_b1_l8192 | 0.132096 | 0.015328 | 8.62x |
| train_bwd_4x256_b1_l8192 | 0.178336 | 0.023456 | 7.60x |
| train_fwd_4x256_b4_l8192 | 0.428096 | 0.037856 | 11.31x |
| train_bwd_4x256_b4_l8192 | 0.688064 | 0.049152 | 14.00x |
| train_fwd_4x256_b1_l16384 | 0.235520 | 0.025536 | 9.22x |
| train_bwd_4x256_b1_l16384 | 0.424096 | 0.031744 | 13.36x |
| prefill_4x256_b1_l16384 | 0.186368 | 0.029696 | 6.28x |
| prefill_4x256_b1_l16384_prev | 0.256160 | 0.035808 | 7.15x |
| prefill_4x256_b1_l4096_prev | 0.102464 | 0.019360 | 5.29x |
| prefill_4x256_b8_l16384 | 1.331200 | 0.151520 | 8.79x |
| decode_4x256_b1 | 0.075776 | 0.013280 | 5.71x |
| decode_4x256_b8 | 0.395264 | 0.045952 | 8.60x |
| train_fwd_8x128_b1_l8192 | 0.131040 | 0.017024 | 7.70x |
| train_bwd_8x128_b1_l8192 | 0.180160 | 0.019456 | 9.26x |
| train_fwd_8x128_b4_l8192 | 0.424960 | 0.039904 | 10.65x |
| train_bwd_8x128_b4_l8192 | 0.692224 | 0.040576 | 17.06x |
| train_fwd_8x128_b1_l16384 | 0.235520 | 0.023520 | 10.01x |
| train_bwd_8x128_b1_l16384 | 0.423424 | 0.033792 | 12.53x |
| prefill_8x128_b1_l16384 | 0.186368 | 0.029664 | 6.28x |
| prefill_8x128_b1_l16384_prev | 0.256960 | 0.035840 | 7.17x |
| prefill_8x128_b1_l4096_prev | 0.102368 | 0.019392 | 5.28x |
| prefill_8x128_b8_l16384 | 1.330208 | 0.151552 | 8.78x |
| decode_8x128_b1 | 0.077760 | 0.013280 | 5.86x |
| decode_8x128_b8 | 0.395296 | 0.044000 | 8.98x |

The JSON companion records raw event latency and CUPTI kernel spans as separate measurements. Decode rows are
particularly sensitive to a few microseconds of overhead. Timings measure these operations, not full-model throughput.

## Numerical agreement

This is **tolerance-based**, not a bit-exact replacement. The table summarizes model-case errors against the fp64
reference rounded to each output dtype. `atol + rtol * abs(reference)` is the per-element criterion. Some regression
checks admit a small fraction of elements outside it; the exact thresholds and achieved fractions are recorded for
every output in the JSON companion. The additional small fp32 native/autograd tests require every element to pass.

| Output | Largest absolute error | Largest relative L2 error | Exactly equal tensors / tested tensors |
|---|---:|---:|---:|
| kkey | 0.0009765625 | 0.00014251201 | 0 / 18 |
| lse_out | 9.5367432e-07 | 4.6961272e-08 | 0 / 18 |
| dkey | 0.00012207031 | 2.7125906e-05 | 0 / 6 |
| prev_kkey | 6.1035156e-05 | 4.5363392e-05 | 0 / 4 |

Repeated calls were bitwise repeatable on the tested model cases; this does not imply bitwise equality to the original or fp64 reference.

Detailed inputs and tolerances live in `tests/kernels/awm_softmax/cases.py`; the measured per-output errors, match fractions and repeatability are in [awm_softmax-measurements.json](awm_softmax-measurements.json).

## Validation

The full native regression suite passes 136 tests. The complete `setup.py build_ext --inplace` build passes.
The 18 native primitive integration cases pass compute-sanitizer memcheck with zero errors. The eight complete working-memory comparisons also pass. A separate sanitizer run through cuBLAS reported its initialization-time selective-recompilation API warning; it reported no invalid device-memory access.

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
| kkey | 0.17214371 | 0.17214372 |
| lse_out | 3.7144113e-06 | 3.7743046e-06 |
| dkey | 0.2490972 | 0.16620178 |
| prev_kkey | 0.16890945 | 0.16890946 |

Per-case outputs, maximum absolute errors, and relative L2 errors are recorded in
[awm_softmax-fp64-errors.json](awm_softmax-fp64-errors.json).
