# Group RMS norm and layer norm

The eight existing group RMS norm and layer norm ops dispatch to the fused implementations on Blackwell.
The Python APIs and their saved statistics are unchanged. The original CUDA algorithm handles other devices,
double precision, deterministic-algorithm mode and shapes declined by the fast paths. Its 32-group check is removed:
the algorithm itself supports more groups, and the regression sweep exercises that fallback too.

Comparisons above 32 groups use the original algorithm with its group-count restriction lifted. Unmodified
upstream rejects those calls; the reported speedup is not a successful execution of unmodified upstream on them.

## Build and check

From the repository root, on a B300 with a CUDA toolchain that supports sm_103a:

```bash
TORCH_CUDA_ARCH_LIST=10.3a python setup.py build_ext --inplace
python -m pytest -q tests/kernels/group_norm
XLLM_TEST_INSTALLED=1 python -m pytest -q tests/kernels/group_norm/test_integrated.py
python tests/kernels/group_norm/benchmark_native.py --output /tmp/group_norm-latency.json
```

The regular setup includes the optimized sources when all selected CUDA targets are Blackwell or newer; mixed/older
architecture builds retain the original ops. Tests skip without a CUDA device of compute capability 10.0 or newer.
The regression extension compiles the production sources, rather than maintaining separate kernel copies.
Hardware validation is limited to NVIDIA B300 (sm_103a), PyTorch 2.16.0.dev20261009+cu134 and CUDA 13.4.

## Latency reproduction

The earlier standalone claim was **7.7x**. The integrated public interfaces reproduce a geometric mean
of **7.69x** across 26 model cases: **1.683010 ms** for the original
interfaces and **0.218970 ms** for the new interfaces. Every case below is faster than its original
counterpart; the smallest measured gain is 1.87x.

These are median CUDA-event latencies over 21 calls with 512 MiB of L2 flushing. The device is kept occupied while
the host queues each call. Native times include output allocation kernels and saved-statistic/layout conversions;
event overhead is retained in both native columns. The standalone-claim column describes a preallocated primitive,
and some claims exclude event/entry overhead. It is not interchangeable with the native-call latency.

| Case | Claimed primitive ms | Reproduced primitive ms | Original native ms | New native ms | Native speedup |
|---|---:|---:|---:|---:|---:|
| rms-q64c64-fwd | 0.651300 | 0.648256 | 8.610816 | 0.651264 | 13.22x |
| rms-q64c64-bwd | 0.941000 | 0.936960 | 13.161504 | 0.940160 | 14.00x |
| rms-q64c128-fwd | 0.322500 | 0.318496 | 2.158592 | 0.321504 | 6.71x |
| rms-q64c128-bwd | 0.474700 | 0.472096 | 3.259392 | 0.475136 | 6.86x |
| ln-q64c128-fwd | 0.347100 | 0.343136 | 2.159616 | 0.348128 | 6.20x |
| ln-q64c128-bwd | 0.483400 | 0.480352 | 3.300576 | 0.484352 | 6.81x |
| rms-o64c24-fwd | 0.261100 | 0.258112 | 8.609824 | 0.261120 | 32.97x |
| rms-o64c24-bwd | 0.375800 | 0.370752 | 8.611840 | 0.374784 | 22.98x |
| rms-o64c40-fwd | 0.117800 | 0.114784 | 2.157568 | 0.118752 | 18.17x |
| rms-o64c40-bwd | 0.164900 | 0.161824 | 2.158592 | 0.165888 | 13.01x |
| rms-o64c64-fwd | 0.167900 | 0.163904 | 2.157568 | 0.168128 | 12.83x |
| rms-o64c64-bwd | 0.242700 | 0.239680 | 2.158592 | 0.242656 | 8.90x |
| rms-b1c1536-fwd | 0.240700 | 0.237632 | 0.592896 | 0.240672 | 2.46x |
| rms-b1c1536-bwd | 0.361500 | 0.359488 | 5.444608 | 0.362464 | 15.02x |
| rms-b2c1280-fwd | 0.107500 | 0.102464 | 0.297184 | 0.106496 | 2.79x |
| rms-b2c1280-bwd | 0.158900 | 0.155712 | 1.529824 | 0.159744 | 9.58x |
| rms-b4c1024-fwd | 0.164800 | 0.159808 | 0.306144 | 0.163840 | 1.87x |
| rms-b4c1024-bwd | 0.244800 | 0.241760 | 1.609664 | 0.245664 | 6.55x |
| rms-b1c4096-fwd | 0.167000 | 0.163904 | 0.318400 | 0.167936 | 1.90x |
| rms-b1c4096-bwd | 0.250900 | 0.248768 | 1.660800 | 0.251904 | 6.59x |
| rms-q32c128-fwd | 0.164900 | 0.161824 | 1.082368 | 0.164768 | 6.57x |
| rms-q32c128-bwd | 0.244900 | 0.241728 | 2.229248 | 0.245792 | 9.07x |
| rms-k8c128-fwd | 0.048100 | 0.045120 | 0.276288 | 0.048128 | 5.74x |
| rms-k8c128-bwd | 0.068600 | 0.065568 | 0.276480 | 0.068608 | 4.03x |
| rms-o32c80-fwd | 0.107500 | 0.104512 | 1.082336 | 0.108512 | 9.97x |
| rms-o32c80-bwd | 0.154800 | 0.153632 | 1.083392 | 0.155648 | 6.96x |

The JSON companion records raw event latency and CUPTI kernel spans as separate measurements. Decode rows are
particularly sensitive to a few microseconds of overhead. Timings measure these operations, not full-model throughput.

## Numerical agreement

This is **tolerance-based**, not a bit-exact replacement. The table summarizes model-case errors against the fp64
reference rounded to each output dtype. `atol + rtol * abs(reference)` is the per-element criterion. Some regression
checks admit a small fraction of elements outside it; the exact thresholds and achieved fractions are recorded for
every output in the JSON companion. The additional small fp32 native/autograd tests require every element to pass.

| Output | Largest absolute error | Largest relative L2 error | Exactly equal tensors / tested tensors |
|---|---:|---:|---:|
| y | 0.0625 | 1.5706339e-05 | 0 / 13 |
| rstd | 2.3841858e-07 | 5.0749964e-08 | 0 / 13 |
| dx | 0.015625 | 1.2742872e-05 | 0 / 13 |
| dgamma | 0.5 | 2.1745099e-05 | 3 / 8 |
| mean | 5.9604645e-08 | 3.1022537e-09 | 0 / 1 |
| dbeta | 0 | 0 | 1 / 1 |

Repeated calls are not universally bitwise repeatable on the fast paths. Original paths are selected when deterministic algorithms are requested.

Detailed inputs and tolerances live in `tests/kernels/group_norm/cases.py`; the measured per-output errors, match fractions and repeatability are in [group_norm-measurements.json](group_norm-measurements.json).

## Validation

The full native regression suite passes 221 tests. The complete `setup.py build_ext --inplace` build passes.
The installed extension's integration cases pass compute-sanitizer memcheck with zero errors.

## Short training comparison

A B300 training comparison combined the group RMS normalization and rotary embedding optimizations.
These measurements describe their combined effect, rather than attributing a model-level gain to either
operation independently. The baseline used the unchanged xllm implementation.

Each of two placements ran 200 steps from the same K2 Horizon 7B final SFT/FTPO weights with fresh AdamW
optimizers. Each arm used four nodes and 32 GPUs, FSDP2, bf16 computation with fp32 master weights,
8192-token sequences, a global batch of 32, seed 42, peak learning rate 1e-5, and gradient clipping at 1.0.
The second placement exchanged the physical node groups between implementations. Every run consumed
52,428,800 training tokens. All 12,800 paired rank/batch input hashes matched.

The text was a fixed sample of [TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2), revision
`a86bdfb101ebaaf71b445f95fb0f9b9bc2a47511`, with 70.6M training tokens and 1.0M held-out tokens counted
with the K2 tokenizer. Held-out perplexity was evaluated every 50 steps, capped at two batches per rank.
This comparison does not reproduce the original pretraining mixture, optimizer state, or batch order.
The checkpoint's rotary base of 10,000,000 was preserved.

| Measurement | Original | Combined optimizations |
|---|---:|---:|
| Geometric mean of median warm step latency | 0.598011 s | 0.576836 s |
| Final held-out perplexity, placement 1 | 11.525846 | 11.525516 |
| Final held-out perplexity, placement 2 | 11.526088 | 11.525371 |

The crossover latency gain is **1.0367x**. Each median covers steps 21 through 200; checkpoint and evaluation
periods are excluded. Input hashing and rank-zero operation-layout logging were enabled in both arms.

This training comparison is not bit-exact. Mean absolute global-loss difference over both placements was
0.000252; the largest difference was 0.026130. Repeating the original implementation alone produced a
maximum difference of 0.026134. Final master-weight relative L2 differences were 0.0571% and 0.0592%;
repeated original and optimized runs differed by 0.0555% and 0.0554%, respectively. Relative to the
parameter updates rather than the full weights, implementation differences were 18.9% to 19.6%, versus
18.3% for repeated runs of either unchanged implementation.

Both implementations produced isolated gradient-norm spikes; maxima across their two runs were 43.7 for
the original and 16.7 for the optimized build. No nonfinite losses, gradient norms, or final parameters were
observed. These are short continued-training checks, not evidence of long-run convergence. They exercise
group RMS normalization and RoPE; they do not exercise group layer normalization, timestep decay
normalization, or AWM softmax. Hardware coverage remains B300 (sm_103a).
