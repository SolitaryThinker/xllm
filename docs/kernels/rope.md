# Rotary position embedding

The existing single-tensor and paired rotary helpers use one fused CUDA launch on Blackwell. The custom
autograd function applies the conjugate rotation in backward and supports incoming expanded/strided gradients.
Table gradients, CPU-only use without the extension, mixed dtypes, complex128 or broadcast tables, and unsupported layouts use the original PyTorch expressions. The table
may be sliced, strided or have a storage offset. Native argument checks verify both output dtypes, and conjugate or
negative table views are resolved before raw-pointer access.

## Build and check

From the repository root, on a B300 with a CUDA toolchain that supports sm_103a:

```bash
TORCH_CUDA_ARCH_LIST=10.3a python setup.py build_ext --inplace
python -m pytest -q tests/kernels/rope
XLLM_TEST_INSTALLED=1 python -m pytest -q tests/kernels/rope/test_integrated.py
python tests/kernels/rope/benchmark_native.py --output /tmp/rope-latency.json
```

The regular setup includes the optimized sources when all selected CUDA targets are Blackwell or newer; mixed/older
architecture builds retain the original ops. Tests skip without a CUDA device of compute capability 10.0 or newer.
The regression extension compiles the production sources, rather than maintaining separate kernel copies.
Hardware validation is limited to NVIDIA B300 (sm_103a), PyTorch 2.16.0.dev20261009+cu134 and CUDA 13.4.

## Latency reproduction

The earlier standalone claim was **4.7x**. The integrated public interfaces reproduce a geometric mean
of **4.00x** across 23 model cases: **0.103248 ms** for the original
interfaces and **0.025813 ms** for the new interfaces. Every case below is faster than its original
counterpart; the smallest measured gain is 2.00x.

These are median CUDA-event latencies over 21 calls with 512 MiB of L2 flushing. The device is kept occupied while
the host queues each call. Native times include output allocation kernels and saved-statistic/layout conversions;
event overhead is retained in both native columns. The standalone-claim column describes a preallocated primitive,
and some claims exclude event/entry overhead. It is not interchangeable with the native-call latency.

| Case | Claimed primitive ms | Reproduced primitive ms | Original native ms | New native ms | Native speedup |
|---|---:|---:|---:|---:|---:|
| train-h32k8d64-fwd | 0.055400 | 0.055424 | 0.300928 | 0.058368 | 5.16x |
| train-h32k8d64-bwd | 0.045100 | 0.045184 | 0.238592 | 0.048128 | 4.96x |
| decode-h32k8d64-b1 | 0.004200 | 0.004224 | 0.014336 | 0.007168 | 2.00x |
| decode-h32k8d64-b64 | 0.004200 | 0.004224 | 0.016384 | 0.007168 | 2.29x |
| train-h32k8d128-fwd | 0.104500 | 0.104544 | 0.583680 | 0.106496 | 5.48x |
| train-h32k8d128-bwd | 0.084000 | 0.084096 | 0.466912 | 0.087040 | 5.36x |
| decode-h32k8d128-b1 | 0.004200 | 0.004224 | 0.014336 | 0.007168 | 2.00x |
| decode-h32k8d128-b64 | 0.004200 | 0.004224 | 0.016384 | 0.007168 | 2.29x |
| train-h8k4d256r128-fwd | 0.034900 | 0.034976 | 0.264192 | 0.037888 | 6.97x |
| train-h8k4d256r128-bwd | 0.024700 | 0.024704 | 0.178176 | 0.027648 | 6.44x |
| decode-h8k4d256r128-b1 | 0.004200 | 0.004224 | 0.014336 | 0.007168 | 2.00x |
| decode-h8k4d256r128-b64 | 0.004200 | 0.004224 | 0.018432 | 0.007200 | 2.56x |
| train-h4k4d256-fwd | 0.045100 | 0.047200 | 0.245760 | 0.048128 | 5.11x |
| train-h4k4d256-bwd | 0.026600 | 0.026720 | 0.129024 | 0.027648 | 4.67x |
| decode-h4k4d256-b1 | 0.004200 | 0.004224 | 0.014336 | 0.007168 | 2.00x |
| decode-h4k4d256-b64 | 0.004200 | 0.004224 | 0.016384 | 0.007168 | 2.29x |
| prefill-h32k8d128-start8192 | 0.030800 | 0.030848 | 0.158688 | 0.033792 | 4.70x |
| prefill-h32k8d128 | 0.055400 | 0.055424 | 0.298784 | 0.058336 | 5.12x |
| prefill-h8k4d256r128 | 0.020500 | 0.020608 | 0.141312 | 0.023520 | 6.01x |
| prefill-h4k4d256 | 0.026700 | 0.026752 | 0.133088 | 0.029664 | 4.49x |
| unbind-q-h32k8d128-fwd | 0.104500 | 0.104544 | 0.785376 | 0.108512 | 7.24x |
| unbind-kq-h32k8d128-fwd | 0.084000 | 0.084096 | 0.665568 | 0.087872 | 7.57x |
| bhld-h32k8d128-bwd | 0.086100 | 0.086144 | 0.447520 | 0.089056 | 5.03x |

The JSON companion records raw event latency and CUPTI kernel spans as separate measurements. Decode rows are
particularly sensitive to a few microseconds of overhead. Timings measure these operations, not full-model throughput.

## Numerical agreement

This is **tolerance-based**, not a bit-exact replacement. The table summarizes model-case errors against the fp64
reference rounded to each output dtype. `atol + rtol * abs(reference)` is the per-element criterion. Some regression
checks admit a small fraction of elements outside it; the exact thresholds and achieved fractions are recorded for
every output in the JSON companion. The additional small fp32 native/autograd tests require every element to pass.

| Output | Largest absolute error | Largest relative L2 error | Exactly equal tensors / tested tensors |
|---|---:|---:|---:|
| out1 | 0.015625 | 1.065366e-05 | 5 / 23 |
| out2 | 0.015625 | 1.2184832e-05 | 8 / 17 |

Repeated calls were bitwise repeatable on the tested model cases; this does not imply bitwise equality to the original or fp64 reference.

Detailed inputs and tolerances live in `tests/kernels/rope/cases.py`; the measured per-output errors, match fractions and repeatability are in [rope-measurements.json](rope-measurements.json).

## Validation

The full native regression suite passes 133 tests. The complete `setup.py build_ext --inplace` build passes.
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

## Combined model training

See [Gekko training comparison](gekko-training.md) for all four kernel families
trained together with nonzero AWM gradients and same-binary repeat controls.
