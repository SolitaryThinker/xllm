# Group timestep decay normalization (CUDA)

CUDA kernels for xllm's group timestep decay normalization (TDN): a decayed running mean and variance of each channel
group along the sequence, with resets at bos tokens, that normalizes the activations of every layer of the Gekko
models. They replace xllm's `group_timestep_decay_norm_cub_fwd/_bwd` and `group_timestep_decay_norm_fwd/_bwd` ops
(xllm 692bc25) with one entry point that serves training and prefill forward, decode, and the backward pass, including
the memory-efficient variant, in one to three launches per call on the main paths. The forward kernels read `x` once
and write `y` once, and the backward kernels read `x` (or `y`) and `dy` twice (the second read comes from L2), so the
time of a call is set by memory traffic and not by the length of a serial scan.

The kernels do not use xllm's original cub and chunkwise backends as a fallback: the dispatch described below never
reaches them, and their sources are not in this directory.

## What it computes

For each row `b` and group `k` of `x [B, L, H]` (`K` groups of `G = H / K` channels), along the sequence
`t = 0 .. L-1`, from the carried state `prev_count [B]`, `prev_mean`, `prev_var [B, K]`:

```
m_t, v_t = mean and biased variance of the G channels of group k at step t
valid_t  = not padding_mask[b, t]            a padded step leaves the state unchanged and writes y = 0
reset_t  = valid_t and bos_mask[b, t]        count = 0, mean = 0, var = 0 before the update
valid step:
    count   += 1
    mean     = beta1 * mean + (1 - beta1) * m_t
    var      = beta2 * var  + (1 - beta2) * v_t
    cummean  = mean / (1 - beta1 ** count)
    cumrstd  = 1 / sqrt(var / (1 - beta2 ** count) + eps)       (cummean = 0, cumrstd = 1 / sqrt(eps) where
                                                                 1 - beta ** count is not positive)
    y        = (x - cummean) * cumrstd * gamma + beta
```

The forward pass returns `y`, the final `count`, `mean` and `var`, and, for training, `cummean` and `cumrstd`
(`[B, L, K]`, fp32). The backward pass is the gradient of `y` and of the final `mean` and `var` with respect to `x`,
`prev_mean`, `prev_var`, `gamma` and `beta`. With `memory_efficient` it receives `y` instead of `x` and recovers
`x = (y - beta) / gamma / cumrstd + cummean`. `tests/reference.py` states all of this in fp64, step by step and in
closed form.

## Interface

The extension has one function, `run(...)`, with destination passing: it takes 19 inputs and 11 output tensors,
positionally, and writes the outputs in place; it returns nothing.

```
run(mode, num_groups, beta1, beta2, eps, memory_efficient,
    x_or_y, bos_mask, padding_mask, prev_count, prev_mean, prev_var, gamma, beta, dy, dmean, dvar, cummean, cumrstd,
    y, count, mean, var, cummean_out, cumrstd_out, dx, dprev_mean, dprev_var, dgamma, dbeta)
```

An input a mode does not take is an empty tensor (`torch.empty(0)`); an output a mode does not produce is a
one-element tensor that is left as given. `T` is the dtype of `x` (bf16, fp16 or fp32).

| Argument | Shape, dtype | mode 0 (inference forward) | mode 1 (training forward) | mode 2 (backward) |
|---|---|---|---|---|
| `mode` | int | 0 | 1 | 2 |
| `num_groups` | int | K, a divisor of H | K | K |
| `beta1`, `beta2`, `eps` | float | decays and epsilon | same | same (`eps` is not read: `cumrstd` carries it) |
| `memory_efficient` | bool | ignored | ignored | `x_or_y` holds `y` |
| `x_or_y` | `[B, L, H]`, T | x | x | x, or y when `memory_efficient` |
| `bos_mask` | `[B, L]`, bool, or empty | bos tokens | same | same |
| `padding_mask` | `[B, L]`, bool, or empty | padded steps | same | same |
| `prev_count` | `[B]`, int64 | count before step 0 (zeros: fresh state) | same | same |
| `prev_mean`, `prev_var` | `[B, K]`, T | state before step 0 (zeros: fresh state) | same | not read |
| `gamma`, `beta` | `[H]`, T | scale and shift | same | same (`beta` is read when `memory_efficient`) |
| `dy` | `[B, L, H]`, T | empty | empty | gradient of y |
| `dmean`, `dvar` | `[B, K]`, T | empty | empty | gradient of the final mean and var |
| `cummean`, `cumrstd` | `[B, L, K]`, fp32 | empty | empty | the statistics saved by mode 1 |
| `y` | `[B, L, H]`, T | written | written | placeholder |
| `count` | `[B]`, int64 | final count | final count | placeholder |
| `mean`, `var` | `[B, K]`, T | final mean and var | final mean and var | placeholder |
| `cummean_out`, `cumrstd_out` | `[B, L, K]`, fp32 | placeholder | written | placeholder |
| `dx` | `[B, L, H]`, T | placeholder | placeholder | written |
| `dprev_mean`, `dprev_var` | `[B, K]`, T | placeholder | placeholder | written |
| `dgamma`, `dbeta` | `[H]`, T | placeholder | placeholder | written |

The placeholders and the empty tensors of the table keep the call signature the same in every mode. Every tensor must be
contiguous and on the same GPU. `y`, `dx`, `mean`, `var`, the gradients and the state have dtype `T`; `count` and
`prev_count` are int64. A padded step writes `y = 0` and `dx = 0`.

## Dispatch, shapes, dtypes and limits

`run` picks the kernel from the mode and the shape. A path that does not take a shape declines before it writes
anything, and the next one takes the call.

| Call | Path | Takes | Declines |
|---|---|---|---|
| forward, `L == 1` | `decode_fwd.cu`: one launch, one warp per (row, group) | bf16, fp16, fp32; any `G`; `B <= 65535`; contiguous operands | other layouts or dtypes |
| forward, `L >= 2` | `fwd_persistent.cu`: one persistent launch | `G == 64`; bf16 or fp16; 16-byte aligned `x`, `y` and statistics | any other shape; during CUDA graph capture; more than 8 live streams |
| forward, anything else | `fwd_general.cu`: at most two launches | any `L >= 1`, any `K`, `G <= 256`, bf16, fp16, fp32 | `G > 256` raises an error |
| backward | `bwd_fused.cu` with `bwd_prep.cu`: a prep launch, the fused kernel, a finalize launch | `G <= 256`; bf16, fp16, fp32; `B <= 65535`; `B * L * H < 2^31`; `cummean` fp32 `[B, L, K]` | other shapes; during CUDA graph capture; more than 8 live streams |
| backward, otherwise | `bwd_general.cu`: five launches | any `L`, `G <= 256`, `0 < beta < 1` | `G > 256` raises an error |

The tests cover lengths from 1 to 16384, 1 to 256 groups of 32 to 256 channels, bos and padding masks (including a bos
at every step), fresh and carried state (counts up to 2e5), nonzero gradients of the final state, and all three
dtypes. The memory-efficient backward is tested with groups of 64 channels. From a fresh state the backward
gradients of `prev_mean` and `prev_var` reach 1e4 to 1e5, which does not fit fp16; fp16 backward is tested from a
carried state.

`tests/cases.py` lists the tolerances. In short: bf16 outputs match the fp64 reference within a few units in the last
place, the statistics within 1e-3 relative, `count` exactly, and `dx` and the parameter gradients within the wider
tolerance of a long reduction.

The environment variable `TDN_BWD_T` overrides the number of steps per tile of the fused backward kernel (default: 16 to
64, chosen from the shape).

## Build

`src/` holds the sources: `binding.cpp` (the `run` entry point and the dispatch) and the kernels listed above. The
extension is built as a PyTorch C++/CUDA extension with these flags:

```
host:  -O3 -std=c++20
nvcc:  -O3 -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -lineinfo
       -gencode=arch=compute_XXXa,code=sm_XXXa          (the GPU's own target, e.g. sm_103a)
```

`tests/build.py` does this with `torch.utils.cpp_extension.load` and takes the target from
`torch.cuda.get_device_capability()`. The kernels use `cp.async`, vector `red.global.add`, programmatic dependent
launch and `ld.acquire`/`st.release` at GPU scope; they were built with CUDA 13 and verified on an NVIDIA B300
(compute capability 10.3). A full build takes about one minute.

## Design

- **Everything is an affine scan.** For one (row, group), a step maps the pair (mean, var) and the count by
  `v -> a * v + c`, with `a = beta` and `c = (1 - beta) * m_t` for a valid step, `a = 1, c = 0` for a padded step and
  `a = 0` for a reset. Such maps compose associatively, so a tile of steps is one aggregate and the state entering a
  tile is the composition of the aggregates of the earlier tiles. The backward pass is the same recurrence read
  backwards: the adjoint of the state obeys `R_t = alpha_{t+1} * R_{t+1} + C_t`.
- **Persistent single-pass forward (`fwd_persistent.cu`).** One launch with one wave of CTAs (three per SM). A tile is
  (one row, four groups, 128 steps) of `x`, streamed once into shared memory with `cp.async` and written back as `y`
  from there, so `x` is read once. The per-step moments are computed one step per lane (a whole 64-channel group
  per lane from a padded shared slot, so there is no bank conflict and no shuffle), the tile's aggregate comes from a
  warp scan over 32 chunks of four steps, and the state entering the tile from a decoupled look-back over the
  predecessors' 16-byte records (aggregate, then inclusive prefix). A record is read and written as one aligned
  16-byte access carrying its payload and an epoch tag, so a reader never sees half of an update and records need not
  be cleared between calls. The next tile's copy is issued slot by slot while `y` is written. The bias correction is
  carried as `1 - beta^count` (an affine recurrence of its own) instead of `1 - p` with `p = beta^count`, which would
  lose digits at small counts.
- **General forward (`fwd_general.cu`).** Two kernels at most: one reduces each (chain, time tile) to its affine
  aggregate, one composes the earlier tiles' aggregates, scans its tile and normalizes. With one tile per chain the
  first launch is skipped. Works for any group width, as a warp walks one chain and the reductions stay in a warp.
  The recurrence is written in delta form (`mean + (1 - beta) * (m - mean)`) with `1 - beta` rounded once from double
  on the host, and the bias correction is `-expm1(count * log(1 - (1 - beta)))`, so the increment and the correction
  use one constant.
- **Decode (`decode_fwd.cu`).** One launch, one warp per (row, group) (four groups per warp with 16-byte loads for
  16-bit groups of 64): everything that does not depend on the group reduction (state, gamma, beta, the count's bias
  correction and its reciprocals) is loaded or computed first so its latency overlaps the reduction.
- **Fused backward (`bwd_fused.cu`).** Tiles of 16 to 64 steps are dispatched last-first. Pass 1 reads `x` (or `y`) and
  `dy` through a four-deep `cp.async` ring, accumulates the per-step group sums, the `dgamma`/`dbeta` partials and the
  tile's aggregate; a warp polls a window of 32 successor tiles for their aggregates or inclusive adjoints
  (flags with release/acquire semantics and the call's epoch in their tag); pass 2 re-reads the tile from L2 through the
  same ring, scans in reverse and writes `dx` with evict-first stores. The memory-efficient variant recovers `x` from `y`
  per element with a reciprocal of `gamma` and keeps `beta` in shared memory, which holds the kernel to 64 registers.
  `dgamma`/`dbeta` are fp32 partials added with packed `red.global.add.v4.f32` and converted by a small launch that also
  clears them for the next call, using programmatic dependent launch to overlap its prologue. Flags and partials are
  kept per stream across calls, so no launch zeroes them.
- **Bias-correction prep (`bwd_prep.cu`).** The backward pass needs `1 / (1 - beta^count)` per step, which is not
  recoverable from the saved statistics. One launch computes it with a block scan over (reset, count) monoids; the count
  entering a tile is found by reading the masks 16 bytes at a time.
- **General backward (`bwd_general.cu`).** Five memory-bound launches (count prefix, per-step adjoints and tile
  aggregate, composition of the aggregates, the fused reverse scan with the element pass, the `dgamma`/`dbeta`
  reduction), for the shapes the fused kernel declines.

## Measured times

NVIDIA B300 (148 SMs, 2032 MHz, 126 MiB L2), bf16, groups of 64 channels, cold L2 cache. Times are GPU time from the
start of the first kernel of a call to the end of the last (the mean of three trials of 50 calls). The xllm column is the
cub backend for the forward and backward passes and the chunkwise backend for the memory-efficient backward (xllm
692bc25). Training: 4 x 8192 tokens; prefill: 16384 tokens from a fresh state; tail: 3616 tokens with the state carried
in, batch 8; decode: one token with the state carried in.

| Case (0.9B / 4B / 7.1B model width) | xllm 692bc25, ms | this implementation, ms | speedup |
|---|---|---|---|
| training forward | 0.776 / 1.157 / 1.720 | 0.0525 / 0.0793 / 0.1167 | 14.8x / 14.6x / 14.7x |
| training backward | 1.579 / 2.172 / 2.224 | 0.1071 / 0.1620 / 0.2425 | 14.7x / 13.4x / 9.2x |
| training backward, memory-efficient | 11.17 / 11.60 / 12.24 | 0.1192 / 0.1834 / 0.2761 | 94x / 63x / 44x |
| prefill, batch 1 | 0.366 / 0.564 / 0.851 | 0.0327 / 0.0480 / 0.0672 | 11.2x / 11.8x / 12.7x |
| prefill, batch 8 | 2.432 / 3.952 / 6.214 | 0.1668 / 0.2655 / 0.4183 | 14.6x / 14.9x / 14.9x |
| tail, batch 8 | 0.544 / 0.882 / 1.383 | 0.0445 / 0.0665 / 0.1029 | 12.2x / 13.3x / 13.4x |
| decode, batch 1 / 8 / 64 | 0.0135 to 0.0322 | 0.0024 to 0.0026 | 5x to 13x |

Geometric mean over the 27 cases: 0.404 ms for xllm, 0.0316 ms here (12.8x). `tests/benchmark.py` times the same cases
with CUDA events bracketing the call; it reports the same GPU work plus about 5 us of event and launch overhead per call,
which dominates the decode rows.

## Tests

`tests/` is self-contained: it needs PyTorch with CUDA, `nvcc` and `ninja`, and a GPU of compute capability 10.0 or
newer (the tests skip otherwise).

```bash
python -m pytest tests/test_timestep_decay_norm.py      # builds ../src, runs 116 tests
python tests/test_timestep_decay_norm.py [case ...]     # the same checks without pytest
python tests/benchmark.py [case ...]                    # ms per case, cold L2
```

- `tests/reference.py`: the fp64 reference, as a step-by-step recurrence and in closed form.
- `tests/cases.py`: the 27 model cases (training, prefill, tail, decode), the 88 edge and random sweep cases, the input
  generators and the tolerances.
- `tests/test_timestep_decay_norm.py`: builds the extension, runs every case in its mode and compares all outputs with
  the reference; one more test checks that the two forms of the reference agree.
- `tests/benchmark.py`: CUDA-event timing with the L2 flushed before every call.
- `tests/build.py`: the JIT build with the flags above.
