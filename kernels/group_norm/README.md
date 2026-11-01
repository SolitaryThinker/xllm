# Group RMS norm and group layer norm

CUDA kernels for xllm's group RMS norm and group layer norm, forward and backward, for any number of groups. They
target NVIDIA Blackwell GPUs (compute capability 10.0 and higher) and serve the activation shapes of xllm's example
models at close to memory bandwidth: the query and key norms (one or two groups per attention head), the
ortho residual norm (one group per residual head) and the residual-stream norms (one to four groups).

## What it computes

The last dimension of `x [B, S, C]` is split into `G` groups of `cpg = C / G` contiguous channels. Each group of each
row (`R = B * S` rows) is normalized on its own, then scaled by `gamma` and, for the layer norm, shifted by `beta`.
All sums are accumulated in fp32; the statistics are stored in fp32.

Forward:

```
RMS norm     r = rsqrt(mean(x^2) + eps)                       y = x * r * gamma
layer norm   mu = mean(x), r = rsqrt(mean((x - mu)^2) + eps)  y = (x - mu) * r * gamma + beta
```

`rstd = r` is written as `[R, G]` fp32, and for the layer norm `mean = mu` as `[R, G]` fp32. Without an affine
transform `gamma = 1` and `beta = 0`. The RMS norm has no mean and no beta (eps is 1e-6 for the RMS norm and 1e-5 for
the layer norm in the tests).

Backward, from `dy`, the saved statistics and `x`:

```
xhat    = (x - mu) * r                      (RMS norm: x * r)
dxhat   = dy * gamma
dx      = r * (dxhat - xhat * mean(dxhat * xhat) - mean(dxhat))      (RMS norm: no `- mean(dxhat)` term)
dgamma  = sum over rows of dy * xhat        dbeta = sum over rows of dy      (layer norm only)
```

With `memory_efficient` the backward pass is given `y` instead of `x` and recovers `xhat = (y - beta) / gamma` (layer
norm) or `xhat = y / gamma` (RMS norm); without an affine transform `xhat = y`. The backward kernels read `x_or_y` and
`dy` once and write `dx` once (3 tensors of traffic, where xllm's own backward moves 5); `dgamma` and `dbeta` are
reduced across rows and CTAs in fp32 and rounded once to the output dtype.

## Interface

`src/binding.cpp` exports one function, `run`, which performs one call of any of the eight operations (forward or
backward, RMS norm or layer norm, with or without the affine transform). It is called positionally; the outputs are
passed in and written in place, and `run` returns `None`.

| # | Argument | Type | Meaning |
|---|---|---|---|
| 1 | `mode` | int | 0 forward, 1 backward |
| 2 | `norm` | int | 0 RMS norm, 1 layer norm |
| 3 | `num_groups` | int | `G`; `C` must be a multiple of it |
| 4 | `eps` | float | added to the variance (or mean square) |
| 5 | `memory_efficient` | bool | backward only: `x_or_y` is `y` rather than `x` |
| 6 | `x_or_y` | CUDA tensor `[B, S, C]` | `x` (forward, and backward unless `memory_efficient`), else `y`; bf16, fp16 or fp32 |
| 7 | `dy` | tensor `[B, S, C]` | backward only; empty tensor in the forward pass |
| 8 | `mean` | fp32 tensor `[R, G]` | backward of the layer norm from `x` only; otherwise empty |
| 9 | `rstd` | fp32 tensor `[R, G]` | backward only; empty in the forward pass |
| 10 | `gamma` | tensor `[C]`, dtype of `x` | affine scale; empty tensor for no affine transform |
| 11 | `beta` | tensor `[C]`, dtype of `x` | layer norm with an affine transform only; otherwise empty |
| 12 | `y` | output `[B, S, C]` | forward; a one-element placeholder in the backward pass |
| 13 | `mean_out` | output fp32 `[R, G]` | forward of the layer norm; a one-element placeholder otherwise |
| 14 | `rstd_out` | output fp32 `[R, G]` | forward; a one-element placeholder in the backward pass |
| 15 | `dx` | output `[B, S, C]` | backward; a one-element placeholder in the forward pass |
| 16 | `dgamma` | output `[C]` | backward with `gamma`; a one-element placeholder otherwise |
| 17 | `dbeta` | output `[C]` | backward of the layer norm with `gamma`; a one-element placeholder otherwise |

An input a call does not take is an empty tensor (`torch.empty(0)`), and an output it does not produce is a
one-element tensor that `run` leaves untouched. `tests/cases.py` builds the arguments of every case.

## Build

All sources in `src/` are compiled into one extension (`binding.cpp` and every `*.cu`). The flags:

| | Flags |
|---|---|
| C++ | `-O3 -std=c++20` |
| CUDA | `-O3 -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -lineinfo` and the GPU's own feature-specific target, `-gencode=arch=compute_XXXa,code=sm_XXXa` (for example `compute_100a` on compute capability 10.0) |

The feature-specific `a` target is required: the TMA backward uses `cp.async.bulk` and mbarrier instructions.
`tests/extension.py` shows the call to `torch.utils.cpp_extension.load` that builds it just in time.

## Supported shapes, dtypes and limits

- `x_or_y` is a 3-dimensional contiguous CUDA tensor `[B, S, C]`; `num_groups >= 1` and `C % num_groups == 0`.
- Any number of groups and any `cpg` up to 8192 are served. Fast kernels cover bf16 and fp16 with `cpg` a multiple of 8 and
  several also fp32; every other call, such as non-contiguous or unaligned
  tensors, falls back to xllm's own kernels (`group_rms_norm_kernel.cu`, `group_layer_norm_kernel.cu`), which are
  correct for all shapes but slow when groups are small and numerous.
- The layer norm with `gamma` needs `beta`.
- The statistics are fp32 and have the shape `[R, G]`.
- Tested: 1 to 256 groups, `cpg` from 24 to 8192, 1 to 262144 rows, bf16 / fp16 / fp32, with and without the affine
  transform and `memory_efficient`.

## Design

A dispatcher (`dispatch.cu`) holds a table of kernels. Each registers a predicate that says which shapes it takes and a
priority; `run` tries the kernels that take the call from the highest priority down, and a kernel may decline at run
time (a misaligned or non-contiguous tensor, a case it does not instantiate), after which the next one is tried and
finally xllm's own kernels. The environment variable `GN_IMPL=off` sends every call to xllm's kernels, and
`GN_IMPL=fwd:<name>,bwd:<name>` restricts the candidates to one kernel per direction.

Techniques shared by the kernels:

- A group is a contiguous run of `cpg` channels, so the tensor is processed as one flat array of `R * G` groups; no
  `(row, group)` grid is launched. A group of up to 32 sixteen-byte vectors is spread over that many lanes of a warp (the
  count need not be a power of two), so a warp holds several whole groups and the group statistics are shuffle
  reductions: no shared memory and no `__syncthreads` on the main path. A wider group is one warp, or one CTA with a
  reduction through shared memory.
- All loads are 16 bytes a lane, held in registers between the reduction and the scale, so the input is read once and
  the output written once; `__ldcs` / `__stcs` streaming hints keep the data from polluting the cache.
- Warps are column-stationary where they can be: a warp keeps a fixed set of channels while it walks rows, so `gamma`
  and `beta` stay in registers, and the backward pass accumulates `dgamma` / `dbeta` in registers over its rows. The
  partials are combined once per CTA or row chunk into an fp32 workspace (atomics, or one writer per element and a
  finalize kernel) and rounded once to the output dtype.
- The layer norm computes the mean first and the variance from the centered values, so the statistics match the
  two-pass definition. `memory_efficient`, the affine flag and the norm are template parameters, so no kernel
  carries registers for a mode it does not run.
- The number of rows in flight per warp, the grid length per SM and the mapping are chosen per group width.

The kernels:

| Kernel (`GN_IMPL` name) | Direction | Takes | Technique |
|---|---|---|---|
| `dense_lanes_nonaffine` (`dense_lanes.cu`) | fwd, bwd | calls without `gamma`; groups of 1 to 32 vectors | a group takes exactly as many lanes as it has vectors (3, 5 and 10 vectors keep every lane busy); with an even vector count two chunks a lane halve the lanes per group |
| `bwd_tma` (`bwd_tma.cu`) | bwd | bf16, `gamma` from `x`, groups of whole 16-byte chunks up to 8192 channels | one persistent CTA per SM; a producer warp streams row tiles with `cp.async.bulk` into an mbarrier ring, consumer warps own fixed columns; tiles are handed out from a global counter so CTAs end together; `dgamma` / `dbeta` partials are combined across the grid in a fixed order (bit-reproducible) |
| `fwd_flat` (`fwd_flat.cu`) | fwd | any width (scalar fallback for widths that are not whole chunks) | grid-stride over flat groups, `TPG` lanes of `Q` chunks each, full or half lane mapping; the layer norm keeps its converted values in registers |
| `fwd_by_width`, `bwd_by_width` (`*_by_width.cu`) | fwd, bwd | bf16 / fp16; groups wider than 256 channels, and 40 and 80 channels | mapping by width: warp per group, CTA per group with a shared-memory reduction; backward in one pass with one vector atomic (`red.global.add.v4.f32`) per CTA for `dgamma` / `dbeta` |
| `fwd_lane_groups` (`fwd_lane_groups.cu`) | fwd | groups of 3 to 16 vectors | lane groups with `gamma` / `beta` in registers, two rows an iteration |
| `wide_odd_fwd`, `wide_odd_bwd` (`wide.cu`, `odd.cu`) | fwd, bwd | wide groups (512 channels and more); widths whose vector count is not a power of two | CTA per group with two barriers a row; lanes at a segment base with a shift-loop reduction |
| `bwd_one_pass` (`bwd_one_pass.cu`) | bwd | 2-byte dtypes, whole 16-byte vectors | `dx`, `dgamma`, `dbeta` from one read; column-stationary warps (small groups) or CTAs (wide groups); row-chunk workspace and a finalize kernel |
| `dense_lanes_affine_bwd` (`dense_lanes.cu`) | bwd | affine, groups of 1 to 32 vectors | fixed column slot per warp, `dgamma` / `dbeta` in registers |
| `fwd_floor` (`fwd_floor.cu`) | fwd | shape-general fallback for 2-byte dtypes | flat walk with a fixed grid; used only where no kernel above takes the call |

## Measured times

Median time of one call, timed with CUDA events and the L2 cache flushed (`tests/benchmark.py`), on an NVIDIA B300 (148 SMs, SM clock 2032 MHz), bf16.
The baseline is xllm's own kernels (xllm 692bc25) with their restriction to at most 32 groups removed, which is what
`GN_IMPL=off` runs.

| Case | Groups x channels per group | Rows | xllm kernels (ms) | This implementation (ms) | Speedup |
|---|---|---|---|---|---|
| rms-q64c64-fwd | 64 x 64 | 262144 | 8.6436 | 0.6513 | 13.3x |
| rms-q64c64-bwd | 64 x 64 | 262144 | 13.1892 | 0.9410 | 14.0x |
| rms-q64c128-fwd | 64 x 128 | 65536 | 2.1667 | 0.3225 | 6.7x |
| rms-q64c128-bwd | 64 x 128 | 65536 | 3.2236 | 0.4747 | 6.8x |
| ln-q64c128-fwd | 64 x 128 | 65536 | 2.1668 | 0.3471 | 6.2x |
| ln-q64c128-bwd | 64 x 128 | 65536 | 3.2717 | 0.4834 | 6.8x |
| rms-o64c24-fwd | 64 x 24 | 262144 | 8.6436 | 0.2611 | 33.1x |
| rms-o64c24-bwd | 64 x 24 | 262144 | 8.6455 | 0.3758 | 23.0x |
| rms-o64c40-fwd | 64 x 40 | 65536 | 2.1668 | 0.1178 | 18.4x |
| rms-o64c40-bwd | 64 x 40 | 65536 | 2.1668 | 0.1649 | 13.1x |
| rms-o64c64-fwd | 64 x 64 | 65536 | 2.1668 | 0.1679 | 12.9x |
| rms-o64c64-bwd | 64 x 64 | 65536 | 2.1668 | 0.2427 | 8.9x |
| rms-b1c1536-fwd | 1 x 1536 | 262144 | 0.5928 | 0.2407 | 2.5x |
| rms-b1c1536-bwd | 1 x 1536 | 262144 | 5.1344 | 0.3615 | 14.2x |
| rms-b2c1280-fwd | 2 x 1280 | 65536 | 0.2974 | 0.1075 | 2.8x |
| rms-b2c1280-bwd | 2 x 1280 | 65536 | 1.5310 | 0.1589 | 9.6x |
| rms-b4c1024-fwd | 4 x 1024 | 65536 | 0.3042 | 0.1648 | 1.8x |
| rms-b4c1024-bwd | 4 x 1024 | 65536 | 1.6099 | 0.2448 | 6.6x |
| rms-b1c4096-fwd | 1 x 4096 | 65536 | 0.3185 | 0.1670 | 1.9x |
| rms-b1c4096-bwd | 1 x 4096 | 65536 | 1.6600 | 0.2509 | 6.6x |
| rms-q32c128-fwd | 32 x 128 | 65536 | 1.0866 | 0.1649 | 6.6x |
| rms-q32c128-bwd | 32 x 128 | 65536 | 2.2334 | 0.2449 | 9.1x |
| rms-k8c128-fwd | 8 x 128 | 65536 | 0.2774 | 0.0481 | 5.8x |
| rms-k8c128-bwd | 8 x 128 | 65536 | 0.2775 | 0.0686 | 4.0x |
| rms-o32c80-fwd | 32 x 80 | 65536 | 1.0865 | 0.1075 | 10.1x |
| rms-o32c80-bwd | 32 x 80 | 65536 | 1.0875 | 0.1548 | 7.0x |
| geomean | | | 1.6814 | 0.2186 | 7.7x |

## Tests

`tests/` depends on nothing outside this directory except PyTorch with CUDA and a compute capability 10.0+ GPU; it is
skipped otherwise.

```
python -m pytest tests/test_group_norm.py     # or: python tests/test_group_norm.py [name-substring ...]
python tests/benchmark.py [name-substring ...] [--sweep]   # per-case milliseconds; GN_IMPL=off for the baseline
```

- `tests/test_group_norm.py` builds the extension from `src/` and runs every case of `cases.shape_cases()` (the shapes
  of xllm's example models) and `cases.sweep()` (group counts 1 to 256, `cpg` 24 to 8192, row counts 1 to 4097,
  fp32 / fp16 / bf16, both norms, forward and backward, affine and not, `memory_efficient`), comparing all six outputs
  with the fp64 reference of `tests/reference.py`.
- `tests/cases.py` defines the shapes, the inputs and the tolerances (bf16: `atol 2e-2`, `rtol 1.6e-2`; the fp32
  statistics `atol 1e-5`, `rtol 1e-4`).
- `tests/benchmark.py` times each case with CUDA events.
