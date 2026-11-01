# Rotary position embedding

A fused CUDA kernel for xllm's rotary position embedding (`apply_rotary_embedding` and `apply_rotary_embeddings` in
`xllm/modules/rotary_positional_embedding.py`), forward and backward, for one tensor or for q and k together, reading
and writing the strided layouts xllm passes. One launch replaces the four to six ATen kernels of the PyTorch
implementation and moves each element once in and once out.

## What it computes

`x` has shape `[B, L, H, R]`, where `R` is the rotated width (the whole head, or its first `R` channels given as a
view). Read as `R / 2` complex numbers `z[b, l, h, j] = x[b, l, h, 2j] + i x[b, l, h, 2j + 1]` and with
`f[l, j] = freqs_cis[l, j] = cos(theta[l, j]) + i sin(theta[l, j])` (complex64), the output `y` of the same shape is

- forward: `y[b, l, h, j] = z[b, l, h, j] * f[l, j]`
- backward (the rotation of an incoming gradient by the conjugate): `y[b, l, h, j] = z[b, l, h, j] * conj(f[l, j])`

with `f` broadcast over `B` and `H`. The product is formed in fp32, as xllm does, with one fused multiply-add per
component, and rounded once (to nearest even) to the dtype of `x`:

```
y_re = z_re * c - z_im * s          (backward: z_re * c + z_im * s)
y_im = z_im * c + z_re * s          (backward: z_im * c - z_re * s)
```

When a second tensor `x2` (k, with its own head count `Hk`) is given, it is rotated by the same table in the same
launch.

## Interface

The module built from `src/` exposes one function. It follows destination passing: the caller allocates the outputs.

```python
run(conj, x1, x2, freqs_cis, out1, out2) -> None
```

| Argument | Type and shape | Meaning |
|---|---|---|
| `conj` | `bool` | `False`: forward. `True`: backward (rotation by the conjugate). |
| `x1` | CUDA tensor `[B, L, H, R]`, bf16, fp16 or fp32 | First tensor (q, or the only one). Any strides, provided the last one is 1. |
| `x2` | CUDA tensor `[B, L, Hk, R]`, same dtype and device as `x1`, or an empty tensor | Second tensor (k), rotated with the same table. Any strides with the last one 1. An empty tensor (`numel() == 0`) means a single tensor. |
| `freqs_cis` | CUDA complex64 tensor `[Lt, R / 2]` with `Lt >= L` | The table, one row per position. Rows `0 .. L - 1` are used. Any strides and any storage offset, so a slice of a longer table (for decode, starting at the token's position) is accepted. |
| `out1` | contiguous tensor, shape and dtype of `x1`, same device | Receives the rotated `x1`. |
| `out2` | contiguous tensor, shape and dtype of `x2`, same device | Receives the rotated `x2`. Not touched when `x2` is empty. |

Accepted layouts of `x1` and `x2` (anything with a last stride of 1 works; these are the ones that matter):

- contiguous `[B, L, H, R]`;
- the first `R` channels of wider heads, `x[..., :R]` of `[B, L, H, D]` (partial rotation);
- a slice of an interleaved tensor, such as `q` or `kq` out of `[B, L, H, 2, R]` by `unbind`;
- a gradient laid out `[B, H, L, R]` and viewed as `[B, L, H, R]` through `permute(0, 2, 1, 3)`.

`x1` and `x2` must agree in `B`, `L`, `R` and dtype. Outputs are always contiguous. The call runs on the current
stream of the device of `x1` and returns without synchronizing.

## Limits

- `R` is even (the rotated channels are `R / 2` complex pairs) and at most 2^31 - 1.
- `B <= 65535`, `L <= 2^31 - 1`, `H <= 2^31 - 1`, and `H + Hk <= 2^31 - 1`.
- dtypes: bf16, fp16, fp32 for `x1`, `x2`, `out1`, `out2`; the table is complex64.
- A call outside these limits raises an error instead of returning with the outputs unwritten.
- Pointers and strides need no alignment: a call whose pointers or strides are not multiples of 16 bytes takes a
  narrower vector (8 bytes, 4 bytes, or one element at a time) and stays correct.
- Wide rows (more than 2048 channels in bf16/fp16, 1024 in fp32, when the vector tier is the widest) and very large
  head counts take dedicated paths of the same file; they are correct but outside the shapes xllm rotates (32 to 256
  channels).

## Build

`src/` holds `binding.cpp` (the entry point and argument checks), `rope.h` and `rope.cu` (the kernels and their
launch). Build it as a PyTorch extension with these flags:

| | Flags |
|---|---|
| C++ | `-O3 -std=c++20` |
| CUDA | `-O3 -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -lineinfo` and the GPU's own `-gencode arch=compute_XXXa,code=sm_XXXa` (for example `compute_103a` on a B300) |

`tests/extension.py` shows the call to `torch.utils.cpp_extension.load` that does this with the GPU target read from
the device. The kernel targets NVIDIA Blackwell (tested on a B300, sm_103a).

## Design

- **One pass, one launch.** q and k (or a single tensor) are rotated by one kernel. A block handles one position and one
  batch index, so a position's table row is read once per thread and kept in registers for every row (head) the thread
  rotates. The row slots of a block are split between the two tensors, and `heads` counts fix how many slots each owns,
  so blocks do not carry idle slots when the head counts are small.
- **Vector loads, streaming stores.** A thread owns one 16-byte vector of a row and the cos/sin pairs of that vector.
  Consecutive lanes take consecutive vectors, so a warp reads and writes a row with coalesced 16-byte accesses
  (`__ldcs` loads, `__stcs` stores), which are not worth keeping in cache since each element is used once.
- **Rows in flight.** A thread takes four consecutive rows of one tensor and issues all four loads before any store, so
  four requests per thread are in flight: a single pass over memory needs that to keep the memory pipe full. Small
  calls (at most 8 MiB moved, as in decode) use one row per thread and many small blocks, since their time is the launch
  and one dependent read.
- **Strides as given.** Inputs are read at their own strides (views of wider heads, interleaved views, `[B, H, L, D]`
  gradients) and outputs are written contiguously; no copy is made. Everything is reduced on the host to raw pointers,
  byte strides and a dtype code, so the host path does no ATen work.
- **Vector tiers.** The widest vector, 16, 8 or 4 bytes, for which `R` is a multiple and every pointer and all six
  strides are aligned is chosen per call; the element-wise path remains for pointers with no alignment. A narrower tier
  keeps more bytes in flight than one element a lane would.
- **Block shape.** The block is `(R / VEC) x row slots x positions`. The number of positions per block is chosen so
  that the block is a whole number of warps (96 to 256 threads for xllm's shapes), by a doubling rule for small blocks
  and, for odd widths, a search for the position count that leaves the fewest idle lanes.
- **Wide rows and many heads.** A row of more than 256 vectors is covered by several blocks along the third grid
  dimension, again with four rows a thread so one table read serves four rows; the element tier and calls whose row
  slots exceed the grid's third dimension use a kernel that strides over rows and pairs and has no such limit.
- **Overflow-safe indexing.** Row counts, grid sizes and offsets that scale with `L`, `R` or the head counts are
  computed in 64 bits on the host and in the kernel where they can pass 2^31, and the limits above are checked up front.
- **Numerics.** fp32 throughout, `fmaf` for the products, one round-to-nearest-even conversion at the end, matching the
  rounding of xllm's `type_as`.

## Measured times

NVIDIA B300 (148 SMs), bf16, L2 flushed before every call, `tests/benchmark.py --torch`. "PyTorch" is
xllm's implementation (commit 692bc25): the tensor cast to fp32, viewed as complex, multiplied by the table, viewed as
real and cast back, 4 to 6 kernels. Shapes: 32 q heads and 8 k heads of 64 or 128 channels, 8 + 4 heads of 256 channels
rotated on the first 128, and 4 + 4 heads of 256 channels; training is 4 x 8192 tokens, prefill 8192 or 16384 tokens,
decode one token at position 20000 (batch 1 and 64). The event timer has about 1 us of granularity, which is most of the
decode figures.

| Case | PyTorch (ms) | Kernel (ms) | Speedup |
|---|---:|---:|---:|
| `train-h32k8d64-fwd` | 0.2940 | 0.0554 | 5.31x |
| `train-h32k8d64-bwd` | 0.2362 | 0.0451 | 5.23x |
| `decode-h32k8d64-b1` | 0.0114 | 0.0042 | 2.71x |
| `decode-h32k8d64-b64` | 0.0134 | 0.0042 | 3.20x |
| `train-h32k8d128-fwd` | 0.5781 | 0.1045 | 5.53x |
| `train-h32k8d128-bwd` | 0.4631 | 0.0840 | 5.51x |
| `decode-h32k8d128-b1` | 0.0114 | 0.0042 | 2.71x |
| `decode-h32k8d128-b64` | 0.0134 | 0.0042 | 3.20x |
| `train-h8k4d256r128-fwd` | 0.2572 | 0.0349 | 7.37x |
| `train-h8k4d256r128-bwd` | 0.1752 | 0.0247 | 7.11x |
| `decode-h8k4d256r128-b1` | 0.0134 | 0.0042 | 3.20x |
| `decode-h8k4d256r128-b64` | 0.0155 | 0.0042 | 3.69x |
| `train-h4k4d256-fwd` | 0.2407 | 0.0451 | 5.34x |
| `train-h4k4d256-bwd` | 0.1260 | 0.0266 | 4.73x |
| `decode-h4k4d256-b1` | 0.0114 | 0.0042 | 2.71x |
| `decode-h4k4d256-b64` | 0.0134 | 0.0042 | 3.20x |
| `prefill-h32k8d128-start8192` | 0.1465 | 0.0308 | 4.75x |
| `prefill-h32k8d128` | 0.2899 | 0.0554 | 5.24x |
| `prefill-h8k4d256r128` | 0.1281 | 0.0205 | 6.24x |
| `prefill-h4k4d256` | 0.1213 | 0.0267 | 4.54x |
| `unbind-q-h32k8d128-fwd` | 0.7796 | 0.1045 | 7.46x |
| `unbind-kq-h32k8d128-fwd` | 0.6619 | 0.0840 | 7.88x |
| `bhld-h32k8d128-bwd` | 0.6708 | 0.0861 | 7.79x |
| **geometric mean** | 0.0963 | 0.0205 | 4.69x |

## Tests

`tests/` depends only on PyTorch and the CUDA compiler. Each test builds the extension from `src/` with
`torch.utils.cpp_extension.load` and skips cleanly without a CUDA device of compute capability 10.0 or above.

```bash
python -m pytest tests/test_rope.py     # 23 shapes and 96 sweep cases, forward and backward, against the fp64 reference
python tests/test_rope.py               # the same without pytest
python tests/benchmark.py --torch       # milliseconds per call, kernel and PyTorch
```

- `tests/reference.py`: the reference, computed in fp64 and rounded once to the dtype of `x`.
- `tests/cases.py`: the shapes, layouts, inputs and tolerances. The shapes are those xllm's models produce; the sweep is
  a fixed pseudo-random draw over batches 1 to 9, lengths 1 to 1027, 1 to 40 heads (k heads differing from q's), widths
  32 to 256 rotated in full or in part, every layout above, table slices starting anywhere up to 100000, and bf16,
  fp16 and fp32.
- A case passes when at least 99.999% of the output elements are within `|out - ref| <= 0.016 + 0.016 |ref|`, there is
  no NaN or Inf, and the output is not all zeros where the reference is not.
