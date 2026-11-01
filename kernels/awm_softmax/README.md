# awm_softmax

CUDA kernels for the chunked softmax and logsumexp over time of xllm's Gekko adaptive working memory (AWM): the
forward pass, the backward pass and the decode-time update. The key is read in its physical `[B, L, K, S]` layout (xllm
views it as `[B, K, L, S]`), transposed once through shared memory, and the softmax and logsumexp of each chunk of time
are computed in one pass. The kernels replace the ATen sequence of xllm 692bc25 (a cast to fp32, a transpose, a
`masked_fill`, `softmax`, a cast back, nine kernels of `logsumexp`, and for the backward pass a recomputed softmax,
`_softmax_backward_data` chunk by chunk, the logsumexp's backward and a `masked_fill`), which moves 20-40x the bytes.

```
src/    awm_softmax_binding.cpp   entry point `run`, choice of kernel and launch geometry, ATen path
        awm_chunk_softmax.cu/.h   forward, decode and the fused one-pass backward kernels
        awm_two_pass_bwd.cu       the two-pass backward kernels
tests/  test_awm_softmax.py       correctness against the fp64 reference, every case, every mode
        cases.py                  shapes, inputs, destinations and tolerances
        reference.py              the fp64 reference
        build.py                  JIT build of ../src
        benchmark.py              per-case timing
```

## What is computed

C is `chunk_size`, `L = N * C` the length (a multiple of the chunk), `t` a step and `c = t // C` its chunk. For each row
`b`, kv head `k` and channel `s`, the key is taken along time:

    x[b, k, s, t] = key[b, t, k, s]            (-inf where key_mask[b, t] is True)

**Mode 0, forward** (`_adaptive_working_memory_fwd` and the same lines of `_adaptive_working_memory_accum_fwd`):

    kkey[b, k, s, t]    = exp(x[t]) / sum_{t' in chunk c} exp(x[t'])        in key's dtype
    lse_out[b, k, s, n] = log sum_{t in chunk n-1} exp(x[t])                n = 1 .. N-1, fp32

With `prev_key [B, C, K, S]` (the previous call's last chunk, never masked), also
`prev_kkey = softmax over time of prev_key` in `[B, K, S, C]` and `lse_out[..., 0] = logsumexp over time of prev_key`.
Without it, slot 0 of `lse_out` is not written (xllm stores -inf there; the caller does). The last chunk's logsumexp
is not computed, so with `N = 1` `lse_out` holds slot 0 only.

**Mode 1, backward** (`_adaptive_working_memory_bwd` on the fp32 softmax that the forward recomputes): with `y` the
softmax above, `g = dkkey` and `d = dlse` of chunk `c = 0 .. N-2`:

    dkey[b, t, k, s] = y[t] * (g[t] - sum_{t' in chunk c} g[t'] y[t'])  +  d[c+1] * exp(x[t] - lse[c+1])

summed in fp32 and rounded once to key's dtype. `dkey` is zero in the last chunk and at masked positions. These zeros
are not written by the fast path: **`dkey` must arrive zeroed** (a buffer that holds other values there keeps them).
Slot 0 of `lse` and `dlse` belongs to `prev_key` and is not read.

**Mode 2, decode** (`update_memory` at a chunk boundary): `key [B, K, C, S]` is one finished chunk, unmasked;
`kkey[b, k, s, j] = softmax_j(key[b, k, j, s])` and `lse_out[b, k, s] = logsumexp_j(key[b, k, j, s])`.

## Interface

One entry point, compiled as a PyTorch extension module:

    run(mode, chunk_size, key, key_mask, prev_key, lse, dkkey, dlse, kkey, lse_out, prev_kkey, dkey) -> None

The call is positional and destination-passing: the last four arguments are written in place. All tensors are
contiguous CUDA tensors on one device; the shape given is the memory layout. `e` is key's dtype (bf16, fp16 or fp32).
An input a mode does not take is an empty tensor (`torch.empty(0)`); an output it does not produce is a one-element
tensor that is left unchanged.

| argument | mode 0 (forward) | mode 1 (backward) | mode 2 (decode) |
|---|---|---|---|
| `mode` (int) | 0 | 1 | 2 |
| `chunk_size` (int) | C | C | C (= key.shape[2]) |
| `key` | in `[B, L, K, S]` e | in `[B, L, K, S]` e | in `[B, K, C, S]` e |
| `key_mask` | in `[B, L]` bool, True = -inf; or empty | in `[B, L]` bool; or empty | empty |
| `prev_key` | in `[B, C, K, S]` e; or empty | empty | empty |
| `lse` | empty (fp32) | in `[B, K, S, N]` fp32 (mode 0's `lse_out`) | empty (fp32) |
| `dkkey` | empty (e) | in `[B, K, S, L-C]` e (gradients of chunks 0 .. N-2) | empty (e) |
| `dlse` | empty (fp32) | in `[B, K, S, N]` fp32 | empty (fp32) |
| `kkey` | out `[B, K, S, L]` e | one element, unchanged | out `[B, K, S, C]` e |
| `lse_out` | out `[B, K, S, N]` fp32 (slot 0 only with `prev_key`) | one element (fp32), unchanged | out `[B, K, S]` fp32 |
| `prev_kkey` | out `[B, K, S, C]` e with `prev_key`; else one element, unchanged | one element, unchanged | one element, unchanged |
| `dkey` | one element, unchanged | out `[B, L, K, S]` e, must arrive zeroed | one element, unchanged |

## Build

`tests/build.py` builds the extension from `src/` with `torch.utils.cpp_extension.load`. The flags:

    host:  -O3 -std=c++20
    nvcc:  -O3 -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -lineinfo
           -gencode=arch=compute_100a,code=sm_100a        (the GPU's own compute target, with its `a` features)

The sources are `src/awm_softmax_binding.cpp`, `src/awm_chunk_softmax.cu` and `src/awm_two_pass_bwd.cu`
(`awm_chunk_softmax.h` is included by the first two); the include path is `src/`. The kernels use thread block
clusters with distributed shared memory and programmatic dependent launch, and are built and tested for Blackwell
(compute capability 10.x).

## Supported shapes, dtypes and limits

- Dtypes: key is bf16, fp16 or fp32 (`kkey`, `prev_kkey` and `dkey` in the same dtype; `lse_out` is fp32). Other
  dtypes are not supported.
- The length is a multiple of `chunk_size`. Batch, kv heads and chunk count are free.
- Forward and decode fast path: `S` a multiple of 8, and `C / P` a multiple of 256 (2-byte key) or 128 (4-byte key),
  where P is the number of CTAs the chunk is split over (1 unless the chunk is long, see Design); the tile of
  `ST x (C/P + 8)` elements plus the combine scratch must fit 199 KiB of shared memory. A whole chunk fits up to about
  C = 12.7k (2-byte key, 8 channels) or 6.3k (4-byte key); longer chunks are split over 2, 4 or 8 CTAs of a cluster
  where the slice meets the same multiple (`sp_ok` and `pick_sp` in `awm_softmax_binding.cpp` list the exact cells).
  The tests cover chunk sizes from 64 to 8192.
- Backward fast path: `S` a multiple of 16 and `N >= 2`, any `C` (the two-pass kernels). For `S` a multiple of 8 but
  not of 16, the fused one-pass kernel takes `C` a multiple of 256 whose tile fits shared memory. With `N = 1` the
  backward pass has nothing to compute and writes nothing.
- **ATen path.** Any call the fast paths do not cover (for example `C` not a multiple of 256, `S` not a multiple of 8,
  or a tile that does not fit) is computed by xllm's own ATen operations, which are kept in
  `awm_softmax_binding.cpp` (`fwd_ref`, `bwd_ref`, `decode_ref`). The result is the same up to rounding; the time is
  that of the ATen sequence. The dispatch uses it, so the code is part of the artifact.
- Masks: any bool mask is correct. The kernels are fastest when each chunk's mask is a masked prefix followed by an
  unmasked suffix (packed documents), and skip the masked prefix. The forward pass is defined for masks that keep at
  least one position per chunk row (xllm's masks never mask a chunk's last position).
- Tolerances the tests apply (against the fp64 reference), at C = 4096: `kkey` and `prev_kkey` atol 1e-6, rtol 1.6e-2,
  99.99% of elements; `lse_out` atol 1e-4, rtol 1e-5, all elements; `dkey` atol 4e-6, rtol 1.6e-2, 99.9%. The absolute
  tolerances scale with `4096 / C`.

## Design

**Forward and decode** (`fwd_kernel`). One CTA of 512 threads per (batch row, kv head, block of ST channels, chunk). The
CTA loads its tile of `ST` channels by `C` steps from `[B, L, K, S]` with 16-byte loads (four rows in flight per thread
to cover memory latency) and stores it into shared memory transposed, time innermost, in rows padded by 8 elements and
with the 8-element block index of the time XORed with a per-channel swizzle, so that the transposing stores and the
vector reads are free of bank conflicts. A warp then takes one channel at a time and runs an online (max, sum) over the
row with `ex2.approx` and one FFMA per element, merges its lanes with shuffles, and the same warp writes the softmax
from the tile in 16-byte stores, 512 bytes per warp along time, into `kkey [B, K, S, L]`. The logsumexp is
`m + log(s)`.
- Masked steps load nothing (their 32-byte sector is needed by no other CTA) and enter the tile as -inf. The CTAs
  detect a masked prefix with a block-wide min/max of the first unmasked and last masked step; the reduce and the write
  then start at the first vector round of the suffix and the write emits zeros for the prefix without reading the tile.
- Long chunks are cut into P slices along time, one per CTA of a thread block cluster. Each rank holds `ST x C/P` in
  shared memory, which allows a wide channel block with a small tile (three CTAs per SM by shared memory, two by
  registers). The ranks publish their (max, sum) pairs into every rank's copy through distributed shared memory and
  combine locally after a single `cluster.sync`.
- The previous chunk (`prev_key`) is a second plane of the grid (`blockIdx.y`) of the same launch; decode is the same
  kernel on a `[B, K, C, S]` key.
- The (ST, P) cell is chosen per call: (16, 2) for unmasked forward and decode at `C >= 4096`, (8, 1) for masked
  forward, (16, 1) and (8, 1) otherwise by grid size, (32, 4) as a fallback, and (8, 2), (8, 4), (8, 8) for chunks
  whose whole-chunk tile does not fit shared memory. Tile addresses are incremented rather than recomputed where the
  register budget allows, and the exponent fold is disabled in the narrow split cells to keep the kernels at the
  register count that holds their CTAs per SM.

**Backward** (`awm_dot`, `awm_write`). `lse` is an input, so `y = exp2(x*log2(e) - lse*log2(e))` is one `ex2` per
element and nothing is recomputed. The dot `sum g*y` runs along time, so two streaming kernels with no barrier between
CTAs carry it: `awm_dot` makes one pass over key and dkkey and writes `nc` partial dots per channel to a scratch, and
`awm_write` reads the partials (summed in shared memory), passes over key and dkkey again from L2, and writes `dkey`
once, rounded once from fp32. A CTA owns 16 channels (32 bytes of a key row) of a piece of a chunk; a thread owns 16
bytes of key (8 channels at 2 bytes) of one step and loads dkkey with time contiguous across a warp. The mask is tested
once per turn of the loop, warp-uniformly, so a turn of masked steps costs two mask bytes. `awm_write` is launched with
programmatic dependent launch: its blocks are resident while `awm_dot` drains, take the `lse` and `dlse` reads before
`griddepcontrol.wait`, and wait only before reading the partials. The number of pieces `nc` targets about 1024 CTAs
(2048 for batched shapes), the kernels are instantiated for 8 or 9 resident CTAs per SM, and the 9-CTA (56 register)
instantiation is chosen only when the grid exceeds one wave of the device. For a channel count that is not a multiple
of 16, a fused one-pass kernel (`bwd_kernel`) keeps the chunk tile in shared memory, computes the dot from it and
writes dkey transposed back through the tile.

## Measured times

GPU time of one call, NVIDIA B300 (148 SMs, SM clock 2032 MHz), bf16 key, `chunk_size` 4096, L2 flushed before every
call, mean of three trials of 50 calls. The reference is xllm 692bc25's PyTorch/ATen implementation measured the same
way (library kernels only). `K x S = 4 x 256`; `8 x 128` is within 1% of it in both columns.

| case | xllm ATen (ms) | this kernel (ms) | speedup |
|---|---|---|---|
| train_fwd_4x256_b1_l8192 | 0.1316 | 0.0097 | 13.6x |
| train_fwd_4x256_b4_l8192 | 0.4388 | 0.0327 | 13.4x |
| train_fwd_4x256_b1_l16384 | 0.2409 | 0.0201 | 12.0x |
| train_bwd_4x256_b1_l8192 | 0.1811 | 0.0139 | 13.0x |
| train_bwd_4x256_b4_l8192 | 0.7121 | 0.0305 | 23.3x |
| train_bwd_4x256_b1_l16384 | 0.4645 | 0.0204 | 22.8x |
| prefill_4x256_b1_l16384 | 0.1902 | 0.0242 | 7.9x |
| prefill_4x256_b1_l16384_prev | 0.2662 | 0.0299 | 8.9x |
| prefill_4x256_b1_l4096_prev | 0.1047 | 0.0128 | 8.2x |
| prefill_4x256_b8_l16384 | 1.3805 | 0.1460 | 9.5x |
| decode_4x256_b1 | 0.0757 | 0.0088 | 8.6x |
| decode_4x256_b8 | 0.4096 | 0.0406 | 10.1x |

The geometric mean of the speedups is 11.8x. `train_*` are the training calls (mask of packed documents masking 57-66%
of the keys), `prefill_*` are generate's calls without a mask, `_prev` with the previous chunk's key, `decode_*` is the
update at a chunk boundary. `tests/benchmark.py` times each call from CUDA event to CUDA event, which adds the launch
latency (about 7 us for the small cases) to these numbers.

## Tests

    python -m pytest tests/test_awm_softmax.py       # or: python tests/test_awm_softmax.py
    python tests/benchmark.py [--reps 50] [--cases substring ...]

The tests build the extension (JIT, a minute or so), skip when there is no CUDA GPU of compute capability 10.0 or later,
and run 24 main shapes (both head shapes of the Gekko models, every mode) and an 86-case sweep (chunk sizes 64 to 8192,
1 to 5 chunks, batches 1 to 3, 1 to 16 kv heads of 16 to 256 channels, no mask, sparse, dense and densest masks, a
chunk masked whole, with and without the previous chunk, bf16, fp16 and fp32) against the fp64 reference in
`tests/reference.py`, at the tolerances above.
