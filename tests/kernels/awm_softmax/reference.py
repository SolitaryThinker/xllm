"""Reference for the chunked softmax and logsumexp of xllm's Gekko adaptive working memory, forward and backward, in
fp64 and vectorized (no loop over time or chunks), with each output rounded once.

Notation: key's dtype e; C = chunk_size; L = N * C (the length is a multiple of the chunk); t the step, c = t // C its
chunk. For each row b, kv head k and channel s, the key is taken along time, in fp64:

    x[b, k, s, t] = key[b, t, k, s]                 (-inf where key_mask[b, t])

mode 0 (FWD), xllm's _adaptive_working_memory_fwd (and the same lines of _adaptive_working_memory_accum_fwd):

    kkey[b, k, s, t]     = exp(x[t]) / sum_{t' in chunk c} exp(x[t'])               rounded to e
    lse_out[b, k, s, n]  = log sum_{t in chunk n - 1} exp(x[t])                       for n = 1 .. N-1, in fp32
    with prev_key [B, C, K, S] (the previous call's last chunk, never masked):
        p[b, k, s, j]        = prev_key[b, j, k, s]
        prev_kkey            = softmax_j(p)                                           rounded to e
        lse_out[..., 0]      = logsumexp_j(p)
    without it, xllm writes -inf into lse_out[..., 0]; the reference leaves it zero, as an output must be finite to be
    compared, and the caller writes the -inf. The last chunk's logsumexp is not computed (it is the next call's prev
    chunk's), so with N = 1 lse_out holds slot 0 only.

mode 1 (BWD), xllm's _adaptive_working_memory_bwd on the fp32 softmax that _adaptive_working_memory_accum_fwd
recomputes: with y = the softmax above in fp64 and, for each chunk c = 0 .. N-2, g = dkkey's chunk c and
d = dlse[..., c + 1]:

    dkey[b, t, k, s] = y[t] * (g[t] - sum_{t' in chunk c} g[t'] y[t'])   (aten._softmax_backward_data)
                     + d * exp(x[t] - lse[..., c + 1])                    (logsumexp_backward; lse is mode 0's, so
                                                                           this factor is y[t] to fp32's precision)
    rounded to e; dkey is zero in the last chunk (its kkey and logsumexp feed nothing in a training call) and at
    every masked position (xllm's key_grad.masked_fill_(key_mask, 0)). lse[..., 0] and dlse[..., 0] belong to
    prev_key, which the backward pass never has: they are not read.

mode 2 (DECODE), update_memory at a chunk boundary of decode: key [B, K, C, S] is one finished chunk, unmasked;

    kkey[b, k, s, j] = softmax_j(key[b, k, j, s])  rounded to e,      lse_out[b, k, s] = logsumexp_j(key[b, k, j, s])

Masking. key_mask is xllm's: True where a position's segment is not the one its chunk ends in, so a chunk's last
position is never masked and no row of a chunk is entirely -inf in any call xllm makes. Where a row is entirely -inf
anyway, the result is torch's: softmax gives NaN (0 / 0) and logsumexp -inf in the forward pass; the backward pass's
masked_fill makes dkey 0 there. Only finite outputs are compared, so forward cases keep at least one unmasked
position per chunk row; backward cases may mask a chunk whole.

fp64 throughout: torch's fp32 softmax and logsumexp, as xllm runs them, are within a few fp32 units of these values,
far inside the tolerances; the fp64 values are the definition, so that no kernel's rounding is preferred.
"""

import torch

FWD, BWD, DECODE = 0, 1, 2


def _masked_time(key, key_mask):
    """x: key [B, L, K, S] as fp64 [B, K, S, L], -inf where key_mask."""
    B, L, K, S = key.shape
    x = key.double().permute(0, 2, 3, 1)
    if key_mask is not None and key_mask.numel():
        x = x.masked_fill(key_mask.bool().view(B, 1, 1, L), float("-inf"))
    return x


def run(mode, chunk_size, key, key_mask, prev_key, lse, dkkey, dlse):
    """The four outputs (kkey, lse_out, prev_kkey, dkey) of one call, as the kernels' `run` writes them: an output a
    mode does not produce is a one-element zero."""
    dtype, device = key.dtype, key.device
    C = chunk_size
    one = torch.zeros(1, device=device, dtype=dtype)
    one32 = torch.zeros(1, device=device, dtype=torch.float32)

    if mode == DECODE:
        x = key.double().transpose(2, 3)  # B x K x S x C
        return (torch.softmax(x, dim=-1).to(dtype), torch.logsumexp(x, dim=-1).float(), one, one)

    B, L, K, S = key.shape
    assert L % C == 0, "the length must be a multiple of chunk_size"
    N = L // C
    x = _masked_time(key, key_mask).reshape(B, K, S, N, C)
    y = torch.softmax(x, dim=-1)

    if mode == FWD:
        kkey = y.reshape(B, K, S, L).to(dtype)
        lse_out = torch.zeros(B, K, S, N, device=device, dtype=torch.float64)
        lse_out[..., 1:] = torch.logsumexp(x[:, :, :, :-1], dim=-1)
        prev_kkey = one
        if prev_key is not None and prev_key.numel():
            p = prev_key.double().permute(0, 2, 3, 1)  # B x K x S x C
            prev_kkey = torch.softmax(p, dim=-1).to(dtype)
            lse_out[..., 0] = torch.logsumexp(p, dim=-1)
        return (kkey, lse_out.float(), prev_kkey, one)

    # The backward pass: chunks 0 .. N-2.
    dx = torch.zeros(B, K, S, N, C, device=device, dtype=torch.float64)
    if N > 1:
        yc = y[:, :, :, :-1]
        g = dkkey.double().reshape(B, K, S, N - 1, C)
        d = dlse.double()[..., 1:].unsqueeze(-1)
        dx[:, :, :, :-1] = yc * (g - (g * yc).sum(dim=-1, keepdim=True)) + d * yc
    dkey = dx.reshape(B, K, S, L).permute(0, 3, 1, 2)  # B x L x K x S
    if key_mask is not None and key_mask.numel():
        dkey = dkey.masked_fill(key_mask.bool().view(B, L, 1, 1), 0.0)
    return (one, one32, one, dkey.to(dtype).contiguous())
