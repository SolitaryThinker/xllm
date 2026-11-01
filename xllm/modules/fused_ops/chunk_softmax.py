"""Fused softmax and logsumexp primitives used by adaptive working memory."""

import torch
from xllm_extension import ops


def available(key):
    """Whether the native extension has a kernel for this tensor's device and dtype."""
    return (hasattr(ops, "chunk_softmax") and key.is_cuda and key.dtype in
            (torch.float16, torch.bfloat16, torch.float32) and
            torch.cuda.get_device_capability(key.device)[0] >= 10)


def forward(key, chunk_size, mask=None, prev_key=None):
    """Return chunk probabilities and shifted logsumexp; key is [B,K,L,S]."""
    b, k, length, s = key.shape
    empty = key.new_empty(0)
    probs = key.new_empty(b, k, s, length)
    lse = torch.full((b, k, s, length // chunk_size), -torch.inf, device=key.device, dtype=torch.float32)
    prev_probs = key.new_empty(b, k, s, chunk_size) if prev_key is not None else empty
    ops.chunk_softmax(0, chunk_size, key.permute(0, 2, 1, 3).resolve_neg().contiguous(),
                     empty if mask is None else mask.reshape(b, length).contiguous(),
                     empty if prev_key is None else prev_key.permute(0, 2, 1, 3).resolve_neg().contiguous(),
                     empty, empty, empty, probs, lse, prev_probs, empty)
    return probs, lse, prev_probs if prev_key is not None else None


def backward(key, chunk_size, mask, lse, dprobs, dlse):
    """Combine float32 probability gradients and logsumexp gradients into [B,K,L,S]."""
    b, k, length, s = key.shape
    empty = key.new_empty(0)
    dkey = torch.zeros((b, length, k, s), device=key.device, dtype=key.dtype)
    ops.chunk_softmax(1, chunk_size, key.permute(0, 2, 1, 3).resolve_neg().contiguous(),
                     empty if mask is None else mask.reshape(b, length).contiguous(), empty,
                     lse.resolve_neg().contiguous(), dprobs.resolve_neg().contiguous(), dlse.resolve_neg().contiguous(), empty, empty, empty, dkey)
    return dkey.permute(0, 2, 1, 3)


def decode(key):
    """Normalize one completed chunk [B,K,C,S], returning probabilities and logsumexp."""
    b, k, c, s = key.shape
    empty = key.new_empty(0)
    probs = key.new_empty(b, k, s, c)
    lse = torch.empty((b, k, s), device=key.device, dtype=torch.float32)
    ops.chunk_softmax(2, c, key.resolve_neg().contiguous(), empty, empty, empty, empty, empty, probs, lse, empty, empty)
    return probs, lse
