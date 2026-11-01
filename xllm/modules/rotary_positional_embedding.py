import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn


class _RotaryFunction(torch.autograd.Function):
    """Rotate q and k together, including the conjugate rotation in autograd."""

    @staticmethod
    def forward(ctx, x1, x2, freqs, conjugate):
        from xllm_extension import ops
        ctx.save_for_backward(freqs)
        ctx.conjugate = conjugate
        y1 = torch.empty_like(x1, memory_format=torch.contiguous_format)
        y2 = torch.empty_like(x2, memory_format=torch.contiguous_format)
        ops.rotary_embedding(conjugate, x1, x2, freqs, y1, y2)
        return y1, y2

    @staticmethod
    def backward(ctx, dy1, dy2):
        (freqs,) = ctx.saved_tensors
        dx1, dx2 = _RotaryFunction.apply(dy1.contiguous(), dy2.contiguous(), freqs, not ctx.conjugate)
        return dx1, dx2, None, None


def _fused_rotary(x1, x2, freqs, backward):
    """Return the fused results when the compiled kernel supports these inputs."""
    tensors = (x1,) if x2 is None else (x1, x2)
    if freqs.requires_grad or freqs.is_conj() or freqs.dtype != torch.complex64 or freqs.ndim != 2:
        return None
    if not all(x.is_cuda and x.ndim == 4 and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
               and not x.is_neg() and x.stride(-1) == 1 and x.shape[-1] % 2 == 0 for x in tensors):
        return None
    if freqs.shape != (x1.size(1), x1.size(3) // 2):
        return None
    if x2 is not None and (x2.dtype != x1.dtype or x2.shape[:2] != x1.shape[:2] or x2.size(3) != x1.size(3)):
        return None
    if torch.cuda.get_device_capability(x1.device)[0] < 10:
        return None
    try:
        from xllm_extension import ops
    except ImportError:
        return None
    if not hasattr(ops, "rotary_embedding"):
        return None
    empty = x1.new_empty(0) if x2 is None else x2
    return _RotaryFunction.apply(x1, empty, freqs, backward)


class RotaryEmbedding(nn.Module):
    def __init__(self, embed_dim, max_positions, base):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.base = base
        self.freqs: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

    def _precompute_freqs(self, device: torch.device):
        freqs = [self.base ** (j / self.embed_dim) for j in range(0, self.embed_dim, 2)]
        freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
        freqs = 1.0 / freqs
        return freqs

    @torch.no_grad()
    def _precompute_until(self, max_positions: int, device: torch.device):
        assert self.max_positions <= max_positions
        self.max_positions = max_positions
        if self.freqs is None:
            self.freqs = self._precompute_freqs(device)
        # C
        t = torch.arange(max_positions, dtype=torch.float32, device=self.freqs.device)
        # C x D/2
        freqs = torch.outer(t, self.freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def get_freqs_cis(self, start: int, end: int, device: torch.device) -> torch.Tensor:
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_until(self.max_positions, device)
        if end > self.freqs_cis.shape[0]:
            warnings.warn('Extending rotary range from {} to {}'.format(self.max_positions, end))
            self.freqs_cis = self._precompute_until(end, device)
        return self.freqs_cis[start:end]  # type: ignore

    def forward(self, xq, xk, start: int):
        # B x C x N x  D
        seq_len = xq.shape[1]
        freqs_cis = self.get_freqs_cis(start, start + seq_len, xq.device)
        return apply_rotary_embeddings(xq, xk, freqs_cis=freqs_cis, backward=False)

    def extra_repr(self) -> str:
        return 'dim={}, max positions={}, base={:.1f}'.format(self.embed_dim, self.max_positions, self.base)


def apply_rotary_embedding(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    backward: bool,
) -> torch.Tensor:
    """
    If backward=False:
        - inputs: x
        - outputs: x_out
    If backward=True:
        - inputs: grad_x_out
        - outputs: grad_x
    """
    fused = _fused_rotary(x, None, freqs_cis, backward)
    if fused is not None:
        return fused[0]

    if backward:
        freqs_cis = freqs_cis.conj()

    # B x C x N x D/2
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # C x 1 x D/2
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


def apply_rotary_embeddings(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    backward: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If backward=False:
        - inputs: (xq, xk)
        - outputs: (xq_out, xk_out)
    If backward=True:
        - inputs: (grad_xq_out, grad_xk_out)
        - outputs: (grad_xq, grad_xk)
    """
    fused = _fused_rotary(xq, xk, freqs_cis, backward)
    if fused is not None:
        return fused

    if backward:
        freqs_cis = freqs_cis.conj()

    # B x C x N x D/2
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # C x 1 x D/2
    freqs_cis = freqs_cis.unsqueeze(1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
