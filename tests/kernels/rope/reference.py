"""Reference for the rotary position embedding, computed in fp64.

x [B, L, H, R] (R is the rotated width: the head, or its first channels given as a view) is read as R / 2 complex
numbers (x[..., 2j], x[..., 2j + 1]) and multiplied by freqs_cis [L, R / 2] (complex64) along L, broadcast over B
and H. The backward pass multiplies the incoming gradient by the conjugate. The product is formed in fp64 and rounded
once to x's dtype.
"""

import torch


def rotate(x, freqs_cis, conj):
    """Rotates one tensor x [B, L, H, R] by freqs_cis [L, R / 2]; conj selects the conjugate (the backward pass)."""
    f = freqs_cis.to(torch.complex128)
    if conj:
        f = f.conj()
    z = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(z * f.unsqueeze(1)).flatten(-2).to(x.dtype)


def reference(conj, x1, x2, freqs_cis):
    """The two outputs of run(): x1 rotated, and x2 rotated (a one-element zero when x2 is empty)."""
    one = torch.zeros(1, device=x1.device, dtype=x1.dtype)
    return rotate(x1, freqs_cis, conj), (rotate(x2, freqs_cis, conj) if x2.numel() else one)
