"""fp64 reference of xllm's group RMS norm and group layer norm, forward and backward.

The inputs are those of cases.make_inputs; the outputs are cases.OUTPUTS, in their dtypes. The backward pass uses the
statistics it is given (as a training step does, from its own forward pass), and, when memory_efficient, recovers the
normalized input from y: (y - beta) / gamma, or y itself without an affine transform.
"""

import torch

FWD, RMS = 0, 0


def run(mode, norm, num_groups, eps, memory_efficient, x_or_y, dy, mean, rstd, gamma, beta):
    """Returns (y, mean, rstd, dx, dgamma, dbeta); the outputs a mode does not produce are zero placeholders."""
    shape, dtype, device = x_or_y.shape, x_or_y.dtype, x_or_y.device
    channels = shape[-1]
    rows = x_or_y.numel() // channels
    cpg = channels // num_groups
    affine = gamma.numel() > 0
    layer = norm != RMS
    one = torch.zeros(1, device=device, dtype=dtype)
    one32 = torch.zeros(1, device=device, dtype=torch.float32)

    def grouped(t):
        return t.double().reshape(rows, num_groups, cpg)

    g = gamma.double().reshape(num_groups, cpg) if affine else None
    b = beta.double().reshape(num_groups, cpg) if affine and layer else None

    if mode == FWD:
        x = grouped(x_or_y)
        mu = x.mean(-1, keepdim=True) if layer else torch.zeros_like(x[..., :1])
        centered = x - mu
        r = torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps)
        y = centered * r
        if affine:
            y = y * g + (b if layer else 0.0)
        mean_out = mu.squeeze(-1).float() if layer else one32
        return (y.reshape(shape).to(dtype), mean_out, r.squeeze(-1).float(), one, one, one)

    r = rstd.double().reshape(rows, num_groups, 1)
    if memory_efficient:
        y = grouped(x_or_y)
        xhat = ((y - b) if layer and affine else y) / (g if affine else 1.0)
    else:
        mu = mean.double().reshape(rows, num_groups, 1) if layer else 0.0
        xhat = (grouped(x_or_y) - mu) * r
    d = grouped(dy)
    dxhat = d * g if affine else d
    proj = (dxhat * xhat).mean(-1, keepdim=True)
    dx = dxhat - xhat * proj
    if layer:
        dx = dx - dxhat.mean(-1, keepdim=True)
    dx = (dx * r).reshape(shape).to(dtype)
    dgamma = (d * xhat).sum(0).reshape(channels).to(dtype) if affine else one
    dbeta = d.sum(0).reshape(channels).to(dtype) if affine and layer else one
    return (one, one32, one32, dx, dgamma, dbeta)
