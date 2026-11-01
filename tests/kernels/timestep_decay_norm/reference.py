"""Reference for xllm's group timestep decay normalization (TDN), forward and backward, in fp64.

forward_loop() is the recurrence step by step, forward() the same in closed form (a cumulative sum of decayed terms),
and run() has the calling convention of the extension's `run` entry point and returns its outputs in the same order.

For each row b and group k of x [B, L, H] (K groups of G = H / K channels), along the sequence t = 0 .. L-1:

    a step is valid unless padding_mask[b, t]; a padded step leaves the state as it is and writes y = 0
    at a valid step with bos_mask[b, t], the state restarts: count = 0, mean = var = 0
    at a valid step:  count += 1
                      mean = beta1 * mean + (1 - beta1) * m_t        m_t, v_t: the group's mean and (biased)
                      var  = beta2 * var  + (1 - beta2) * v_t        variance at step t
                      cummean = mean / (1 - beta1 ** count),  cumrstd = 1 / sqrt(var / (1 - beta2 ** count) + eps)
                      (where 1 - beta ** count is not positive: cummean = 0, cumrstd = 1 / sqrt(eps))
                      y = (x - cummean) * cumrstd * gamma + beta

starting from prev_count [B], prev_mean, prev_var [B, K]. The forward pass returns y, the final count, mean and var,
and (for training) cummean and cumrstd [B, L, K] in fp32, which the backward pass takes back. The backward pass is
the gradient of y and of the final mean and var with respect to x, prev_mean, prev_var, gamma and beta; given
memory_efficient, it gets y in place of x and recovers x from y and the saved statistics.
"""

import torch

INFER, TRAIN, BWD = 0, 1, 2


def _empty(t):
    return t is None or t.numel() == 0


def forward_loop(x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, num_groups, beta1, beta2, eps):
    """The recurrence step by step, as written above; forward() computes the same in closed form."""
    B, L, H = x.shape
    K = num_groups
    xg = x.reshape(B, L, K, H // K)
    var_t, mean_t = torch.var_mean(xg, dim=-1, correction=0)  # B x L x K
    bos = torch.zeros(B, L, dtype=torch.bool, device=x.device) if _empty(bos_mask) else bos_mask.bool()
    pad = torch.zeros(B, L, dtype=torch.bool, device=x.device) if _empty(padding_mask) else padding_mask.bool()
    valid = ~pad
    bos = bos & valid

    count = prev_count.to(torch.int64)
    mean, var = prev_mean, prev_var
    # beta ** count, kept as a running product so that a long run of steps does not lose it
    p1 = torch.full((B,), beta1, dtype=torch.float64, device=x.device) ** count.double()
    p2 = torch.full((B,), beta2, dtype=torch.float64, device=x.device) ** count.double()
    cummean, cumrstd = [], []
    for t in range(L):
        v, r = valid[:, t], bos[:, t]
        keep = (~r).double().unsqueeze(1)
        count = torch.where(r, torch.zeros_like(count), count)
        p1 = torch.where(r, torch.ones_like(p1), p1)
        p2 = torch.where(r, torch.ones_like(p2), p2)
        mean, var = mean * keep, var * keep
        vv = v.double().unsqueeze(1)
        mean = vv * (beta1 * mean + (1 - beta1) * mean_t[:, t]) + (1 - vv) * mean
        var = vv * (beta2 * var + (1 - beta2) * var_t[:, t]) + (1 - vv) * var
        count = count + v.to(torch.int64)
        p1 = torch.where(v, p1 * beta1, p1)
        p2 = torch.where(v, p2 * beta2, p2)
        d1, d2 = (1 - p1).unsqueeze(1), (1 - p2).unsqueeze(1)
        ok = (d1 > 0) & (d2 > 0)
        cm = torch.where(ok, mean / torch.where(ok, d1, torch.ones_like(d1)), torch.zeros_like(mean))
        cv = torch.where(ok, var / torch.where(ok, d2, torch.ones_like(d2)), torch.zeros_like(var))
        cummean.append(cm)
        cumrstd.append(torch.rsqrt(cv + eps))
    cummean = torch.stack(cummean, 1)  # B x L x K
    cumrstd = torch.stack(cumrstd, 1)
    y = (xg - cummean.unsqueeze(-1)) * cumrstd.unsqueeze(-1)
    y = y.reshape(B, L, H) * gamma + beta
    y = y * valid.double().unsqueeze(-1)
    return y, count, mean, var, cummean, cumrstd


def forward(x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, num_groups, beta1, beta2, eps):
    """The recurrence in fp64 on x's device, in closed form; differentiable in x, prev_mean, prev_var, gamma, beta.

    Within a segment (from a reset, or from the start with the state carried in), after n valid steps,
        mean_t = beta1 ** n_t * mean_0 + (1 - beta1) * sum_{valid s <= t} beta1 ** (n_t - n_s) * m_s
    which is beta1 ** n_t * (mean_0 + (1 - beta1) * (C_t - C_start)) with C the running sum of beta1 ** -n_s * m_s.
    n stays below the sequence's length, where beta1 ** -n (~1e7 at 16384 steps) is exact enough in fp64.
    """
    B, L, H = x.shape
    K = num_groups
    dev = x.device
    xg = x.reshape(B, L, K, H // K)
    var_t, mean_t = torch.var_mean(xg, dim=-1, correction=0)  # B x L x K
    bos = torch.zeros(B, L, dtype=torch.bool, device=dev) if _empty(bos_mask) else bos_mask.bool()
    pad = torch.zeros(B, L, dtype=torch.bool, device=dev) if _empty(padding_mask) else padding_mask.bool()
    valid = ~pad
    reset = bos & valid

    # The step each step's segment starts at (-1: the first segment, which starts from the state carried in).
    steps = torch.arange(L, device=dev).expand(B, L)
    start = torch.cummax(torch.where(reset, steps, torch.full_like(steps, -1)), dim=1).values
    first = start < 0
    V = torch.cumsum(valid.long(), dim=1)  # valid steps so far
    V0 = torch.where(first, torch.zeros_like(V), torch.gather(V, 1, start.clamp(min=0)) - 1)
    n = (V - V0).double()  # valid steps in the segment, up to and including t
    count = torch.where(first, prev_count.view(B, 1) + (V - V0), V - V0)

    def decayed(beta_, stat_t, init):
        w = valid.double().unsqueeze(-1) * beta_ ** (-n.unsqueeze(-1)) * stat_t  # B x L x K
        C = torch.cumsum(w, dim=1)
        C0 = torch.gather(C, 1, (start.clamp(min=1) - 1).unsqueeze(-1).expand(B, L, K))
        C0 = torch.where((start <= 0).unsqueeze(-1), torch.zeros_like(C0), C0)
        base = torch.where(first.unsqueeze(-1), init.unsqueeze(1).expand(B, L, K), torch.zeros_like(C))
        return beta_ ** n.unsqueeze(-1) * (base + (1 - beta_) * (C - C0))

    mean = decayed(beta1, mean_t, prev_mean)
    var = decayed(beta2, var_t, prev_var)
    p1 = (beta1 ** count.double()).unsqueeze(-1)
    p2 = (beta2 ** count.double()).unsqueeze(-1)
    d1, d2 = 1 - p1, 1 - p2
    ok = (d1 > 0) & (d2 > 0)
    cummean = torch.where(ok, mean / torch.where(ok, d1, torch.ones_like(d1)), torch.zeros_like(mean))
    cv = torch.where(ok, var / torch.where(ok, d2, torch.ones_like(d2)), torch.zeros_like(var))
    cumrstd = torch.rsqrt(cv + eps)
    y = (xg - cummean.unsqueeze(-1)) * cumrstd.unsqueeze(-1)
    y = y.reshape(B, L, H) * gamma + beta
    y = y * valid.double().unsqueeze(-1)
    return y, count[:, -1], mean[:, -1], var[:, -1], cummean, cumrstd


def run(mode, num_groups, beta1, beta2, eps, memory_efficient, x_or_y, bos_mask, padding_mask, prev_count, prev_mean,
        prev_var, gamma, beta, dy, dmean, dvar, cummean, cumrstd):
    """The expected outputs of the extension's `run` for these inputs, in its output order: y, count, mean, var,
    cummean, cumrstd, dx, dprev_mean, dprev_var, dgamma, dbeta. An output a mode does not produce is a one-element
    zero tensor, as in the extension. Statistics and gradients are computed in fp64 and rounded to the output dtype."""
    dtype, device = x_or_y.dtype, x_or_y.device
    B, L, H = x_or_y.shape
    K = num_groups
    one = torch.zeros(1, device=device, dtype=dtype)
    one32 = torch.zeros(1, device=device, dtype=torch.float32)
    args = (bos_mask, prev_count, prev_mean.double(), prev_var.double(), gamma.double(), beta.double(), padding_mask,
            K, beta1, beta2, eps)

    if mode in (INFER, TRAIN):
        with torch.no_grad():
            y, count, mean, var, cm, cr = forward(x_or_y.double(), *args)
        stats = (cm.float(), cr.float()) if mode == TRAIN else (one32, one32)
        return (y.to(dtype), count, mean.to(dtype), var.to(dtype), *stats, one, one, one, one, one)

    # The backward pass: given y (memory_efficient), x is what the forward pass normalized into it.
    if memory_efficient:
        G = H // K
        yg = ((x_or_y.double() - beta.double()) / gamma.double()).reshape(B, L, K, G)
        x = yg / cumrstd.double().unsqueeze(-1) + cummean.double().unsqueeze(-1)
        x = x.reshape(B, L, H)
    else:
        x = x_or_y.double()
    leaves = [x.detach().requires_grad_(), prev_mean.double().requires_grad_(), prev_var.double().requires_grad_(),
              gamma.double().requires_grad_(), beta.double().requires_grad_()]
    xl, pm, pv, g, b = leaves
    with torch.enable_grad():
        y, _, mean, var, _, _ = forward(xl, bos_mask, prev_count, pm, pv, g, b, padding_mask, K, beta1, beta2, eps)
        torch.autograd.backward([y, mean, var], [dy.double(), dmean.double(), dvar.double()])
    dx = xl.grad.to(dtype)
    if not _empty(padding_mask):
        dx = dx * (~padding_mask.bool()).unsqueeze(-1)  # a padded step's x reaches nothing
    return (one, torch.zeros(1, device=device, dtype=torch.int64), one, one, one32, one32,
            dx, pm.grad.to(dtype), pv.grad.to(dtype), g.grad.to(dtype), b.grad.to(dtype))
