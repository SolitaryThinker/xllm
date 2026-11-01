"""Test cases for the group norms: shapes, inputs, outputs and tolerances.

The operation is one entry point,

    run(mode, norm, num_groups, eps, memory_efficient, x_or_y, dy, mean, rstd, gamma, beta,
        y, mean_out, rstd_out, dx, dgamma, dbeta)

- mode 0 is the forward pass: x_or_y is x [B, S, C]; it writes y [B, S, C], rstd [B*S, G] (fp32) and, for the layer
  norm, mean [B*S, G] (fp32).
- mode 1 is the backward pass: x_or_y is x, or y when memory_efficient; dy [B, S, C]; mean and rstd as the forward
  pass wrote them. It writes dx [B, S, C], and with gamma also dgamma [C] and, for the layer norm, dbeta [C].
- norm 0 is the RMS norm, norm 1 the layer norm. gamma (and beta, layer norm only) [C] are the affine parameters as
  the op receives them; empty tensors mean no affine transform.
- An input a call does not take is an empty tensor; an output it does not produce is a one-element zero tensor that
  must stay zero.

Two groups of cases are defined here. SHAPES are the calls xllm makes in its example models (bf16, B x 8192 rows),
with more than 32 groups and with the 32 groups and fewer that the models already use. `sweep()` is a fixed sweep
of shapes beyond those: any number of groups from 1 to 256 (also not a power of two), channels per group from 24 to
4096 (also not a multiple of 16 bytes), row counts of 1, 7 and not a multiple of any tile, fp32, fp16 and bf16, both
norms, forward and backward, with and without the affine transform and memory_efficient.
"""

import itertools
import random

import torch

OUTPUTS = ["y", "mean", "rstd", "dx", "dgamma", "dbeta"]

FWD, BWD = 0, 1
RMS, LAYER = 0, 1
EPS = {RMS: 1e-6, LAYER: 1e-5}

# bf16 outputs: two units in the last place, plus an absolute floor where cancellation leaves small values. The
# statistics are fp32 from fp32 accumulation. dgamma and dbeta are sums over all rows (magnitude ~ sqrt(rows)).
TOL = {
    "atol": 2e-2,
    "rtol": 1.6e-2,
    "match": 0.99999,
    "outputs": {
        "mean": {"atol": 1e-5, "rtol": 1e-4, "match": 1.0},
        "rstd": {"atol": 1e-5, "rtol": 1e-4, "match": 1.0},
        "dgamma": {"atol": 1.0, "rtol": 1.6e-2, "match": 1.0},
        "dbeta": {"atol": 1.0, "rtol": 1.6e-2, "match": 1.0},
    },
}

# name: (batch, groups, channels per group, affine, memory_efficient, norms). Rows are batch x 8192. Every shape is a
# call that xllm 692bc25 makes in its example models (bf16).
SHAPES = {
    # More than 32 groups.
    "q64c64": (32, 64, 64, True, False, (RMS,)),  # query norm, qknorm + softdelta: 2 x 32 heads of 64
    "q64c128": (8, 64, 128, True, False, (RMS, LAYER)),  # query norm, qknorm + softdelta: 2 x 32 heads of 128
    "o64c24": (32, 64, 24, False, True, (RMS,)),  # ortho residual norm, 64 residual heads: model_dim 1536
    "o64c40": (8, 64, 40, False, True, (RMS,)),  # ortho residual norm, 64 heads: model_dim 2560
    "o64c64": (8, 64, 64, False, True, (RMS,)),  # ortho residual norm, 64 heads: model_dim 4096
    # 32 groups and fewer.
    "b1c1536": (32, 1, 1536, True, False, (RMS,)),  # residual-stream norm
    "b2c1280": (8, 2, 1280, True, False, (RMS,)),  # residual-stream norm (1 x 1280 per rank at TP2)
    "b4c1024": (8, 4, 1024, True, False, (RMS,)),  # residual-stream norm
    "b1c4096": (8, 1, 4096, True, False, (RMS,)),  # residual-stream norm
    "q32c128": (8, 32, 128, True, False, (RMS,)),  # query norm with qknorm (softmax): 32 heads of 128
    "k8c128": (8, 8, 128, False, True, (RMS,)),  # key norm with qknorm: 8 kv heads of 128
    "o32c80": (8, 32, 80, False, True, (RMS,)),  # ortho residual norm, 32 heads
}

SEQ = 8192


def _case(name, mode, norm, batch, seq, groups, cpg, affine, memory_efficient, dtype, tol):
    return {
        "name": name, "mode": mode, "norm": norm, "batch": batch, "seq": seq, "groups": groups,
        "channels_per_group": cpg, "affine": affine, "memory_efficient": memory_efficient and mode == BWD,
        "dtype": dtype, **tol,
    }


def shape_cases():
    """The cases of SHAPES: every shape, every norm it lists, forward and backward."""
    out = []
    for name, (batch, groups, cpg, affine, me, norms) in SHAPES.items():
        for norm, mode in itertools.product(norms, (FWD, BWD)):
            key = f"{'rms' if norm == RMS else 'ln'}-{name}-{'fwd' if mode == FWD else 'bwd'}"
            out.append(_case(key, mode, norm, batch, SEQ, groups, cpg, affine, me, "bfloat16", TOL))
    return out


SWEEP_SALT = 20261031


def sweep():
    """The sweep of shapes beyond the example models: a grid of edge shapes, then random shapes from a fixed salt."""
    specs = []
    edges = [
        # (groups, channels per group, rows)
        (1, 8192, 7), (1, 1536, 1), (2, 1280, 129), (3, 96, 1000), (4, 1024, 4097), (8, 128, 1),
        (24, 64, 7), (32, 128, 129), (33, 64, 1000), (48, 128, 7), (64, 36, 129), (64, 100, 7),
        (96, 128, 4097), (128, 64, 1), (256, 32, 129), (256, 48, 1000), (5, 640, 7), (12, 160, 129),
        (64, 24, 129), (64, 40, 7), (32, 80, 1000), (32, 48, 129),
    ]
    rng = random.Random(SWEEP_SALT)
    for groups, cpg, rows in edges:
        for norm, mode in itertools.product((RMS, LAYER), (FWD, BWD)):
            specs.append(dict(groups=groups, cpg=cpg, rows=rows, norm=norm, mode=mode, affine=rng.random() < 0.6,
                              me=mode == BWD and rng.random() < 0.5,
                              dtype=rng.choice(["bfloat16", "float16", "float32"])))
    for _ in range(48):
        groups = rng.choice([1, 2, 3, 4, 6, 8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 192, 256])
        cpg = rng.choice([32, 36, 48, 64, 96, 100, 128, 160, 256, 384, 512, 640, 1024, 1280, 1536, 2048, 4096])
        if groups * cpg > 32768:
            continue
        mode = rng.choice([FWD, BWD])
        specs.append(dict(groups=groups, cpg=cpg, rows=rng.choice([1, 2, 7, 33, 129, 1000, 2049]),
                          norm=rng.choice([RMS, LAYER]), mode=mode, affine=rng.random() < 0.6,
                          me=mode == BWD and rng.random() < 0.5, dtype=rng.choice(["bfloat16", "float16", "float32"])))

    out = []
    for i, c in enumerate(specs):
        name = (f"sweep{i:03d}-{'rms' if c['norm'] == RMS else 'ln'}-{'fwd' if c['mode'] == FWD else 'bwd'}"
                f"-g{c['groups']}x{c['cpg']}-r{c['rows']}-{c['dtype']}{'-affine' if c['affine'] else ''}"
                f"{'-me' if c['me'] else ''}")
        # dgamma and dbeta are sums over the rows: their absolute floor follows their magnitude, ~sqrt(rows).
        floor = 2e-2 * max(1.0, c["rows"] ** 0.5)
        tol = {**TOL, "outputs": {**TOL["outputs"], "dgamma": {"atol": floor, "rtol": 1.6e-2, "match": 1.0},
                                  "dbeta": {"atol": floor, "rtol": 1.6e-2, "match": 1.0}}}
        out.append(_case(name, c["mode"], c["norm"], 1, c["rows"], c["groups"], c["cpg"], c["affine"], c["me"],
                         c["dtype"], tol))
    return out


def all_cases():
    """The shape cases followed by the sweep."""
    return shape_cases() + sweep()


def _stats(x, norm, groups, eps):
    """mean (layer norm) and rstd [B*S, G] in fp32, from fp32 accumulation, as the forward pass writes them."""
    xg = x.float().reshape(-1, groups, x.shape[-1] // groups)
    if norm == RMS:
        return None, torch.rsqrt(xg.square().mean(-1) + eps)
    mean = xg.mean(-1)
    var = (xg - mean.unsqueeze(-1)).square().mean(-1)
    return mean, torch.rsqrt(var + eps)


def make_inputs(case, salt, device):
    """The arguments of run() before the outputs, drawn from `salt` (the same salt gives the same inputs)."""
    dtype = getattr(torch, case["dtype"])
    groups, channels = case["groups"], case["groups"] * case["channels_per_group"]
    norm, eps = case["norm"], EPS[case["norm"]]
    g = torch.Generator(device=device).manual_seed(salt)
    shape = (case["batch"], case["seq"], channels)
    # An offset and a scale per channel, so that the mean is not zero and the groups differ.
    scale = 0.5 + 1.5 * torch.rand(channels, generator=g, device=device)
    shift = 0.5 * torch.randn(channels, generator=g, device=device)
    x = (torch.randn(shape, generator=g, device=device) * scale + shift).to(dtype)
    empty = torch.empty(0, device=device, dtype=dtype)
    empty32 = torch.empty(0, device=device, dtype=torch.float32)
    gamma = (0.5 + torch.rand(channels, generator=g, device=device)).to(dtype) if case["affine"] else empty
    beta = (0.2 * torch.randn(channels, generator=g, device=device)).to(dtype) if case["affine"] and norm == LAYER else empty
    if case["mode"] == FWD:
        return [FWD, norm, groups, eps, False, x, empty, empty32, empty32, gamma, beta]
    dy = torch.randn(shape, generator=g, device=device).to(dtype)
    mean, rstd = _stats(x, norm, groups, eps)
    x_or_y = x
    if case["memory_efficient"]:
        xg = x.float().reshape(-1, groups, channels // groups)
        y = (xg - mean.unsqueeze(-1)) if norm == LAYER else xg
        y = (y * rstd.unsqueeze(-1)).reshape(shape)
        if case["affine"]:
            y = y * gamma.float() + (beta.float() if norm == LAYER else 0.0)
        x_or_y = y.to(dtype)
    return [BWD, norm, groups, eps, case["memory_efficient"], x_or_y, dy,
            mean if mean is not None else empty32, rstd, gamma, beta]


def make_outputs(inputs):
    """Zero-filled output tensors for a call on these inputs, in OUTPUTS order."""
    mode, norm, groups, _, _, x, _, _, _, gamma, beta = inputs
    rows = x.numel() // x.shape[-1]
    one = torch.zeros(1, device=x.device, dtype=x.dtype)
    one32 = torch.zeros(1, device=x.device, dtype=torch.float32)
    if mode == FWD:
        stat = torch.zeros(rows, groups, device=x.device, dtype=torch.float32)
        return [torch.zeros_like(x), stat.clone() if norm == LAYER else one32, stat, one.clone(), one.clone(), one.clone()]
    affine = gamma.numel() > 0
    return [one, one32, one32.clone(), torch.zeros_like(x),
            torch.zeros_like(gamma) if affine else one.clone(),
            torch.zeros_like(gamma) if affine and norm == LAYER else one.clone()]


def tolerance(case, name):
    """The tolerance output `name` is judged with: the case's atol, rtol and match, with what its "outputs" entry gives
    that output instead."""
    return {"atol": case["atol"], "rtol": case["rtol"], "match": case["match"], **case.get("outputs", {}).get(name, {})}


def compare(outs, refs, case):
    """One row per output: (name, passed, matched ratio, max abs error, note).

    An output passes when it has the reference's shape and dtype, no NaN or Inf, is not all zeros where the reference
    is not, and at least `match` of its elements satisfy |out - ref| <= atol + rtol * |ref|.
    """
    rows = []
    for name, out, ref in zip(OUTPUTS, outs, refs):
        if out.shape != ref.shape or out.dtype != ref.dtype:
            rows.append((name, False, 0.0, float("inf"), f"{tuple(out.shape)} {out.dtype}, expected {tuple(ref.shape)} {ref.dtype}"))
            continue
        x, y = out.float(), ref.to(out.device).float()
        if not torch.isfinite(x).all():
            rows.append((name, False, 0.0, float("inf"), "NaN or Inf"))
            continue
        if y.abs().sum() > 0 and x.abs().sum() == 0:
            rows.append((name, False, 0.0, float("inf"), "all zeros"))
            continue
        err = (x - y).abs()
        own = tolerance(case, name)
        # A count, not a float32 mean: over 1e8 elements a mean of all ones can round below 1.
        ratio = (err <= own["atol"] + own["rtol"] * y.abs()).sum().item() / err.numel()
        rows.append((name, ratio >= own["match"], ratio, err.max().item(), ""))
    return rows
