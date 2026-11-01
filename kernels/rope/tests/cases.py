"""Shapes, layouts, inputs and tolerances for the rotary position embedding.

A case is a dict with the keys

    batch, length          B and L
    start                  the table is sliced at [start, start + length), as xllm slices it for a position offset
    conj                   False: forward; True: backward (the rotation by the conjugate, applied to a gradient)
    pair                   True: q and k are rotated together (x2 is k); False: one tensor (x2 is empty)
    layout                 how x1 (and x2) are laid out in memory:
                             contig     contiguous [B, L, H, R]
                             partial    the first R channels of wider heads [B, L, H, D], as a view
                             unbind-q   q out of a [B, L, H, 2, R] tensor (a view; the last stride is 1)
                             unbind-kq  the second tensor of the same unbind
                             bhld       a gradient laid out [B, H, L, R] and viewed as [B, L, H, R]
    heads, kv_heads        H of x1 and of x2
    head_dim, rope         D (the full head) and R (the rotated width)
    dtype                  "bfloat16", "float16" or "float32"

CASES lists the shapes that xllm's models produce (training, prefill, decode); sweep_cases() returns a fixed,
pseudo-random sweep over batches, lengths, head counts, head widths, layouts, table offsets and dtypes.
"""

import random

import torch

ROPE_BASE = 500000.0

# Outputs are rounded once from a product computed in fp32: one rounding of bf16 is within 2**-8 relative.
TOL = {"atol": 1.6e-2, "rtol": 1.6e-2, "match": 0.99999}

# name: (q heads, k heads, head dim, rotated width)
FAMILIES = {
    "h32k8d64": (32, 8, 64, 64),
    "h32k8d128": (32, 8, 128, 128),
    "h8k4d256r128": (8, 4, 256, 128),  # partial: the first 128 channels of 256-wide heads, as a view
    "h4k4d256": (4, 4, 256, 256),
}


def _main_cases():
    # name: (family, batch, length, start position, conj, pair, layout)
    shapes = {}
    for f in FAMILIES:
        shapes[f"train-{f}-fwd"] = (f, 4, 8192, 0, False, True, "contig")
        shapes[f"train-{f}-bwd"] = (f, 4, 8192, 0, True, False, "contig")
        shapes[f"decode-{f}-b1"] = (f, 1, 1, 20000, False, True, "contig")
        shapes[f"decode-{f}-b64"] = (f, 64, 1, 20000, False, True, "contig")
    shapes["prefill-h32k8d128-start8192"] = ("h32k8d128", 1, 8192, 8192, False, True, "contig")
    for f in ("h32k8d128", "h8k4d256r128", "h4k4d256"):
        shapes[f"prefill-{f}"] = (f, 1, 16384, 0, False, True, "contig")
    # attention variants whose q and kq are unbind views of [B, L, H, 2, D] and whose gradients are [B, H, L, D]
    shapes["unbind-q-h32k8d128-fwd"] = ("h32k8d128", 4, 8192, 0, False, True, "unbind-q")
    shapes["unbind-kq-h32k8d128-fwd"] = ("h32k8d128", 4, 8192, 0, False, False, "unbind-kq")
    shapes["bhld-h32k8d128-bwd"] = ("h32k8d128", 4, 8192, 0, True, False, "bhld")

    cases = {}
    for name, (f, b, length, start, conj, pair, layout) in shapes.items():
        h, hk, d, r = FAMILIES[f]
        cases[name] = {**TOL, "batch": b, "length": length, "start": start, "conj": conj, "pair": pair,
                       "layout": "partial" if r < d and layout == "contig" else layout,
                       "heads": h, "kv_heads": hk, "head_dim": d, "rope": r, "dtype": "bfloat16"}
    return cases


CASES = _main_cases()

SWEEP_KEY = 20261031


def sweep_cases():
    """96 fixed cases: forward and backward, one tensor and q with k, batches 1-9, lengths 1-1027, 1-40 heads (k heads
    differing from q's), head widths 32-256 rotated in full or in part (a view of wider heads), every layout, table
    slices starting anywhere up to 1e5, and bf16, fp16 and fp32."""
    rng = random.Random(SWEEP_KEY)
    out = []
    for _ in range(96):
        d = rng.choice([32, 64, 96, 128, 256])
        r = d if rng.random() < 0.6 else rng.choice([x for x in (16, 32, 64, 128) if x < d] or [d])
        layout = rng.choice(["contig", "contig", "partial", "unbind-q", "unbind-kq", "bhld"])
        if layout == "partial" and r == d:
            layout = "contig"
        if layout != "partial":
            d = r  # the views other than partial rotate their whole width
        h = rng.choice([1, 2, 4, 8, 32, 40])
        out.append(dict(batch=rng.choice([1, 2, 3, 9]), length=rng.choice([1, 2, 7, 64, 129, 1027]),
                        start=rng.choice([0, 0, 1, 4095, 100000]), conj=rng.random() < 0.5,
                        pair=layout in ("contig", "partial", "unbind-q") and rng.random() < 0.6, layout=layout,
                        heads=h, kv_heads=rng.choice([x for x in (1, 2, 4, 8) if x <= max(h, 1)]), head_dim=d, rope=r,
                        dtype=rng.choice(["bfloat16", "float16", "float32"])))
    return {sweep_name(i, c): {**TOL, **c} for i, c in enumerate(out)}


def sweep_name(i, c):
    return (f"{i:03d}-{'bwd' if c['conj'] else 'fwd'}-{'pair' if c['pair'] else 'one'}-{c['layout']}-b{c['batch']}"
            f"-l{c['length']}-s{c['start']}-h{c['heads']}x{c['kv_heads']}-d{c['head_dim']}r{c['rope']}-{c['dtype']}")


def freqs_table(rope, start, length, device):
    """xllm's table (cos + i sin of position * ROPE_BASE ** (-j / rope)), sliced at [start, start + length)."""
    freqs = 1.0 / torch.tensor([ROPE_BASE ** (j / rope) for j in range(0, rope, 2)], dtype=torch.float32, device=device)
    t = torch.arange(start + length, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)[start:start + length]


def _tensor(g, c, heads, layout, device, dtype):
    """One input in the given layout."""
    B, L, D, R = c["batch"], c["length"], c["head_dim"], c["rope"]
    if layout == "contig":
        return torch.randn(B, L, heads, R, generator=g, device=device).to(dtype)
    if layout == "partial":
        return torch.randn(B, L, heads, D, generator=g, device=device).to(dtype)[..., :R]
    if layout in ("unbind-q", "unbind-kq"):
        base = torch.randn(B, L, heads, 2, R, generator=g, device=device).to(dtype)
        return base[..., 0, :] if layout == "unbind-q" else base[..., 1, :]
    if layout == "bhld":
        return torch.randn(B, heads, L, R, generator=g, device=device).to(dtype).permute(0, 2, 1, 3)
    raise ValueError(layout)


def make_inputs(case, key, device):
    """The arguments of run() before the outputs: [conj, x1, x2, freqs_cis]. `key` fixes the random draw."""
    dtype = getattr(torch, case["dtype"])
    g = torch.Generator(device=device).manual_seed(key)
    x1 = _tensor(g, case, case["heads"], case["layout"], device, dtype)
    # k: contiguous, or the same view of wider heads as q in partial rope
    if case["pair"]:
        x2 = _tensor(g, case, case["kv_heads"], "partial" if case["layout"] == "partial" else "contig", device, dtype)
    else:
        x2 = torch.empty(0, device=device, dtype=dtype)
    return [case["conj"], x1, x2, freqs_table(case["rope"], case["start"], case["length"], device)]


def make_outputs(inputs):
    """Zero-filled, contiguous outputs for a call on these inputs (out2 is a one-element zero when x2 is empty)."""
    _, x1, x2, _ = inputs
    out2 = torch.zeros(x2.shape if x2.numel() else (1,), device=x1.device, dtype=x1.dtype)
    return [torch.zeros(x1.shape, device=x1.device, dtype=x1.dtype), out2]
