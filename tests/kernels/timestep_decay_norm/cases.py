"""Cases for the timestep decay normalization extension: shapes, input generation, output buffers and tolerances.

A case is a dict: mode (INFER, TRAIN or BWD), batch, length, features (H), groups (K), bos (probability of a bos
token per step; 0 means no bos mask), padding (probability of a padded step; 0 means no padding mask), carried (the
call starts from a carried state instead of a fresh one), memory_efficient, dstate (nonzero gradients of the final
state in the backward pass), dtype, and the tolerance of every output.

SHAPES are the calls of xllm's Gekko models (gekko-0.9B, -4B, -7.1B: H = 1536, 2560, 4096 in groups of 64 channels):
training (one call of the whole sequence per layer, from a fresh state, a packed document's bos about every 2-3k
tokens), prefill (chunks of up to 16384 tokens, the first from a fresh state and the next with the state carried
in) and decode (one token per call with the state carried). SWEEP covers shapes and inputs beyond those: lengths 1 to
4097, 1 to 256 groups of 32 to 256 channels, no bos, sparse bos and a bos at every step, padding masks, fresh and
carried state, nonzero gradients of the final state, fp32, fp16 and bf16. Both are fixed: the same cases, with the
same inputs, in every run.
"""

import math
import random

import torch

import reference

OUTPUTS = ["y", "count", "mean", "var", "cummean", "cumrstd", "dx", "dprev_mean", "dprev_var", "dgamma", "dbeta"]

INFER, TRAIN, BWD = reference.INFER, reference.TRAIN, reference.BWD
BETA1, BETA2, EPS = 0.999, 0.9999, 1e-5
MODELS = {"0.9b": 1536, "4b": 2560, "7.1b": 4096}  # model dimension; groups of 64 channels

# bf16 outputs: a few units in the last place; the statistics are fp32. dx and the parameter gradients come out of
# an adjoint scan over the sequence, so they get the wider tolerance of a long reduction.
TOL = {
    "atol": 2e-2,
    "rtol": 1.6e-2,
    "match": 0.9999,
    "outputs": {
        "count": {"atol": 0, "rtol": 0, "match": 1.0},
        "mean": {"atol": 1e-3, "rtol": 1e-2, "match": 1.0},
        "var": {"atol": 1e-3, "rtol": 1e-2, "match": 1.0},
        "cummean": {"atol": 1e-4, "rtol": 1e-3, "match": 0.99999},
        "cumrstd": {"atol": 1e-4, "rtol": 1e-3, "match": 0.99999},
        "dx": {"atol": 5e-2, "rtol": 3e-2, "match": 0.9999},
        "dprev_mean": {"atol": 5e-2, "rtol": 3e-2, "match": 1.0},
        "dprev_var": {"atol": 5e-2, "rtol": 3e-2, "match": 1.0},
    },
}

SWEEP_KEY = 20261031  # fixes the sweep's random choices; case i draws its inputs from key SWEEP_KEY + i


def _tolerances(mode, batch, length, carried):
    tol = {**TOL, "outputs": dict(TOL["outputs"])}
    # dgamma and dbeta are sums over all rows: their absolute floor follows their magnitude, ~sqrt(rows).
    floor = 2e-2 * max(1.0, math.sqrt(batch * length))
    tol["outputs"]["dgamma"] = {"atol": floor, "rtol": 3e-2, "match": 1.0}
    tol["outputs"]["dbeta"] = {"atol": floor, "rtol": 3e-2, "match": 1.0}
    if mode == BWD and not carried:
        # From a fresh state the bias correction 1 / (1 - beta ** count) scales the first steps by up to ~1 / (1 -
        # beta1) = 1000, so dprev_mean and dprev_var reach 1e4-1e5, where 16-bit steps are 64-512 apart.
        for name in ("dprev_mean", "dprev_var"):
            tol["outputs"][name] = {"atol": 512.0, "rtol": 3e-2, "match": 1.0}
    return tol


def make_case(name, mode, batch, length, groups, cpg, bos, padding, carried, memory_efficient, dstate, dtype, key):
    return {"name": name, "key": key, "mode": mode, "batch": batch, "length": length, "features": groups * cpg,
            "groups": groups, "bos": bos, "padding": padding, "carried": carried, "memory_efficient": memory_efficient,
            "dstate": dstate, "dtype": dtype, **_tolerances(mode, batch, length, carried)}


def shapes():
    """The model cases: (name -> case)."""
    out = {}
    for model, h in MODELS.items():
        # name: mode, batch, length, bos probability per token, carried state, memory_efficient
        spec = {
            f"train-{model}-fwd": (TRAIN, 4, 8192, 1 / 2048, False, False),
            f"train-{model}-bwd": (BWD, 4, 8192, 1 / 2048, False, False),
            f"train-{model}-bwd-me": (BWD, 4, 8192, 1 / 2048, False, True),
            f"prefill-{model}-b1": (INFER, 1, 16384, 0.0, False, False),
            f"prefill-{model}-b8": (INFER, 8, 16384, 0.0, False, False),
            f"tail-{model}-b8": (INFER, 8, 3616, 0.0, True, False),
        }
        for b in (1, 8, 64):
            spec[f"decode-{model}-b{b}"] = (INFER, b, 1, 0.0, True, False)
        for name, (mode, b, length, bos, carried, me) in spec.items():
            out[name] = make_case(name, mode, b, length, h // 64, 64, bos, 0.0, carried, me, False, "bfloat16", 0)
    return out


def sweep():
    """The sweep: a grid of edge shapes, then random cases from a fixed key (name -> case)."""
    cases = []
    rng = random.Random(SWEEP_KEY)
    edges = [
        # (batch, length, groups, channels per group)
        (1, 1, 24, 64), (9, 1, 64, 64), (1, 2, 1, 64), (3, 7, 40, 64), (2, 255, 24, 64), (1, 257, 8, 128),
        (2, 1000, 3, 96), (1, 4097, 64, 64), (4, 129, 256, 32), (1, 2049, 2, 256), (5, 33, 12, 48), (1, 513, 16, 160),
    ]
    for b, length, k, cpg in edges:
        for mode in (INFER, TRAIN, BWD):
            for me in ((False, True) if mode == BWD else (False,)):
                cases.append(dict(batch=b, length=length, groups=k, cpg=cpg, mode=mode, me=me,
                                  bos=rng.choice([0.0, 1 / 64, 1.0]), padding=rng.choice([0.0, 0.0, 0.1]),
                                  carried=rng.random() < 0.5, dstate=rng.random() < 0.5,
                                  dtype=rng.choice(["bfloat16", "float16", "float32"])))
    for _ in range(40):
        mode = rng.choice([INFER, TRAIN, BWD])
        cases.append(dict(batch=rng.choice([1, 2, 3, 8]), length=rng.choice([1, 2, 31, 128, 300, 1024, 1500]),
                          groups=rng.choice([1, 4, 24, 40, 64, 100, 128]), cpg=rng.choice([32, 64, 64, 64, 96, 128]),
                          mode=mode, me=mode == BWD and rng.random() < 0.5,
                          bos=rng.choice([0.0, 1 / 256, 1 / 16, 1.0]), padding=rng.choice([0.0, 0.0, 0.05]),
                          carried=rng.random() < 0.5, dstate=rng.random() < 0.5,
                          dtype=rng.choice(["bfloat16", "float16", "float32"])))
    out = {}
    for i, c in enumerate(cases):
        if c["me"]:
            c["cpg"] = 64  # the memory-efficient backward recovers x from y with 64-channel groups
        if c["mode"] == BWD and c["dtype"] == "float16":
            # fp16 cannot hold the gradients of a fresh previous state (1e5 > 65504): from a fresh state it overflows
            # in the reference as in any kernel, so fp16 backward cases carry their state.
            c["carried"] = True
        kind = {INFER: "infer", TRAIN: "train", BWD: "bwd"}[c["mode"]]
        name = (f"{i:03d}-{kind}{'-me' if c['me'] else ''}-b{c['batch']}-l{c['length']}-g{c['groups']}x{c['cpg']}"
                f"-bos{c['bos']:.3g}-pad{c['padding']:g}{'-carried' if c['carried'] else ''}"
                f"{'-dstate' if c['dstate'] else ''}-{c['dtype']}")
        out[name] = make_case(name, c["mode"], c["batch"], c["length"], c["groups"], c["cpg"], c["bos"], c["padding"],
                          c["carried"], c["me"], c["dstate"], c["dtype"], SWEEP_KEY + i)
        # With a carried state the gradients of the previous state stay small (no bias-correction blow-up).
        if c["mode"] == BWD:
            for o in ("dprev_mean", "dprev_var"):
                out[name]["outputs"][o] = {"atol": 512.0 if not c["carried"] else 0.5, "rtol": 3e-2, "match": 1.0}
    return out


def _forward_stats(x, bos, pad, prev_count, prev_mean, prev_var, gamma, beta, groups):
    """What the forward pass saves for the backward pass: y in x's dtype and the statistics in fp32."""
    with torch.no_grad():
        y, _, _, _, cm, cr = reference.forward(x.double(), bos, prev_count, prev_mean.double(), prev_var.double(),
                                               gamma.double(), beta.double(), pad, groups, BETA1, BETA2, EPS)
    return y.to(x.dtype), cm.float(), cr.float()


def make_inputs(case, device="cuda"):
    """The positional arguments of `run` that precede the outputs, for a case."""
    dtype = getattr(torch, case["dtype"])
    B, L, H, K = case["batch"], case["length"], case["features"], case["groups"]
    g = torch.Generator(device=device).manual_seed(case["key"])
    scale = 0.5 + 1.5 * torch.rand(H, generator=g, device=device)
    shift = 0.3 * torch.randn(H, generator=g, device=device)
    x = (torch.randn(B, L, H, generator=g, device=device) * scale + shift).to(dtype)
    bos = (torch.rand(B, L, generator=g, device=device) < case["bos"]) if case["bos"] else torch.empty(
        0, device=device, dtype=torch.bool)
    pad = (torch.rand(B, L, generator=g, device=device) < case["padding"]) if case["padding"] else torch.empty(
        0, device=device, dtype=torch.bool)
    if case["carried"]:
        # Every row of a batch carries the same count, as when a prompt is processed in slices.
        count = int(torch.randint(1, 200_000, (1,), generator=g, device=device))
        prev_count = torch.full((B,), count, device=device, dtype=torch.int64)
        prev_mean = (0.3 * torch.randn(B, K, generator=g, device=device)).to(dtype)
        prev_var = (0.5 + torch.rand(B, K, generator=g, device=device)).to(dtype)
    else:
        prev_count = torch.zeros(B, device=device, dtype=torch.int64)
        prev_mean = torch.zeros(B, K, device=device, dtype=dtype)
        prev_var = torch.zeros(B, K, device=device, dtype=dtype)
    gamma = (0.5 + torch.rand(H, generator=g, device=device)).to(dtype)
    beta = (0.2 * torch.randn(H, generator=g, device=device)).to(dtype)
    empty = torch.empty(0, device=device, dtype=dtype)
    empty32 = torch.empty(0, device=device, dtype=torch.float32)
    head = [case["mode"], K, BETA1, BETA2, EPS, case["memory_efficient"]]
    if case["mode"] != BWD:
        return head + [x, bos, pad, prev_count, prev_mean, prev_var, gamma, beta, empty, empty, empty, empty32, empty32]
    y, cm, cr = _forward_stats(x, bos, pad, prev_count, prev_mean, prev_var, gamma, beta, K)
    dy = torch.randn(B, L, H, generator=g, device=device).to(dtype)
    if case["dstate"]:  # the final state feeds a later chunk
        dmean = torch.randn(B, K, generator=g, device=device).to(dtype)
        dvar = torch.randn(B, K, generator=g, device=device).to(dtype)
    else:  # in training the final mean and var feed nothing: their gradients are zero
        dmean = torch.zeros(B, K, device=device, dtype=dtype)
        dvar = torch.zeros(B, K, device=device, dtype=dtype)
    x_or_y = y if case["memory_efficient"] else x
    return head + [x_or_y, bos, pad, prev_count, prev_mean, prev_var, gamma, beta, dy, dmean, dvar, cm, cr]


def make_outputs(inputs):
    """Zero-filled output buffers for a call on these inputs, in the order of OUTPUTS."""
    mode, K = inputs[0], inputs[1]
    x, prev_count, prev_mean, gamma = inputs[6], inputs[9], inputs[10], inputs[12]
    B, L, H = x.shape
    dev, dt = x.device, x.dtype

    def one(dtype=dt):
        return torch.zeros(1, device=dev, dtype=dtype)

    if mode in (INFER, TRAIN):
        stats = ([torch.zeros(B, L, K, device=dev, dtype=torch.float32) for _ in range(2)] if mode == TRAIN
                 else [one(torch.float32), one(torch.float32)])
        return [torch.zeros_like(x), torch.zeros_like(prev_count), torch.zeros_like(prev_mean),
                torch.zeros_like(prev_mean), *stats, one(), one(), one(), one(), one()]
    return [one(), one(torch.int64), one(), one(), one(torch.float32), one(torch.float32), torch.zeros_like(x),
            torch.zeros_like(prev_mean), torch.zeros_like(prev_mean), torch.zeros_like(gamma), torch.zeros_like(gamma)]


def tolerance(case, name):
    """The tolerance output `name` is judged with: the case's atol, rtol and match, overridden by its entry."""
    return {"atol": case["atol"], "rtol": case["rtol"], "match": case["match"], **case["outputs"].get(name, {})}


def compare(outs, refs, case):
    """One row per output: (name, passed, matched ratio, max abs error, note).

    An output passes when at least `match` of its elements satisfy |out - ref| <= atol + rtol * |ref|, it has no NaN
    or Inf, and it is not all zeros where the reference is not."""
    rows = []
    for name, out, ref in zip(OUTPUTS, outs, refs):
        if out.shape != ref.shape or out.dtype != ref.dtype:
            rows.append((name, False, 0.0, math.inf, f"{tuple(out.shape)} {out.dtype}, expected "
                                                      f"{tuple(ref.shape)} {ref.dtype}"))
            continue
        x, y = out.to(ref.device, torch.float32), ref.float()
        if not torch.isfinite(x).all():
            rows.append((name, False, 0.0, math.inf, "NaN or Inf"))
            continue
        if y.abs().sum() > 0 and x.abs().sum() == 0:
            rows.append((name, False, 0.0, math.inf, "all zeros"))
            continue
        err = (x - y).abs()
        own = tolerance(case, name)
        # A count, not a float32 mean: over 1e8 elements a mean of all ones can round below 1.
        ratio = (err <= own["atol"] + own["rtol"] * y.abs()).sum().item() / err.numel()
        rows.append((name, ratio >= own["match"], ratio, err.max().item(), ""))
    return rows
