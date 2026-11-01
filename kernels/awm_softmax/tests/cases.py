"""Cases of the chunked softmax and logsumexp of xllm's Gekko adaptive working memory: shapes, inputs, destinations
and tolerances.

The entry point `run` of the extension (see ../README.md) is called positionally:

    run(mode, chunk_size, key, key_mask, prev_key, lse, dkkey, dlse,   kkey, lse_out, prev_kkey, dkey)

Every tensor is contiguous; its shape below is its memory layout. C is chunk_size, L = N * C, e is key's dtype.

- mode 0 (FWD): from key [B, L, K, S] (e) and key_mask [B, L] (bool, True: the position is filled with -inf; or
  empty) it writes kkey [B, K, S, L] (e), the softmax over each chunk of C steps, and lse_out [B, K, S, N] (fp32),
  whose slot n >= 1 is the logsumexp of chunk n - 1. With prev_key [B, C, K, S] (e, the previous call's last chunk,
  unmasked; or empty) it also writes prev_kkey [B, K, S, C] (e), that chunk's softmax, and lse_out[..., 0], its
  logsumexp. Without prev_key, lse_out[..., 0] is left as given (zero) and prev_kkey is a one-element zero that stays
  zero.
- mode 1 (BWD): from key and key_mask as in mode 0, lse [B, K, S, N] (fp32, mode 0's lse_out), dkkey
  [B, K, S, L - C] (e), the gradients of kkey's chunks 0 .. N-2, and dlse [B, K, S, N] (fp32), the gradients of lse
  (slot 0 belongs to prev_key and is not read), it writes dkey [B, L, K, S] (e). dkey must arrive zeroed.
- mode 2 (DECODE): from key [B, K, C, S] (e, one finished chunk) it writes kkey [B, K, S, C] and lse_out
  [B, K, S] (fp32).

An input a mode does not take is an empty tensor; an output it does not produce is a one-element zero that must stay
zero. reference.py says exactly what is computed.

The shapes are the calls xllm makes in Gekko (gekko-0.9B and -7.1B: K = 4 kv heads of S = 256; gekko-4B: K = 8 of
S = 128; chunk_size 4096): training is one call per layer of the whole sequence (8192 or 16384 tokens) with packed
documents, whose segments mask about 60-70% of the keys; generate() prefills in calls of up to 16384 tokens without a
mask, the second and later with the previous chunk's key; decode runs update_memory once per 4096 tokens. SWEEP adds
shapes beyond those: chunk sizes from 64 to 8192, 1 to 5 chunks, batches 1 to 3, 1 to 16 kv heads of 16 to 256
channels (also not a power of two), no mask, sparse and dense segment masks and the densest one (every chunk keeps
only its last position), a chunk masked whole in the backward pass, the previous chunk's key or none, and fp32, fp16
and bf16.
"""

import random

import torch

FWD, BWD, DECODE = 0, 1, 2
OUTPUTS = ["kkey", "lse_out", "prev_kkey", "dkey"]

CHUNK = 4096
# kv heads x head dim: gekko-0.9B and gekko-7.1B share 4 x 256, gekko-4B has 8 x 128
HEADS = {"4x256": (4, 256), "8x128": (8, 128)}
# The eos period of the training masks: an eos every 3000 tokens, one more for each row of the batch.
EOS_EVERY = 3000
SWEEP_SALT = 20261031
MAX_ELEMS = 1 << 25  # B * K * S * L per sweep case: the sweep is about shapes, not size


# kkey and dkey are e (bf16 in the main shapes), values of about 1 / C (2.4e-4 at C = 4096, where a bf16 unit is
# 9.5e-7): a few units in the last place, with an absolute floor that scales with 1 / C, below which values are the
# softmax's negligible tail. lse is fp32 of magnitude ~10: tight. dkey is the sum of the softmax's gradient and the
# logsumexp's, which xllm rounds to e each and adds in e: where the two terms cancel, its error relative to the sum is
# unbounded (up to a unit of the larger term, 4e-3 where a masked chunk leaves its softmax peaked), so 0.03-0.05% of
# xllm's own dkey elements are off by more than 4 units: dkey's match is 0.999. A kernel that sums in fp32 and rounds
# once matches all.
def tolerances(chunk_size):
    """The tolerances at this chunk size: the floors of kkey and dkey follow their values, about 1 / chunk_size."""
    f = CHUNK / chunk_size
    return {
        "atol": 1e-6 * f,
        "rtol": 1.6e-2,
        "match": 0.9999,
        "outputs": {
            "kkey": {"atol": 1e-6 * f, "rtol": 1.6e-2, "match": 0.9999},
            "lse_out": {"atol": 1e-4, "rtol": 1e-5, "match": 1.0},
            "prev_kkey": {"atol": 1e-6 * f, "rtol": 1.6e-2, "match": 0.9999},
            "dkey": {"atol": 4e-6 * f, "rtol": 1.6e-2, "match": 0.999},
        },
    }


def tolerance(tol, name):
    """The tolerance output `name` is compared with: the case's atol, rtol and match, with what its "outputs" entry
    gives that output instead."""
    return {"atol": tol["atol"], "rtol": tol["rtol"], "match": tol["match"], **tol.get("outputs", {}).get(name, {})}


# name: (mode, batch, length, heads, segments, prev_key). segments: "eos" is the training mask (an eos every
# EOS_EVERY + b tokens in row b), None no mask.
_SHAPES = {}
for _h in HEADS:
    for _b, _l in ((1, 8192), (4, 8192), (1, 16384)):
        _SHAPES[f"train_fwd_{_h}_b{_b}_l{_l}"] = (FWD, _b, _l, _h, "eos", False)
        _SHAPES[f"train_bwd_{_h}_b{_b}_l{_l}"] = (BWD, _b, _l, _h, "eos", False)
    _SHAPES[f"prefill_{_h}_b1_l16384"] = (FWD, 1, 16384, _h, None, False)
    _SHAPES[f"prefill_{_h}_b1_l16384_prev"] = (FWD, 1, 16384, _h, None, True)
    _SHAPES[f"prefill_{_h}_b1_l4096_prev"] = (FWD, 1, 4096, _h, None, True)
    _SHAPES[f"prefill_{_h}_b8_l16384"] = (FWD, 8, 16384, _h, None, False)
    for _b in (1, 8):
        _SHAPES[f"decode_{_h}_b{_b}"] = (DECODE, _b, CHUNK, _h, None, False)

# case name -> its description: mode, batch, length, chunk_size, kv_heads, head_dim, segments, prev_key, masked_chunk,
# dtype, and the tolerances at its chunk size.
CASES = {}
SHAPE_NAMES = list(_SHAPES)
for _name, (_mode, _b, _l, _h, _seg, _prev) in _SHAPES.items():
    _k, _s = HEADS[_h]
    CASES[_name] = {**tolerances(CHUNK), "mode": _mode, "batch": _b, "length": _l, "chunk_size": CHUNK,
                    "kv_heads": _k, "head_dim": _s, "segments": _seg, "prev_key": _prev, "masked_chunk": False,
                    "dtype": "bfloat16"}


def sweep_specs():
    """The sweep: a fixed grid of the edge cases, then random cases from a fixed generator, identical on every invocation."""
    out = []
    rng = random.Random(SWEEP_SALT)
    edges = [
        # (batch, chunks, chunk_size, kv heads, head dim)
        (1, 1, 4096, 4, 256), (1, 2, 4096, 8, 128), (2, 3, 1024, 4, 256), (3, 2, 256, 1, 16), (1, 5, 128, 2, 64),
        (2, 4, 64, 16, 32), (1, 2, 8192, 2, 96), (1, 1, 512, 3, 48), (2, 2, 2048, 4, 160), (1, 3, 4096, 1, 256),
    ]
    for b, n, c, k, s in edges:
        for mode in (FWD, BWD):
            for segs in ((None, "dense") if mode == FWD else ("sparse", "only_last")):
                out.append(dict(batch=b, chunks=n, chunk=c, heads=k, dim=s, mode=mode, segments=segs,
                                prev=mode == FWD and rng.random() < 0.5, masked_chunk=False,
                                dtype=rng.choice(["bfloat16", "float16", "float32"])))
        out.append(dict(batch=b, chunks=1, chunk=c, heads=k, dim=s, mode=DECODE, segments=None, prev=False,
                        masked_chunk=False, dtype=rng.choice(["bfloat16", "float16", "float32"])))
    for _ in range(36):
        mode = rng.choice([FWD, FWD, BWD, BWD, DECODE])
        c = dict(batch=rng.choice([1, 2, 3]), chunks=rng.choice([1, 2, 2, 3, 4]),
                 chunk=rng.choice([64, 128, 256, 1024, 4096]), heads=rng.choice([1, 2, 4, 5, 8]),
                 dim=rng.choice([16, 32, 64, 80, 128, 256]), mode=mode,
                 segments=None if mode == DECODE else rng.choice([None, "sparse", "dense", "only_last"]),
                 prev=mode == FWD and rng.random() < 0.5,
                 masked_chunk=mode == BWD and rng.random() < 0.3,
                 dtype=rng.choice(["bfloat16", "float16", "float32"]))
        out.append(c)
    for c in out:
        if c["mode"] == DECODE:
            c["chunks"] = 1
        if c["masked_chunk"] and c["chunks"] < 2:
            c["chunks"] = 2
        while c["batch"] * c["heads"] * c["dim"] * c["chunks"] * c["chunk"] > MAX_ELEMS:
            c["chunks"] = max(1, c["chunks"] - 1)
            if c["batch"] > 1:
                c["batch"] -= 1
            elif c["chunks"] == 1:
                c["chunk"] //= 2
    return out


def eos_rate(kind, chunk):
    """The probability of an eos at each step, per kind of mask relative to the chunk: segments of about 3 chunks
    ("sparse"), of about a third of a chunk ("dense"), and of one step ("only_last": each chunk keeps one position)."""
    return {"sparse": 1 / (3 * chunk), "dense": 3 / chunk, "only_last": 1.0}[kind]


def _sweep_name(i, c):
    mode = {FWD: "fwd", BWD: "bwd", DECODE: "decode"}[c["mode"]]
    return (f"{i:03d}-{mode}-b{c['batch']}-n{c['chunks']}-c{c['chunk']}-k{c['heads']}-s{c['dim']}"
            f"-{c['segments'] or 'nomask'}{'-prev' if c['prev'] else ''}{'-maskedchunk' if c['masked_chunk'] else ''}"
            f"-{c['dtype']}")


SWEEP_NAMES = []
for _i, _c in enumerate(sweep_specs()):
    _name = _sweep_name(_i, _c)
    CASES[_name] = {**tolerances(_c["chunk"]), "mode": _c["mode"], "batch": _c["batch"],
                    "length": _c["chunks"] * _c["chunk"], "chunk_size": _c["chunk"], "kv_heads": _c["heads"],
                    "head_dim": _c["dim"],
                    "segments": None if _c["segments"] is None else eos_rate(_c["segments"], _c["chunk"]),
                    "prev_key": _c["prev"], "masked_chunk": _c["masked_chunk"], "dtype": _c["dtype"]}
    SWEEP_NAMES.append(_name)


def rng_salt(name):
    """The salt of a case's random inputs: its position among the shapes, or SWEEP_SALT + its position in the sweep."""
    if name in SWEEP_NAMES:
        return SWEEP_SALT + SWEEP_NAMES.index(name)
    return SHAPE_NAMES.index(name)


def chunk_mask(segment_idx, chunk_size):
    """xllm's key_mask from segment_idx [B, L]: a position is masked when its segment is not the one its chunk ends in.
    A chunk's last position is never masked, so no row of a chunk is entirely -inf."""
    B, L = segment_idx.shape
    seg = segment_idx.view(B, L // chunk_size, chunk_size)
    return torch.ne(seg, seg[:, :, -1:]).view(B, L)


def segments(w, g, device):
    """segment_idx [B, L] for the case's mask, or None. "eos": the training eos period; a float p: an eos at each step
    with probability p (p = 1 makes every position its own segment, so that each chunk keeps only its last
    position)."""
    B, L, kind = w["batch"], w["length"], w["segments"]
    if kind is None:
        return None
    t = torch.arange(L, device=device)
    if kind == "eos":
        rows = []
        for b in range(B):
            period = EOS_EVERY + b
            phase = int(torch.randint(0, period, (1,), generator=g, device=device))
            rows.append((t + phase) // period)
        return torch.stack(rows)
    eos = torch.rand(B, L, generator=g, device=device) < float(kind)
    return torch.cumsum(eos.long(), dim=1)


def _key(shape, K, S, dtype, g, device):
    """A key with a scale and an offset of its own per (head, channel), along an axis of time; shape names K and S."""
    scale = 0.5 + 1.5 * torch.rand(K, S, generator=g, device=device)
    shift = 0.5 * torch.randn(K, S, generator=g, device=device)
    x = torch.randn(*shape, generator=g, device=device)
    if shape[1] == K:  # [B, K, C, S]
        x = x * scale[None, :, None, :] + shift[None, :, None, :]
    else:  # [B, L, K, S]
        x = x * scale + shift
    return x.to(dtype)


def chunk_lse(key, key_mask, chunk_size):
    """lse_out as mode 0 writes it for key [B, L, K, S], slot 0 left zero: the logsumexp of chunk n - 1 in slot n,
    computed in fp64 and rounded to fp32."""
    B, L, K, S = key.shape
    N = L // chunk_size
    k = key.double().permute(0, 2, 3, 1)  # B x K x S x L
    if key_mask is not None and key_mask.numel():
        k = k.masked_fill(key_mask.view(B, 1, 1, L), float("-inf"))
    out = torch.zeros(B, K, S, N, dtype=torch.float32, device=key.device)
    if N > 1:
        out[..., 1:] = torch.logsumexp(k.reshape(B, K, S, N, chunk_size)[:, :, :, :-1], dim=-1).float()
    return out


def make_inputs(name, device="cuda"):
    """The positional arguments of one call of case `name` before the destinations: mode, chunk_size, key, key_mask,
    prev_key, lse, dkkey, dlse."""
    w = CASES[name]
    dtype = getattr(torch, w["dtype"])
    B, L, C, K, S = w["batch"], w["length"], w["chunk_size"], w["kv_heads"], w["head_dim"]
    N = L // C
    g = torch.Generator(device=device).manual_seed(rng_salt(name))
    empty = torch.empty(0, device=device, dtype=dtype)
    empty32 = torch.empty(0, device=device, dtype=torch.float32)
    nomask = torch.empty(0, device=device, dtype=torch.bool)
    head = [w["mode"], C]

    if w["mode"] == DECODE:
        key = _key((B, K, C, S), K, S, dtype, g, device)
        return head + [key, nomask, empty, empty32, empty, empty32]

    key = _key((B, L, K, S), K, S, dtype, g, device)
    seg = segments(w, g, device)
    mask = chunk_mask(seg, C) if seg is not None else nomask
    if w["mode"] == FWD:
        prev = _key((B, C, K, S), K, S, dtype, g, device) if w["prev_key"] else empty
        return head + [key, mask, prev, empty32, empty, empty32]

    if w["masked_chunk"] and N > 1:
        # A chunk masked whole: its softmax and logsumexp are torch's NaN and -inf, and the backward pass's
        # masked_fill makes its dkey zero.
        c = int(torch.randint(0, N - 1, (1,), generator=g, device=device))
        mask = mask.clone() if mask.numel() else torch.zeros(B, L, device=device, dtype=torch.bool)
        mask[:, c * C:(c + 1) * C] = True
    lse = chunk_lse(key, mask, C)
    # The gradients as xllm's backward pass makes them: kkey's from a bf16 bmm, the logsumexp's in fp32.
    dkkey = torch.randn(B, K, S, L - C, generator=g, device=device).to(dtype)
    dlse = torch.randn(B, K, S, N, generator=g, device=device)
    return head + [key, mask, empty, lse, dkkey, dlse]


def make_outputs(inputs):
    """Zero-filled destinations [kkey, lse_out, prev_kkey, dkey] for a call on these inputs."""
    mode, C, key, prev = inputs[0], inputs[1], inputs[2], inputs[4]
    dev, dt = key.device, key.dtype
    one = lambda dtype=dt: torch.zeros(1, device=dev, dtype=dtype)  # noqa: E731
    if mode == DECODE:
        B, K, _, S = key.shape
        return [torch.zeros(B, K, S, C, device=dev, dtype=dt), torch.zeros(B, K, S, device=dev, dtype=torch.float32),
                one(), one()]
    B, L, K, S = key.shape
    if mode == FWD:
        prev_kkey = torch.zeros(B, K, S, C, device=dev, dtype=dt) if prev.numel() else one()
        return [torch.zeros(B, K, S, L, device=dev, dtype=dt),
                torch.zeros(B, K, S, L // C, device=dev, dtype=torch.float32), prev_kkey, one()]
    return [one(), one(torch.float32), one(), torch.zeros_like(key)]
