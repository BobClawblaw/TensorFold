"""MiniMax-M3.1-Flash weight requantizations against the exact release: full-model logit KL, many variants, one pass."""


import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimax_m3_ref import BF, Layer, bf16_from_bits, gemma_norm, rope_tables  # noqa: E402
from minimax_m3_nvfp4_mix import LM, Source  # noqa: E402

USAGE = """
  python minimax_m3_kl_study.py MODEL_DIR --ids corpus.json --out results/ [--layers 4] [--sanity]

Each layer's weights load once (minimax_m3_ref.py's Layer: MXFP8 dense dequantized, router, norms); the experts load
once per source (the release's NVFP4, the checkpoint's stored 3-bit affine refit, or the refit computed on the fly for
layers that ship NVFP4 only); then every variant applies its own weight transform (fake-quant of dense projections)
and runs the layer on its own residual stream. At the end each variant's head (bf16, MXFP8 or NVFP4 fake-quant) gives
logits in row chunks and the metrics against `exact` (and `served`) are accumulated; full logits are never kept.
"""

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
E2M1_MID = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)

# variant: (experts: nv = the release's NVFP4, served = 3-bit on affine3_layers, aff = 3-bit; dense groups; head)
DENSE_GROUPS = {"qo": ("q", "o"), "mlp": ("gate_w", "up_w", "down_w", "s_gate", "s_up", "s_down"),
                "q": ("q",), "o": ("o",), "gu": ("gate_w", "up_w", "s_gate", "s_up"), "down": ("down_w", "s_down"),
                "kvi": ("k", "v", "iq", "ik")}
VARIANTS = {
    "exact": ("nv", (), "bf16"),
    "served": ("served", (), "bf16"),
    "served+head8": ("served", (), "mx8"),
    "served+head4": ("served", (), "nv4"),
    "served+qo4": ("served", ("qo",), "bf16"),
    "served+mlp4": ("served", ("mlp",), "bf16"),
    "served+kvi4": ("served", ("kvi",), "bf16"),
    "served+dense4": ("served", ("qo", "mlp", "kvi"), "mx8"),
    "all3": ("aff", (), "bf16"),
}
# round 2: dense groups as (group, method); "mse" = NVFP4 with the MSE-searched block scale, "nv4" = plain
VARIANTS2 = {
    "exact": ("nv", (), "bf16"),
    "served": ("served", (), "bf16"),
    "served+head8": ("served", (), "mx8"),
    "served+head8+q4": ("served", (("q", "mse"),), "mx8"),
    "served+head8+o4": ("served", (("o", "mse"),), "mx8"),
    "served+head8+gu4": ("served", (("gu", "mse"),), "mx8"),
    "served+head8+down4": ("served", (("down", "mse"),), "mx8"),
    "served+head8+mlp4": ("served", (("mlp", "mse"),), "mx8"),
    "served+head8+qo4": ("served", (("qo", "mse"),), "mx8"),
    "served+head8+qo4+mlp4": ("served", (("qo", "mse"), ("mlp", "mse")), "mx8"),
    "served+head8+qo4+mlp4(plain)": ("served", (("qo", "nv4"), ("mlp", "nv4")), "mx8"),
}

# round 3: the converter's --dense-fp4 kinds, scales searched, head MXFP8 (one global a tensor, not a group)
VARIANTS3 = {
    "exact": ("nv", (), "bf16"),
    "served": ("served", (), "bf16"),
    "served+head8": ("served", (), "mx8"),
    "served+head8+all4": ("served", (("q", "mse"), ("kvi", "mse"), ("o", "mse"), ("mlp", "mse")), "mx8"),
    "served+head8+qkv4+mlp4": ("served", (("q", "mse"), ("kvi", "mse"), ("mlp", "mse")), "mx8"),
}


def dense_spec(groups):
    """A variant's dense groups as [(group, method)]; round 1's bare names mean plain NVFP4."""
    return [g if isinstance(g, tuple) else (g, "nv4") for g in groups]


# ---------------------------------------------------------------- fake quants (dequantized values, bf16 out)

def fq_mxfp8(w: torch.Tensor, rows=8192) -> torch.Tensor:
    """e4m3 codes, one UE8M0 scale per 32 inputs: scale = 2^ceil(log2(amax/448)), codes = e4m3 cast of w/scale."""
    out = torch.empty(w.shape, dtype=BF, device=w.device)
    n, k = w.shape
    for r0 in range(0, n, rows):
        x = w[r0:r0 + rows].float().view(-1, k // 32, 32)
        amax = x.abs().amax(-1, keepdim=True)
        e = torch.ceil(torch.log2(torch.clamp(amax, min=1e-30) / 448.0))
        s = torch.where(amax > 0, torch.exp2(e), torch.ones_like(amax))
        q = (x / s).to(torch.float8_e4m3fn).float()
        out[r0:r0 + rows] = (q * s).view(-1, k).to(BF)
    return out


def e2m1_round(a: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    """|y| (clamped to 6) to the nearest E2M1 code index 0..7, exact ties to the even code index."""
    lo = torch.bucketize(a, mid)  # ties to the lower code
    hi = torch.bucketize(a, mid, right=True)  # ties to the upper code
    return torch.where((lo != hi) & (lo % 2 == 1), hi, lo)


def fq_nvfp4_mse(w: torch.Tensor, rows=4096, down=6, up=1) -> torch.Tensor:
    """NVFP4, each block's e4m3 code from c0+up down to c0-down by least squared error (ties: larger); s2 as plain."""
    dev = w.device
    grid, mid = E2M1.to(dev), E2M1_MID.to(dev)
    out = torch.empty(w.shape, dtype=BF, device=dev)
    n, k = w.shape
    s2 = w.float().abs().amax() / (448.0 * 6.0)
    s2 = s2 if s2 > 0 else torch.ones((), device=dev)
    for r0 in range(0, n, rows):
        x = w[r0:r0 + rows].float().view(-1, k // 16, 16)
        amax = x.abs().amax(-1, keepdim=True)
        c0 = torch.clamp(amax / 6.0 / s2, max=448.0).to(torch.float8_e4m3fn).view(torch.uint8).long()  # 0..126
        best_err = torch.full_like(amax, float("inf"))
        best = torch.zeros_like(x)
        for dc in range(up, -down - 1, -1):  # larger scales first: a later code must be strictly better
            c = c0 + dc
            ok = (c >= 1) & (c <= 126)
            sc = torch.clamp(c, 1, 126).to(torch.uint8).view(torch.float8_e4m3fn).float()
            d = sc * s2
            y = x / d
            q = grid[e2m1_round(torch.clamp(y.abs(), max=6.0), mid)] * torch.sign(y)
            deq = (q * d).to(BF).float()  # judged as stored (bf16), so it is never worse than plain
            err = ((deq - x) ** 2).sum(-1, keepdim=True)
            err = torch.where(ok, err, torch.full_like(err, float("inf")))
            better = err < best_err
            best = torch.where(better, deq, best)
            best_err = torch.where(better, err, best_err)
        best = torch.where(torch.isinf(best_err), torch.zeros_like(best), best)  # no valid code: an all-zero block
        out[r0:r0 + rows] = best.view(-1, k).to(BF)
    return out


def fq_nvfp4(w: torch.Tensor, rows=8192) -> torch.Tensor:
    """NVFP4: s2 = amax/(448*6); e4m3 block scales (16 inputs) of amax_b/6/s2; E2M1 codes round to nearest."""
    dev = w.device
    grid, mid = E2M1.to(dev), E2M1_MID.to(dev)
    out = torch.empty(w.shape, dtype=BF, device=dev)
    n, k = w.shape
    s2 = w.float().abs().amax() / (448.0 * 6.0)
    s2 = s2 if s2 > 0 else torch.ones((), device=dev)
    for r0 in range(0, n, rows):
        x = w[r0:r0 + rows].float().view(-1, k // 16, 16)
        amax = x.abs().amax(-1, keepdim=True)
        sc = torch.clamp(amax / 6.0 / s2, max=448.0).to(torch.float8_e4m3fn).float()
        d = sc * s2
        safe = torch.where(d > 0, d, torch.ones_like(d))
        y = x / safe
        a = torch.clamp(y.abs(), max=6.0)
        idx = e2m1_round(a, mid)
        q = grid[idx] * torch.sign(y)
        q = torch.where(d > 0, q, torch.zeros_like(q))
        out[r0:r0 + rows] = (q * d).view(-1, k).to(BF)
    return out


def sanity(dev):
    g = torch.Generator(device="cpu").manual_seed(0)
    w = torch.randn(4096, 4096, generator=g).to(dev)
    res = {}
    for name, f in (("mxfp8", fq_mxfp8), ("nvfp4", fq_nvfp4), ("nvfp4_mse", fq_nvfp4_mse)):
        y = f(w).float()
        res[name] = float((y - w).norm() / w.norm())
    # the e2m1 grid itself: values on it survive nvfp4 exactly when scales land exactly (amax 6 per block)
    v = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -6, -4, -3, -2, -1.5, -1, -.5, 6], device=dev).repeat(64, 1)
    res["nvfp4_grid_roundtrip_maxerr"] = float((fq_nvfp4(v).float() - v).abs().max())
    res["nvfp4_mse_grid_roundtrip_maxerr"] = float((fq_nvfp4_mse(v).float() - v).abs().max())
    res["nvfp4_mse_never_worse"] = bool((((fq_nvfp4_mse(w).float() - w) ** 2).view(-1, 16).sum(-1)
                                         <= ((fq_nvfp4(w).float() - w) ** 2).view(-1, 16).sum(-1) + 1e-9).all())
    v8 = torch.tensor([2.0 ** i for i in range(-4, 4)] * 4, device=dev).repeat(32, 1)
    res["mxfp8_pow2_roundtrip_maxerr"] = float((fq_mxfp8(v8).float() - v8).abs().max())
    print("sanity (relative Frobenius error, gaussian 4096x4096):", json.dumps(res), flush=True)
    return res


# ---------------------------------------------------------------- experts

def refit_gpu(w: torch.Tensor, bits=3, group=64, iters=8) -> torch.Tensor:
    """minimax_m3_nvfp4_mix.refit(values=True) for a GPU tensor: the 3-bit affine refit's values (bf16 rounded)."""
    n = 2 ** bits - 1
    x = w.float().reshape(-1, group)
    lo, hi = x.min(1, keepdim=True).values, x.max(1, keepdim=True).values
    s, b = torch.clamp((hi - lo) / n, min=1e-8), lo
    xm = x.mean(1, keepdim=True)
    for _ in range(iters):
        q = torch.clamp(torch.round((x - b) / s), 0, n)
        qm = q.mean(1, keepdim=True)
        var = ((q - qm) ** 2).sum(1, keepdim=True)
        fit = ((q - qm) * (x - xm)).sum(1, keepdim=True) / torch.clamp(var, min=1e-12)
        s = torch.where((var > 0) & (fit > 1e-8), fit, s)
        b = xm - s * qm
    s, b = s.to(torch.bfloat16).float(), b.to(torch.bfloat16).float()
    q = torch.clamp(torch.round((x - b) / s), 0, n)
    return (q * s + b).to(torch.bfloat16).reshape(w.shape)


def nvfp4_gpu(src: Source, name: str, dev) -> torch.Tensor:
    """minimax_m3_nvfp4_mix.dequant_nvfp4 on the GPU: packed codes, e4m3 block scales, dividing global scale."""
    from minimax_m3_nvfp4_mix import E4M3, FP4
    p = torch.from_numpy(src.np(name + ".weight_packed").copy()).to(dev)
    s = torch.from_numpy(src.np(name + ".weight_scale").copy()).to(dev).long()
    gs = float(src.np(name + ".weight_global_scale"))
    fp4, e4 = torch.from_numpy(FP4).to(dev), torch.from_numpy(E4M3).to(dev)
    v = torch.stack([fp4[(p & 0xF).long()], fp4[(p >> 4).long()]], -1).view(p.shape[0], -1)
    return v * (e4[s] / gs).repeat_interleave(16, dim=-1)


def unpack_affine(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, bits=3, group=64) -> torch.Tensor:
    """MLX affine words [n, k*bits/32] (32 codes in `bits` words, low bits first), bf16 scales/biases: bf16 [n, k]."""
    n = words.shape[0]
    w = words.long().view(n, -1, bits)  # [n, k/32, bits]
    codes = []
    mask = (1 << bits) - 1
    for i in range(32):
        o = i * bits
        j, sh = o // 32, o % 32
        c = (w[..., j] >> sh) & mask
        if sh + bits > 32:
            c = c | ((w[..., j + 1] << (32 - sh)) & mask)
        codes.append(c)
    q = torch.stack(codes, -1).view(n, -1).float()  # [n, k]
    k = q.shape[1]
    x = q.view(n, k // group, group) * scales.float()[..., None] + biases.float()[..., None]
    return x.view(n, k).to(BF)


def load_experts(src: Source, layer: int, text: dict, dev, want_nv: bool, want_aff: bool, log):
    """{e: (w1, w3, w2) bf16} for the NVFP4 release values and/or the 3-bit refit (stored if present, else computed)."""
    m = f"{LM}model.layers.{layer}.block_sparse_moe."
    E, I = text["num_local_experts"], text["intermediate_size"]
    nv, aff = {}, {}
    stored = f"{m}experts_affine.gate_up_proj.weight" in src.where
    if want_aff and stored:
        t = {}
        for part in ("gate_up_proj", "down_proj"):
            for kind in ("weight", "scales", "biases"):
                nm = f"{m}experts_affine.{part}.{kind}"
                if kind == "weight":  # U32 words (Source.np has no U32): read as int32, widened when unpacked
                    a = np.frombuffer(src.raw(nm), dtype=np.int32).reshape(src.info(nm)["shape"])
                else:
                    a = src.np(nm).view(np.int16)
                t[part, kind] = torch.from_numpy(a.copy()).to(dev)
        for e in range(E):
            gu = unpack_affine(t["gate_up_proj", "weight"][e].long() & 0xFFFFFFFF,
                               t["gate_up_proj", "scales"][e].view(BF), t["gate_up_proj", "biases"][e].view(BF))
            dn = unpack_affine(t["down_proj", "weight"][e].long() & 0xFFFFFFFF,
                               t["down_proj", "scales"][e].view(BF), t["down_proj", "biases"][e].view(BF))
            aff[e] = [gu[:I], gu[I:], dn]
        del t
    for e in range(E):
        if not want_nv and (stored or not want_aff):
            break
        ws = [nvfp4_gpu(src, f"{m}experts.{e}.{w}", dev) for w in ("w1", "w3", "w2")]
        if want_aff and not stored:  # the ref tool's on-the-fly refit (values as the kernels compute them)
            aff[e] = [refit_gpu(x).to(BF) for x in ws]
        if want_nv:
            nv[e] = [x.to(BF) for x in ws]
        del ws
    if want_aff and stored and want_nv and layer == 3:  # the stored refit is the ref tool's refit
        for e in (0, 77):
            for j, w in enumerate(("w1", "w3", "w2")):
                x = nvfp4_gpu(src, f"{m}experts.{e}.{w}", dev)
                r = refit_gpu(x).to(BF)
                d = (r.float() - aff[e][j].float()).abs()
                log(f"  check layer 3 expert {e} {w}: stored vs on-the-fly refit max|d| {float(d.max()):.3g}, "
                    f"mismatch frac {float((d > 0).float().mean()):.2e}")
    return nv, aff


# ---------------------------------------------------------------- the pass

def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers (smoke run)")
    ap.add_argument("--variants", default="", help="comma list (default: the round's whole set)")
    ap.add_argument("--round", type=int, default=1, help="1: the first variant set, 2: the mse/split set, 3: the engine's plans")
    ap.add_argument("--tail", type=int, default=1000, help="also report over the last N rows")
    ap.add_argument("--chunk", type=int, default=128, help="logit rows per chunk")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t_start = time.time()
    global VARIANTS
    if a.round == 2:
        VARIANTS = VARIANTS2
    elif a.round == 3:
        VARIANTS = VARIANTS3
    names = a.variants.split(",") if a.variants else list(VARIANTS)
    assert names[0] == "exact", "exact is the reference and runs first"
    for v in names:
        assert v in VARIANTS, v
    sanity_res = sanity(dev)
    log(f"sanity {json.dumps(sanity_res)}")
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    text = cfg.get("text_config", cfg)
    mix = json.load(open(os.path.join(a.model_dir, "mix.json")))
    affine3 = set(mix["affine3_layers"])
    log(f"nvfp4_layers {mix['nvfp4_layers']} affine3_layers {sorted(affine3)}")
    src = Source(a.model_dir)
    ids = json.load(open(a.ids))
    ids = ids["ids"] if isinstance(ids, dict) else ids
    T = len(ids)
    n_layers = a.layers or text["num_hidden_layers"]
    log(f"{T} tokens, {n_layers} layers, variants {names}")
    pos = torch.arange(T, device=dev)
    cos, sin = rope_tables(pos, text["rotary_dim"], text["rope_theta"], dev)
    embed = bf16_from_bits(src.np(f"{LM}model.embed_tokens.weight"))
    r0 = embed[torch.tensor(ids)].to(dev)
    del embed
    res = {v: r0.clone() for v in names}

    with torch.no_grad():
        for i in range(n_layers):
            t = time.time()
            layer = Layer(src, i, text, dev, False)
            base = {n: getattr(layer, n) for g in DENSE_GROUPS.values() for n in g if hasattr(layer, n)}
            fq = {}  # lazily: nvfp4 fake-quant of each dense projection, shared by the variants that use it
            nv, aff = {}, {}
            if layer.moe:
                kinds = {VARIANTS[v][0] for v in names}
                need_nv = "nv" in kinds or ("served" in kinds and i not in affine3)
                need_aff = "aff" in kinds or ("served" in kinds and i in affine3)
                nv, aff = load_experts(src, i, text, dev, need_nv, need_aff, log)
            t_load = time.time() - t
            for v in names:
                ex, groups, _ = VARIANTS[v]
                for n, w in base.items():
                    setattr(layer, n, w)
                for g, method in dense_spec(groups):
                    for n in DENSE_GROUPS[g]:
                        if n in base:
                            if (n, method) not in fq:
                                fq[n, method] = (fq_nvfp4_mse if method == "mse" else fq_nvfp4)(base[n])
                            setattr(layer, n, fq[n, method])
                if layer.moe:
                    use_aff = ex == "aff" or (ex == "served" and i in affine3)
                    layer.cache = aff if use_aff else nv
                res[v] = layer.forward(res[v], pos, cos, sin)
            d = {v: float((res[v].float() - res["exact"].float()).norm() / res["exact"].float().norm())
                 for v in names if v != "exact"} if "exact" in names else {}
            log(f"layer {i:2d} {'moe' if layer.moe else 'dense'} "
                f"{'3bit' if i in affine3 else 'nvfp4' if layer.moe else ''} load {t_load:.1f} s total "
                f"{time.time() - t:.1f} s mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB | resid rel-err "
                + " ".join(f"{k}={x:.4f}" for k, x in d.items()))
            del layer, base, fq, nv, aff
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ------------------------------------------------------------ logits and metrics
        t = time.time()
        norm_w = bf16_from_bits(src.np(f"{LM}model.norm.weight")).to(dev)
        head_bf = bf16_from_bits(src.np(f"{LM}lm_head.weight")).to(dev)
        heads = {"bf16": head_bf}
        need = {VARIANTS[v][2] for v in names}
        if "mx8" in need:
            heads["mx8"] = fq_mxfp8(head_bf)
        if "nv4" in need:
            heads["nv4"] = fq_nvfp4(head_bf)
        for k, h in heads.items():
            if k != "bf16":
                log(f"head {k} relative error {float((h.float() - head_bf.float()).norm() / head_bf.float().norm()):.4f}")
        xs = {v: gemma_norm(res[v], norm_w) for v in names}
        tail0 = max(0, T - a.tail)
        # per row: KL(exact||v), KL(served||v), top1 agree with exact
        kl_e = {v: torch.zeros(T, device=dev) for v in names}
        kl_s = {v: torch.zeros(T, device=dev) for v in names}
        top1 = {v: torch.zeros(T, dtype=torch.bool, device=dev) for v in names}
        for c0 in range(0, T, a.chunk):
            c1 = min(T, c0 + a.chunk)
            lp = {}
            lp["exact"] = torch.log_softmax((xs["exact"][c0:c1] @ heads[VARIANTS["exact"][2]].T).float(), -1)
            if "served" in names:
                lp["served"] = torch.log_softmax((xs["served"][c0:c1] @ heads["bf16"].T).float(), -1)
            ref_e = lp["exact"]
            pe = ref_e.exp()
            am_e = ref_e.argmax(-1)
            ps = lp["served"].exp() if "served" in lp else None
            for v in names:
                lv = lp[v] if v in lp else torch.log_softmax((xs[v][c0:c1] @ heads[VARIANTS[v][2]].T).float(), -1)
                kl_e[v][c0:c1] = (pe * (ref_e - lv)).sum(-1)
                if ps is not None:
                    kl_s[v][c0:c1] = (ps * (lp["served"] - lv)).sum(-1)
                top1[v][c0:c1] = lv.argmax(-1) == am_e
                del lv
            del lp, pe, ps
        out = {}
        for v in names:
            row = {}
            for tag, sl in (("all", slice(0, T)), ("tail", slice(tail0, T))):
                row[f"kl_exact_{tag}"] = float(kl_e[v][sl].mean())
                row[f"top1_{tag}"] = 100.0 * float(top1[v][sl].float().mean())
                if v.startswith("served+") or v == "all3":  # vs served
                    row[f"kl_served_{tag}"] = float(kl_s[v][sl].mean())
                row[f"kl_exact_{tag}_p99"] = float(torch.quantile(kl_e[v][sl], 0.99))
            out[v] = row
        log(f"logits+metrics {time.time() - t:.1f} s")
    wall = time.time() - t_start
    meta = {"model_dir": os.path.basename(os.path.normpath(a.model_dir)), "tokens": T, "tail_rows": T - tail0, "layers": n_layers, "wall_s": wall,
            "sanity": sanity_res, "nvfp4_layers": mix["nvfp4_layers"], "affine3_layers": sorted(affine3),
            "variants": {v: {"experts": VARIANTS[v][0], "nvfp4_dense": [f"{n}:{m}" for g, m in dense_spec(VARIANTS[v][1]) for n in DENSE_GROUPS[g]],
                             "head": VARIANTS[v][2]} for v in names}}
    json.dump({"meta": meta, "metrics": out}, open(os.path.join(a.out, "kl_study.json"), "w"), indent=1)
    lines = [f"# MiniMax-M3.1-Flash requant KL study ({n_layers} layers, {T} tokens, tail = last {T - tail0} rows)", "",
             "KL in nats (mean over rows, full vocab, fp32); top-1 = argmax agreement with `exact`.", "",
             "| variant | KL(exact‖v) all | KL(exact‖v) tail | top-1 % all | top-1 % tail | KL(served‖v) all | KL(served‖v) tail | p99 KL all |",
             "|---|---|---|---|---|---|---|---|"]
    for v in names:
        r = out[v]
        ks = lambda k: f"{r[k]:.5f}" if k in r else "-"
        lines.append(f"| {v} | {r['kl_exact_all']:.5f} | {r['kl_exact_tail']:.5f} | {r['top1_all']:.2f} | "
                     f"{r['top1_tail']:.2f} | {ks('kl_served_all')} | {ks('kl_served_tail')} | {r['kl_exact_all_p99']:.4f} |")
    lines += ["", f"wall time {wall / 60:.1f} min"]
    open(os.path.join(a.out, "kl_study.md"), "w").write("\n".join(lines) + "\n")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
