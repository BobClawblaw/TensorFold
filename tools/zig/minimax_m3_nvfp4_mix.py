"""MiniMax-M3.1-Flash's release as the Sparks' mixed checkpoint: NVFP4 experts on chosen layers, 3-bit elsewhere."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time

USAGE = """
Every tensor of the release (MiniMaxAI/MiniMax-M3.1-Flash) kept as released, except the routed experts of the listed layers, refit to 3-bit affine (groups of 64) so two 121 GiB
GB10 ranks hold the model beside a usable cache. The layers that keep nvfp4 are the ones whose experts lose most at 3 bits
(tools/zig/minimax_m3_requant_study.py ranks them); see docs/recipes/minimax-m3.1-flash.md.

  python tools/zig/minimax_m3_nvfp4_mix.py HF_DIR OUT_DIR --nvfp4-layers 20-29,31,52,59 [--layers 3-10] [--check]
  python tools/zig/minimax_m3_nvfp4_mix.py --check OUT_DIR          # every file against minimax_m3_checkpoints.json
  python tools/zig/minimax_m3_nvfp4_mix.py HF_DIR OUT_DIR --nvfp4-layers ... --dense-fp4 [--affine-from EARLIER_OUT]

Output (OUT_DIR):
  model-NNNNN.safetensors, model.safetensors.index.json   release names and dtypes for every kept tensor; a 3-bit layer's
      128 experts as language_model.model.layers.I.block_sparse_moe.experts_affine.{gate_up_proj,down_proj}.{weight,
      scales,biases}: MLX affine words u32 [128, n, k*3/32] (32 codes fill 3 words, low bits first; w = s*q + b), bf16
      scales and biases [128, n, k/64]; gate_up stacks each expert's w1 (gate) rows then w3 (up) rows
  with --dense-fp4: every decoder projection (q, k, v, index q/k, o; the dense and shared MLPs) NVFP4 in the release's
      expert format instead of its MXFP8 .weight/.weight_scale_inv: .weight_packed u8 [n, k/2] (low nibble first),
      .weight_scale e4m3 [n, k/16], .weight_global_scale f32 [] (dividing). Each block's scale is the e4m3 code, of
      the one its largest value fills, the next above and six below, that leaves the least squared error (ties: the
      larger). Projections the engine stacks share one global (q|k|v|index q|index k; gate|up), 2688/amax over the
      group. The head, lm_head, becomes MXFP8: .weight e4m3 [vocab, hidden], .weight_scale_inv u8 [vocab, hidden/32]
      (e8m0: the power of two that keeps each 32 at or under 448). The DSpark drafter stays as released.
  mix.json                                                the source revision, the layer sets, the group, every output
      file's sha256 (the --check input)
  config.json, tokenizer and template files, dspark/      copied from the release
"""

import numpy as np

try:
    import torch
except ImportError:  # the refit runs in numpy, slowly
    torch = None

LM = "language_model."
KEPT_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
              "preprocessor_config.json", "configuration_minimax_m3_vl.py", "image_processor.py",
              "processing_minimax.py", "video_processor.py", "LICENSE", "README.md")
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


class Source:
    """A safetensors folder read by name: headers once, tensors as raw bytes or numpy."""

    def __init__(self, folder: str, index="model.safetensors.index.json"):
        self.folder = folder
        wm = json.load(open(os.path.join(folder, index)))["weight_map"]
        self.where = {}
        self.headers = {}
        for shard in sorted(set(wm.values())):
            with open(os.path.join(folder, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            self.headers[shard] = (8 + n, h)
            for name in h:
                self.where[name] = shard

    def info(self, name: str) -> dict:
        return self.headers[self.where[name]][1][name]

    def raw(self, name: str) -> bytes:
        shard = self.where[name]
        base, h = self.headers[shard]
        a, b = h[name]["data_offsets"]
        with open(os.path.join(self.folder, shard), "rb") as f:
            f.seek(base + a)
            return f.read(b - a)

    def np(self, name: str) -> np.ndarray:
        info = self.info(name)
        dt = {"F32": np.float32, "BF16": np.uint16, "F8_E4M3": np.uint8, "U8": np.uint8, "F16": np.float16}[info["dtype"]]
        return np.frombuffer(self.raw(name), dtype=dt).reshape(info["shape"])

    def names(self, prefix: str) -> list[str]:
        return sorted(n for n in self.where if n.startswith(prefix))


def e4m3_table() -> np.ndarray:
    """e4m3fn: 256 byte codes to float (no infinities; 0x7f/0xff are NaN in the format, unused by the release)."""
    t = np.zeros(256, dtype=np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 0xF, c & 7
        v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        t[c] = -v if s else v
    return t


E4M3 = e4m3_table()


def dequant_nvfp4(packed: np.ndarray, scale: np.ndarray, gscale: np.ndarray, block=16) -> np.ndarray:
    """Codes [..., n, k/2] (low nibble first), e4m3 scales [..., n, k/16], a dividing global scale: fp32 [..., n, k]."""
    lo, hi = packed & 0xF, packed >> 4
    vals = np.stack([FP4[lo], FP4[hi]], axis=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    s = E4M3[scale] / gscale.reshape(*gscale.shape, *([1] * (scale.ndim - gscale.ndim))).astype(np.float32)
    return vals * np.repeat(s, block, axis=-1)


def to_bf16_bits(x) -> np.ndarray:
    """fp32 to bf16 bit patterns, round to nearest even (as torch and MLX round)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def pack_codes(q: np.ndarray, bits: int) -> np.ndarray:
    """Codes [..., K] (0 <= q < 2^bits) in MLX's affine words: each 32 codes fill `bits` u32 words, low bits first."""
    c = q.astype(np.uint64)
    k = c.shape[-1]
    c = c.reshape(*c.shape[:-1], k // 32, 32)
    words = np.zeros((*c.shape[:-1], bits), dtype=np.uint64)
    for i in range(32):
        o = i * bits
        j, sh = o // 32, o % 32
        words[..., j] |= (c[..., i] << np.uint64(sh)) & np.uint64(0xFFFFFFFF)
        if sh + bits > 32:
            words[..., j + 1] |= c[..., i] >> np.uint64(32 - sh)
    return words.astype(np.uint32).reshape(*c.shape[:-2], k // 32 * bits)


def refit(w: np.ndarray, bits: int, group: int, iters: int = 8, device=None, values: bool = False):
    """Affine codes, bf16 scales and biases by least squares (minimax_m3_convert.py's refit): words, scales, biases."""
    n = 2 ** bits - 1
    if torch is not None:
        x = torch.from_numpy(np.ascontiguousarray(w)).to(device).float().reshape(-1, group)
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
        s, b = s.to(torch.bfloat16), b.to(torch.bfloat16)
        q = torch.clamp(torch.round((x - b.float()) / s.float()), 0, n)
        if values:  # the dequantized values as the kernels compute them (bf16), fp32: a study's
            return (q * s.float() + b.float()).to(torch.bfloat16).float().reshape(w.shape)
        q = q.to(torch.uint8).cpu().numpy()
        s = s.view(torch.int16).cpu().numpy().view(np.uint16)
        b = b.view(torch.int16).cpu().numpy().view(np.uint16)
    else:
        x = np.ascontiguousarray(w, dtype=np.float32).reshape(-1, group)
        lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
        s, b = np.maximum((hi - lo) / n, 1e-8), lo
        xm = x.mean(1, keepdims=True)
        for _ in range(iters):
            q = np.clip(np.round((x - b) / s), 0, n)
            qm = q.mean(1, keepdims=True)
            var = ((q - qm) ** 2).sum(1, keepdims=True)
            fit = ((q - qm) * (x - xm)).sum(1, keepdims=True) / np.maximum(var, 1e-12)
            s = np.where((var > 0) & (fit > 1e-8), fit, s)
            b = xm - s * qm
        s, b = to_bf16_bits(s), to_bf16_bits(b)
        sf = (s.astype(np.uint32) << 16).view(np.float32)
        bf = (b.astype(np.uint32) << 16).view(np.float32)
        q = np.clip(np.round((x - bf) / sf), 0, n).astype(np.uint8)
    shape = (*w.shape[:-1], w.shape[-1] // group)
    return pack_codes(q.reshape(w.shape), bits), s.reshape(shape), b.reshape(shape)


def dequant_mxfp8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """e4m3 codes [n, k] under e8m0 bytes [n, k/32]: fp32 [n, k] (exact)."""
    s = np.ldexp(np.float32(1), scales.astype(np.int32) - 127).astype(np.float32)
    return E4M3[codes] * np.repeat(s, 32, axis=-1)


E2M1_EDGES = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))


def e2m1(y):
    """|y| to the nearest E2M1 code index 0..7 (0, .5, 1, 1.5, 2, 3, 4, 6), ties to the even index."""
    idx = torch.zeros(y.shape, dtype=torch.uint8, device=y.device)
    for edge, closed in E2M1_EDGES:  # a tie at an edge goes up when the upper index is even
        idx += (y >= edge) if closed else (y > edge)
    return idx


def nvfp4_group(ws: list, device, rows=2048) -> tuple:
    """fp32 [n_i, k] under one global to NVFP4 [(codes, scales)] and the dividing global; scales searched (USAGE)."""
    # the error is judged under the multiplying 1/g in fp32, as the kernels apply it, summed in fp64 in a fixed order
    amax = max(float(np.abs(w).max()) for w in ws)
    g = np.float32(2688.0) / np.float32(amax) if amax > 0 else np.float32(1)
    m = float(np.float32(1) / g)
    grid = torch.tensor(FP4[:8], device=device)
    e4m3 = torch.tensor(E4M3, device=device)
    out = []
    for w in ws:
        n, k = w.shape
        codes = np.empty((n, k // 2), dtype=np.uint8)
        scales = np.empty((n, k // 16), dtype=np.uint8)
        for r0 in range(0, n, rows):
            x = torch.from_numpy(np.ascontiguousarray(w[r0:r0 + rows])).to(device).view(-1, k // 16, 16)
            bmax = x.abs().amax(-1, keepdim=True)
            c0 = torch.clamp(bmax / 6.0 / m, max=448.0).to(torch.float8_e4m3fn).view(torch.uint8).long()
            best = c0.clone()
            best_err = torch.full(bmax.shape, float("inf"), dtype=torch.float64, device=device)
            for dc in range(1, -7, -1):  # larger codes first: a smaller one must be strictly better
                c = c0 + dc
                ok = (c >= 1) & (c <= 126) & (c0 > 0)
                step = e4m3[torch.clamp(c, 1, 126)] * m
                y = x / step
                d = (x - grid[e2m1(y.abs()).long()] * torch.sign(y) * step).double()
                sq = d[..., 0:1] * d[..., 0:1]
                for j in range(1, 16):
                    sq = sq + d[..., j:j + 1] * d[..., j:j + 1]
                err = torch.where(ok, sq, torch.full_like(best_err, float("inf")))
                better = err < best_err
                best = torch.where(better, c, best)
                best_err = torch.where(better, err, best_err)
            step = e4m3[best] * m
            y = torch.where(step > 0, x / torch.where(step > 0, step, 1.0), torch.zeros_like(x))
            q = e2m1(y.abs()) | ((y < 0) & (e2m1(y.abs()) > 0)).to(torch.uint8) << 3
            q = q.view(-1, k)
            codes[r0:r0 + rows] = (q[:, 0::2] | q[:, 1::2] << 4).cpu().numpy()
            scales[r0:r0 + rows] = best.view(-1, k // 16).to(torch.uint8).cpu().numpy()
        out.append((codes, scales))
    return out, g


def mxfp8(w: np.ndarray) -> tuple:
    """fp32 [n, k] to MXFP8: e4m3 codes (nearest even), e8m0 bytes a 32 (the power of two keeping them <= 448)."""
    n, k = w.shape
    x = w.reshape(n, k // 32, 32)
    amax = np.abs(x).max(-1).astype(np.float32)
    f, e = np.frexp(amax / np.float32(448.0))
    e = np.where(amax > 0, np.where(f == 0.5, e - 1, e), 0).astype(np.int32)
    e = np.clip(e, -127, 127)
    y = torch.from_numpy(np.ascontiguousarray(x * np.ldexp(np.float32(1), -e)[..., None].astype(np.float32)))
    codes = y.to(torch.float8_e4m3fn).view(torch.uint8).numpy().reshape(n, k)
    return codes, (e + 127).astype(np.uint8)


# the projections the engine stacks into one lane weight, under one global, by --dense-fp4 kind
FP4_GROUPS = (("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.index_q_proj", "self_attn.index_k_proj")),
              ("o", ("self_attn.o_proj",)), ("mlp", ("mlp.gate_proj", "mlp.up_proj")), ("mlp", ("mlp.down_proj",)),
              ("mlp", ("block_sparse_moe.shared_experts.gate_proj", "block_sparse_moe.shared_experts.up_proj")),
              ("mlp", ("block_sparse_moe.shared_experts.down_proj",)))


def dense_fp4(src: Source, p: str, t: dict, device, kinds: set):
    """Layer prefix p's MXFP8 projections of `kinds` (qkv, o, mlp) in t replaced by NVFP4 (USAGE)."""
    for kind, group in FP4_GROUPS:
        if kind not in kinds:
            continue
        names = [p + n for n in group if p + n + ".weight" in t]
        if not names:
            continue
        assert len(names) == len(group), names
        ws = [dequant_mxfp8(src.np(n + ".weight"), src.np(n + ".weight_scale_inv")) for n in names]
        parts, g = nvfp4_group(ws, device)
        for n, (codes, scales) in zip(names, parts):
            del t[n + ".weight"], t[n + ".weight_scale_inv"]
            t[n + ".weight_packed"] = ("U8", codes.shape, codes.tobytes())
            t[n + ".weight_scale"] = ("F8_E4M3", scales.shape, scales.tobytes())
            t[n + ".weight_global_scale"] = ("F32", [], np.float32(g).tobytes())


def parse_ranges(spec: str) -> list[int]:
    out = set()
    for part in filter(None, spec.split(",")):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "U8": 1, "U32": 4, "I32": 4}


class Writer:
    """Shards of raw tensors (name: dtype, shape, bytes) and the index; a partial run keeps other shards' entries."""

    def __init__(self, out_dir: str):
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.weight_map = {}
        index = os.path.join(out_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            self.weight_map = json.load(open(index))["weight_map"]

    def save(self, shard: str, tensors: dict):
        name = f"model-{shard}.safetensors"
        path, part = os.path.join(self.out, name), os.path.join(self.out, name + ".part")
        header, offset = {}, 0
        for n, (dtype, shape, data) in tensors.items():
            assert len(data) == DTYPE_BYTES[dtype] * int(np.prod(shape, dtype=np.int64)), n
            header[n] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
            offset += len(data)
        header["__metadata__"] = {"format": "pt"}
        h = json.dumps(header, separators=(",", ":")).encode()
        h += b" " * (-len(h) % 8)
        with open(part, "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            for _, _, data in tensors.values():
                f.write(data)
        os.replace(part, path)
        self.weight_map = {k: v for k, v in self.weight_map.items() if v != name}
        for n in tensors:
            self.weight_map[n] = name
        self.finish()

    def finish(self):
        files = {os.path.join(self.out, f) for f in self.weight_map.values()}
        total = sum(os.path.getsize(f) for f in files if os.path.exists(f))
        tmp = os.path.join(self.out, "model.safetensors.index.json.part")
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(self.weight_map.items()))},
                  open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(self.out, "model.safetensors.index.json"))


def kept(src: Source, name: str, out: dict):
    info = src.info(name)
    out[name] = (info["dtype"], info["shape"], src.raw(name))


def affine_experts(src: Source, m: str, experts: int, group: int, device) -> dict:
    """Layer prefix m's 128 nvfp4 experts refit to 3-bit affine: the experts_affine tensors."""
    out = {}
    for ws_, n in ((("w1", "w3"), "gate_up_proj"), (("w2",), "down_proj")):
        codes, scales, biases = [], [], []
        for e0 in range(0, experts, 16):
            es = range(e0, min(e0 + 16, experts))
            parts = []
            for w in ws_:  # gate (w1) rows then up (w3) rows of each expert
                ws = np.stack([src.np(f"{m}experts.{e}.{w}.weight_packed") for e in es])
                ss = np.stack([src.np(f"{m}experts.{e}.{w}.weight_scale") for e in es])
                gs = np.stack([src.np(f"{m}experts.{e}.{w}.weight_global_scale") for e in es])
                parts.append(dequant_nvfp4(ws, ss, gs))
            cq, cs, cb = refit(np.concatenate(parts, axis=1), 3, group, device=device)
            codes.append(cq), scales.append(cs), biases.append(cb)
        base = f"{m}experts_affine.{n}"
        q, s, b = np.concatenate(codes), np.concatenate(scales), np.concatenate(biases)
        out[base + ".weight"] = ("U32", q.shape, q.tobytes())
        out[base + ".scales"] = ("BF16", s.shape, s.tobytes())
        out[base + ".biases"] = ("BF16", b.shape, b.tobytes())
    return out


def layer_tensors(src: Source, i: int, text: dict, affine: bool, group: int, device, keep_nvfp4=False, fp4=frozenset(),
                  affine_from=None) -> dict:
    """Layer i as released, experts refit when `affine` (or from `affine_from`), projections of kinds `fp4` NVFP4."""
    # keep_nvfp4 keeps the NVFP4 experts beside the refit (a superset)
    p = f"{LM}model.layers.{i}."
    t = {}
    for name in src.names(p):
        if affine and not keep_nvfp4 and ".block_sparse_moe.experts." in name:
            continue
        kept(src, name, t)
    if affine and affine_from is not None:
        for name in affine_from.names(p + "block_sparse_moe.experts_affine."):
            kept(affine_from, name, t)
    elif affine:
        t.update(affine_experts(src, p + "block_sparse_moe.", text["num_local_experts"], group, device))
    if fp4:
        dense_fp4(src, p, t, device, fp4)
    return t


DENSE_FP4 = ("the dense_fp4 kinds' projections NVFP4 (e2m1, e4m3 scales a 16 searched by least squared error, in fp64 in "
             "a fixed order, over the plain code's next above and six below; one global per stacked group: "
             "q|k|v|index_q|index_k, gate|up); lm_head MXFP8 (e8m0 a 32)")


def flat_config(cfg: dict, nvfp4: list, affine3: list, group: int, keep: bool, revision: str, fp4=None) -> dict:
    """The engine's config.json: text_config at the top (model_type minimax_m3), vision, quantization, the mix."""
    out = dict(cfg["text_config"])
    for k in ("architectures", "image_token_index", "video_token_index", "projector_hidden_size",
              "img_token_compression_config", "vision_config", "quantization_config", "torch_dtype", "dtype"):
        if k in cfg:
            out[k] = cfg[k]
    out["model_type"] = "minimax_m3"
    out["nvfp4_mix"] = {"nvfp4_layers": nvfp4, "affine3_layers": affine3, "affine3_group": group, "affine3_bits": 3,
                        "keep_nvfp4": keep, "experts_affine": "mlx affine words, w = scales * q + biases; gate_up = w1 rows then w3 rows"}
    if fp4:
        out["nvfp4_mix"].update(fp4)
    out["source"] = {"repo": "MiniMaxAI/MiniMax-M3.1-Flash", "revision": revision}
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check(out: str, hashes: dict) -> int:
    bad = 0
    for name, want in sorted(hashes.items()):
        path = os.path.join(out, name)
        got = sha256(path) if os.path.exists(path) else "missing"
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} {name}")
    print(f"{len(hashes) - bad}/{len(hashes)} files match")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir", nargs="?", help="the release folder (MiniMaxAI/MiniMax-M3.1-Flash at the pinned revision)")
    ap.add_argument("out_dir")
    ap.add_argument("--nvfp4-layers", default="", help="layers whose experts stay nvfp4, e.g. 20-29,31,52,59")
    ap.add_argument("--group", type=int, default=64, help="the 3-bit affine group (64)")
    ap.add_argument("--layers", default="", help="only these layers (a partial run), e.g. 3-10")
    ap.add_argument("--keep-nvfp4", action="store_true",
                    help="3-bit layers keep their nvfp4 experts too (a superset to choose each layer's width at load)")
    ap.add_argument("--only", default="", help="blocks,head,files: skip the rest")
    ap.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--check", action="store_true", help="compare OUT_DIR's files with mix.json's hashes")
    ap.add_argument("--revision", default="", help="the release revision, recorded in mix.json")
    ap.add_argument("--dense-fp4", nargs="?", const="all", default="",
                    help="the decoder's projections NVFP4 and the head MXFP8 (USAGE): all, or a list of qkv, o, mlp")
    ap.add_argument("--affine-from", default="",
                    help="an earlier output of the same release and group whose 3-bit experts to copy instead of refitting")
    a = ap.parse_args()
    if a.check and not a.hf_dir:
        mix = json.load(open(os.path.join(a.out_dir, "mix.json")))
        raise SystemExit(check(a.out_dir, mix["sha256"]))
    if not a.hf_dir:
        raise SystemExit("converting needs HF_DIR")
    src = Source(a.hf_dir)
    cfg = json.load(open(os.path.join(a.hf_dir, "config.json")))
    text = cfg["text_config"]
    n_layers = text["num_hidden_layers"]
    nvfp4 = set(parse_ranges(a.nvfp4_layers))
    moe = [i for i in range(n_layers) if text["moe_layer_freq"][i]]
    affine3 = [i for i in moe if i not in nvfp4]
    if nvfp4 - set(moe):
        raise SystemExit(f"not MoE layers: {sorted(nvfp4 - set(moe))}")
    only = set(a.only.split(",")) if a.only else {"blocks", "head", "files"}
    layers = parse_ranges(a.layers) if a.layers else list(range(n_layers))
    kinds = {"qkv", "o", "mlp"} if a.dense_fp4 == "all" else set(filter(None, a.dense_fp4.split(",")))
    if kinds - {"qkv", "o", "mlp"}:
        raise SystemExit(f"--dense-fp4: not qkv, o or mlp: {sorted(kinds - {'qkv', 'o', 'mlp'})}")
    if kinds and torch is None:
        raise SystemExit("--dense-fp4 needs torch")
    # what the engine reads (the kinds stored NVFP4, the head's format) and how they were made
    fp4 = {"dense_fp4": [k for k in ("qkv", "o", "mlp") if k in kinds], "head_format": "mxfp8",
           "dense_fp4_method": DENSE_FP4} if kinds else None
    affine_from = None
    if a.affine_from:
        prev_cfg = json.load(open(os.path.join(a.affine_from, "config.json")))
        prev = prev_cfg["nvfp4_mix"]
        if prev["affine3_group"] != a.group or not set(affine3) <= set(prev["affine3_layers"]):
            raise SystemExit(f"{a.affine_from}: its 3-bit experts are not this run's (group, layers)")
        if prev_cfg.get("source", {}).get("revision") != a.revision:
            raise SystemExit(f"{a.affine_from}: made from revision {prev_cfg.get('source', {}).get('revision')!r}, not {a.revision!r}")
        affine_from = Source(a.affine_from)
        for i in affine3:
            if len(affine_from.names(f"{LM}model.layers.{i}.block_sparse_moe.experts_affine.")) != 6:
                raise SystemExit(f"{a.affine_from}: layer {i} lacks its 3-bit experts")
    w = Writer(a.out_dir)
    t0 = time.time()
    if "blocks" in only:
        for i in layers:
            t = time.time()
            tensors = layer_tensors(src, i, text, i in affine3, a.group, a.device, a.keep_nvfp4, frozenset(kinds), affine_from)
            w.save(f"{i:05d}", tensors)
            size = sum(len(d) for _, _, d in tensors.values())
            print(f"layer {i:2d} {'3-bit ' if i in affine3 else 'nvfp4' if i in moe else 'dense'} {size / 2**30:6.2f} GiB "
                  f"{time.time() - t:5.1f} s", flush=True)
    if "head" in only:
        t = {}
        for name in src.where:
            if not name.startswith(f"{LM}model.layers."):
                kept(src, name, t)
        if kinds:
            head = (src.np(f"{LM}lm_head.weight").astype(np.uint32) << 16).view(np.float32)
            codes, scales = mxfp8(head)
            t[f"{LM}lm_head.weight"] = ("F8_E4M3", codes.shape, codes.tobytes())
            t[f"{LM}lm_head.weight_scale_inv"] = ("U8", scales.shape, scales.tobytes())
        w.save("head", t)
        print(f"head+embed+vision {sum(len(d) for _, _, d in t.values()) / 2**30:.2f} GiB", flush=True)
    if "files" in only:
        for f in KEPT_FILES:
            if f != "config.json" and os.path.exists(os.path.join(a.hf_dir, f)):
                shutil.copy2(os.path.join(a.hf_dir, f), os.path.join(a.out_dir, f))
        json.dump(flat_config(cfg, sorted(nvfp4), affine3, a.group, a.keep_nvfp4, a.revision, fp4),
                  open(os.path.join(a.out_dir, "config.json"), "w"), indent=1)
        if os.path.isdir(os.path.join(a.hf_dir, "dspark")):
            shutil.copytree(os.path.join(a.hf_dir, "dspark"), os.path.join(a.out_dir, "dspark"), dirs_exist_ok=True)
    if not a.layers and "files" in only:  # every file present hashed (a files-only run keeps earlier shards)
        files = sorted(f for f in os.listdir(a.out_dir)
                       if f.endswith(".safetensors") or f.endswith(".index.json") or f == "config.json")
        if os.path.isdir(os.path.join(a.out_dir, "dspark")):
            files += sorted(os.path.join("dspark", f) for f in os.listdir(os.path.join(a.out_dir, "dspark")))
        mix = {"source_repo": "MiniMaxAI/MiniMax-M3.1-Flash", "source_revision": a.revision,
               "nvfp4_layers": sorted(nvfp4), "affine3_layers": affine3, "group": a.group, "keep_nvfp4": a.keep_nvfp4,
               "experts_affine": "mlx affine, 3 bits, w = scales * q + biases; gate_up = w1 rows then w3 rows",
               **(fp4 or {}),
               "sha256": {f: sha256(os.path.join(a.out_dir, f)) for f in files}}
        json.dump(mix, open(os.path.join(a.out_dir, "mix.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
