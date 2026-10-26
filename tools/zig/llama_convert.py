"""A Llama bf16 release to the family's affine checkpoint: projections MLX affine, q|k|v and gate|up fused."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

GROUP = 64
COPIED = ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "LICENSE", "README.md")
SHARD_BYTES = 2 << 30


class Shards:
    """The source's tensors by name, read from the mapped shards on demand."""

    def __init__(self, source: Path) -> None:
        index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        self.where: dict[str, tuple[Path, str, list[int], int, int]] = {}
        self.maps: dict[Path, np.memmap] = {}
        for shard in sorted(set(index.values())):
            path = source / shard
            with path.open("rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            base = 8 + n
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.where[k] = (path, v["dtype"], v["shape"], base + a, base + b)
            self.maps[path] = np.memmap(path, dtype=np.uint8, mode="r")

    def names(self) -> list[str]:
        return sorted(self.where)

    def get(self, name: str) -> tuple[str, list[int], bytes]:
        path, dt, shape, a, b = self.where[name]
        return dt, shape, bytes(self.maps[path][a:b])


def bf16_to_f32(raw: bytes, shape: list[int]) -> np.ndarray:
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to nearest even, as an fp32 to bf16 cast does (no NaNs reach here)."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def round_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def quantize(w: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MLX's affine rule in fp32 (bias at the larger edge, zero on the grid): w [n, k] to words, scales, biases."""
    n, k = w.shape
    if k % GROUP:
        raise SystemExit(f"width {k} is not a multiple of {GROUP}")
    g = w.reshape(n, k // GROUP, GROUP)
    w_max = g.max(axis=2, keepdims=True)
    w_min = g.min(axis=2, keepdims=True)
    bins = np.float32((1 << bits) - 1)
    mask = np.abs(w_min) > np.abs(w_max)
    scales = np.maximum((w_max - w_min) / bins, np.float32(1e-7)).astype(np.float32)
    scales = np.where(mask, scales, -scales)
    edge = np.where(mask, w_min, w_max).astype(np.float32)
    q0 = round_away(edge / scales)
    scales = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), scales).astype(np.float32)
    biases = np.where(q0 == 0, np.float32(0), edge).astype(np.float32)
    q = np.clip(round_away((g - biases) / scales), 0, bins).astype(np.uint32).reshape(n, k)
    per = 32 // bits
    q = q.reshape(n, k // per, per)
    words = np.zeros((n, k // per), dtype=np.uint32)
    for j in range(per):
        words |= q[:, :, j] << np.uint32(bits * j)
    return words, f32_to_bf16(scales[..., 0]), f32_to_bf16(biases[..., 0])


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {}
    at = 0
    for k, (dt, shape, raw) in tensors.items():
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [at, at + len(raw)]}
        at += len(raw)
    header["__metadata__"] = {"format": "pt"}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for _, _, raw in tensors.values():
            f.write(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="the bf16 release as downloaded (e.g. Nanbeige/Nanbeige4.3-33B)")
    p.add_argument("output", type=Path)
    p.add_argument("--bits", type=int, default=8, choices=(4, 8))
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    config = json.loads((args.source / "config.json").read_text())
    if config.get("model_type") != "llama":
        raise SystemExit("expected a llama checkpoint")
    if config.get("quantization") or config.get("quantization_config"):
        raise SystemExit("expected the bf16 release, not a quantized checkpoint")
    src = Shards(args.source)
    layers = config["num_hidden_layers"]
    args.output.mkdir(parents=True)
    shards: list[dict] = [{}]
    size = 0
    weight_map = {}

    def emit(k: str, v: tuple[str, list[int], bytes]) -> None:
        nonlocal size
        if size + len(v[2]) > SHARD_BYTES and shards[-1]:
            shards.append({})
            size = 0
        shards[-1][k] = v
        size += len(v[2])

    def quantized(base: str, parts: list[str]) -> None:
        mats = []
        for name in parts:
            dt, shape, raw = src.get(name)
            if dt != "BF16":
                raise SystemExit(f"{name} is {dt}, expected BF16")
            mats.append(bf16_to_f32(raw, shape))
        w = mats[0] if len(mats) == 1 else np.concatenate(mats, axis=0)
        words, s, b = quantize(w, args.bits)
        emit(base + ".weight", ("U32", list(words.shape), words.tobytes()))
        emit(base + ".scales", ("BF16", list(s.shape), s.tobytes()))
        emit(base + ".biases", ("BF16", list(b.shape), b.tobytes()))
        print(f"{base} {list(w.shape)} quantized from {len(parts)}", flush=True)

    def copied(name: str) -> None:
        emit(name, src.get(name))
        print(f"{name} {src.get(name)[1]}", flush=True)

    copied("model.embed_tokens.weight")
    for i in range(layers):
        a = f"model.layers.{i}.self_attn."
        m = f"model.layers.{i}.mlp."
        copied(f"model.layers.{i}.input_layernorm.weight")
        copied(f"model.layers.{i}.post_attention_layernorm.weight")
        quantized(a + "qkv_proj", [a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"])
        quantized(a + "o_proj", [a + "o_proj.weight"])
        quantized(m + "gate_up_proj", [m + "gate_proj.weight", m + "up_proj.weight"])
        quantized(m + "down_proj", [m + "down_proj.weight"])
    copied("model.norm.weight")
    quantized("lm_head", ["lm_head.weight"])
    expected = 3 + 9 * layers  # the embedding, the final norm, the head; two norms and seven projections a layer
    if len(src.names()) != expected:
        raise SystemExit(f"{len(src.names())} tensors in the source, expected {expected} (a plain Llama release)")
    names = [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]
    total = 0
    for fname, tensors in zip(names, shards):
        write_shard(args.output / fname, tensors)
        for k, v in tensors.items():
            weight_map[k] = fname
            total += len(v[2])
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=4) + "\n")
    q = {"group_size": GROUP, "bits": args.bits, "mode": "affine"}
    out_config = dict(config, quantization=q, quantization_config=q)
    out_config.pop("auto_map", None)
    (args.output / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")
    for name in COPIED:
        if (args.source / name).exists():
            shutil.copy(args.source / name, args.output / name)
    print(f"wrote {args.output}: llama, affine {args.bits}-bit groups of {GROUP}, {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
