"""Qwen3.8 Flash Next's image tower against transformers' Qwen4ExpVisionModel, on a set of images (the Metal port's gate).

    python -I tools/zig/flashnext_vision_fixtures.py --model DIR --images DIR --out DIR [--tower tf-flashnext-vision]

For each PNG or JPEG in --images: transformers' Qwen2VLImageProcessor with the checkpoint's preprocessor_config.json
(sized to --max-tokens visual tokens) gives the patches, and Qwen4ExpVisionModel, built from the checkpoint's
vision_config and loaded with its `vision_tower.*` tensors (bf16, the patch convolution moved from MLX's channels-last
layout), runs them in bf16 on the CPU, step by step as its forward does, writing to OUT/<image>/:
  patches.f32   [P, 1536] the processor's patches (channel, frame, row, column), merge windows contiguous
  embed.f32     [P, 1152] the patch projection plus the resampled position table
  block0.f32    [P, 1152] after the first block
  blocks.f32    [P, 1152] after the last block
  features.f32  [P/4, 2560] the merger's rows, which replace the image's placeholder tokens
  features_fp32.f32  the same tower in fp32: how far bf16 itself sits from exact
and meta.json. With --tower it then runs the native tower on each image's patches and prints its comparison lines.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import Qwen2VLImageProcessor
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


def load_tower(model: Path, dtype: torch.dtype) -> Qwen4ExpVisionModel:
    cfg = Qwen4ExpVisionConfig(**json.loads((model / "config.json").read_text())["vision_config"])
    cfg._attn_implementation = "eager"
    tower = Qwen4ExpVisionModel(cfg).to(dtype).eval()
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for name in sorted(n for n in index if n.startswith("vision_tower.")):
        with safe_open(model / index[name], framework="pt") as f:
            t = f.get_tensor(name)
        key = name[len("vision_tower."):]
        if key == "patch_embed.proj.weight":  # MLX's Conv3d (O, T, H, W, C) to torch's (O, C, T, H, W)
            t = t.permute(0, 4, 1, 2, 3).contiguous()
        state[key] = t.to(dtype)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"tower weights: missing {missing}, unexpected {unexpected}")
    return tower


@torch.no_grad()
def stages(tower: Qwen4ExpVisionModel, pixels: torch.Tensor, grid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Qwen4ExpVisionModel.forward, with its intermediate rows kept."""
    cfg = tower.config
    idx, wts = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=tower.num_grid_per_side, mode=tower.interpolation_mode,
        align_corners=tower.interpolation_align_corners, spatial_merge_size=cfg.spatial_merge_size)
    pos_ids = get_vision_position_ids(grid, tower.spatial_merge_size)
    cu, max_len = get_vision_attention_seqlens(grid, cfg)
    h = tower.patch_embed(pixels)
    pos = (tower.pos_embed(idx) * wts[:, :, None]).sum(1)
    h = h + pos.to(h.dtype)
    out = {"embed": h}
    rope = tower.rotary_pos_emb(h, pos_ids)
    for i, blk in enumerate(tower.blocks):
        h = blk(h, cu_seqlens=cu, max_seqlen=max_len, position_embeddings=rope)
        if i == 0:
            out["block0"] = h
    out["blocks"] = h
    out["features"] = tower.merger(h)
    return out


def write(path: Path, t: torch.Tensor) -> None:
    path.write_bytes(t.float().contiguous().numpy().astype(np.float32).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="the Flash Next checkpoint folder (its vision_tower.* tensors)")
    ap.add_argument("--images", required=True, help="a folder of PNG and JPEG images")
    ap.add_argument("--out", required=True, help="where each image's references go")
    ap.add_argument("--max-tokens", type=int, default=4096, help="the visual-token budget an image is sized to")
    ap.add_argument("--tower", help="tf-flashnext-vision: also run the native tower on each image's patches")
    a = ap.parse_args()
    model, out = Path(a.model), Path(a.out)
    pp = json.loads((model / "preprocessor_config.json").read_text())
    merge, patch = pp["merge_size"], pp["patch_size"]
    longest = min(pp["size"]["longest_edge"], a.max_tokens * merge * merge * patch * patch)
    proc = Qwen2VLImageProcessor(
        size={"shortest_edge": pp["size"]["shortest_edge"], "longest_edge": longest}, patch_size=patch,
        temporal_patch_size=pp["temporal_patch_size"], merge_size=merge, image_mean=pp["image_mean"],
        image_std=pp["image_std"])
    towers = {"bf16": load_tower(model, torch.bfloat16), "fp32": load_tower(model, torch.float32)}
    failed = 0
    for img in sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
        dest = out / img.name.replace(".", "-")
        dest.mkdir(parents=True, exist_ok=True)
        enc = proc(images=Image.open(img).convert("RGB"), return_tensors="pt")
        pixels, grid = enc["pixel_values"].float(), enc["image_grid_thw"]
        write(dest / "patches.f32", pixels)
        ref = stages(towers["bf16"], pixels.to(torch.bfloat16), grid)
        for name, t in ref.items():
            write(dest / f"{name}.f32", t)
        exact = stages(towers["fp32"], pixels, grid)["features"]
        write(dest / "features_fp32.f32", exact)
        f16, f32 = ref["features"].float(), exact.float()
        cos = torch.nn.functional.cosine_similarity(f16, f32, dim=1)
        size = Image.open(img).size
        meta = {"image": img.name, "size": list(size), "grid_thw": grid[0].tolist(), "max_tokens": a.max_tokens,
                "tokens": int(f16.shape[0]), "patch_width": int(pixels.shape[1]),
                "bf16_vs_fp32_cosine": {"mean": float(cos.mean()), "worst": float(cos.min())}}
        (dest / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{img.name}: {size[0]}x{size[1]} -> grid {meta['grid_thw']}, {meta['tokens']} tokens; "
              f"bf16 against fp32 cosine mean {cos.mean():.6f} worst {cos.min():.6f}", flush=True)
        if a.tower:
            for stage in ("embed", "block0", "features"):
                run = subprocess.run([a.tower, str(model), str(dest), stage], capture_output=True, text=True)
                tail = (run.stdout + run.stderr).strip().splitlines()
                print("   ", tail[-1] if tail else f"exit {run.returncode}", flush=True)
                failed += run.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
