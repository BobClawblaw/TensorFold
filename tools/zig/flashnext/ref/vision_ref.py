#!/usr/bin/env python3
"""The Python vision path's prepared prompts, patches and tower features, as references for the native port."""
# usage (Python image): vision_ref.py <model dir> <out dir> [cpu|cuda] [folder of PNGs + questions.json]
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

model, out = Path(sys.argv[1]), Path(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
out.mkdir(parents=True, exist_ok=True)
from tensorfold.vision.qwen_cuda import QwenCudaVision

vision = QwenCudaVision(model, device)
frontend = vision.frontend


def picture(w, h, seed):
    """A drawn test image: a colour gradient, shapes and a line of text (deterministic)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 127 // max(w + h, 1))], -1)
    img = Image.fromarray(a.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    for _ in range(6):
        x0, y0 = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        d.ellipse([x0, y0, x0 + int(rng.integers(20, 80)), y0 + int(rng.integers(20, 80))],
                  fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    d.rectangle([w // 4, h // 3, w // 2, h // 2], outline=(255, 255, 255), width=4)
    d.text((10, 10), "TensorFold 42", fill=(0, 0, 0))
    return img


cases = [("small", 320, 240, 1), ("wide", 640, 300, 2), ("tall", 280, 520, 3)]
pngs = Path(sys.argv[4]) if len(sys.argv) > 4 else None   # a folder of PNGs instead of the drawn cases
if pngs is not None:
    cases = [(f.stem, 0, 0, 0) for f in sorted(pngs.glob("*.png"))]
questions = json.loads((pngs / "questions.json").read_text()) if pngs is not None and (pngs / "questions.json").exists() else {}
index = []
for name, w, h, seed in cases:
    img = Image.open(pngs / f"{name}.png").convert("RGB") if pngs is not None else picture(w, h, seed)
    w, h = img.size
    img.save(out / f"{name}.png")
    question = questions.get(name, "Describe this image in one sentence.")
    rendered = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + question +
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    import io
    from tensorfold.vision.images import DEFAULT_LIMITS, _decode
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    decoded = _decode(buf.getvalue(), "auto", DEFAULT_LIMITS, DEFAULT_LIMITS.max_total_pixels)  # the server's decode
    prepared = frontend.prepare(rendered, [decoded])
    pix = np.ascontiguousarray(prepared.pixel_values, dtype=np.float32)
    grid = prepared.image_grid_thw.astype(np.int64)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if device != "cpu" else [SDPBackend.MATH]
    with torch.no_grad(), sdpa_kernel(backends):   # the engine's backends on CUDA (qwen_cuda.py)
        import os
        if os.environ.get("VISION_FP32") == "1":     # a near-exact reference: the tower in fp32
            vision.tower.float()
            x = torch.from_numpy(pix).to(device)
        else:
            x = torch.from_numpy(pix).to(torch.bfloat16).to(device)
        res = vision.tower(x, grid_thw=torch.from_numpy(grid).to(device))
        feats = getattr(res, "pooler_output", res)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = feats.to(torch.bfloat16).contiguous().cpu()
    pix.tofile(out / f"{name}.patches.f32")
    feats.view(torch.int16).numpy().tofile(out / f"{name}.features.bf16")
    pos = np.ascontiguousarray(prepared.position_ids, dtype=np.int32)
    pos.tofile(out / f"{name}.positions.i32")
    meta = {"name": name, "width": w, "height": h, "grid": grid.tolist(), "patches": list(pix.shape),
            "features": list(feats.shape), "tokens": list(prepared.token_ids), "positions": list(pos.shape),
            "rope_delta": int(prepared.rope_delta), "spans": [list(s) for s in prepared.image_spans]}
    (out / f"{name}.json").write_text(json.dumps(meta))
    index.append({k: meta[k] for k in ("name", "grid", "patches", "features", "rope_delta", "spans")})
    print(name, meta["grid"], "patches", meta["patches"], "features", meta["features"], "delta", meta["rope_delta"],
          flush=True)
(out / "index.json").write_text(json.dumps(index, indent=1))
