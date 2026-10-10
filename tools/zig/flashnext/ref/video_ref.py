#!/usr/bin/env python3
"""A synthetic video and the Python frontend's preparation of it, as references for the native video path."""
# usage (Python image, no GPU): video_ref.py <model dir> <out dir>
import json, sys
from pathlib import Path
import av
import numpy as np

model, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
path = out / "test.mp4"
with av.open(str(path), "w") as c:
    s = c.add_stream("mpeg4", rate=10)
    s.width, s.height, s.pix_fmt = 320, 240, "yuv420p"
    for i in range(37):                       # 3.7 s at 10 fps: an odd frame count after sampling
        img = np.zeros((240, 320, 3), np.uint8)
        img[:, :, 2] = 40 + i * 3
        x = 20 + i * 7
        img[100:150, x:x + 50] = (230, 40, 40)
        img[30:60, 250 - i * 4:280 - i * 4] = (40, 200, 60)
        f = av.VideoFrame.from_ndarray(img, format="rgb24")
        for p in s.encode(f):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
from tensorfold.vision.qwen_processing import QwenImageProcessor
from tensorfold.vision.videos import DEFAULT_VIDEO_LIMITS, decode_video
import time
proc = QwenImageProcessor.from_directory(model)
data = path.read_bytes()
video = decode_video(data, proc.video_size, DEFAULT_VIDEO_LIMITS, time.monotonic() + 60)
rendered = ("<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>What moves in this video?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
prepared = proc.prepare(rendered, [], videos=[video])
pix = np.ascontiguousarray(prepared.video_pixel_values, dtype=np.float32)
pix.tofile(out / "video.patches.f32")
np.ascontiguousarray(video.frames).tofile(out / "video.frames.u8")
# Python's expanded text, as prepare builds it before tokenizing
grid = [int(x) for x in prepared.video_grid_thw[0]]
seqlen = grid[1] * grid[2] // 4
blocks = "".join(f"<{t:.1f} seconds><|vision_start|>{'<|video_pad|>' * seqlen}<|vision_end|>" for t in video.timestamps(2))
expanded = rendered.replace("<|vision_start|><|video_pad|><|vision_end|>", blocks)
meta = {"indices": list(map(int, video.indices)), "fps": float(video.fps), "frames": list(video.frames.shape),
        "grid": grid, "patches": list(pix.shape), "tokens": list(map(int, prepared.token_ids)), "expanded": expanded,
        "rope_delta": int(prepared.rope_delta)}
(out / "video.json").write_text(json.dumps(meta))
print("frames", video.frames.shape, "indices", video.indices, "fps", video.fps, "grid", grid, "tokens", len(prepared.token_ids))
