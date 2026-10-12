"""Flash Next video input on a running `tensorfold-native --vision` against transformers' Qwen3VLProcessor.

    python -I tools/zig/flashnext_video_check.py --url http://127.0.0.1:8888 --model DIR --clips DIR [--max-frames 256]

For each clip (tools/zig/glm_video_fixtures.py --make draws the set: a yellow square crossing a gradient): the server's
/tokenize ids for a user message holding the clip as a video_url data URL equal the processor's input_ids for the same
message (the frames it samples at 2 fps, each pair's "<X.Y seconds>", <|vision_start|>, its <|video_pad|> rows,
<|vision_end|>), thinking off; then the model, asked what moves and which way, names the square and its direction.
A clip too short to sample is refused by both. Needs transformers with Qwen3-VL and PyAV.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import av
import numpy as np


def post(url: str, path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def load(path: Path, vp, max_frames: int):
    """The frames the processor samples (PyAV), with their metadata; None when it samples none."""
    from transformers.video_utils import VideoMetadata

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        total = int(st.frames) or sum(1 for p in c.demux(st) if p.size)
        c.seek(0)
        fps = st.average_rate
        meta = VideoMetadata(total_num_frames=total, fps=float(fps), duration=float(total / fps) if fps else 0.0,
                             video_backend="pyav", height=st.height, width=st.width)
        vp.max_frames = max_frames
        try:
            indices = vp.sample_frames(meta)
        except ValueError:
            return None, meta
        frames = []
        for i, frame in enumerate(c.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(frame)
        meta.frames_indices = indices
        return np.stack([f.to_ndarray(format="rgb24") for f in frames]), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--max-frames", type=int, default=256, help="the server's --vision-max-frames")
    a = ap.parse_args()
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(a.model)
    vp = proc.video_processor
    failures = 0
    for path in sorted(p for p in Path(a.clips).iterdir() if p.suffix in {".mp4", ".mov", ".webm", ".mkv"}):
        mime = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska"}[path.suffix]
        part = {"type": "video_url", "video_url": {"url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()}}
        question = "What moves in this video, what color is it, and which way does it go? One sentence."
        msgs = [{"role": "user", "content": [part, {"type": "text", "text": question}]}]
        status, tok = post(a.url, "/tokenize", {"model": "x", "messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})
        frames, meta = load(path, vp, a.max_frames)
        if frames is None:
            ok = status >= 400
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {path.name}: the processor samples no frames; the server {'refuses' if ok else 'answers'} ({status})")
            continue
        ref_msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(ref_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = proc(text=[text], videos=[frames], video_metadata=[meta], do_sample_frames=False, return_tensors="np")
        ref = [int(x) for x in enc["input_ids"][0]]
        mine = tok.get("tokens") or []
        same = mine == ref
        grid = [int(v) for v in enc["video_grid_thw"][0]]
        first = next((i for i, (x, y) in enumerate(zip(mine, ref)) if x != y), min(len(mine), len(ref)))
        status2, chat = post(a.url, "/v1/chat/completions", {"model": "x", "messages": msgs, "max_tokens": 60, "temperature": 0})
        reply = chat["choices"][0]["message"]["content"] if status2 == 200 else f"HTTP {status2} {json.dumps(chat)[:120]}"
        low = reply.lower()
        sees = "yellow" in low and ("right" in low or "left to right" in low)
        ok = same and status2 == 200
        failures += not ok
        note = "" if same else f" (first difference at {first}: {mine[first:first + 6]} vs {ref[first:first + 6]}; {len(mine)} vs {len(ref)})"
        print(f"{'ok  ' if ok else 'FAIL'} {path.name}: {len(meta.frames_indices)} frames, grid {grid}, {len(ref)} prompt ids "
              f"{'equal' if same else 'DIFFER'}{note}; sees the yellow square going right: {sees}; {reply[:110]!r}", flush=True)
    print("PASS" if failures == 0 else f"FAIL: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
