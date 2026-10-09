#!/usr/bin/env python3
"""Image prompts through the Python engine on two ranks: their Triton launches and greedy MTP replies."""
# usage (both ranks): capture_vision.py <rank> <out root> <model dir> <vision_ref.py dir>; env TF_MASTER, CAP_KV
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
MASTER = os.environ.get("TF_MASTER") or sys.exit("TF_MASTER is not set: rank 0's address as rank 1 reaches it")
PORT = int(os.environ.get("TF_PORT", "29572"))                  # the engines' rendezvous port


rank, root, model, refdir = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out = root / f"rank{rank}"
out.mkdir(parents=True, exist_ok=True)
DEPTH, CONF = 15, 0.7
from tensorfold.families.qwen4_exp.cuda import decode as D
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.vision.qwen_cuda import EncodedVision
from tensorfold.vision.qwen_processing import image_positions

config = json.loads((model / "config.json").read_text())
IMAGE = int(config["image_token_id"])
t0 = time.time()
eng = FlashNextEngine(model, depth=DEPTH, confidence=CONF, max_len=1048576, context_explicit=True, tp=2, rank=rank,
                      master=MASTER, port=PORT, graphs=False, kv_dtype=os.environ.get("CAP_KV", "int8"), prefetch=True)
e, w = eng.e, eng.w
sys.path.insert(0, str(root))
import triton_aot_manifest as aot
REC = aot.Recorder().install()
print(f"rank {rank}: engine up in {time.time() - t0:.0f} s", flush=True)


def encoded(meta, tokens):
    feats = np.fromfile(refdir / f"{meta['name']}.features.bf16", dtype=np.int16).reshape(meta["features"])
    feats = torch.from_numpy(feats).view(torch.bfloat16).cuda()
    pos, delta, _ = image_positions(tokens, meta["grid"], config)
    pos = torch.as_tensor(np.asarray(pos).reshape(3, -1), dtype=torch.int32).cuda()
    rows = tuple(i for i, t in enumerate(tokens) if t == IMAGE)
    return EncodedVision(rows=rows, features=feats, positions=pos, rope_delta=int(delta))


refs = []
index = json.loads((refdir / "index.json").read_text())
filler = 1782  # a plain text token ("the" or similar), added after the image to shift the prompt length
for case in index:
    meta = json.loads((refdir / f"{case['name']}.json").read_text())
    base = list(meta["tokens"])
    cut = max(i for i, t in enumerate(base) if t == IMAGE) + 2   # after <|vision_end|>
    for extra in (0, (16 - len(base) % 16) % 16 or 16):
        tokens = base[:cut] + [filler] * extra + base[cut:]
        enc = encoded(meta, tokens)
        first = D.prefill(e, tokens, None, mtp=True, vision=enc)
        res = D.mtp_decode(e, first, 48, None, depth=DEPTH, confidence=CONF)
        refs.append({"name": meta["name"], "extra": extra, "prompt": tokens, "first": int(first), "tokens": list(map(int, res.tokens)),
                     "rounds": res.rounds, "drafted": res.drafted, "accepted": res.accepted, "rope_delta": enc.rope_delta,
                     "length": len(tokens)})
        print(f"rank {rank}: {meta['name']}+{extra} ({len(tokens)} tokens): {res.tokens[:8]} rounds {res.rounds}", flush=True)
torch.cuda.synchronize()

# every verify width and commit at both parities, and every MTP head width, on an image state (decode past the prompt)
meta = json.loads((refdir / f"{index[0]['name']}.json").read_text())
tokens = list(meta["tokens"])
first = D.prefill(e, tokens, None, mtp=True, vision=encoded(meta, tokens))
toks = [first] + [1] * 16
for R in range(1, DEPTH + 2):
    for _ in range(2):
        e.forward(toks[:R]); D.commit(w, e.st, e.buf, R, R)
    for keep in range(1, R):
        e.forward(toks[:R]); D.commit(w, e.st, e.buf, R, keep)
for n in range(1, DEPTH + 2):
    e.mtp_forward(toks[:n], e.buf.streams[:n]); e.st.set_mtp_len(e.st.mtp_len + n)
torch.cuda.synchronize()
print(f"rank {rank}: image-state widths covered", flush=True)
REC.dump(out / "launches.json")
(out / "references.json").write_text(json.dumps(refs))


def specialization():
    from triton.runtime.jit import JITFunction
    import gc
    d = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            d[f"{fn.fn.__module__}.{fn.fn.__qualname__}"] = {
                "params": [p.name for p in fn.params],
                "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    return d
(out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
print(f"rank {rank}: done in {time.time() - t0:.0f} s", flush=True)
