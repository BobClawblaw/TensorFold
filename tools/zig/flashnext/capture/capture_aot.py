#!/usr/bin/env python3
"""Every Triton kernel the Python Flash Next engine launches on two ranks, recorded for the native kernel set."""
# usage (both ranks): capture_aot.py <rank> <out root> <model dir>; env TF_MASTER, TF_PORT, CAP_KV, CAP_SWEEP=1
import json, os, sys, time
from pathlib import Path
import torch
MASTER = os.environ.get("TF_MASTER") or sys.exit("TF_MASTER is not set: rank 0's address as rank 1 reaches it")
PORT = int(os.environ.get("TF_PORT", "29571"))                  # the engines' rendezvous port


rank, root, model = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out = root / f"rank{rank}"
out.mkdir(parents=True, exist_ok=True)
DEPTH, CONF = 15, 0.7                      # the served profile: mtp=15@0.70
from tensorfold.families.qwen4_exp.cuda import decode as D
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

t0 = time.time()
eng = FlashNextEngine(model, depth=DEPTH, confidence=CONF, max_len=1048576, context_explicit=True, tp=2, rank=rank,
                      master=MASTER, port=PORT, graphs=False, kv_dtype=os.environ.get("CAP_KV", "int8"), prefetch=True)
e, w = eng.e, eng.w
sys.path.insert(0, str(root))
import triton_aot_manifest as aot
REC = aot.Recorder().install()             # after startup: NCCL and the loaders are left alone
print(f"rank {rank}: engine up in {time.time() - t0:.0f} s", flush=True)

from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(model / "tokenizer.json"))
text = json.loads((root / "request.json").read_text())["messages"][0]["content"]
ids = tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                 add_special_tokens=False).ids
filler = (ids * 80)[:140000]

def prompt(n):
    return (ids * (n // len(ids) + 1))[:n]

# CAP_SWEEP=1: every prompt length a chunk can leave: 1..300, then each multiple of 16 to 2048 and its neighbours
if os.environ.get("CAP_SWEEP") == "1":
    lengths = sorted(set(range(1, 301)) | {m + d for m in range(304, 2049, 16) for d in (-1, 0, 1) if m + d <= 2048})
    for n in lengths:
        e.reset()
        D.prefill(e, prompt(n), None, mtp=True)
    torch.cuda.synchronize()
    print(f"rank {rank}: swept {len(lengths)} prompt lengths", flush=True)

# prompt passes: one row, odd and divisible rows, a full chunk plus a partial one, and the reference reply with drafts
for n in (1, 17, 300, 2048, 2695):
    e.reset()
    first = D.prefill(e, prompt(n), None, mtp=True)
res = D.mtp_decode(e, first, 48, None, depth=DEPTH, confidence=CONF)
print(f"rank {rank}: reference {res.tokens[:8]} rounds {res.rounds}", flush=True)

# every verify width and commit (rows, kept) at both parities, every MTP head width
toks = list(res.tokens[-16:]) + [1] * 16
for R in range(1, DEPTH + 2):
    for _ in range(2):
        e.forward(toks[:R]); D.commit(w, e.st, e.buf, R, R)
    for keep in range(1, R):
        for _ in range(2):
            e.forward(toks[:R]); D.commit(w, e.st, e.buf, R, keep)
for n in range(1, DEPTH + 2):
    e.mtp_forward(toks[:n], e.buf.streams[:n]); e.st.set_mtp_len(e.st.mtp_len + n)
torch.cuda.synchronize()
print(f"rank {rank}: widths covered", flush=True)

# selector widths: a step and a head step in every power-of-two band of key blocks (positions set directly)
pos0 = e.st.pos
for k in range(0, 18):
    p = 4 * (2 ** k) + 3
    if p + 32 >= e.st.capacity:
        break
    e.st.set_pos(p)
    e.forward(toks[:1]); D.commit(w, e.st, e.buf, 1, 1)
    e.st.set_mtp_len(p)
    e.mtp_forward(toks[:1], e.buf.streams[:1])
torch.cuda.synchronize()
print(f"rank {rank}: selector bands to position {e.st.pos}", flush=True)

# a long prompt (the tiled selector on prompt rows, and every attention chunk count along the way)
e.reset()
D.prefill(e, filler, None, mtp=True)
torch.cuda.synchronize()
print(f"rank {rank}: long prompt {len(filler)} tokens", flush=True)
REC.dump(out / "launches.json")

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
