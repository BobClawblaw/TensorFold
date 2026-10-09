#!/usr/bin/env python3
"""Python's MTP reply to a reference prompt at a given cache dtype (both ranks): reference.json for fn-native."""
# usage (both ranks): ref_kv.py <rank> <out root> <model dir> <reference.json with the prompt> <int8|int4>
import json, os, sys, time
from pathlib import Path
import torch
MASTER = os.environ.get("TF_MASTER") or sys.exit("TF_MASTER is not set: rank 0's address as rank 1 reaches it")
PORT = int(os.environ.get("TF_PORT", "29573"))                  # the engines' rendezvous port
rank, root, model, src, kv = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
out = root / f"rank{rank}"
out.mkdir(parents=True, exist_ok=True)
DEPTH, CONF, COUNT = 6, 0.7, 48
from tensorfold.families.qwen4_exp.cuda import decode as D
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
eng = FlashNextEngine(model, depth=DEPTH, max_len=16384, context_explicit=True, tp=2, rank=rank, master=MASTER,
                      port=PORT, graphs=False, kv_dtype=kv, prefetch=True)
e, w = eng.e, eng.w
ids = json.loads(src.read_text())["prompt"]
first = D.prefill(e, ids, None, mtp=True)
res = D.mtp_decode(e, first, COUNT, None, depth=DEPTH, confidence=CONF)
torch.cuda.synchronize()
print(f"rank {rank}: kv {kv} first {first} reference {res.tokens[:12]} rounds {res.rounds} accepted {res.accepted}", flush=True)
(out / "reference.json").write_text(json.dumps({"prompt": ids, "tokens": res.tokens, "first": first, "rounds": res.rounds,
    "drafted": res.drafted, "accepted": res.accepted, "keeps": res.keeps, "widths": res.widths,
    "vocab_offset": int(w.meta.get("vocab_offset", 0)), "depth": DEPTH, "confidence": CONF, "kv_dtype": kv}))
