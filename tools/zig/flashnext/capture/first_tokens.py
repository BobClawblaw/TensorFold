#!/usr/bin/env python3
"""The Python engine's greedy first token after prompts of several lengths, on two ranks (prefill check)."""
# usage (both ranks): first_tokens.py <rank> <root> <model dir> <length>...; prompt: root/prompt.json or reference
import json, os, sys
from pathlib import Path
import torch
MASTER = os.environ.get("TF_MASTER") or sys.exit("TF_MASTER is not set: rank 0's address as rank 1 reaches it")
PORT = int(os.environ.get("TF_PORT", "29571"))                  # the engines' rendezvous port
rank, root, model = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
lengths = [int(x) for x in sys.argv[4:]]
from tensorfold.families.qwen4_exp.cuda import decode as D
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
eng = FlashNextEngine(model, depth=6, max_len=16384, context_explicit=True, tp=2, rank=rank, master=MASTER,
                      port=PORT, graphs=False, kv_dtype=os.environ.get("CAP_KV", "int8"), prefetch=True)
e = eng.e
ref = json.loads((root / f"rank{rank}" / "reference.json").read_text()) if (root / f"rank{rank}" / "reference.json").exists() else None
ids = json.loads((root / "prompt.json").read_text()) if (root / "prompt.json").exists() else ref["prompt"]
out = {}
for n in lengths:
    p = (ids * (n // len(ids) + 1))[:n]
    first = D.prefill(e, p, None, mtp=True)
    out[n] = first
    print(f"rank {rank}: prompt of {n} tokens -> first token {first}", flush=True)
(root / f"rank{rank}").mkdir(exist_ok=True)
(root / f"rank{rank}" / "first_tokens.json").write_text(json.dumps(out))
