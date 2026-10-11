#!/usr/bin/env python3
"""The Python engine with CUDA graphs on the reference prompt: MTP decode of the same reply, timed three times."""
import json, os, sys, time
from pathlib import Path
import torch
MASTER = os.environ.get("TF_MASTER") or sys.exit("TF_MASTER is not set: rank 0's address as rank 1 reaches it")
PORT = int(os.environ.get("TF_PORT", "29571"))                  # the engines' rendezvous port
rank, root, model = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
from tensorfold.families.qwen4_exp.cuda import decode as D
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
eng = FlashNextEngine(model, depth=6, max_len=16384, context_explicit=True, tp=2, rank=rank, master=MASTER,
                      port=PORT, graphs=True, kv_dtype=os.environ.get("CAP_KV", "int8"), prefetch=True)
e = eng.e
from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(model / "tokenizer.json"))
text = json.loads((root / "request.json").read_text())["messages"][0]["content"]
ids = tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
for i in range(3):
    torch.cuda.synchronize(); t = time.perf_counter()
    first = D.prefill(e, ids, None, mtp=True)
    torch.cuda.synchronize(); tp = time.perf_counter() - t
    res = D.mtp_decode(e, first, 48, None, depth=6, confidence=0.7)
    print(f"rank {rank}: run {i}: prefill {tp*1000:.0f} ms; decode {len(res.tokens)-1} tokens in {res.seconds:.3f} s = "
          f"{(len(res.tokens)-1)/res.seconds:.1f} tok/s; rounds {res.rounds} drafted {res.drafted} accepted {res.accepted}; "
          f"tokens {res.tokens[:6]}", flush=True)
