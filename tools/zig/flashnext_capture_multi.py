#!/usr/bin/env python3
"""The Triton kernels the Python Flash Next engine's shared rounds launch (attn_multi's prep, pool, chunks, merge
for the main forward and the MTP head), recorded on two ranks for the native engine's kernel set: waves of 2 .. 16
concurrent greedy requests through the engine's scheduler, the served settings (1M context, mtp=15@0.70, int8 KV).
Pack the result with the other captures' manifests through flashnext_aot_pack_dtypes.py (KT packs as an integer).
usage (both ranks): flashnext_capture_multi.py <rank> <out root> <model dir>   (env: CAP_MASTER, the rank 0 address)"""
import json, os, sys, threading, time
from pathlib import Path
import torch

rank, root, model = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out = root / f"rank{rank}"
out.mkdir(parents=True, exist_ok=True)
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

t0 = time.time()
eng = FlashNextEngine(model, depth=15, confidence=0.7, max_len=1048576, context_explicit=True, tp=2, rank=rank,
                      master=os.environ.get("CAP_MASTER", "127.0.0.1"), port=29573, graphs=False, streams=16,
                      kv_dtype=os.environ.get("CAP_KV", "int8"), prefetch=True)
sys.path.insert(0, str(root))
import triton_aot_manifest as aot
REC = aot.Recorder().install()
print(f"rank {rank}: engine up in {time.time() - t0:.0f} s", flush=True)
if rank == 1:
    eng.follow()
else:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    text = json.loads((root / "request.json").read_text())["messages"][0]["content"]
    ids = tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                     add_special_tokens=False).ids
    def prompt(k):
        n = 40 + 97 * k
        return (ids * (n // len(ids) + 1))[k:k + n]
    for n in (2, 3, 5, 8, 16, 16):
        reqs = [{"prompt": prompt(k), "count": 24 + 8 * (k % 5), "sampling": None} for k in range(n)]
        res = eng.scheduler.submit_many(reqs)
        print(f"rank 0: wave of {n}: {[len(r.get('tokens', [])) if isinstance(r, dict) else r for r in res][:4]} ...", flush=True)
    eng.shutdown()
torch.cuda.synchronize()
REC.dump(out / "launches.json")
names = sorted({k["name"] for k in REC.kernels.values()})
print(f"rank {rank}: {len(REC.kernels)} specializations: {', '.join(n for n in names if 'multi' in n)}", flush=True)

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
os._exit(0)
