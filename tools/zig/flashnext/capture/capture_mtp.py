#!/usr/bin/env python3
"""Two-rank oracle for MTP decode: traced prefill, the logged MTP reply, every round program, the weight pack."""
# usage (both ranks): capture_mtp.py <rank> <out root> <model dir>; env TF_MASTER, TF_PORT, CAP_KV, SKIP_PACK=1
import dataclasses, hashlib, json, os, sys, time
from pathlib import Path


import numpy as np
import torch
MASTER = os.environ.get("TF_MASTER") or sys.exit("TF_MASTER is not set: rank 0's address as rank 1 reaches it")
PORT = int(os.environ.get("TF_PORT", "29571"))                  # the engines' rendezvous port

rank, root, model = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out = root / f"rank{rank}"
out.mkdir(parents=True, exist_ok=True)
DEPTH, CONF, COUNT = 6, 0.7, 48

from tensorfold.families.qwen4_exp.cuda import decode as D
from tensorfold.families.qwen4_exp.cuda import forward as F
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

t0 = time.time()
eng = FlashNextEngine(model, depth=DEPTH, max_len=16384, context_explicit=True, tp=2, rank=rank, master=MASTER,
                      port=PORT, graphs=False, kv_dtype=os.environ.get("CAP_KV", "int8"), prefetch=True)
e, w = eng.e, eng.w
assert e.mbuf is not None and w.mtp is not None
sys.path.insert(0, str(root))
import capture_trace as ct
TR = ct.Tracer()
from tensorfold.cuda import experts as _ex
from tensorfold.cuda.kernels import gdn as _gdn, qmm as _qmm
from tensorfold.families.qwen4_exp.cuda import gdn as _fgdn, gdn_io as _fio
ct.install(TR, w.comm, [("experts", _ex._ext()), ("gdn_v2", _gdn._ext()), ("qmm", _qmm._ext()),
                        ("qwen4_exp_gdn", _fgdn._ext()), ("gdn_io", _fio._ext())])
print(f"rank {rank}: engine up in {time.time() - t0:.0f} s", flush=True)

from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(model / "tokenizer.json"))
text = json.loads((root / "request.json").read_text())["messages"][0]["content"]
ids = tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                 add_special_tokens=False).ids

ple = next(layer.ple for layer in w.layers if layer.ple is not None)
ng, tb = ple.ngram, ple.table
(out / "ngram.json").write_text(json.dumps({
    "n": ng.n, "per_ngram": ng.per_ngram, "heads": ng.heads, "eos": ng.eos, "dims": ng.dims,
    "head_sizes": ng.head_sizes.tolist(), "head_offsets": ng.head_offsets.tolist(),
    "multipliers": [int(x) for x in ng.multipliers.tolist()], "width": tb.width, "rows": tb.rows,
    "starts": tb.starts.tolist(), "lut": tb.lut.tolist(),
    "shards": [{"file": str(v.filename), "offset": int(v.offset), "rows": int(v.shape[0])} for v in tb.values],
    "initial_history": ng.initial_history().tolist(), "vocab_offset": int(w.meta.get("vocab_offset", 0))}))
w.draft_ids.to(torch.int32).cpu().numpy().tofile(out / "draft_ids.bin")       # draft column -> global token id
marks = {}

def dump_state(path_prefix, roots):
    torch.cuda.synchronize()
    seen, entries, off_ = set(), [], 0
    with open(f"{path_prefix}.bin", "wb") as f:
        for root_name, root_ in roots:
            for n in ct.named_map([(root_name, root_)]):
                if n["base"] in seen:
                    continue
                seen.add(n["base"])
                obj = root_
                for part in n["name"].split(".")[1:]:
                    obj = obj[int(part)] if isinstance(obj, (list, tuple)) else (obj[part] if isinstance(obj, dict) else getattr(obj, part))
                st_ = obj.untyped_storage()
                raw = torch.empty(0, dtype=torch.uint8, device="cuda").set_(st_, 0, (st_.nbytes(),)).cpu().numpy().tobytes()
                f.write(raw)
                entries.append({"name": n["name"], "base": n["base"], "bytes": len(raw), "offset": off_})
                off_ += len(raw)
    Path(f"{path_prefix}.json").write_text(json.dumps(entries))

ROOTS = [("buf", e.buf), ("pbuf", e.pbuf), ("mbuf", e.mbuf), ("st", e.st)]
# every upload's host bytes, by phase and order (the Zig side builds its own and compares)
upl = out / "uploads"
upl.mkdir(exist_ok=True)
_add = TR.add
_seq = {}
def add(kind, name, **kw):
    _add(kind, name, **kw)
dump_state(str(out / "pre_prefill"), ROOTS)
_chunk = D.prefill_chunk
def prefill_chunk(e_, prompt, start, **kw):
    last = _chunk(e_, prompt, start, **kw)
    if last is not None:
        marks["prefill"] = {"logits": ct.tdesc(last), "last_streams": ct.tdesc(e_.last_streams)}
    return last
D.prefill_chunk = prefill_chunk
_stage = F.stage
_n = [0]
def stage(w_, b, windows):
    segs = _stage(w_, b, windows)
    if TR.on and TR.phase == "prefill":
        torch.cuda.synchronize()
        R = segs[-1][2]
        b.ids[:R].cpu().numpy().tofile(upl / f"prefill_c{_n[0]}_ids.bin")
        b.ple_v.contiguous().view(torch.uint8).cpu().numpy().tofile(upl / f"prefill_c{_n[0]}_ple_v.bin")
        _n[0] += 1
    return segs
F.stage = stage
with TR.scope("prefill"):
    first = D.prefill(e, ids, None, mtp=True)
D.prefill_chunk = _chunk
print(f"rank {rank}: prefill {len(ids)} tokens, first {first}, mtp_len {e.st.mtp_len}", flush=True)
dump_state(str(out / "after_prefill"), [("st", e.st)])

# the reference reply with MTP drafts, every round logged (untraced)
log = []
_fwd, _commit, _sd, _mf = e.forward, D.commit, e.sample_draft, e.mtp_forward
def lfwd(tokens):
    log.append({"verify": list(tokens), "pos": e.st.pos})
    return _fwd(tokens)
def lcommit(w_, st, b, R, keep, *a, **kw):
    log.append({"commit": [R, keep]})
    return _commit(w_, st, b, R, keep, *a, **kw)
def lsd(logits, position, sampling):
    t, p = _sd(logits, position, sampling)
    log.append({"draft": t, "p": p})
    return t, p
def lmf(next_tokens, streams):
    log.append({"mtp": list(next_tokens), "mtp_len": e.st.mtp_len})
    return _mf(next_tokens, streams)
e.forward, D.commit, e.sample_draft, e.mtp_forward = lfwd, lcommit, lsd, lmf
torch.cuda.synchronize()
s = time.perf_counter()
res = D.mtp_decode(e, first, COUNT, None, depth=DEPTH, confidence=CONF)
torch.cuda.synchronize()
secs = time.perf_counter() - s
e.forward, D.commit, e.sample_draft, e.mtp_forward = _fwd, _commit, _sd, _mf
print(f"rank {rank}: reference {res.tokens[:12]} ... rounds {res.rounds} drafted {res.drafted} accepted {res.accepted} "
      f"({secs:.2f} s eager)", flush=True)
(out / "reference.json").write_text(json.dumps({"prompt": ids, "tokens": res.tokens, "first": first,
    "rounds": res.rounds, "drafted": res.drafted, "accepted": res.accepted, "keeps": res.keeps, "widths": res.widths,
    "log": log, "vocab_offset": int(w.meta.get("vocab_offset", 0)), "depth": DEPTH, "confidence": CONF}))

# coverage: every program a round can need, traced from the state the reply left (outputs unused)
toks = list(res.tokens[-7:]) + [1] * 7
for R in range(1, DEPTH + 2):
    for _ in range(2):
        p = e.st.cur[0]
        with TR.scope(f"fwd_R{R}_p{p}"):
            lg = e.forward(toks[:R])
        marks[f"fwd_R{R}_p{p}"] = {"logits": ct.tdesc(lg)}
        with TR.scope(f"commit_R{R}_k{R}_p{p}"):
            D.commit(w, e.st, e.buf, R, R)
    for keep in range(1, R):
        for _ in range(2):
            p = e.st.cur[0]
            e.forward(toks[:R])
            with TR.scope(f"commit_R{R}_k{keep}_p{p}"):
                D.commit(w, e.st, e.buf, R, keep)
for n in range(1, DEPTH + 2):
    with TR.scope(f"mtp_n{n}"):
        lg = e.mtp_forward(toks[:n], e.buf.streams[:n])
    marks[f"mtp_n{n}"] = {"logits": ct.tdesc(lg)}
    e.st.set_mtp_len(e.st.mtp_len + n)
torch.cuda.synchronize()
(out / "trace.json").write_text(json.dumps({"events": TR.events, "kernels": TR.kernels}))
(out / "named.json").write_text(json.dumps(ct.named_map([("w", w)] + ROOTS)))
(out / "marks.json").write_text(json.dumps(marks))
print(f"rank {rank}: trace {len(TR.events)} events", flush=True)

if os.environ.get("SKIP_PACK") == "1":
    raise SystemExit(0)
index, seen = [], set()
pack = open(out / "weights.bin", "wb")
offset = 0
def walk(name, v, depth=0):
    global offset
    if depth > 12 or id(v) in seen:
        return
    if isinstance(v, torch.Tensor):
        seen.add(id(v))
        if not v.is_cuda:
            index.append({"name": name, "host": True, "dtype": str(v.dtype), "shape": list(v.shape)})
            return
        raw = v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        pack.write(raw)
        index.append({"name": name, "dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape),
                      "stride": list(v.stride()), "contiguous": v.is_contiguous(), "offset": offset, "bytes": len(raw)})
        offset += len(raw)
        return
    if isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            walk(f"{name}.{i}", x, depth + 1)
        return
    if isinstance(v, (int, float, str, bool)) or v is None or name.endswith(("comm", ".meta")):
        return
    seen.add(id(v))
    fields = ([f.name for f in dataclasses.fields(v)] if dataclasses.is_dataclass(v)
              else list(vars(v)) if hasattr(v, "__dict__") else [])
    for f in fields:
        walk(f"{name}.{f}" if name else f, getattr(v, f, None), depth + 1)
walk("", w)
pack.close()
(out / "weights.json").write_text(json.dumps({"tensors": index}, indent=0))
print(f"rank {rank}: {len(index)} tensors, {offset / 2**30:.1f} GiB packed; done in {time.time() - t0:.0f} s", flush=True)
