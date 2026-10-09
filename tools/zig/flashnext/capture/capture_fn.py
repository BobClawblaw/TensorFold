#!/usr/bin/env python3
"""Two-rank oracle for the native port: reference tokens, per-layer dumps, traces and the prepared-weight pack."""
# usage (both ranks): capture_fn.py <rank> <out root> <model dir>; env TF_MASTER, CAP_RECORD=1|2, SKIP_PACK=1
import dataclasses, hashlib, json, os, sys, time
from pathlib import Path


import numpy as np
import torch
MASTER = os.environ.get("TF_MASTER") or sys.exit("TF_MASTER is not set: rank 0's address as rank 1 reaches it")
PORT = int(os.environ.get("TF_PORT", "29571"))                  # the engines' rendezvous port

rank, root, model = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out = root / f"rank{rank}"
out.mkdir(parents=True, exist_ok=True)

REC = None
TR = None


def phase(name, detail):
    from contextlib import nullcontext
    if TR is not None:
        return TR.scope(name) if detail else nullcontext()
    return REC.scope(name, detail) if REC is not None else nullcontext()


from tensorfold.families.qwen4_exp.cuda import decode as D
from tensorfold.families.qwen4_exp.cuda import forward as F
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

TEACHER, DUMP_STEPS, CHUNK = 16, (0, 1, 15), 300


class Dumps:
    dir: Path | None = None

    def save(self, name, t):
        if self.dir is None or t is None:
            return
        torch.cuda.synchronize()
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tofile(self.dir / f"{name}.bin")
        meta = self.dir / "index.json"
        idx = json.loads(meta.read_text()) if meta.exists() else {}
        idx[name] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}
        meta.write_text(json.dumps(idx, indent=0))


dumps = Dumps()
_embed, _pre_moe, _moe_block, _finish, _stage = F._embed, F._pre_moe, F.moe_block, F.finish, F.stage


def embed(w, ids, copies, out_):
    r = _embed(w, ids, copies, out_)
    dumps.save("embed_h", out_)
    return r


def pre_moe(layer, w, segs, b, R, pending, **kw):
    _pre_moe(layer, w, segs, b, R, pending, **kw)
    dumps.save(f"L{layer.index:02d}_h", b.h[:R])
    dumps.save(f"L{layer.index:02d}_mixed", b.mixed[:R])


def moe_block(layer, w, b, R):
    mode, a, wts = _moe_block(layer, w, b, R)
    dumps.save(f"L{layer.index:02d}_moe_y", a)
    dumps.save(f"L{layer.index:02d}_moe_wts", wts)
    return mode, a, wts


def finish(w, mixer, b, R, pending, logits=True, ends=()):
    res = _finish(w, mixer, b, R, pending, logits=logits, ends=ends)
    dumps.save("final_streams", b.streams[:R])
    dumps.save("final_mixed", b.mixed[:max(1, len(ends)) if b.prefill else R])
    dumps.save("logits", res)
    return res


_chunk_no = [0]


def stage(w, b, windows):
    segs = _stage(w, b, windows)
    R = segs[-1][2]
    if TR is not None and TR.on and TR.phase == "prefill":
        torch.cuda.synchronize()
        k = _chunk_no[0]
        _chunk_no[0] += 1
        b.ids[:R].cpu().numpy().tofile(out / "prefill_inputs" / f"c{k}_ids.bin")
        b.ple_v.contiguous().view(torch.uint8).cpu().numpy().tofile(out / "prefill_inputs" / f"c{k}_ple_v.bin")
    dumps.save("ids", b.ids[:R])
    for name in ("ple_w", "ple_s", "ple_b", "ple_v"):
        t = getattr(b, name, None)
        if t is not None:
            dumps.save(f"staged_{name}", t)
    return segs


F._embed, F._pre_moe, F.moe_block, F.finish, F.stage = embed, pre_moe, moe_block, finish, stage

_ple_block, _hc_block, _gdn_block, _attn_block = F.ple_block, F.hc_block, F.gdn_block, F.attn_block
_fine = {"n": 0}


def ple_block(layer, w, segs, b, R):
    _ple_block(layer, w, segs, b, R)
    dumps.save(f"F{_fine['n']:04d}_ple_h", b.h[:R]); _fine["n"] += 1


def hc_block(hc, b, R, eps, streams, low, mode, inject_prev, inject_out, h, branch=None, y=None, wts=None):
    _hc_block(hc, b, R, eps, streams, low, mode, inject_prev, inject_out, h, branch=branch, y=y, wts=wts)
    dumps.save(f"F{_fine['n']:04d}_hc_h", h[:R]); _fine["n"] += 1
    dumps.save(f"F{_fine['n']:04d}_hc_normed", b.normed[:R]); _fine["n"] += 1
    dumps.save(f"F{_fine['n']:04d}_hc_act", b.act[:R]); _fine["n"] += 1
    dumps.save(f"F{_fine['n']:04d}_hc_mixed", b.mixed[:R]); _fine["n"] += 1


def gdn_block(layer, w, segs, b, R, cuts=()):
    mode, branch = _gdn_block(layer, w, segs, b, R, cuts)
    dumps.save(f"F{_fine['n']:04d}_gdn_gout", b.gout[:R]); _fine["n"] += 1
    dumps.save(f"F{_fine['n']:04d}_gdn_branch", branch if not isinstance(branch, tuple) else branch[0]); _fine["n"] += 1
    return mode, branch


def attn_block(layer, w, segs, b, R, mtp, context=None):
    mode, branch = _attn_block(layer, w, segs, b, R, mtp, context)
    dumps.save(f"F{_fine['n']:04d}_attn_branch", branch if not isinstance(branch, tuple) else branch[0]); _fine["n"] += 1
    return mode, branch


F.ple_block, F.hc_block, F.gdn_block, F.attn_block = ple_block, hc_block, gdn_block, attn_block


def dump_state(path_prefix, roots):
    """Every CUDA storage reachable from roots, raw: <prefix>.bin + <prefix>.json (name, base, bytes, offset)."""
    torch.cuda.synchronize()
    seen, entries, off_ = set(), [], 0
    with open(f"{path_prefix}.bin", "wb") as f:
        for root_name, root in roots:
            for n in ct.named_map([(root_name, root)]):
                if n["base"] in seen:
                    continue
                seen.add(n["base"])
                obj = root
                for part in n["name"].split(".")[1:]:
                    obj = obj[int(part)] if isinstance(obj, (list, tuple)) else (obj[part] if isinstance(obj, dict) else getattr(obj, part))
                st_ = obj.untyped_storage()
                raw = torch.empty(0, dtype=torch.uint8, device="cuda").set_(st_, 0, (st_.nbytes(),)).cpu().numpy().tobytes()
                f.write(raw)
                entries.append({"name": n["name"], "base": n["base"], "bytes": len(raw), "offset": off_})
                off_ += len(raw)
    Path(f"{path_prefix}.json").write_text(json.dumps(entries))


t0 = time.time()
eng = FlashNextEngine(model, depth=0, max_len=16384, context_explicit=True, tp=2, rank=rank, master=MASTER,
                      port=PORT, graphs=False, kv_dtype=os.environ.get("CAP_KV", "int8"), prefetch=True)
e, w = eng.e, eng.w
if os.environ.get("CAP_RECORD") == "1":           # after startup: NCCL and the loaders are left alone
    sys.path.insert(0, str(root))
    import triton_aot_manifest as aot
    REC = aot.Recorder().install()
    from tensorfold.cuda import experts as _ex
    from tensorfold.cuda.kernels import gdn as _gdn, qmm as _qmm
    from tensorfold.families.qwen4_exp.cuda import gdn as _fgdn, gdn_io as _fio
    for _label, _mod in (("experts", _ex), ("gdn_v2", _gdn), ("qmm", _qmm), ("qwen4_exp_gdn", _fgdn), ("gdn_io", _fio)):
        _m = _mod._ext()
        REC.wrap(_m, tuple(n for n in dir(_m) if not n.startswith("_") and callable(getattr(_m, n))), _label)
TR = None
if os.environ.get("CAP_RECORD") == "2":           # the full trace (capture_trace.py), after startup
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
body = json.loads((root / "request.json").read_text())
text = body["messages"][0]["content"]
ids = tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                 add_special_tokens=False).ids

# 1) reference: a fresh prefill and 48 greedy tokens, one a step
if TR is not None:                                  # the n-gram constants the Zig host side needs
    ple = next(layer.ple for layer in w.layers if layer.ple is not None)
    ng, tb = ple.ngram, ple.table
    (out / "ngram.json").write_text(json.dumps({
        "n": ng.n, "per_ngram": ng.per_ngram, "heads": ng.heads, "eos": ng.eos, "dims": ng.dims,
        "head_sizes": ng.head_sizes.tolist(), "head_offsets": ng.head_offsets.tolist(),
        "multipliers": [int(x) for x in ng.multipliers.tolist()], "width": tb.width, "rows": tb.rows,
        "starts": tb.starts.tolist(), "lut": tb.lut.tolist(),
        "shards": [{"file": str(v.filename), "offset": int(v.offset), "rows": int(v.shape[0])} for v in tb.values],
        "initial_history": ng.initial_history().tolist(), "vocab_offset": int(w.meta.get("vocab_offset", 0))}))
    prefill_inputs = out / "prefill_inputs"
    prefill_inputs.mkdir(exist_ok=True)
if TR is not None:                                  # the state the traced prefill starts from (after warm-up)
    dump_state(str(out / "pre_prefill"), [("buf", e.buf), ("pbuf", e.pbuf), ("st", e.st)])
with phase("prefill", True):
    first = D.prefill(e, ids, None, mtp=False)
if TR is not None:                                  # the state right after the prefill, every st storage raw
    torch.cuda.synchronize()
    seen, entries, off_ = set(), [], 0
    with open(out / "after_prefill.bin", "wb") as f:
        for n in ct.named_map([("st", e.st), ("buf", e.buf)]):
            if n["base"] in seen:
                continue
            seen.add(n["base"])
            obj = {"st": e.st, "buf": e.buf}[n["name"].split(".")[0]]
            for part in n["name"].split(".")[1:]:
                obj = obj[int(part)] if isinstance(obj, (list, tuple)) else (obj[part] if isinstance(obj, dict) else getattr(obj, part))
            st_ = obj.untyped_storage()
            raw = torch.empty(0, dtype=torch.uint8, device="cuda").set_(st_, 0, (st_.nbytes(),)).cpu().numpy().tobytes()
            f.write(raw)
            entries.append({"name": n["name"], "base": n["base"], "bytes": len(raw), "offset": off_})
            off_ += len(raw)
    (out / "after_prefill.json").write_text(json.dumps(entries))
if TR is not None:                                  # each decode step after the prefill as its own phase
    _fwd, _commit, _n = e.forward, D.commit, {"i": 0}

    decdir = out / "dec"
    decdir.mkdir(exist_ok=True)

    def _traced_forward(tokens):
        TR.phase = f"dec{_n['i']:02d}"
        lg = _fwd(tokens)
        torch.cuda.synchronize()
        TR.on = False
        k_ = _n["i"]
        e.buf.ids[:1].cpu().numpy().tofile(decdir / f"d{k_:02d}_ids.bin")
        e.buf.ple_v.contiguous().view(torch.uint8).cpu().numpy().tofile(decdir / f"d{k_:02d}_ple_v.bin")
        lg[:1].contiguous().view(torch.uint8).cpu().numpy().tofile(decdir / f"d{k_:02d}_logits.bin")
        TR.on = True
        return lg

    def _traced_commit(*a, **kw):
        TR.phase = f"dcommit{_n['i']:02d}"
        _n["i"] += 1
        return _commit(*a, **kw)
    e.forward, D.commit = _traced_forward, _traced_commit
    with TR.scope("dec00"):
        res = D.serial_decode(e, first, int(os.environ.get("CAP_DEC_STEPS", "48")), None)
    e.forward, D.commit = _fwd, _commit
else:
    res = D.serial_decode(e, first, 48, None)
reference = {"prompt": ids, "tokens": res.tokens if hasattr(res, "tokens") else res[0], "first": first}
print(f"rank {rank}: reference {len(reference['tokens'])} tokens: {reference['tokens'][:12]} ...", flush=True)

# 2) teacher-forced single-row steps from an empty state over the prompt's first tokens
e.reset()
if TR is not None:                                  # the state after reset, every buffer and state storage raw
    torch.cuda.synchronize()
    seen, entries, off = set(), [], 0
    with open(out / "reset_storages.bin", "wb") as f:
        for root_name, root in (("buf", e.buf), ("pbuf", e.pbuf), ("st", e.st)):
            for n in ct.named_map([(root_name, root)]):
                if n["base"] in seen:
                    continue
                seen.add(n["base"])
                tensor = None
                # find the tensor object again to read its storage bytes
                obj = root
                for part in n["name"].split(".")[1:]:
                    obj = obj[int(part)] if isinstance(obj, (list, tuple)) else (obj[part] if isinstance(obj, dict) else getattr(obj, part))
                st_ = obj.untyped_storage()
                raw = torch.empty(0, dtype=torch.uint8, device="cuda").set_(st_, 0, (st_.nbytes(),)).cpu().numpy().tobytes()
                f.write(raw)
                entries.append({"name": n["name"], "base": n["base"], "bytes": len(raw), "offset": off})
                off += len(raw)
    (out / "reset_storages.json").write_text(json.dumps(entries))
    stepdir = out / "steps"
    stepdir.mkdir(exist_ok=True)
steps = []
for i, t in enumerate(ids[:TEACHER]):
    dumps.dir = out / f"teacher{i:02d}" if i in DUMP_STEPS else None
    if dumps.dir is not None:
        dumps.dir.mkdir(parents=True, exist_ok=True)
    with phase(f"teacher{i:02d}", i in (0, 1, 2)):
        logits = e.forward([t])
        torch.cuda.synchronize()
    steps.append(hashlib.sha256(logits[:1].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
    if TR is not None:                              # every step's inputs as staged and its logits
        b_ = e.buf
        b_.ids[:1].cpu().numpy().tofile(stepdir / f"s{i:02d}_ids.bin")
        for nm in ("ple_v", "ple_w", "ple_s", "ple_b"):
            tt = getattr(b_, nm, None)
            if tt is not None:
                tt.contiguous().view(torch.uint8).cpu().numpy().tofile(stepdir / f"s{i:02d}_{nm}.bin")
        logits[:1].contiguous().view(torch.uint8).cpu().numpy().tofile(stepdir / f"s{i:02d}_logits.bin")
    with phase(f"commit{i:02d}", i in (0, 1, 2)):
        F.commit(w, e.st, e.buf, 1, 1)
dumps.dir = None

# 3) one prompt chunk of CHUNK rows from an empty state (the prompt kernels)
e.reset()
dumps.dir = out / "chunk"
dumps.dir.mkdir(parents=True, exist_ok=True)
with phase("chunk", True):
    F.forward(w, e.st, e.pbuf, ids[:CHUNK], logits=True)
    torch.cuda.synchronize()
dumps.dir = None
if REC is not None:
    REC.dump(out / "launches.json")
if TR is not None:
    (out / "trace.json").write_text(json.dumps({"events": TR.events, "kernels": TR.kernels}))
    (out / "named.json").write_text(json.dumps(ct.named_map([("w", w), ("buf", e.buf), ("pbuf", e.pbuf), ("st", e.st)])))
    print(f"rank {rank}: trace {len(TR.events)} events", flush=True)
(out / "reference.json").write_text(json.dumps({**reference, "teacher_logits_sha256": steps,
                                                 "vocab_offset": int(w.meta.get("vocab_offset", 0)),
                                                 "chunk_rows": CHUNK, "teacher_steps": TEACHER}))

if os.environ.get("SKIP_PACK") == "1":
    print(f"rank {rank}: done (no pack) in {time.time() - t0:.0f} s", flush=True)
    raise SystemExit(0)
# 4) the prepared weights: every CUDA tensor reachable from the weights, raw bytes plus an index
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
                      "stride": list(v.stride()), "contiguous": v.is_contiguous(), "offset": offset, "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()})
        offset += len(raw)
        return
    if isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            walk(f"{name}.{i}", x, depth + 1)
        return
    if isinstance(v, (int, float, str, bool)) or v is None:
        return
    if name.endswith(("comm", ".meta")):
        return
    seen.add(id(v))
    fields = ([f.name for f in dataclasses.fields(v)] if dataclasses.is_dataclass(v)
              else list(vars(v)) if hasattr(v, "__dict__") else [])
    for f in fields:
        walk(f"{name}.{f}" if name else f, getattr(v, f, None), depth + 1)


walk("", w)
pack.close()
cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(w.cfg).items()}
(out / "weights.json").write_text(json.dumps({"tensors": index, "cfg": cfg,
                                              "meta": {k: v for k, v in w.meta.items() if isinstance(v, (int, float, str, bool))}},
                                             indent=0))
print(f"rank {rank}: {len(index)} tensors, {offset / 2**30:.1f} GiB packed; done in {time.time() - t0:.0f} s", flush=True)
