#!/usr/bin/env python3
"""A rank's decode program from capture_fn.py's trace: storages, temporaries, step and commit ops by parity."""
# usage: gen_program.py <capture rank dir> <pack rank dir> <out.json>; env RESET_DUMP (default reset_storages)
import bisect, json, sys
from pathlib import Path

META = {"aten.slice.Tensor", "aten.select.int", "aten.detach.default", "aten.view.default", "aten.view.dtype",
        "aten.unsqueeze.default", "aten.alias.default", "aten.as_strided.default", "aten.expand.default",
        "aten.t.default", "aten.reshape.default", "aten._unsafe_view.default", "aten.permute.default",
        "aten.squeeze.dim", "aten.transpose.int", "aten.split.Tensor", "aten.unbind.int", "aten.chunk.default",
        "aten.flatten.using_ints", "aten.narrow.default", "aten.split_with_sizes.default", "aten.lift_fresh.default"}
ALLOC = {"aten.empty.memory_format", "aten.empty_like.default", "aten.empty_strided.default"}
SIZE = {"bfloat16": 2, "float16": 2, "float32": 4, "int32": 4, "int64": 8, "uint8": 1, "int8": 1, "bool": 1,
        "int16": 2, "float8_e4m3fn": 1}

cap, pack, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
trace = json.load(open(cap / "trace.json"))
named = json.load(open(cap / "named.json"))
import os
RESET = os.environ.get("RESET_DUMP", "reset_storages")       # or pre_prefill: the state the traced prefill began from
reset = {e["base"]: e for e in json.load(open(cap / f"{RESET}.json"))}
packidx = {t["name"]: t for t in json.load(open(pack / "weights.json"))["tensors"] if not t.get("host")}

# storages: one per distinct base among named tensors
storages, by_base = [], {}
for n in sorted(named, key=lambda n: n["base"]):
    if n["base"] not in by_base:
        by_base[n["base"]] = len(storages)
        storages.append({"name": n["name"], "base": n["base"], "bytes": n["bytes"], "fills": []})
    s = storages[by_base[n["base"]]]
    if n["name"].startswith("w."):
        p = packidx.get(n["name"][2:])
        if p is None:
            raise SystemExit(f"{n['name']}: not in the pack")
        s["fills"].append({"pack_offset": p["offset"], "bytes": p["bytes"], "at": n["ptr"] - n["base"]})
for s in storages:
    if s["name"].startswith("w."):
        s["init"] = "pack"
    elif s["base"] in reset:
        s["init"] = "reset"
        s["reset_offset"] = reset[s["base"]]["offset"]
        assert reset[s["base"]]["bytes"] == s["bytes"], s["name"]
    else:
        s["init"] = "zero"
starts = sorted(by_base)


def extent(t):
    if not t["shape"] or 0 in t["shape"]:
        return SIZE[t["dtype"]]
    return (sum((d - 1) * st for d, st in zip(t["shape"], t["stride"])) + 1) * SIZE[t["dtype"]]


class Temps:
    def __init__(self):
        self.live, self.sizes = {}, []

    def new(self, base):
        self.live[base] = len(self.sizes)
        self.sizes.append(0)
        return self.live[base]

    def ref(self, base):
        return self.live[base] if base in self.live else self.new(base)


def named_ref(base):
    i = bisect.bisect_right(starts, base) - 1
    if i >= 0:
        s = storages[by_base[starts[i]]]
        if s["base"] <= base < s["base"] + max(s["bytes"], 1):
            return by_base[starts[i]], base - s["base"]
    return None


SAMPLER = {"aten._to_copy.default", "aten.topk.default", "aten.add.Tensor", "aten.logsumexp.default",
           "aten.argmax.default", "aten.gather.default", "aten.cat.default"}


def phase_ops(phase, temps, check_names=None):
    ops, sampler, uploads = [], False, 0
    checks = list(check_names or [])
    ret_base = None            # the last extension call's returned storage: its own allocation shows up after it

    def sym(v):
        if isinstance(v, dict) and v.get("t") == 1:
            if v["dev"] != "cuda":
                return {"host": True, "dtype": v["dtype"], "shape": v["shape"]}
            r = named_ref(v["base"])
            if r is not None:
                sid, _ = r
                ref = {"s": sid, "off": v["ptr"] - storages[sid]["base"]}
            else:
                tid = temps.ref(v["base"])
                off = v["ptr"] - v["base"]
                temps.sizes[tid] = max(temps.sizes[tid], off + extent(v))
                ref = {"tmp": tid, "off": off}
            return {**ref, "dtype": v["dtype"], "shape": v["shape"], "stride": v["stride"]}
        if isinstance(v, list):
            return [sym(x) for x in v]
        if isinstance(v, dict):
            return {k: sym(x) for k, x in v.items()}
        return v

    def addr_ref(value):
        """A pointer-valued fill: the storage or temporary the address points into."""
        r = named_ref(value)
        if r is not None:
            return {"s": r[0], "off": value - storages[r[0]]["base"]}
        for base, tid in temps.live.items():
            if base <= value < base + max(temps.sizes[tid], 1):
                return {"tmp": tid, "off": value - base}
        return None

    for e in trace["events"]:
        if e["phase"] != phase:
            continue
        name = e["name"]
        if e["kind"] == "aten" and name in META:
            continue
        if e["kind"] == "aten" and name == "aten._to_copy.default" and (e.get("kwargs") or {}).get("device") == "cpu":
            if checks:                                 # a dump: compare here (its name from the dump index, in order)
                ops.append({"kind": "check", "name": checks.pop(0), "args": [sym(e["args"][0])]})
            continue                                   # the capture's own dumps
        if e["kind"] == "aten" and name in ALLOC:
            if e["out"]["base"] == ret_base:
                ret_base = None                        # the call's own output allocation, already a temporary
                continue
            temps.new(e["out"]["base"])
            continue
        if e["kind"] == "aten" and name == "aten._to_copy.default" and \
                (e.get("kwargs") or {}).get("dtype") == "torch.float32" and not sampler:
            sampler = True                             # the sampler: the Zig side picks greedily itself
        if e["kind"] == "aten" and name in SAMPLER:
            continue                                   # sampler arithmetic, anywhere
        if sampler:
            if e["kind"] in ("comm",) or (e["kind"] == "aten" and name in ("aten.copy_.default", "aten.clone.default")):
                continue                               # the candidates' copies and their exchange
            if e["kind"] in ("triton", "ext"):
                sampler = False
        op = {"kind": e["kind"], "name": name}
        if e["kind"] == "triton":
            op.update(hash=e["hash"], grid=e["grid"], args=[{"type": a["type"], "v": sym(a["v"])} for a in e["args"]])
        elif e["kind"] == "aten" and name == "aten.clone.default":
            temps.new(e["out"]["base"])
            op = {"kind": "aten", "name": "aten.copy_.default", "args": [sym(e["out"]), sym(e["args"][0])], "kwargs": {}}
        else:
            op.update(args=sym(e.get("args", [])), kwargs=sym(e.get("kwargs", {})))
            if e.get("ret") is not None:
                temps.new(e["ret"]["base"])
                op["ret"] = sym(e["ret"])
                ret_base = e["ret"]["base"]
            if e["kind"] == "aten" and name == "aten.copy_.default":
                src = e["args"][1]
                if isinstance(src, dict) and src.get("dev") == "cpu":
                    op["kind"] = "upload"
                    op["input"] = storages[op["args"][0]["s"]]["name"]
                    op["seq"] = uploads
                    uploads += 1
            if e["kind"] == "aten" and name == "aten.fill_.Scalar":
                tgt = e["args"][0]
                op["fill_name"] = storages[op["args"][0]["s"]]["name"] if "s" in op["args"][0] else None
                if tgt["dtype"] == "int64" and isinstance(e["args"][1], int) and e["args"][1] > 1 << 32:
                    ref = addr_ref(e["args"][1])
                    if ref is None:
                        raise SystemExit(f"a pointer fill {e['args'][1]:#x} resolves to nothing")
                    op["fill_ptr"] = ref
        ops.append(op)
    return ops


programs = {}
for step, commit in (("teacher01", "commit01"), ("teacher02", "commit02")):
    temps = Temps()
    fwd = phase_ops(step, temps)
    com = phase_ops(commit, temps)
    programs["odd" if step.endswith("1") else "even"] = {"forward": fwd, "commit": com, "temps": temps.sizes}
if (cap / "chunk" / "index.json").exists() and any(e["phase"] == "chunk" for e in trace["events"]):
    temps = Temps()
    names = list(json.load(open(cap / "chunk" / "index.json")))
    programs["chunk"] = {"forward": phase_ops("chunk", temps, names), "commit": [], "temps": temps.sizes}
if any(e["phase"] == "prefill" for e in trace["events"]):
    temps = Temps()
    programs["prefill"] = {"forward": phase_ops("prefill", temps), "commit": [], "temps": temps.sizes}
consts = {}
for e in trace["events"]:
    if e["kind"] == "triton" and e["hash"] not in consts:
        consts[e["hash"]] = {k: v for k, v in (e.get("consts") or {}).items() if isinstance(v, (int, bool))}
kernels = {h: {"name": k["name"], "files": {leaf: p.replace("/cache/tf/", "") for leaf, p in k["files"].items()},
               "attrs": k["attrs"], "consts": consts.get(h, {})} for h, k in trace["kernels"].items()}
outp.write_text(json.dumps({"storages": storages, "programs": programs, "kernels": kernels}))
kinds = {}
for name, p in programs.items():
    from collections import Counter
    kinds[name] = dict(Counter(o["kind"] for o in p["forward"] + p["commit"]))
print(json.dumps({"storages": len(storages), "bytes_GiB": round(sum(s["bytes"] for s in storages) / 2**30, 2),
                  "temps": {k: (len(p["temps"]), sum(p["temps"])) for k, p in programs.items()}, "ops": kinds}))
