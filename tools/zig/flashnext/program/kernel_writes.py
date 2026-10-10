#!/usr/bin/env python3
"""Which parameters each traced Triton kernel writes (from its TTIR), added to a program JSON."""
# usage: kernel_writes.py <Triton cache dir> <program.json> [verbose]
import json, re, sys
from pathlib import Path

FOLLOW0 = {"tt.addptr", "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "ttg.convert_layout",
           "tt.make_tensor_ptr", "tt.advance", "tt.bitcast", "tt.trans", "tt.make_tensor_descriptor"}
NAME = r"%[\w$.-]+(?:#\d+)?"

def analyze(text):
    m = re.search(r"tt\.func public @\w+\((.*?)\)\s*(?:attributes|\{)", text, re.S)
    params = re.findall(r"(%[\w$.-]+): ", m.group(1))
    defs, carried = {}, {}
    for line in text.splitlines():
        line = line.split(" loc(")[0].strip()
        fm = re.match(r"(%[\w$.-]+)(?::\d+)? = scf\.(for|while)\b(.*)", line)
        if fm:
            ia = re.search(r"iter_args\((.*?)\)\s*->", line) or re.search(r"\((.*?)\)\s*:", fm.group(3))
            if ia:
                for a, b in re.findall(rf"({NAME}) = ({NAME})", ia.group(1)):
                    carried[a] = b
            continue
        dm = re.match(rf"({NAME})(?::\d+)? = ([\w.]+) (.*)", line)
        if dm:
            name, op, rest = dm.groups()
            ops = re.findall(NAME, rest.split(" : ")[0])
            defs[name] = (op, ops)
    def roots(v, seen):
        v = v.split("#")[0] if v not in defs and v.split("#")[0] in defs else v
        if v in seen:
            return set()
        seen.add(v)
        if v in params:
            return {params.index(v)}
        if v in carried:
            return roots(carried[v], seen)
        if v not in defs:
            return None
        op, ops = defs[v]
        if op in FOLLOW0 and ops:
            return roots(ops[0], seen)
        if op == "arith.select" and len(ops) >= 3:
            a, b = roots(ops[1], seen), roots(ops[2], seen)
            return None if a is None or b is None else a | b
        return None
    writes = set()
    for line in text.splitlines():
        s = line.split(" loc(")[0].strip()
        ptr = None
        if s.startswith("tt.store ") or s.startswith("tt.descriptor_store "):
            ptr = re.findall(NAME, s)[0]
        elif "tt.atomic_rmw" in s or "tt.atomic_cas" in s:
            ptr = re.findall(NAME, s.split("=", 1)[1] if "=" in s else s)[0]
        elif "tt.elementwise_inline_asm" in s or "tt.call" in s:
            return "all", params
        if ptr is None:
            continue
        r = roots(ptr, set())
        if r is None:
            return "all", params
        writes |= r
    return sorted(writes), params

cache, prog_path = Path(sys.argv[1]), Path(sys.argv[2])
prog = json.load(open(prog_path))
stats = {"all": 0, "exact": 0}
nargs = {}
for name, p in prog["programs"].items():
    for o in p["forward"]:
        if o["kind"] == "triton":
            nargs[o["hash"]] = len(o["args"])
for h, k in prog["kernels"].items():
    ttir = next((v for f, v in k["files"].items() if f.endswith(".ttir")), None)
    if ttir is None:
        k["writes"] = "all"; stats["all"] += 1; continue
    w, params = analyze((cache / ttir).read_text())
    if w != "all" and h in nargs and len(params) != nargs[h]:
        w = "all"                                       # parameters and traced arguments disagree: stay safe
    k["writes"] = w
    stats["all" if w == "all" else "exact"] += 1
    if len(sys.argv) > 3:
        print(k["name"], w, [params[i] for i in w] if w != "all" else "", len(params), nargs.get(h))
json.dump(prog, open(prog_path, "w"))
print(json.dumps(stats))
