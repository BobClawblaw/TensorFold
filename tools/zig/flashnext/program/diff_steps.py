#!/usr/bin/env python3
"""Which decode-step arguments change with the position: diff phases' symbolic ops field by field."""
# usage: diff_steps.py <capture rank dir with trace.json and named.json> <phase>,<phase>[,...]
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_lib as T

d = sys.argv[1]
phases = sys.argv[2].split(",")
named = json.load(open(d + "/named.json"))
progs = {}
for ph in phases:
    res = T.Resolver(named)
    w = T.work(T.load(d, ph))
    progs[ph] = [(e["kind"], e["name"], json.loads(json.dumps(T.symbolic({k: e.get(k) for k in ("args", "kwargs", "grid")}, res)))) for e in w]
lens = {ph: len(p) for ph, p in progs.items()}
print("work ops:", lens)


def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f"{p}[{i}]")
    else:
        yield p, o


base = phases[0]
var = defaultdict(list)
n = min(lens.values())
for i in range(n):
    kinds = {(progs[ph][i][0], progs[ph][i][1]) for ph in phases}
    if len(kinds) > 1:
        print("structure differs at", i, kinds)
        break
    fl = {ph: dict(flat(progs[ph][i][2])) for ph in phases}
    for key in fl[base]:
        vals = [fl[ph].get(key) for ph in phases]
        if len(set(map(str, vals))) > 1:
            var[(progs[base][i][1], key)].append((i, vals))
print(len(var), "varying fields")
for (name, key), occ in sorted(var.items(), key=lambda kv: kv[1][0][0])[:60]:
    print(f"{len(occ):3d}x {name:22s} {key:28s} e.g. op {occ[0][0]}: {occ[0][1]}")
