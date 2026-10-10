#!/usr/bin/env python3
"""Compares cuobjdump -sass of the Python extension builds and our fatbins kernel by kernel: functions keyed by their
demangled name with namespaces dropped, every instruction and encoding compared. usage: sass_compare.py PY.sass ZIG.sass"""
import re, subprocess, sys

def kernels(path):
    out, cur = {}, None
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1); out[cur] = []; continue
        if cur is not None and re.match(r"\s*(/\*[0-9a-f]{4}\*/|/\* 0x)", line):
            out[cur].append(re.sub(r"\s+", " ", line.strip()))
    names = list(out)
    dem = subprocess.run(["cu++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    key = lambda d: re.sub(r"(\(anonymous namespace\)|<unnamed>|tf_fn)::", "", d).replace("(int)", "")
    return {key(d): out[n] for n, d in zip(names, dem)}

py, zg = kernels(sys.argv[1]), kernels(sys.argv[2])
bad = 0
for k in sorted(py):
    if k not in zg:
        print(f"MISSING in ours: {k[:110]}"); bad += 1
    elif py[k] != zg[k]:
        print(f"DIFFERS: {k[:110]} ({len(py[k])} vs {len(zg[k])} lines)"); bad += 1
extra = [k for k in zg if k not in py]
print(f"{len(py)} Python kernels, {len(py) - bad} identical; {len(extra)} only in ours")
sys.exit(1 if bad else 0)
