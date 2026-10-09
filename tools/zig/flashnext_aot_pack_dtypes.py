#!/usr/bin/env python3
"""flashnext_aot_pack.py with Triton dtype constexprs: attn_multi's KT (the cache's element type) packs as an integer
code the native engine compares (kern.zig KT_INT8): int8 1, uint8 2. A constexpr None packs with neither int nor f32."""
import runpy, sys
mod = runpy.run_path(sys.argv.pop(1), run_name="aot_pack")      # first argument: the path of aot_pack.py
base = mod["const"]
DTYPES = {"int8": 1, "uint8": 2}
def const(v):
    if v is None:
        return {}
    if isinstance(v, str) and v in DTYPES:
        return {"int": DTYPES[v]}
    return base(v)
mod["main"].__globals__["const"] = const
sys.exit(mod["main"]())
