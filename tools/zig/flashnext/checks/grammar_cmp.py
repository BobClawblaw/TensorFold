#!/usr/bin/env python3
"""Walk grammars with xgrammar (py) or libtfgrammar (shim): each step's mask hash and the chosen token."""
# usage: grammar_cmp.py py|shim <tokenizer.json> <out.json> [libtfgrammar.so]
import ctypes, hashlib, json, sys
mode, tok_path, out = sys.argv[1:4]
VOCAB, STOPS = 248320, [248046, 248044]          # Qwen3.x Flash Next: padded vocabulary, <|im_end|> and <|endoftext|>
SPECS = [(0, ""), (1, json.dumps({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "age"]})),
         (2, r"\d{3}-[a-z]+ (yes|no)"), (3, json.dumps(["Paris", "Lyon", "Marseille/Nord", "Zürich"])), (4, 'root ::= "a" b+ "!"\nb ::= [0-9] | "x"')]
words = (VOCAB + 31) // 32
res = []
if mode == "py":
    import numpy as np, xgrammar as xgr
    from transformers import PreTrainedTokenizerFast
    info = xgr.TokenizerInfo.from_huggingface(PreTrainedTokenizerFast(tokenizer_file=tok_path), vocab_size=VOCAB, stop_token_ids=STOPS)
    comp = xgr.GrammarCompiler(info, max_threads=8, cache_limit_bytes=256 << 20)
    def build(kind, text):
        if kind == 0: return comp.compile_json_schema('{"type": "object"}', max_whitespace_cnt=32)
        if kind == 1: return comp.compile_json_schema(text, max_whitespace_cnt=32)
        if kind == 2: return comp.compile_regex(text)
        if kind == 3: return comp.compile_grammar("root ::= " + " | ".join(json.dumps(v, ensure_ascii=False) for v in json.loads(text)))
        return comp.compile_grammar(text)
    class M:
        def __init__(s, g): s.m = xgr.GrammarMatcher(g); s.bits = np.zeros((1, words), dtype=np.int32)
        def fill(s): s.m.fill_next_token_bitmask(s.bits, 0); return s.bits.tobytes()
        def accept(s, t): return s.m.accept_token(t)
        def done(s): return s.m.is_terminated()
    make = lambda k, t: M(build(k, t))
else:
    lib = ctypes.CDLL(sys.argv[4])
    lib.tfg_open.restype = lib.tfg_compile.restype = lib.tfg_matcher.restype = ctypes.c_void_p
    lib.tfg_open.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    lib.tfg_compile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    for f in ("tfg_matcher", "tfg_terminated"): getattr(lib, f).argtypes = [ctypes.c_void_p]
    lib.tfg_accept.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.tfg_fill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    err = ctypes.create_string_buffer(512)
    data = open(tok_path, "rb").read()
    c = lib.tfg_open(data, len(data), VOCAB, (ctypes.c_int32 * 2)(*STOPS), 2, err, 512)
    assert c, err.value
    class M:
        def __init__(s, k, t):
            b = t.encode(); g = lib.tfg_compile(c, k, b, len(b), err, 512); assert g, err.value
            s.m = lib.tfg_matcher(g); s.bits = (ctypes.c_int32 * words)()
        def fill(s): assert lib.tfg_fill(s.m, s.bits, words) == 0; return bytes(s.bits)
        def accept(s, t): return lib.tfg_accept(s.m, t) == 1
        def done(s): return lib.tfg_terminated(s.m) == 1
    make = M
for k, t in SPECS:
    for seed in range(3):
        m, walk = make(k, t), []
        for step in range(80):
            if m.done(): break
            raw = m.fill()
            allowed = [i for i in range(VOCAB) if raw[i // 8] >> (i % 8) & 1]
            tok = allowed[(seed * 7919 + step * 104729) % len(allowed)] if seed else min(allowed, key=lambda i: (i not in STOPS, i))
            walk.append([hashlib.sha1(raw).hexdigest()[:12], len(allowed), tok])
            assert m.accept(tok)
        res.append({"kind": k, "seed": seed, "walk": walk})
json.dump(res, open(out, "w"))
print(mode, sum(len(r["walk"]) for r in res), "steps")
