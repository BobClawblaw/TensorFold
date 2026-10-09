#!/usr/bin/env python3
"""Structured-output replies from a server: each case's text and whether it obeys its grammar."""
# usage: structured_set.py <port|url> <out.json>
import json, re, sys, urllib.request
BASE = sys.argv[1].rstrip("/") if "://" in sys.argv[1] else f"http://127.0.0.1:{sys.argv[1]}"   # a port or a URL
out = sys.argv[2]
person = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "hobbies": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
          "required": ["name", "age", "hobbies"], "additionalProperties": False}
cases = [
    ("json_object", "Describe a cat as JSON with a few fields.", {"response_format": {"type": "json_object"}}, lambda t: isinstance(json.loads(t), dict)),
    ("json_schema", "Invent a person.", {"response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": person}}},
     lambda t: (lambda d: set(d) == {"name", "age", "hobbies"} and isinstance(d["age"], int))(json.loads(t))),
    ("guided_json", "Invent a person who likes the sea.", {"guided_json": person}, lambda t: isinstance(json.loads(t)["age"], int)),
    ("regex", "Give me a US phone number.", {"guided_regex": r"\(\d{3}\) \d{3}-\d{4}"}, lambda t: re.fullmatch(r"\(\d{3}\) \d{3}-\d{4}", t)),
    ("choice", "Is the sky green? Answer.", {"guided_choice": ["yes", "no", "it depends"]}, lambda t: t in ("yes", "no", "it depends")),
    ("grammar", "Count to five.", {"guided_grammar": 'root ::= num (", " num)*\nnum ::= [1-9]'}, lambda t: re.fullmatch(r"[1-9](, [1-9])*", t)),
]
res, bad = {}, 0
for name, prompt, extra, ok in cases:
    for mode in ("greedy", "sampled", "thinking"):
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400 if mode == "thinking" else 160,
                "chat_template_kwargs": {"enable_thinking": mode == "thinking"}, **extra}
        body.update({"temperature": 0} if mode != "sampled" else {"temperature": 0.8, "top_k": 20, "top_p": 0.95, "seed": 3})
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
            c = d["choices"][0]
            text, fin = c["message"]["content"] or "", c["finish_reason"]
            try: good = bool(ok(text.strip())) if fin == "stop" else None
            except Exception: good = False
        except urllib.error.HTTPError as e:
            text, fin, good = f"HTTP {e.code}: {e.read()[:300]!r}", "error", False
        res[f"{name}/{mode}"] = {"text": text, "finish": fin, "ok": good}
        bad += good is False
        print(f"{name:12s} {mode:9s} {fin:7s} ok={good}  {text[:70]!r}")
# a grammar that cannot compile: a 400
body = {"model": "x", "messages": [{"role": "user", "content": "x"}], "guided_regex": "(unclosed", "max_tokens": 5}
try:
    urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=60); code = 200
except urllib.error.HTTPError as e:
    code = e.code; print("bad regex:", code, e.read()[:200])
res["bad_regex_status"] = code
json.dump(res, open(out, "w"), indent=1)
print("failures:", bad, "| bad regex ->", code)
