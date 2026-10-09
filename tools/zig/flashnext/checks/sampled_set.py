#!/usr/bin/env python3
"""Seeded sampled replies from a server (prompts x sampling rules), for comparing engines."""
# usage: sampled_set.py <port|url> <out.json> [draft on|off]
import json, sys, urllib.request
BASE = sys.argv[1].rstrip("/") if "://" in sys.argv[1] else f"http://127.0.0.1:{sys.argv[1]}"   # a port or a URL
out = sys.argv[2]
draft = (sys.argv[3] if len(sys.argv) > 3 else "on") == "on"
prompts = ["Write a short poem about the sea.", "Give three creative names for a coffee shop.",
           "Explain recursion to a child in two sentences.", "Invent a new board game and describe its rules briefly."]
rules = [{"temperature": 0.8, "top_k": 20, "top_p": 0.95, "seed": 1}, {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 2},
         {"temperature": 0.7, "top_k": 5, "seed": 3}, {"temperature": 1.0, "top_k": 20, "top_p": 0.8, "min_p": 0.05, "seed": 4},
         {"seed": 5}]   # the last: the server's defaults (generation_config)
res = {}
for i, p in enumerate(prompts):
    for j, r in enumerate(rules):
        body = {"model": "Qwen3.8-Flash-Next-INT4-Mixed", "messages": [{"role": "user", "content": p}], "max_tokens": 96,
                "chat_template_kwargs": {"enable_thinking": False}, **r}
        if not draft:
            body["draft"] = False
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions",
                                                                        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
            res[f"{i}/{j}"] = d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            res[f"{i}/{j}"] = f"HTTP {e.code}: {e.read()[:200]!r}"
json.dump(res, open(out, "w"), indent=1)
print(len(res), "replies;", sum(1 for v in res.values() if v.startswith("HTTP")), "errors")
