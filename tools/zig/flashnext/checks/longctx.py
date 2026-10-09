#!/usr/bin/env python3
"""Long prompts against a server: time to first token, prefill and decode tok/s, a needle's answer."""
# usage: longctx.py <port|url> <out.json> [target tokens ...]
import json, random, sys, time, urllib.request
BASE = sys.argv[1].rstrip("/") if "://" in sys.argv[1] else f"http://127.0.0.1:{sys.argv[1]}"   # a port or a URL
out = sys.argv[2]
targets = [int(x) for x in sys.argv[3:]] or [32000, 128000, 225000]
rng = random.Random(7)
words = "river stone cloud lantern harbor meadow copper violet signal garden orbit candle marble forest thunder".split()
res = []
for tgt in targets:
    n = int(tgt / 1.33)
    body_words = [rng.choice(words) for _ in range(n)]
    code = f"{rng.randint(100000, 999999)}"
    body_words.insert(n // 3, f". The secret code is {code}. ")
    text = " ".join(body_words) + "\n\nWhat is the secret code mentioned above? Then write a long story about a lighthouse keeper."
    req = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 256, "temperature": 0, "stream": True,
           "ignore_eos": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; last = None; pieces = []; usage = {}
    with urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(req).encode(),
                                                       {"Content-Type": "application/json"}), timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                t = (c.get("delta") or {}).get("content")
                if t:
                    now = time.time(); first = first or now; last = now; pieces.append(t)
    reply = "".join(pieces)
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", len(pieces))
    row = {"target": tgt, "prompt_tokens": pt, "ttft_s": first - t0, "prefill_tok_s": pt / (first - t0),
           "decode_tok_s": (ct - 1) / (last - first) if ct > 1 and last > first else None, "completion_tokens": ct,
           "needle_found": code in reply[:200], "reply_head": reply[:120]}
    print(json.dumps(row), flush=True)
    res.append(row)
json.dump(res, open(out, "w"), indent=1)
