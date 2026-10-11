#!/usr/bin/env python3
"""Multi-turn prompt reuse: replies, prompt and cached tokens, time to first token; resumed equals fresh."""
# usage: turns_check.py <port|url> <out.json>
import json, sys, time, urllib.request
BASE = sys.argv[1].rstrip("/") if "://" in sys.argv[1] else f"http://127.0.0.1:{sys.argv[1]}"   # a port or a URL
out = sys.argv[2]
system = "You are a careful assistant. " + " ".join(f"Rule {i}: answer plainly and keep facts straight about topic {i}." for i in range(300))
def ask(messages, n=48):
    body = {"model": "x", "messages": messages, "max_tokens": n, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0, first, text, usage = time.time(), None, "", None
    r = urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices", []):
            piece = (c.get("delta") or {}).get("content")
            if piece:
                first = first or time.time() - t0
                text += piece
    return {"text": text, "ttft": round(first or 0, 3), "prompt": (usage or {}).get("prompt_tokens"),
            "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}
res = {}
conv = [{"role": "system", "content": system}, {"role": "user", "content": "Name three rivers in Europe."}]
a1 = ask(conv); res["turn1"] = a1
conv2 = conv + [{"role": "assistant", "content": a1["text"]}, {"role": "user", "content": "Which of them is longest?"}]
res["turn2_resumed"] = ask(conv2)
conv3 = conv2 + [{"role": "assistant", "content": res["turn2_resumed"]["text"]}, {"role": "user", "content": "And the shortest?"}]
res["turn3_resumed"] = ask(conv3)
# the same turns with a different system prompt variant sent cold, then the identical turn 3 again
res["turn3_again"] = ask(conv3)
cold = [{"role": "system", "content": system + " "}] + conv3[1:]
res["turn3_cold_variant"] = ask(cold)
for k, v in res.items():
    print(f"{k:20s} prompt={v['prompt']} cached={v['cached']} ttft={v['ttft']}s  {v['text'][:60]!r}")
print("turn3 resumed == again:", res["turn3_resumed"]["text"] == res["turn3_again"]["text"])
json.dump(res, open(out, "w"), indent=1)
