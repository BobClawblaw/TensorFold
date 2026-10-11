#!/usr/bin/env python3
"""N distinct long prompts at once: how many the server holds together."""
# usage: capacity.py <port|url> <n> <tokens> <out.json>
import json, random, sys, time, urllib.request, concurrent.futures as cf
BASE = sys.argv[1].rstrip("/") if "://" in sys.argv[1] else f"http://127.0.0.1:{sys.argv[1]}"   # a port or a URL
n, tgt, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
words = "river stone cloud lantern harbor meadow copper violet signal garden orbit candle marble forest thunder".split()
def one(i):
    rng = random.Random(1000 + i)
    text = f"Document {i}. " + " ".join(rng.choice(words) for _ in range(int(tgt / 1.33))) + "\n\nSummarize in one sentence."
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 32, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=3600))
        return {"i": i, "ok": True, "s": time.time() - t0, "prompt_tokens": d["usage"]["prompt_tokens"], "finish": d["choices"][0]["finish_reason"]}
    except urllib.error.HTTPError as e:
        return {"i": i, "ok": False, "s": time.time() - t0, "error": f"HTTP {e.code}: {e.read()[:300]!r}"}
    except Exception as e:
        return {"i": i, "ok": False, "s": time.time() - t0, "error": repr(e)[:300]}
with cf.ThreadPoolExecutor(n) as ex:
    res = list(ex.map(one, range(n)))
json.dump(res, open(out, "w"), indent=1)
print(f"{sum(r['ok'] for r in res)}/{n} answered;", [r.get('error', r.get('finish')) for r in res])
