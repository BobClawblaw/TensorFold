#!/usr/bin/env python3
"""Cache exhaustion: concurrent long replies must each end cleanly and the server must keep serving."""
# usage: exhaust_check.py <port|url> <out.json> [waves] [streams] [prompt words] [max_tokens] [timeout s]
import json, sys, threading, time, urllib.request, urllib.error
BASE = sys.argv[1].rstrip("/") if "://" in sys.argv[1] else f"http://127.0.0.1:{sys.argv[1]}"   # a port or a URL
out = sys.argv[2]
waves = int(sys.argv[3]) if len(sys.argv) > 3 else 2
streams = int(sys.argv[4]) if len(sys.argv) > 4 else 6
words = int(sys.argv[5]) if len(sys.argv) > 5 else 5600
max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 1200
TIMEOUT = int(sys.argv[7]) if len(sys.argv) > 7 else 900
URL = f"{BASE}/v1/chat/completions"

def ask(content, n, timeout=900):
    body = {"model": "x", "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0}
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout)
        d = json.loads(r.read())
        return {"status": 200, "finish": d["choices"][0]["finish_reason"], "tokens": d["usage"]["completion_tokens"],
                "prompt": d["usage"]["prompt_tokens"], "s": round(time.time() - t0, 1)}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode()[:300], "s": round(time.time() - t0, 1)}
    except Exception as e:  # a timeout or a dropped connection: the failure this check exists for
        return {"status": -1, "error": repr(e)[:300], "s": round(time.time() - t0, 1)}

def long_prompt(i, w):
    filler = " ".join(f"item{i}-{k}" for k in range(w))
    return (f"Here is list {i}: {filler}\n\nIgnore the list. Write the numbers from 1 to 4000 in words, one per line, "
            f"without stopping or commenting.")

res = {"waves": [], "after": []}
ok = True
for wv in range(waves):
    got = [None] * streams
    def go(i):
        got[i] = ask(long_prompt(wv * 100 + i, words), max_tokens, TIMEOUT)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(streams)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    replies = sum(g["status"] == 200 for g in got)
    refused = sum(g["status"] != 200 and "no memory left" in g.get("error", "") for g in got)
    other = streams - replies - refused
    after = ask("Say hello in five words.", 32, timeout=min(300, TIMEOUT))
    health = None
    try:
        health = json.loads(urllib.request.urlopen(f"{BASE}/health", timeout=30).read()).get("status")
    except Exception as e:
        health = repr(e)[:100]
    wave_ok = other == 0 and after["status"] == 200 and health == "ok"
    ok = ok and wave_ok
    res["waves"].append({"requests": got, "replies": replies, "refused": refused, "other": other, "s": round(time.time() - t0, 1),
                         "after": after, "health": health, "ok": wave_ok})
    print(f"wave {wv}: {replies} replies, {refused} refused in the server's words, {other} other; "
          f"after: {after['status']} {after.get('finish', after.get('error', ''))[:60]}; health {health}; "
          f"{'OK' if wave_ok else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    for g in got:
        print("   ", json.dumps(g)[:200], flush=True)
res["ok"] = ok
print("EXHAUST", "PASS" if ok else "FAIL")
json.dump(res, open(out, "w"), indent=1)
