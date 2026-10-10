#!/usr/bin/env python3
"""Cache exhaustion under concurrent long replies (Flash Next native, two ranks): waves of concurrent requests whose
caches outgrow the server's growth budget, reduced on one rank with TENSORFOLD_GROWTH_LIMIT_MIB (e.g. 1100 at int8:
six ~7k-token prompts' first steps and two more steps; on rank 0 or rank 1 alone, the refusals each rank makes). Each request must end cleanly: a reply, or a refusal in the server's words (no memory left);
the server must keep serving (a short request after each wave, and health). usage: exhaust_check.py <port> <out.json>
[waves] [streams] [prompt words] [max_tokens]"""
import hashlib, json, sys, threading, time, urllib.request, urllib.error
port, out = int(sys.argv[1]), sys.argv[2]
waves = int(sys.argv[3]) if len(sys.argv) > 3 else 2
streams = int(sys.argv[4]) if len(sys.argv) > 4 else 6
words = int(sys.argv[5]) if len(sys.argv) > 5 else 5600
max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 1200
TIMEOUT = int(sys.argv[7]) if len(sys.argv) > 7 else 900
URL = f"http://127.0.0.1:{port}/v1/chat/completions"

def ask(content, n, timeout=900):
    body = {"model": "x", "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0}
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout)
        d = json.loads(r.read())
        text = d["choices"][0]["message"]["content"] or ""
        return {"status": 200, "finish": d["choices"][0]["finish_reason"], "tokens": d["usage"]["completion_tokens"],
                "prompt": d["usage"]["prompt_tokens"], "sha": hashlib.sha256(text.encode()).hexdigest()[:12],
                "s": round(time.time() - t0, 1)}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode()[:300], "s": round(time.time() - t0, 1)}
    except Exception as e:  # a timeout or a dropped connection: the failure this check exists for
        return {"status": -1, "error": repr(e)[:300], "s": round(time.time() - t0, 1)}

def long_prompt(i, w):
    filler = " ".join(f"item{i:03d}-{k}" for k in range(w))
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
        health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30).read()).get("status")
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
