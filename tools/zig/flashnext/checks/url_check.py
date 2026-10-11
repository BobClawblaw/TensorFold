#!/usr/bin/env python3
"""Image URLs in a chat request, each reply or refusal: a public https image, a loopback address, a plain-http URL."""
# usage: url_check.py <port|url>
import json, sys, urllib.request
BASE = sys.argv[1].rstrip("/") if "://" in sys.argv[1] else f"http://127.0.0.1:{sys.argv[1]}"   # a port or a URL
for url in ["https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png", "https://127.0.0.1/x.png", "http://example.com/x.png"]:
    body = {"model": "x", "max_tokens": 40, "temperature": 0, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "What is in this image? One sentence."}]}]}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
        print("OK ", url[:60], "->", repr(d["choices"][0]["message"]["content"][:100]))
    except urllib.error.HTTPError as e:
        print(e.code, url[:60], "->", e.read()[:160])
