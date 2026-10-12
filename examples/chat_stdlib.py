"""One chat turn with nothing but the standard library. No installs.

    python chat_stdlib.py [--base-url http://localhost:8000]

For scripts on a box where pip is not an option. The request and reply are
plain OpenAI JSON; `reasoning_content` holds a thinking model's reasoning,
`content` the answer.
"""
import argparse
import json
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://localhost:8000")
args = p.parse_args()

body = {"messages": [{"role": "user", "content": "What is the capital of Norway? One word."}]}
req = urllib.request.Request(
    args.base_url + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    reply = json.load(r)
    device = r.headers.get("X-Device")

print(reply["choices"][0]["message"]["content"].strip(), f"(answered on {device})")
