"""Drive a Glimmer server through agent-shaped long-context sessions.

Why: Glimmer on the 24 GB B60 leaves ~6 GB for KV + activations, the same
total that killed Qwen3.8 on request 34. This replays a realistic load --
the captured OpenCode system prompt plus ~90k chars of real code, sessions
that grow turn by turn, a cold prefix every new session -- and records
every request so a failure can be matched to the memory log.

In: --mode openai (NoLlama, /v1/chat/completions) or ollama (/api/chat,
num_ctx passed explicitly, since Ollama's /v1 ignores it and would silently
truncate). Out: one JSON line per request on stdout and in --log; stops
early if the server stops answering at all (connection refused).
"""
import argparse
import json
import time
import urllib.error
import urllib.request

QUESTIONS = [
    "Summarise what the functions in this excerpt do, one line each.",
    "Which function in this excerpt is most likely to have a bug? Explain.",
    "List every command-line flag referenced in this excerpt.",
    "What would break if the locking in this excerpt were removed?",
    "Write a short docstring for the longest function in this excerpt.",
]


def build_session(system, source, offset, chars):
    """Seed one session's messages: agent system prompt + a code excerpt.

    Why: a new offset per session means a new prefix, so each session's
    first turn is a cold ~40k-token prefill and later turns are cache hits,
    which is what an agent client actually does.

    In: offset wraps around the source. Out: [system, user] messages; the
    user turn carries the excerpt and the first question.
    """
    src = (source * 2)[offset % len(source):][:chars]
    user = ("Here is part of nollama.py for context:\n```python\n" + src
            + "\n```\n\n" + QUESTIONS[0])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def post(url, body, timeout):
    """POST JSON and return (parsed reply, seconds).

    Why: stdlib only, so the same file runs on both boxes with no installs.

    In: body dict. Out: raises urllib errors unchanged; the caller decides
    whether an HTTP error is a data point or the end of the run.
    """
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t


def ask(args, messages):
    """Send one chat turn in the server's own dialect.

    Why: token counts live in different fields per server; normalising here
    keeps the log comparable across NoLlama and Ollama.

    In: full message list. Out: (reply text, prompt_tokens, completion_tokens,
    seconds).
    """
    if args.mode == "openai":
        body = {"model": args.model, "messages": messages,
                "max_tokens": args.max_tokens, "temperature": 0.2}
        d, s = post(args.url + "/v1/chat/completions", body, args.timeout)
        u = d.get("usage", {})
        return (d["choices"][0]["message"].get("content") or "",
                u.get("prompt_tokens"), u.get("completion_tokens"), s)
    body = {"model": args.model, "messages": messages, "stream": False,
            "options": {"num_ctx": args.num_ctx, "num_predict": args.max_tokens,
                        "temperature": 0.2}}
    d, s = post(args.url + "/api/chat", body, args.timeout)
    return (d["message"].get("content") or "", d.get("prompt_eval_count"),
            d.get("eval_count"), s)


def main():
    """Run sessions x turns requests and log each one.

    Why: the Qwen3.8 failure needed 34 requests, so the default is 7 x 5.

    In: CLI flags. Out: JSONL log; exit after the last request or the first
    connection-level failure.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["openai", "ollama"], required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="glimmer")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--source-file", required=True)
    p.add_argument("--code-chars", type=int, default=90000)
    p.add_argument("--sessions", type=int, default=7)
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--num-ctx", type=int, default=65536)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--log", required=True)
    args = p.parse_args()

    system = json.load(open(args.prompt_file, encoding="utf-8"))[0]["content"]
    source = open(args.source_file, encoding="utf-8").read()
    n = 0
    with open(args.log, "a", encoding="utf-8") as log:
        for s in range(args.sessions):
            msgs = build_session(system, source, s * 37000, args.code_chars)
            for t in range(args.turns):
                n += 1
                rec = {"n": n, "session": s, "turn": t, "ts": time.strftime("%H:%M:%S")}
                try:
                    text, pt, ct, secs = ask(args, msgs)
                    rec.update(ok=True, prompt_tokens=pt, completion_tokens=ct,
                               seconds=round(secs, 1), reply=text[:80])
                    msgs += [{"role": "assistant", "content": text},
                             {"role": "user", "content": QUESTIONS[(t + 1) % len(QUESTIONS)]}]
                except urllib.error.HTTPError as e:
                    rec.update(ok=False, error=f"HTTP {e.code}: {e.read()[:600]!r}")
                except Exception as e:  # connection-level: server is gone
                    rec.update(ok=False, error=repr(e)[:600], fatal=True)
                line = json.dumps(rec, ensure_ascii=False)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                if rec.get("fatal"):
                    return


if __name__ == "__main__":
    main()
