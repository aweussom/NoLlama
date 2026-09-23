#!/usr/bin/env python3
"""Tier 1 model check — one SVG, one request, comparable with the whole world.

Asks a served model for "an SVG of a pelican riding a bicycle" and saves what
comes back. It is Simon Willison's test, and the reason to use his prompt
verbatim rather than a better one of our own is the corpus: hundreds of
published results across frontier and local models, so a NoLlama result lands
on a scale people already read.

What it is for: catching a model that is simply weak, cheaply, across the whole
registry. Seconds per model, no agent, no fixture, no tools.

What it CANNOT do, and must never be used for: deciding whether a model can
drive an agent loop. Qwen2.5-Coder-14B draws a perfectly reasonable picture and
cannot call a tool [OBSERVED 2026-09-23]; a model is promoted to `agent` in
models.json only by `agent-probe.ps1`, never by this.

    venv\\Scripts\\python scripts\\pelican-probe.py                     # localhost:8000
    venv\\Scripts\\python scripts\\pelican-probe.py --url http://100.81.4.88:8000/v1
    venv\\Scripts\\python scripts\\pelican-probe.py --model Qwen3-14B@GPU --out bench/pelicans

Out: the raw SVG, an HTML wrapper to open it, and one line of timings. Judging
it is a human's job — this reports, it does not score. Exits 0 unless the
request itself failed, so a bad drawing is data rather than a broken run.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "Generate an SVG of a pelican riding a bicycle"


def fetch_models(base):
    """The model ids a server is currently serving.

    Why: the probe is usually pointed at a box rather than a model, and the id
    carries the device suffix (`Qwen3-14B@GPU`), which nobody wants to type.

    In: an OpenAI-compatible base URL. Out: a list of ids, empty if the server
    answers but lists nothing. Raises on connection failure — a probe that
    cannot reach the server has nothing to report.
    """
    with urllib.request.urlopen(f"{base}/models", timeout=30) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def ask(base, model, timeout):
    """One non-streaming completion, timed.

    In: base URL, model id, timeout in seconds. Out: (text, wall_seconds,
    usage dict). A server that streams by default is not used here on purpose:
    the number that matters is total wall clock to a finished drawing.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 4096,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d["choices"][0]["message"]["content"], time.time() - t0,
            d.get("usage") or {})


def extract_svg(text):
    """Pull the <svg> element out of a chat answer.

    Why: models wrap it in prose, in ```svg fences, or emit it bare, and a file
    that is half English does not open in a browser.

    In: the raw answer. Out: the SVG source, or None when the answer contains
    no <svg> element at all — which is itself a result worth printing.
    """
    m = re.search(r"<svg\b.*?</svg>", text, re.S | re.I)
    return m.group(0) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible base URL (default localhost:8000)")
    ap.add_argument("--model", default=None, help="model id; default: every model served")
    ap.add_argument("--out", default="bench/pelicans", help="where to write the SVGs")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    base = args.url.rstrip("/")
    try:
        models = [args.model] if args.model else fetch_models(base)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"ERROR: no server at {base}: {e}", file=sys.stderr)
        return 1
    if not models:
        print(f"ERROR: {base} serves no models", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    print(f"pelican probe — {base}")
    for model in models:
        try:
            text, wall, usage = ask(base, model, args.timeout)
        except Exception as e:                      # noqa: BLE001 — report, don't crash
            print(f"  {model:<44} FAILED  {type(e).__name__}: {e}")
            continue
        svg = extract_svg(text)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        # NoLlama answers -1 when the runtime did not report usage, and a
        # negative token count printed as a rate reads like a broken server.
        tok = usage.get("completion_tokens") or 0
        rate = f", {tok / wall:.1f} tok/s" if tok > 0 and wall > 0 else ""
        if svg is None:
            (out / f"{stamp}-{safe}.txt").write_text(text, encoding="utf-8")
            print(f"  {model:<44} NO SVG  {wall:6.1f}s{rate}  (answer saved as .txt)")
            continue
        svg_path = out / f"{stamp}-{safe}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        (out / f"{stamp}-{safe}.html").write_text(
            f"<!doctype html><meta charset=utf-8><title>{model}</title>"
            f"<h3 style='font:14px sans-serif'>{model} — {wall:.1f}s</h3>{svg}",
            encoding="utf-8")
        print(f"  {model:<44} ok      {wall:6.1f}s{rate}  {len(svg):>6} B  {svg_path}")

    print("\nOpen the .html files and look. This probe does not score — and it "
          "cannot tell you whether a model can drive an agent loop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
