#!/usr/bin/env python3
"""NoLlama benchmark — compare workloads across devices and backends.

Usage:
    # NoLlama server (auto-detects devices/models from /health):
    python benchmark.py --label npu              # tag this run for later compare
    python benchmark.py --url http://host:9000

    # Ollama backend (e.g. RTX 5090):
    python benchmark.py --backend ollama --model qwen3:8b --label rtx5090

    # Cross-stack workflow (same model on each device):
    #   1. ollama:    benchmark.py --backend ollama --model qwen3:8b --label rtx5090
    #   2. NPU:       (start nollama --device NPU)  benchmark.py --label npu
    #   3. iGPU:      (start nollama --device GPU)  benchmark.py --label igpu
    #   4. CPU:       (start nollama --device CPU)  benchmark.py --label cpu
    # Results land in bench-results/<label>-<model>-<timestamp>.json

Each test: 1 warmup + N runs. Reports avg elapsed, TTFT (prefill latency),
overall tok/s, and decode-only tok/s (excluding prefill). Discards IQR outliers.
"""

import argparse
import base64
import json
import os
import statistics
import sys
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# VLM tests use 4 image paths: two pairs (same-vehicle, different-vehicle).
# Pass them via --same-1/--same-2/--diff-1/--diff-2 or set NOLLAMA_IMAGES_DIR
# and put matching files named same-1.jpg, same-2.jpg, diff-1.jpg, diff-2.jpg.
DEFAULT_IMAGES_DIR = os.environ.get("NOLLAMA_IMAGES_DIR", "")

NO_THINK_SYSTEM = "Respond directly and concisely. Do not use <think> blocks or internal reasoning."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_to_b64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def stream_request(url, body):
    """Send a streaming chat request. Returns (text, tokens, elapsed, ttft).

    Counts both `delta.content` and `delta.reasoning` (Ollama's thinking
    channel) as tokens. NoLlama doesn't separate <think> from content, so
    summing both gives apples-to-apples decode throughput across backends.

    ttft = time-to-first-token (prefill latency); first of either channel.
    `tokens` is the total (content + reasoning).
    `text` is content only — `body["_capture_reasoning"] = True` includes both.
    """
    import urllib.request

    capture_reasoning = body.pop("_capture_reasoning", False) if isinstance(body, dict) else False

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft = None
    content_tokens = 0
    reasoning_tokens = 0
    content_text = ""
    reasoning_text = ""

    with urllib.request.urlopen(req, timeout=600) as resp:
        while True:
            # readline, not read(4096): HTTPResponse.read(n) blocks until
            # it has n bytes, so on a stream it waited for ~4 KB of SSE
            # frames - 20+ tokens - before returning anything, and TTFT
            # included generating them (0.50s vs 0.20s for the true first
            # token). read1() returns early but does NOT reliably
            # terminate a chunked response: it blocked forever after the
            # final frame, hanging the whole benchmark. SSE is
            # line-oriented, so readline is both prompt and correct.
            raw = resp.readline()
            if not raw:
                break
            lines = [raw.decode("utf-8", errors="replace").rstrip()]
            for line in lines:
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        content_text += content
                        content_tokens += 1
                    if reasoning:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        reasoning_text += reasoning
                        reasoning_tokens += 1
                except json.JSONDecodeError:
                    pass

    elapsed = time.perf_counter() - t0
    if ttft is None:
        ttft = elapsed
    total_tokens = content_tokens + reasoning_tokens
    full_text = (f"<think>{reasoning_text}</think>\n{content_text}"
                 if capture_reasoning and reasoning_text else content_text)
    return full_text, total_tokens, elapsed, ttft


def non_stream_request(url, body):
    """Send a non-streaming chat request. Returns (text, tokens, elapsed, ttft)."""
    import urllib.request

    body["stream"] = False
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode())

    elapsed = time.perf_counter() - t0
    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    # Prefer real token count from usage.completion_tokens if present
    usage = result.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    tokens = completion_tokens if isinstance(completion_tokens, int) and completion_tokens > 0 \
        else len(text.split())
    return text, tokens, elapsed, elapsed


def get_models(url):
    """Fetch available models from the server."""
    import urllib.request
    req = urllib.request.Request(url + "/v1/models")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return data.get("data", [])


def check_health(url):
    """Check server health."""
    import urllib.request
    req = urllib.request.Request(url + "/health")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def make_vlm_test(name, img1_path, img2_path, prompt):
    """VLM test: two images + question. Non-streaming (VLM limitation)."""
    def run(url, model):
        b64_1 = image_to_b64(img1_path)
        b64_2 = image_to_b64(img2_path)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_1}},
                {"type": "image_url", "image_url": {"url": b64_2}},
            ]}],
            "max_tokens": 512,
        }
        return non_stream_request(url, body)
    return name, run


def make_llm_test(name, prompt, no_think=False, max_tokens=4096, force_no_stream=False):
    """LLM test: text prompt, streaming unless force_no_stream."""
    def run(url, model):
        messages = []
        if no_think:
            messages.append({"role": "system", "content": NO_THINK_SYSTEM})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
            "stream": not force_no_stream,
            "max_tokens": max_tokens,
            # Send temperature explicitly. Omitting it lets each server apply
            # its OWN default, which is not the same default: NoLlama treats a
            # missing temperature as 0.0 (greedy), Ollama uses 0.8 (sampling).
            # Measured on Ollama qwen3:8b — without this, two identical
            # requests returned 1871 and 1052 completion tokens; with it, 1755
            # both times. Comparing a sampled backend against a greedy one and
            # averaging the result is not a comparison.
            "temperature": 0,
        }
        if force_no_stream:
            return non_stream_request(url, body)
        return stream_request(url, body)
    return name, run


def build_vlm_tests(same_1, same_2, diff_1, diff_2):
    return [
        make_vlm_test(
            "VLM: same vehicle?",
            same_1, same_2,
            "Are these two images of the same vehicle? Answer yes or no, then explain briefly.",
        ),
        make_vlm_test(
            "VLM: different vehicles?",
            diff_1, diff_2,
            "Are these two images of the same vehicle? Answer yes or no, then explain briefly.",
        ),
    ]

TESTS_GPU_TEXT = [
    make_llm_test(
        "GPU text: say hello (thinking)",
        "Say hello.",
        max_tokens=2048,
        force_no_stream=True,
    ),
    make_llm_test(
        "GPU text: say hello (no-think)",
        "Say hello.",
        no_think=True,
        max_tokens=2048,
        force_no_stream=True,
    ),
    make_llm_test(
        "GPU text: what is 2+2 (thinking)",
        "What is 2+2?",
        max_tokens=2048,
        force_no_stream=True,
    ),
    make_llm_test(
        "GPU text: what is 2+2 (no-think)",
        "What is 2+2?",
        no_think=True,
        max_tokens=2048,
        force_no_stream=True,
    ),
]

# Cross-stack tip: Qwen3 only honors `/no_think` literally in the user prompt,
# not via system message. The no_think system prompt below is best-effort and
# may still trigger thinking on some Qwen3 builds — that's fine, we capture
# both content and reasoning channels and count them together for tok/s.
TESTS_LLM = [
    make_llm_test(
        "LLM: say hello (thinking)",
        "Say hello.",
        max_tokens=2048,
    ),
    make_llm_test(
        "LLM: say hello (no-think)",
        "Say hello. /no_think",
        no_think=True,
        max_tokens=2048,
    ),
    make_llm_test(
        "LLM: what is 2+2 (thinking)",
        "What is 2+2?",
        max_tokens=2048,
    ),
    make_llm_test(
        "LLM: what is 2+2 (no-think)",
        "What is 2+2? /no_think",
        no_think=True,
        max_tokens=2048,
    ),
    # Decode-bound test: predictable long output, exercises steady-state tok/s.
    make_llm_test(
        "LLM: count 1-100 (no-think)",
        "List the integers from 1 to 100, separated by commas, on a single line. /no_think",
        no_think=True,
        max_tokens=4096,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test(url, model, test_name, test_fn, n_runs, warmup=True):
    """Run a test N times with optional warmup.

    Returns list of dicts with keys: tokens, elapsed, ttft, tps, decode_tps.
    decode_tps excludes prefill: tokens / (elapsed - ttft).
    """
    results = []

    if warmup:
        print(f"    warmup...", end="", flush=True)
        try:
            _, _, w_elapsed, _ = test_fn(url, model)
            print(f" {w_elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f" FAILED: {e}")
            return []

    for i in range(n_runs):
        print(f"    run {i+1}/{n_runs}...", end="", flush=True)
        try:
            text, tokens, elapsed, ttft = test_fn(url, model)
            tps = tokens / elapsed if elapsed > 0 else 0
            decode_time = elapsed - ttft
            decode_tps = tokens / decode_time if decode_time > 0.01 else tps
            results.append({
                "tokens": tokens, "elapsed": elapsed, "ttft": ttft,
                "tps": tps, "decode_tps": decode_tps,
            })
            print(f" {tokens} tok, {elapsed:.1f}s, ttft={ttft:.2f}s, "
                  f"{tps:.1f} tok/s ({decode_tps:.1f} decode)", flush=True)
        except Exception as e:
            print(f" FAILED: {e}")

    return results


def discard_outliers(values):
    """Remove values more than 1.5 IQR from Q1/Q3."""
    if len(values) < 4:
        return values
    q1 = statistics.quantiles(values, n=4)[0]
    q3 = statistics.quantiles(values, n=4)[2]
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    filtered = [v for v in values if lo <= v <= hi]
    return filtered if filtered else values


def summarize(results):
    """Compute averages after discarding outliers."""
    if not results:
        return None
    tok_vals = [r["tokens"] for r in results]
    elapsed_vals = [r["elapsed"] for r in results]
    ttft_vals = [r["ttft"] for r in results]
    tps_vals = [r["tps"] for r in results]
    decode_tps_vals = [r["decode_tps"] for r in results]

    elapsed_clean = discard_outliers(elapsed_vals)
    tps_clean = discard_outliers(tps_vals)
    decode_tps_clean = discard_outliers(decode_tps_vals)
    ttft_clean = discard_outliers(ttft_vals)

    return {
        "runs": len(results),
        "runs_after_outliers": len(elapsed_clean),
        "avg_tokens": statistics.mean(tok_vals),
        "avg_elapsed": statistics.mean(elapsed_clean),
        "avg_ttft": statistics.mean(ttft_clean),
        "avg_tps": statistics.mean(tps_clean),
        "avg_decode_tps": statistics.mean(decode_tps_clean),
        "stdev_elapsed": statistics.stdev(elapsed_clean) if len(elapsed_clean) > 1 else 0,
        "stdev_tps": statistics.stdev(tps_clean) if len(tps_clean) > 1 else 0,
        "raw_runs": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["nollama", "ollama"], default="nollama",
                        help="Server type. nollama (default) auto-detects via /health; "
                             "ollama uses --model directly and skips health check.")
    parser.add_argument("--url", default=None,
                        help="Server URL. Default: http://localhost:8000 for nollama, "
                             "http://localhost:11434 for ollama.")
    parser.add_argument("--model", default=None,
                        help="Model name to test (required for --backend ollama, "
                             "optional override for nollama).")
    parser.add_argument("--label", default=None,
                        help="Tag for output JSON (e.g. 'rtx5090', 'npu', 'igpu', 'cpu'). "
                             "Defaults to backend name.")
    parser.add_argument("--output-dir", default="bench-results",
                        help="Directory for JSON results (default: bench-results)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to JSON")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of test runs per workload (default: 5)")
    parser.add_argument("--llm-only", action="store_true",
                        help="Skip VLM tests")
    parser.add_argument("--vlm-only", action="store_true",
                        help="Skip LLM tests")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR,
                        help="Directory with same-1/2.jpg, diff-1/2.jpg "
                             "(or set NOLLAMA_IMAGES_DIR)")
    parser.add_argument("--same-1", help="Path to 'same vehicle' image A")
    parser.add_argument("--same-2", help="Path to 'same vehicle' image B")
    parser.add_argument("--diff-1", help="Path to 'different vehicle' image A")
    parser.add_argument("--diff-2", help="Path to 'different vehicle' image B")
    args = parser.parse_args()

    # Default URL per backend
    if args.url is None:
        args.url = "http://localhost:11434" if args.backend == "ollama" \
            else "http://localhost:8000"
    if args.label is None:
        args.label = args.backend

    if args.backend == "ollama":
        if not args.model:
            print("ERROR: --model required for --backend ollama (e.g. --model qwen3:8b)")
            sys.exit(1)
        if args.vlm_only:
            print("ERROR: --vlm-only not supported for ollama backend")
            sys.exit(1)
        # Ollama is text-only here — VLM tests are NoLlama-specific
        args.llm_only = True

    # Resolve image paths (explicit flags win over --images-dir)
    def resolve(explicit, default_name):
        if explicit:
            return explicit
        if args.images_dir:
            return os.path.join(args.images_dir, default_name)
        return None

    same_1 = resolve(args.same_1, "same-1.jpg")
    same_2 = resolve(args.same_2, "same-2.jpg")
    diff_1 = resolve(args.diff_1, "diff-1.jpg")
    diff_2 = resolve(args.diff_2, "diff-2.jpg")

    # Only require images if VLM tests will actually run (--llm-only skips them)
    vlm_paths = (same_1, same_2, diff_1, diff_2)
    vlm_paths_ok = all(p and os.path.isfile(p) for p in vlm_paths)
    if not args.llm_only and not vlm_paths_ok:
        missing = [p or "(unset)" for p in vlm_paths if not p or not os.path.isfile(p)]
        print("ERROR: VLM images missing or unset:")
        for m in missing:
            print(f"  {m}")
        print("\nPass --images-dir, --same-1/--same-2/--diff-1/--diff-2, "
              "or set NOLLAMA_IMAGES_DIR.")
        print("Or use --llm-only to skip vision tests.")
        sys.exit(1)

    TESTS_VLM = build_vlm_tests(same_1, same_2, diff_1, diff_2) if vlm_paths_ok else []

    print(f"Connecting to {args.url} ({args.backend})...")

    vlm_model = None
    llm_model = None

    if args.backend == "ollama":
        # Sanity-check: hit /v1/models, confirm requested model exists
        try:
            models = get_models(args.url)
        except Exception as e:
            print(f"ERROR: Cannot reach Ollama at {args.url}: {e}")
            sys.exit(1)
        ids = [m.get("id") for m in models]
        if args.model not in ids:
            print(f"ERROR: model '{args.model}' not found on server. Available:")
            for i in ids:
                print(f"  {i}")
            sys.exit(1)
        llm_model = args.model
        print(f"Ollama ready. Testing model: {llm_model}\n")
    else:
        # NoLlama: use /health to discover devices and types
        try:
            health = check_health(args.url)
        except Exception as e:
            print(f"ERROR: Cannot reach server: {e}")
            sys.exit(1)

        if health["status"] != "ready":
            print(f"ERROR: Server not ready (status: {health['status']})")
            sys.exit(1)

        models = get_models(args.url)
        print(f"Server ready. Models:")
        for m in models:
            print(f"  {m['id']}")
        print()

        devices = health.get("devices", {})
        for dev_name, dev_info in devices.items():
            if dev_info.get("status") != "ready":
                continue
            model_id = f"{dev_info['model']}@{dev_name.upper()}"
            if dev_info.get("type") == "vlm":
                vlm_model = model_id
            elif dev_info.get("type") == "llm":
                llm_model = model_id

        # Allow --model to override discovered LLM model
        if args.model:
            llm_model = args.model

    all_results = {}

    # --- VLM tests ---
    if vlm_model and not args.llm_only:
        print(f"=== VLM Benchmark: {vlm_model} ===\n")
        for test_name, test_fn in TESTS_VLM:
            print(f"  [{test_name}]")
            results = run_test(args.url, vlm_model, test_name, test_fn, args.runs)
            summary = summarize(results)
            all_results[test_name] = summary
            if summary:
                print(f"    => avg: {summary['avg_elapsed']:.1f}s "
                      f"({summary['avg_tps']:.1f} tok/s), "
                      f"{summary['avg_tokens']:.0f} tokens\n")
            else:
                print(f"    => NO RESULTS\n")
    elif not args.llm_only:
        print("No VLM model found, skipping vision tests.\n")

    # --- GPU text tests (text-only queries to VLM, non-streaming) ---
    if vlm_model and not args.llm_only:
        print(f"=== GPU Text Benchmark: {vlm_model} ===\n")
        for test_name, test_fn in TESTS_GPU_TEXT:
            print(f"  [{test_name}]")
            results = run_test(args.url, vlm_model, test_name, test_fn, args.runs)
            summary = summarize(results)
            all_results[test_name] = summary
            if summary:
                print(f"    => avg: {summary['avg_elapsed']:.1f}s "
                      f"({summary['avg_tps']:.1f} tok/s), "
                      f"{summary['avg_tokens']:.0f} tokens\n")
            else:
                print(f"    => NO RESULTS\n")

    # --- LLM tests ---
    if llm_model and not args.vlm_only:
        print(f"=== LLM Benchmark: {llm_model} ===\n")
        for test_name, test_fn in TESTS_LLM:
            print(f"  [{test_name}]")
            results = run_test(args.url, llm_model, test_name, test_fn, args.runs)
            summary = summarize(results)
            all_results[test_name] = summary
            if summary:
                print(f"    => avg: {summary['avg_elapsed']:.1f}s "
                      f"({summary['avg_tps']:.1f} tok/s), "
                      f"{summary['avg_tokens']:.0f} tokens\n")
            else:
                print(f"    => NO RESULTS\n")
    elif not args.vlm_only:
        # Text tests may already have run on the VLM slot above (the GPU
        # text block). Saying "skipping text tests" there is false, and it
        # was read as doubt about the numbers themselves — NoLlama#24.
        if vlm_model and not args.llm_only:
            print("No dedicated LLM slot — the text tests above ran on "
                  "the VLM slot.\n")
        else:
            print("No LLM model found, skipping text tests.\n")

    # --- Summary table ---
    print("=" * 88)
    print(f"{'Test':<45} {'Time':>7} {'TTFT':>6} {'Tok/s':>7} {'Decode':>7} {'Tokens':>7}")
    print("-" * 88)
    for name, s in all_results.items():
        if s:
            print(f"{name:<45} {s['avg_elapsed']:>6.1f}s {s['avg_ttft']:>5.2f}s "
                  f"{s['avg_tps']:>6.1f} {s['avg_decode_tps']:>6.1f} {s['avg_tokens']:>7.0f}")
        else:
            print(f"{name:<45} {'FAILED':>7}")
    print("=" * 88)

    # --- Save raw results to JSON ---
    if not args.no_save and all_results:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        # Sanitize model name for filename: qwen3:8b -> qwen3-8b
        model_for_file = (llm_model or vlm_model or "unknown").replace(":", "-").replace("/", "-").replace("@", "-at-")
        out_path = os.path.join(
            args.output_dir,
            f"{args.label}-{model_for_file}-{ts}.json",
        )
        payload = {
            "label": args.label,
            "backend": args.backend,
            "url": args.url,
            "llm_model": llm_model,
            "vlm_model": vlm_model,
            "runs_per_test": args.runs,
            "timestamp": ts,
            "results": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
