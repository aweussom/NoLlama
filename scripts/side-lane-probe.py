r"""Does a CPU side-lane server starve while the iGPU beside it prefills?

Why: OpenCode sends small side requests (titles, summaries) beside every
turn, and the two-server recipe puts them on a second NoLlama on the CPU. On
a 140V laptop that side request went 1.6 s idle -> 15 s while Qwen3-8B
prefilled on the iGPU -> 40 s beside Qwen3-Coder-30B [OBSERVED 2026-09-23,
OPENCODE-PLAN.md], and the installer stopped offering the split on iGPU
boxes on the strength of that one machine. This asks the same question of
any box with the same scripted load, so a second machine's answer is
comparable rather than anecdotal.

It starts both servers itself (coder on GPU, side model on CPU), measures
fixed side requests with the box idle and then while a large, never-cached
prompt prefills on the GPU, and stops both. Stdlib only, so it runs from the
venv on any box.

    venv\Scripts\python scripts\side-lane-probe.py --gpu-model ~\models\Qwen3-8B-int4-cw-ov `
        --cpu-model ~\models\Phi-3.5-mini-instruct-int4-cw-ov

Out: one line per request, a summary per request shape, and a JSON record
(provenance included) written to bench/side-lane-<host>-<stamp>.json. Exit 0
unless a server never became ready.
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Two shapes, fixed so every box and every run asks for the same work.
# "title" is what OpenCode actually sends -- a few tokens out, so it is mostly
# prefill and TTFT. "summary" is decode-bound (~128 tokens). A summary alone
# measured ~2.2x slower under contention on the 140V where the original
# title-shaped OpenCode traffic measured ~9x [OBSERVED 2026-09-24], so both
# are kept. Phi-3.5-mini has no thinking channel, so the output length is the
# instruction's, not the model's mood.
SIDE_PROMPTS = {
    "title": ("Generate a title of at most six words for this conversation: fixing "
              "a billing calculator that applied discounts as fractions instead "
              "of percentages.", 16),
    "summary": ("Summarise in about 80 words: a billing calculator applied "
                "discounts as fractions instead of percentages, so two tests "
                "failed; the fix divides the percentage by 100.", 128),
}

FILLER = ("def apply_discount(amount, percent):\n    return round(amount - amount * "
          "percent / 100, 2)\n# The billing report sums discounted line items. ")


def post(url, body, timeout=3600):
    """POST JSON and return (seconds, parsed reply or error string)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return time.perf_counter() - t0, json.loads(r.read())
    except Exception as e:
        return time.perf_counter() - t0, f"error: {e}"


def side_request(base, kind):
    """One measured side request of the given shape: non-streamed, greedy.

    Out: (seconds, characters of answer text; -1 when the call failed).
    """
    prompt, cap = SIDE_PROMPTS[kind]
    secs, reply = post(base + "/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cap, "temperature": 0, "stream": False})
    if isinstance(reply, str):
        return secs, -1
    return secs, len(reply["choices"][0]["message"].get("content") or "")


def big_prompt(nonce, chars):
    """A prompt of about `chars` characters that no earlier run has cached.

    Why the nonce comes first: prefix caching matches from the start, so a
    unique first line forces a full cold prefill every run -- the load an
    agent's new suffix puts on the GPU, with no cache hit to hide it.
    """
    body = (FILLER * (chars // len(FILLER) + 1))[:chars]
    return f"[run {nonce}] Read this code, then answer in one word: does it divide by 100?\n{body}"


def wait_ready(base, proc, limit=900):
    """Poll /health until 'ready'. Out: True, or False if it died or timed out."""
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ready":
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def start_server(model, device, port, log):
    """Start a NoLlama server as a child process. Out: the Popen."""
    return subprocess.Popen(
        [sys.executable, os.path.join(REPO, "nollama.py"), "--model-dir", model,
         "--device", device, "--port", str(port), "--ollama-port", "0",
         "--idle-timeout", "0", "--no-prewarm", "--log-file", log],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_server(proc):
    """Stop a server AND its children.

    Why the tree: a venv's python.exe on Windows is a launcher that spawns the
    real interpreter, so terminating the Popen leaves the server holding its
    port [OBSERVED 2026-09-23, laptop]. taskkill /T takes the whole tree.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.terminate()
    proc.wait(timeout=60)


def provenance():
    """Where and on what this ran -- the part a results table cannot carry."""
    info = {"host": platform.node(), "cpu": platform.processor(),
            "when": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        import openvino
        import openvino_genai
        info["openvino"] = openvino.__version__
        info["openvino_genai"] = openvino_genai.__version__
        core = openvino.Core()
        info["gpu"] = core.get_property("GPU", "FULL_DEVICE_NAME")
        info["gpu_caps"] = core.get_property("GPU", "OPTIMIZATION_CAPABILITIES")
    except Exception as e:
        info["openvino_error"] = str(e)
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController | Where-Object Name -match 'Intel').DriverVersion"],
                capture_output=True, text=True, timeout=60).stdout.strip()
            info["gpu_driver"] = out
        except Exception:
            pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-model", required=True)
    ap.add_argument("--cpu-model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prefill-chars", type=int, default=40000)
    ap.add_argument("--head-start", type=float, default=1.0,
                    help="seconds between starting the GPU prefill and the side request")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    host = platform.node()
    gpu_base, cpu_base = "http://127.0.0.1:8000", "http://127.0.0.1:8002"
    record = {"provenance": provenance(), "args": vars(args), "idle": [], "contended": []}
    print(json.dumps(record["provenance"], indent=1), flush=True)

    gpu = start_server(os.path.expanduser(args.gpu_model), "GPU", 8000,
                       os.path.join(REPO, "bench", f"side-lane-gpu-{host}.log"))
    cpu = start_server(os.path.expanduser(args.cpu_model), "CPU", 8002,
                       os.path.join(REPO, "bench", f"side-lane-cpu-{host}.log"))
    try:
        if not (wait_ready(gpu_base, gpu) and wait_ready(cpu_base, cpu)):
            print("a server never became ready -- see bench/side-lane-*.log", flush=True)
            return 1
        side_request(cpu_base, "title")  # first request pays compilation; not measured

        for kind in SIDE_PROMPTS:
            for i in range(args.runs):
                secs, n = side_request(cpu_base, kind)
                record["idle"].append({"kind": kind, "side_s": round(secs, 2), "chars": n})
                print(f"idle       {kind:7} run {i + 1}: side {secs:6.1f} s ({n} chars)", flush=True)

        for kind in SIDE_PROMPTS:
            for i in range(args.runs):
                gpu_result = {}

                def load():
                    s, reply = post(gpu_base + "/v1/chat/completions", {
                        "messages": [{"role": "user",
                                      "content": big_prompt(f"{stamp}-{kind}-{i}", args.prefill_chars)}],
                        "max_tokens": 16, "temperature": 0, "stream": False})
                    gpu_result.update(gpu_s=round(s, 2), ok=not isinstance(reply, str),
                                      end=time.perf_counter())
                t = threading.Thread(target=load)
                t.start()
                time.sleep(args.head_start)
                secs, n = side_request(cpu_base, kind)
                side_end = time.perf_counter()
                t.join()
                overlapped = gpu_result.get("end", 0) > side_end
                record["contended"].append({"kind": kind, "side_s": round(secs, 2), "chars": n,
                                            "gpu_s": gpu_result.get("gpu_s"),
                                            "gpu_ok": gpu_result.get("ok"),
                                            "gpu_still_busy_at_side_end": overlapped})
                print(f"contended  {kind:7} run {i + 1}: side {secs:6.1f} s ({n} chars), GPU "
                      f"prefill+16 {gpu_result.get('gpu_s')} s, overlapped the whole side "
                      f"request: {overlapped}", flush=True)
    finally:
        stop_server(gpu)
        stop_server(cpu)

    print(flush=True)
    for kind in SIDE_PROMPTS:
        idle = [r["side_s"] for r in record["idle"] if r["kind"] == kind]
        cont = [r["side_s"] for r in record["contended"] if r["kind"] == kind]
        ratio = (sum(cont) / len(cont)) / (sum(idle) / len(idle))
        print(f"{kind:7}  idle {min(idle):.1f}-{max(idle):.1f} s   contended "
              f"{min(cont):.1f}-{max(cont):.1f} s   ratio ~{ratio:.1f}x", flush=True)
    out = os.path.join(REPO, "bench", f"side-lane-{host}-{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
