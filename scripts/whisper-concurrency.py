"""Does Whisper on one device cost the chat model on another anything?

    python scripts/whisper-concurrency.py [--base-url URL] [--set DIR] [--n 6]

Why: T-042's whole point is that transcription on the NPU leaves the GPU to
the chat model. That is a claim about shared memory bandwidth, the power
budget and our own locks, none of which can be read off the code. This
measures it: transcription latency idle vs during a chat stream, and chat
decode speed alone vs with transcriptions running beside it.

Needs a running server with a chat slot and a Whisper slot, and the clip set
from scripts/whisper-build-set.py. Stdlib only.

Out: one JSON object on stdout. Prints progress to stderr.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
import uuid

STORY = ("Write a detailed, long story about a lighthouse keeper on the coast "
         "of northern Norway. At least 600 words.")


def post_audio(base, path, language):
    """POST one WAV to /v1/audio/transcriptions as multipart; return (s, text).

    Why: urllib has no multipart helper, and pulling in requests would break
    the stdlib-only rule for scripts that run on every box.

    In: base URL, WAV path, ISO language or None. Out: wall seconds and the
    transcript; raises on HTTP errors.
    """
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        audio = f.read()
    parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"{os.path.basename(path)}\"\r\nContent-Type: audio/wav\r\n\r\n".encode(),
             audio, b"\r\n"]
    if language:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n"
                     f"{language}\r\n".encode())
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{base}/v1/audio/transcriptions", b"".join(parts),
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.loads(r.read())["text"]
    return time.time() - t, text


def stream_chat(base, result, started):
    """Stream one long chat answer; record first-token and decode timing.

    Why: decode speed is where shared memory bandwidth would show; TTFT is
    reported separately so prefill does not dilute it.

    In: base URL, a dict to fill, an Event set at the first content chunk.
    Out: fills result with ttft_s, chunks, decode_s, chunks_per_s. Chunks,
    not tokens: the server sends about one token per chunk, and the same
    count is used for both arms, so the ratio holds either way.
    """
    body = {"model": "x", "stream": True, "temperature": 0, "max_tokens": 700,
            "messages": [{"role": "user", "content": STORY}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                delta = json.loads(line[5:])["choices"][0]["delta"]
            except (ValueError, KeyError, IndexError):
                continue
            if delta.get("content") or delta.get("reasoning_content"):
                n += 1
                if first is None:
                    first = time.time()
                    started.set()
    end = time.time()
    result.update(ttft_s=round(first - t0, 2) if first else None, chunks=n,
                  decode_s=round(end - first, 2) if first else None,
                  chunks_per_s=round((n - 1) / (end - first), 2) if first and n > 1 else None)
    started.set()


def main():
    """Run idle-transcribe, chat-alone, then both together; print JSON.

    Why: the three arms are the minimum that separates "Whisper is slow"
    from "Whisper slows the chat" from "the chat slows Whisper".

    In: argv. Out: JSON on stdout. Arm 3 keeps transcribing for the whole
    chat stream, so the chat's decode rate is measured under full overlap,
    not diluted by idle stretches. A transcript differing from its idle
    twin is flagged: the arms would not be measuring the same work.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--set", default=os.path.expanduser("~/audio-samples/set"))
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    man = [json.loads(l) for l in open(os.path.join(a.set, "manifest.jsonl"), encoding="utf-8")]
    clips = [c for c in man if c["group"] in ("smoke", "fleurs-no")][:a.n]
    health = json.loads(urllib.request.urlopen(f"{a.base_url}/health", timeout=10).read())
    out = {"health_devices": health.get("devices"), "whisper": health.get("whisper"),
           "clips": [c["path"] for c in clips]}

    def transcribe_all(tag):
        """Transcribe every chosen clip once; return per-clip records."""
        recs = []
        for c in clips:
            s, text = post_audio(a.base_url, os.path.join(a.set, c["path"]), c["lang"])
            recs.append({"path": c["path"], "audio_s": c["seconds"], "wall_s": round(s, 3), "text": text})
            print(f"  [{tag}] {c['path']} {s:.2f}s", file=sys.stderr, flush=True)
        return recs

    print("warm-up", file=sys.stderr)
    post_audio(a.base_url, os.path.join(a.set, clips[0]["path"]), clips[0]["lang"])
    print("arm 1: transcribe, idle", file=sys.stderr)
    out["idle"] = transcribe_all("idle")
    print("arm 2: chat alone", file=sys.stderr)
    alone, ev = {}, threading.Event()
    stream_chat(a.base_url, alone, ev)
    out["chat_alone"] = alone
    print(f"  {alone}", file=sys.stderr)
    print("arm 3: chat + transcribe until the stream ends", file=sys.stderr)
    both, ev = {}, threading.Event()
    th = threading.Thread(target=stream_chat, args=(a.base_url, both, ev))
    th.start()
    ev.wait(600)
    t = time.time()
    conc = []
    while th.is_alive():
        for c in clips:
            if not th.is_alive():
                break
            s, text = post_audio(a.base_url, os.path.join(a.set, c["path"]), c["lang"])
            conc.append({"path": c["path"], "wall_s": round(s, 3), "text": text})
            print(f"  [conc] {c['path']} {s:.2f}s", file=sys.stderr, flush=True)
    out["concurrent_span_s"] = round(time.time() - t, 2)
    th.join()
    out["concurrent"] = conc
    out["chat_with_whisper"] = both
    by_path = {r["path"]: r for r in out["idle"]}
    out["transcript_mismatch"] = sorted({r["path"] for r in conc if r["text"] != by_path[r["path"]]["text"]})
    ratios = [r["wall_s"] / by_path[r["path"]]["wall_s"] for r in conc]
    out["summary"] = {"transcriptions_during_chat": len(conc),
                      "whisper_slowdown_mean": round(sum(ratios) / len(ratios), 3) if ratios else None,
                      "chat_cps_alone": alone.get("chunks_per_s"),
                      "chat_cps_with_whisper": both.get("chunks_per_s")}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
