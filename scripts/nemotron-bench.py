r"""Same prompts through bare openvino_genai and through Ollama: answers, decode
speed, and a needle at 1k/4k/16k tokens.

    venv-nemotron\Scripts\python scripts\nemotron-bench.py --genai <ir-dir> --device GPU --label lightning-int4-b60
    python scripts\nemotron-bench.py --ollama nemotron-3.5-lightning:30b-a3b --label lightning-q4-5090

Why: optimum-intel#2085 brings NemotronH (Nemotron 3 Nano, 3.5 Lightning) to
OpenVINO, and the useful thing to say on that PR is how the export does on
hardware its author does not have, next to a stack that already runs the
model (llama.cpp via Ollama on the RTX 5090). The needle is there because the
export clamps Mamba-2's dt differently from transformers, and the PR shows
that only moves things at long context.

Out: one JSON per arm in --out-dir, plus a line per case on stdout. Reports,
never asserts: a wrong answer or a failed length is a row, not a crash.
"""
import argparse
import json
import os
import platform
import time
import urllib.request

QUESTIONS = (
    ("capital", "What is the capital of Norway? Answer with one word.", "oslo"),
    ("arith", "What is 17 * 23? Answer with the number only.", "391"),
    ("days", "How many days are there in a leap year? Answer with the number only.", "366"),
)
STORY = ("Write a long, detailed story about a lighthouse keeper on a remote "
         "Arctic island. Keep going for at least 800 words.")
NEEDLE = "The access code for the north gate is 7341."
NEEDLE_Q = "What is the access code for the north gate? Answer with the number only."
FILLER = ("The ferry left the quay at seven and the gulls followed it out past "
          "the breakwater, where the swell picked up and the deck hands went "
          "below to drink coffee and complain about the weather forecast. ")


def needle_prompt(n_tokens, count):
    """Filler text of about n_tokens with the needle in the middle, question last.

    Why: a fixed, boring haystack makes TTFT comparable across stacks and the
    answer checkable; the middle is the hardest place to recall from.

    In: target token count, a function counting tokens of a string.
    Out: the prompt string; its real token count is measured by the caller,
    since each stack tokenises with its own tokenizer.
    """
    per = max(1, count(FILLER))
    reps = max(2, n_tokens // per)
    half = reps // 2
    return FILLER * half + NEEDLE + " " + FILLER * (reps - half) + "\n\n" + NEEDLE_Q


class GenaiArm:
    """Bare openvino_genai LLMPipeline (the paged path, where SelectiveSSM runs).

    Why: NoLlama serves through LLMPipeline, so this is the number that would
    hold under the server; OVModelForCausalLM runs the slow ov::Loop path.

    In: IR dir and device. Out: run(prompt, max_new) -> dict with text,
    prompt/gen token counts, TTFT and decode tok/s from genai's perf metrics.
    """

    def __init__(self, ir, device):
        import openvino as ov
        import openvino_genai as ovg
        self.ovg = ovg
        t0 = time.time()
        self.pipe = ovg.LLMPipeline(ir, device)
        self.load_s = time.time() - t0
        self.tok = self.pipe.get_tokenizer()
        self.versions = {"openvino": ov.__version__, "genai": ovg.__version__}

    def count(self, text):
        """Token count of text under the model's own tokenizer."""
        return int(self.tok.encode(text).input_ids.get_shape()[1])

    def run(self, prompt, max_new):
        """Greedy generate one chat turn and return text plus metrics."""
        hist = self.ovg.ChatHistory()
        hist.append({"role": "user", "content": prompt})
        templated = self.tok.apply_chat_template(hist, add_generation_prompt=True)
        cfg = self.ovg.GenerationConfig()
        cfg.max_new_tokens = max_new
        cfg.do_sample = False
        t0 = time.time()
        res = self.pipe.generate([templated], cfg)
        wall = time.time() - t0
        pm = res.perf_metrics
        return {"text": res.texts[0], "prompt_tokens": pm.get_num_input_tokens(),
                "gen_tokens": pm.get_num_generated_tokens(),
                "ttft_s": pm.get_ttft().mean / 1000, "tok_s": pm.get_throughput().mean,
                "wall_s": wall}


class OllamaArm:
    """Ollama's native /api/chat, greedy, one request per case.

    Why: Ollama's own counters (prompt_eval_*, eval_*) are exact and exclude
    HTTP time, so they compare fairly with genai's perf metrics.

    In: model tag, base URL, num_ctx. Out: same dict shape as GenaiArm.run;
    token counts use a ~4 chars/token estimate only for building prompts.
    """

    def __init__(self, model, url, num_ctx):
        self.model, self.url, self.num_ctx = model, url.rstrip("/"), num_ctx
        self.load_s = None
        with urllib.request.urlopen(self.url + "/api/version", timeout=10) as r:
            self.versions = {"ollama": json.load(r)["version"]}

    def count(self, text):
        """Rough token estimate, used only to size the needle prompts."""
        return max(1, len(text) // 4)

    def run(self, prompt, max_new):
        """Greedy chat turn through Ollama; durations come from Ollama itself."""
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "seed": 1, "num_predict": max_new,
                            "num_ctx": self.num_ctx}}
        req = urllib.request.Request(self.url + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.load(r)
        wall = time.time() - t0
        msg = d.get("message", {})
        text = msg.get("content", "")
        if msg.get("thinking"):
            text = "<think>" + msg["thinking"] + "</think>" + text
        ev, evd = d.get("eval_count", 0), d.get("eval_duration", 1)
        return {"text": text, "prompt_tokens": d.get("prompt_eval_count"),
                "gen_tokens": ev, "ttft_s": d.get("prompt_eval_duration", 0) / 1e9,
                "tok_s": ev / (evd / 1e9) if evd else None, "wall_s": wall}


def answer_ok(text, want):
    """True when the expected answer appears after any <think> span.

    Why: a reply cut off inside <think> often names the answer while still
    reasoning; that is a token-budget failure, not a correct answer.

    In: generated text, expected lowercase substring. Out: False for an
    unclosed <think>, else whether the tail contains it (commas ignored).
    """
    if "<think>" in text and "</think>" not in text:
        return False
    tail = text.split("</think>")[-1].lower()
    return want in tail.replace(",", "")


def main():
    """Run the case list against one arm and write its JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--genai", metavar="IR_DIR")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--ollama", metavar="TAG")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--label", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--lengths", default="1000,4000,16000")
    ap.add_argument("--max-new", type=int, default=2048, help="cap for answers (thinking models need room)")
    ap.add_argument("--out-dir", default="bench-results")
    a = ap.parse_args()

    arm = GenaiArm(a.genai, a.device) if a.genai else OllamaArm(a.ollama, a.url, 20480)
    out = {"label": a.label, "note": a.note, "host": platform.node(),
           "arm": {"genai": a.genai, "device": a.device} if a.genai else {"ollama": a.ollama},
           "versions": arm.versions, "load_s": arm.load_s, "cases": []}
    print(f"[{a.label}] {out['versions']} load {arm.load_s}", flush=True)

    def rec(name, r, ok=None):
        """Append one case, print a one-line summary of it."""
        r = dict(r, case=name, ok=ok)
        out["cases"].append(r)
        print(f"  {name:12} ok={ok} in={r['prompt_tokens']} out={r['gen_tokens']} "
              f"ttft={r['ttft_s']:.2f}s {r['tok_s'] or 0:.1f} tok/s  "
              f"{r['text'].split('</think>')[-1].strip()[:80]!r}", flush=True)

    arm.run("Say hi.", 16)                                  # warmup, not recorded
    for name, q, want in QUESTIONS:
        r = arm.run(q, a.max_new)
        rec(name, r, answer_ok(r["text"], want))
    for i in range(3):
        rec(f"decode{i}", arm.run(STORY, 256))
    for n in [int(x) for x in a.lengths.split(",") if x]:
        try:
            r = arm.run(needle_prompt(n, arm.count), a.max_new)
            rec(f"needle{n // 1000}k", r, answer_ok(r["text"], "7341"))
        except Exception as e:                              # a length that fails is a result
            out["cases"].append({"case": f"needle{n // 1000}k", "error": str(e)[:500]})
            print(f"  needle{n // 1000}k ERROR {str(e)[:200]}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"nemotron-{a.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
