r"""Exercise a model hard enough to expose the failures short probes miss.

Why this exists separately from bare-probe.py: that script sweeps the right
knobs (repetition_penalty, presence, frequency, sampling) but runs every one
against "Say hello." — far too short for the failures that actually shipped.
On 2026-09-13 it passed Phi-3.5-mini clean while the model degenerated into
word salad at repetition_penalty 1.1, and passed SmolLM3-3B clean while that
model returned EMPTY content on real work. Both were found by a human reading
output. This script is the attempt to stop needing that.

The two failures look opposite and must both be caught:

  - RUNAWAY: never emits EOS. Fills the token budget with text that decays
    into concatenated words and then free association. Phi-3.5-mini at
    repetition_penalty 1.1 ended "...urbane verdancy wrath workmanship Xanadu
    Yggdrasil Zamians" and was still going at the cap.
  - EMPTY: reasons past its budget and answers nothing at all. SmolLM3-3B on
    NPU 4 returned '' 3/3 with 1430 chars of <think> and no closing tag.

Neither raises. Both are a 200 with bad content, which is why only length and
shape catch them.

    venv\Scripts\python scripts\model-test.py <model-dir>
    venv\Scripts\python scripts\model-test.py <model-dir> --device CPU
    venv\Scripts\python scripts\model-test.py <model-dir> --json out.json

Out: one line per case, a verdict, and exit 1 if any case failed — so it can
gate a run. Long by design: the whole point is generating enough tokens for
decay to show, so budget minutes per model per device, not seconds.
"""
import argparse
import json
import os
import re
import sys
import time

import openvino_genai as ovg

# Prompts chosen to demand LENGTH, because length is what exposes decay. The
# last is a follow-up chain: context growth is its own axis, and an agent's
# later turns are where a small model is most likely to come apart.
PROMPTS = [
    ("short", ["Reply with exactly the word: ready"]),
    ("medium", ["Demonstrate Python functions by writing three examples "
                "and call them from main()"]),
    ("long", ["Write a detailed explanation of Python decorators, with at "
              "least four worked examples and common pitfalls."]),
    ("follow-ups", ["Write a Python function that reverses a string.",
                    "Now add type hints and a docstring.",
                    "Now write three unit tests for it.",
                    "Now explain what could still go wrong."]),
]

# Only the knobs that have actually bitten. Each costs a full generation, so
# the list stays short enough that people run it.
KNOBS = [
    ("default", {}),
    ("rep_penalty=1.05", {"repetition_penalty": 1.05}),
    ("rep_penalty=1.1", {"repetition_penalty": 1.1}),
    ("sampled t=0.7", {"do_sample": True, "temperature": 0.7, "top_p": 0.9}),
]

# Must be generous enough that a HEALTHY answer finishes inside it, or
# "hit the cap" stops discriminating and condemns everything. Calibrated
# 2026-09-13 on Phi-3.5-mini: at 700 the good answer (repetition_penalty
# 1.05) was truncated mid-code and looked identical to the bad one; at 1024
# it stops cleanly at ~3100 chars while 1.1 runs past 4700 and keeps going.
MAX_TOKENS = 1024
# Weak signal, reported but never condemning on its own. Real Python hits 50
# characters with a format string; the degenerate run measured 62. Too close
# to separate, which an earlier draft of this file got wrong.
RUN_LEN_LIMIT = 45
UNIQUE_RATIO_MIN = 0.30   # below this the output is looping


def looks_degenerate(text, hit_cap):
    """Classify one completion without a human reading it.

    The decisive signal is whether it STOPPED. Given a cap a healthy answer
    finishes inside (see MAX_TOKENS), running to the cap means no EOS was ever
    emitted, which is what both shipped failures have in common. A collapsed
    unique-word ratio is the other condemning signal, because looping output
    can still terminate.

    Word-run length is REPORTED but never condemns alone: measured 2026-09-13,
    healthy Python reached 50 characters (a format string) against the
    degenerate run's 62 — too close to separate. An earlier draft of this file
    treated it as decisive and flagged a known-good completion.

    In: the completion and whether generation stopped because it ran out of
    budget. Out: (verdict, reasons) where verdict is 'ok', 'empty' or
    'runaway'.
    """
    stripped = text.strip()
    if not stripped:
        return "empty", ["no content at all"]

    reasons = []
    words = re.findall(r"\S+", stripped)
    longest = max((len(w) for w in words), default=0)
    if longest > RUN_LEN_LIMIT:
        reasons.append(f"longest whitespace-free run {longest} chars")

    if len(words) > 60:
        ratio = len(set(w.lower() for w in words)) / len(words)
        if ratio < UNIQUE_RATIO_MIN:
            reasons.append(f"unique-word ratio {ratio:.2f}")

    # Condemning signals, evaluated after the weak ones are collected so they
    # still appear in `reasons` for a human reading the report.
    condemning = []
    if hit_cap:
        condemning.append(f"hit the {MAX_TOKENS}-token cap without emitting EOS")
    if len(words) > 60:
        ratio = len(set(w.lower() for w in words)) / len(words)
        if ratio < UNIQUE_RATIO_MIN:
            condemning.append(f"unique-word ratio {ratio:.2f} (looping)")

    if condemning:
        return "runaway", condemning + reasons
    return "ok", reasons


def run_case(pipe, tok, turns, knob, results):
    """Run one prompt (or follow-up chain) under one knob setting.

    Why the whole chain shares a config: an agent does not change sampling
    between turns, and the failure we are hunting appears as context grows.

    In: pipeline, tokenizer, the list of user turns, the knob dict, and the
    list to append to. Out: nothing; appends one record per turn. Exceptions
    are recorded as a case rather than raised — a model that throws is a
    result, not a crashed test run.
    """
    history = []
    for i, user in enumerate(turns, 1):
        history.append({"role": "user", "content": user})
        cfg = ovg.GenerationConfig()
        cfg.max_new_tokens = MAX_TOKENS
        for k, v in knob[1].items():
            setattr(cfg, k, v)
        try:
            prompt = tok.apply_chat_template(history, add_generation_prompt=True)
            t0 = time.time()
            out = str(pipe.generate(prompt, cfg))
            dt = time.time() - t0
        except Exception as e:
            results.append(dict(knob=knob[0], turn=i, verdict="error",
                                reasons=[str(e)[:120]], chars=0, secs=0.0))
            return
        n_tok = len(tok.encode(out).input_ids.data[0]) if out else 0
        hit_cap = n_tok >= MAX_TOKENS - 2
        verdict, reasons = looks_degenerate(out, hit_cap)
        results.append(dict(knob=knob[0], turn=i, verdict=verdict, reasons=reasons,
                            chars=len(out), secs=round(dt, 1),
                            tail=out.strip()[-90:]))
        history.append({"role": "assistant", "content": out})
        if verdict != "ok":
            return          # the chain is already compromised; stop burning time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args()

    print(f"model  : {os.path.basename(os.path.normpath(a.model_dir))}")
    print(f"device : {a.device}")
    print(f"genai  : {ovg.__version__}")
    t0 = time.time()
    pipe = ovg.LLMPipeline(a.model_dir, device=a.device)
    tok = pipe.get_tokenizer()
    print(f"loaded : {time.time() - t0:.0f}s\n")

    report = {}
    failures = 0
    for name, turns in PROMPTS:
        for knob in KNOBS:
            results = []
            run_case(pipe, tok, turns, knob, results)
            for r in results:
                key = f"{name}/{knob[0]}/turn{r['turn']}"
                report[key] = r
                flag = "    " if r["verdict"] == "ok" else ">>> "
                print(f"{flag}{key:<42} {r['verdict']:<8} "
                      f"{r['chars']:>6}ch {r['secs']:>6}s")
                if r["reasons"]:
                    for why in r["reasons"]:
                        print(f"        - {why}")
                    if r["verdict"] != "ok":
                        print(f"        tail: {r.get('tail', '')!r}")
                if r["verdict"] != "ok":
                    failures += 1

    print(f"\n{len(report)} cases, {failures} failed")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"report -> {a.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
