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

# Windows gives a redirected stdout cp1252, which cannot encode most of what a
# misbehaving model emits — and printing a tail then raises UnicodeEncodeError
# and kills the run. That cost a 40-minute Qwen3-8B sweep on 2026-09-13, whose
# results were lost entirely because the report was only written at the end.
# errors="replace" keeps a mangled character from being fatal: this harness
# exists to look at bad output, so it must be able to print anything.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Prompts chosen to demand LENGTH, because length is what exposes decay. The
# last is a follow-up chain: context growth is its own axis, and an agent's
# later turns are where a small model is most likely to come apart.
# (name, budget, turns). The BUDGET IS PER PROMPT and must be generous enough
# that a healthy answer finishes inside it, or "hit the cap" stops meaning
# anything and condemns every long answer. Calibrated 2026-09-13 on
# Phi-3.5-mini/CPU, and got this wrong twice on the way: 700 truncated the
# medium answer mid-code, then a flat 1024 truncated the long one the same way
# while the medium one was fine. A cap tuned on one prompt does not transfer to
# a longer one.
PROMPTS = [
    ("short", 48, ["Reply with exactly the word: ready"]),
    ("medium", 1024, ["Demonstrate Python functions by writing three examples "
                      "and call them from main()"]),
    ("long", 3072, ["Write a detailed explanation of Python decorators, with at "
                    "least four worked examples and common pitfalls."]),
    ("follow-ups", 1536, ["Write a Python function that reverses a string.",
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

# Two thresholds, because one number cannot do both jobs. Calibrated against
# 27 judged cases on Phi-3.5-mini (NPU 3 and NPU 4), grouped by the judge's
# verdict:
#     ok          n=11   longest run 10 - 35
#     truncated   n=5                12 - 43
#     off_topic   n=1                47
#     runaway     n=4                28 - 582
# So the ranges OVERLAP below ~50 and separate cleanly at the top: the two
# real runaways measured 454 and 582 while healthy output never passed 35.
RUN_LEN_LIMIT = 45     # report only — useless as a verdict, see the overlap
RUN_LEN_DAMNING = 150  # nothing healthy came close; condemns on its own
UNIQUE_RATIO_MIN = 0.30   # below this the output is looping


def looks_degenerate(text, hit_cap, budget):
    """Classify one completion without a human reading it.

    The decisive signal is whether it STOPPED. Given a cap a healthy answer
    finishes inside (see each prompt's budget), running to the cap means no EOS
    was ever emitted, which is what both shipped failures have in common. A
    collapsed unique-word ratio is the other condemning signal, because looping
    output can still terminate.

    The budget is passed in rather than global: a cap calibrated on one prompt
    does not transfer to a longer one, and a too-small cap makes this check
    condemn every healthy long answer (got that wrong twice, 2026-09-13).

    Word-run length carries TWO thresholds. Below RUN_LEN_DAMNING it is
    reported but never condemns: healthy Python reaches 50 characters with a
    format string, and one real degenerate run measured 62 — too close to
    separate, which an earlier draft got wrong by treating it as decisive.
    Above RUN_LEN_DAMNING it condemns alone: across 27 judged cases nothing
    healthy exceeded 35 while the two clear runaways hit 454 and 582.

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
    if longest > RUN_LEN_DAMNING:
        condemning.append(f"whitespace-free run of {longest} chars")
    if hit_cap:
        condemning.append(f"hit the {budget}-token cap without emitting EOS")
    if len(words) > 60:
        ratio = len(set(w.lower() for w in words)) / len(words)
        if ratio < UNIQUE_RATIO_MIN:
            condemning.append(f"unique-word ratio {ratio:.2f} (looping)")

    if condemning:
        return "runaway", condemning + reasons
    return "ok", reasons


def run_case(pipe, tok, turns, budget, knob, results, judge_fn=None):
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
        cfg.max_new_tokens = budget
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
        hit_cap = n_tok >= budget - 2
        verdict, reasons = looks_degenerate(out, hit_cap, budget)
        rec = dict(knob=knob[0], turn=i, verdict=verdict, reasons=reasons,
                   chars=len(out), secs=round(dt, 1),
                   tail=out.strip()[-90:], prompt=user, output=out)
        # Second opinion. The heuristics cannot tell repetitive CODE from
        # looping, and have no expression at all for "fluent answer to the
        # wrong question" - both of which a reader spots instantly. Recording
        # BOTH verdicts is the point: a disagreement is either a heuristic gap
        # or a false positive, and that is what calibrates the thresholds from
        # data instead of from someone's guess.
        if judge_fn:
            j = judge_fn(user, out)
            rec["judge"] = j.get("verdict")
            rec["judge_why"] = j.get("why", "")[:160]
            rec["judge_by"] = j.get("judged_by")
            if j.get("verdict") not in (None, "unavailable"):
                # "truncated" is NOT a fault — the rubric says so explicitly
                # (coherent text that ran out of budget). Treating it as one
                # made 5 of 6 "disagreements" in the first judged sweep bogus
                # and buried the 4 real ones.
                HEALTHY = ("ok", "truncated")
                rec["agree"] = (j["verdict"] in HEALTHY) == (verdict == "ok")
        results.append(rec)
        history.append({"role": "assistant", "content": out})
        if verdict != "ok":
            return          # the chain is already compromised; stop burning time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--judge", action="store_true",
                    help="also grade each completion with a capable cloud model "
                         "(needs OLLAMA_API_KEY). The heuristics stay the local "
                         "pass; this is the deep one, and disagreements between "
                         "them are the interesting result.")
    a = ap.parse_args()

    judge_fn = None
    if a.judge:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from judge import judge as judge_fn      # noqa: F401
        if not os.environ.get("OLLAMA_API_KEY"):
            print("ERROR: --judge needs OLLAMA_API_KEY in the environment.")
            return 2

    print(f"model  : {os.path.basename(os.path.normpath(a.model_dir))}")
    print(f"device : {a.device}")
    print(f"genai  : {ovg.__version__}")
    t0 = time.time()
    pipe = ovg.LLMPipeline(a.model_dir, device=a.device)
    tok = pipe.get_tokenizer()
    print(f"loaded : {time.time() - t0:.0f}s\n")

    report = {}
    failures = 0
    for name, budget, turns in PROMPTS:
        for knob in KNOBS:
            results = []
            run_case(pipe, tok, turns, budget, knob, results, judge_fn)
            for r in results:
                key = f"{name}/{knob[0]}/turn{r['turn']}"
                report[key] = r
            # Flush after every knob, not at the end. Each case costs minutes
            # of generation; losing the lot to a late crash is not acceptable
            # for a run measured in hours.
            if a.json:
                with open(a.json, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2)
                jv = r.get("judge")
                disagree = r.get("agree") is False
                flag = "!!! " if disagree else ("    " if r["verdict"] == "ok" else ">>> ")
                jtxt = f" judge={jv}" if jv else ""
                print(f"{flag}{key:<42} {r['verdict']:<8} "
                      f"{r['chars']:>6}ch {r['secs']:>6}s{jtxt}")
                if disagree:
                    print(f"        DISAGREE - judge says {jv}: {r.get('judge_why','')[:90]}")
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
