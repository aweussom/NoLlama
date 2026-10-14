r"""Measure whether a model's answers get WORSE as a conversation grows.

Why: we already measure that long context still runs (`benchmark.py --long`,
pass/fail on a ~100k-char prompt) and how much slower it gets (82k chars ->
58 s TTFT, 140k -> 108 s, 199k -> 175 s, `docs/BENCHMARKS.md`). Nothing
measured whether the ANSWERS decay. The one case on record was found by a
human reading output: a Phi-3.5-mini run answering the previous turn's
question using this turn's research (`scripts/judge.py` header). T-047.

**The control arm is the point.** Asking a probe at turn 20 and seeing it
fail proves nothing on its own: the prompt is also long by then, and "long
prompt" and "many turns" are different bugs with different fixes. So every
checkpoint is run twice -- once at the end of a real multi-turn conversation,
and once cold in a fresh conversation padded to the same size. Decay that
appears in both arms is a context-length problem; decay that appears only in
the conversational arm is a turn-count problem.

Three probe families, because they fail independently:

  recall  - a fact planted in the briefing. Mechanical: is the token there?
  format  - an instruction given once, in the briefing, that every later
            reply must obey. This is the first thing to rot and the cheapest
            to check.
  reason  - a self-contained task with a deterministic answer, carrying no
            dependency on the briefing. Isolates raw capability from
            retrieval.

Mechanical checks first, judge second, same split `judge.py` documents: the
heuristics are cheap and local, the judge is the deep pass, and the
interesting signal is where they disagree.

    python scripts/decay-probe.py http://127.0.0.1:8000
    python scripts/decay-probe.py http://127.0.0.1:8000 --checkpoints 1,5,10
    python scripts/decay-probe.py http://127.0.0.1:8000 --no-control

Out: a row per checkpoint per arm, then a JSON dump. Exit 0 always -- this
reports, it does not assert, so a failing row is data rather than a broken
run.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import long_corpus  # noqa: E402
from judge import ask  # noqa: E402

# A distinctive, unguessable token. A model that has lost the briefing cannot
# produce it by luck, and grep-ing for it needs no fuzzy matching.
SECRET = "GRAVLAKS-4417"
END_TOKEN = "###END"

BRIEFING = f"""You are helping with a long technical session. Three standing \
instructions, which apply to EVERY reply you give for the rest of this \
conversation:

1. The project access code is {SECRET}. Remember it.
2. The lead engineer is based in Verdal.
3. End every single reply with the exact token {END_TOKEN} on its own line.

Acknowledge with one short sentence, then {END_TOKEN}."""

# `reference` is handed to the judge verbatim, so it states the criterion
# rather than just the answer. The format probe measures obedience to a
# standing instruction, NOT whether the sentence about reverse proxies is any
# good -- a reference of "ends with ###END" let the judge grade the content
# instead and return ok against a mechanical FAIL, which made the two checks
# disagree about nothing [OBSERVED 2026-10-14, Phi-3.5-mini smoke run].
PROBES = (
    {
        "key": "recall",
        "ask": "What is the project access code? Answer with the code only.",
        "reference": f"The answer must state the access code {SECRET}. "
                     f"Judge only that; any wording around it is fine.",
        "check": lambda text: SECRET in text,
    },
    {
        "key": "reason",
        "ask": ("A server writes 3 log lines per request and keeps 4 days of "
                "logs. It handles 1200 requests a day. How many log lines are "
                "retained? Show the arithmetic in one line."),
        "reference": "The answer must arrive at 14400 (3 x 1200 x 4) and must "
                     "not contradict that figure later in the same reply.",
        "check": lambda text: "14400" in text.replace(",", "").replace(" ", ""),
    },
    {
        "key": "format",
        "ask": "In one sentence, what does a reverse proxy do?",
        "reference": f"IGNORE whether the content is correct. Judge ONLY "
                     f"whether the reply's last non-empty line is exactly "
                     f"{END_TOKEN}, the standing instruction from the briefing.",
        "check": lambda text: text.strip().endswith(END_TOKEN),
    },
)

CORRECTNESS_RUBRIC = """You are grading one answer from a local language model \
against a known-correct reference. Judge ONLY correctness of the substance.

Reply with ONLY a JSON object, no prose, no code fence:
{"correct": true|false, "why": "one short sentence"}

Rules:
  - Extra explanation around a correct answer is still correct.
  - A correct answer in a different format is still correct.
  - If the answer contradicts the reference, or omits it entirely, it is false.
  - If the answer is empty or degenerate, it is false."""


def post_chat(base, model, messages, max_tokens, timeout=1800):
    """One non-streaming chat turn. Returns (text, prompt_tokens, seconds, finish).

    Why non-streaming here when the rest of the repo streams: this script
    cares about the finished answer and the prompt size, not about frame
    shape, and a non-streaming call gives `usage.prompt_tokens` directly --
    which is the x-axis of the whole measurement.

    **`finish` is not optional telemetry, it is the control variable.** The
    first GPU sweep (LFM2.5-24B-A2B, B60, 2026-10-14) read as a decay curve
    and was not one: a reasoning-only model spends its budget on chain of
    thought, hits the cap, and a half-written word is scored exactly like a
    wrong answer. One row ended `'Thu'`. Without finish_reason there is no
    way to tell "the model degraded" from "I set max_tokens too low", and
    those have opposite fixes.

    In: server base URL, model id, OpenAI-shaped messages, token cap. Out:
    the tuple; text is "[error: ...]", tokens 0 and finish "error" when the
    turn failed, so one dead turn records a row instead of killing the sweep.
    """
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0,
                       "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return f"[error: HTTP {e.code} {e.read()[:200]!r}]", 0, time.time() - t0, "error"
    except Exception as e:
        return f"[error: {type(e).__name__} {str(e)[:200]}]", 0, time.time() - t0, "error"
    choice = d["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    return (text, (d.get("usage") or {}).get("prompt_tokens", 0),
            time.time() - t0, choice.get("finish_reason"))


def first_model(base):
    """Name of the first model the server advertises.

    Why: the id is the model directory's name plus @DEVICE and differs per
    machine, so it cannot be hardcoded. In: base URL. Out: the id string;
    raises if the server is not up, which is the right time to stop.
    """
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def _excerpt(text, max_chars=6000):
    """Trim a reply for the judge, keeping BOTH ends.

    Why not `text[:max_chars]`: the format probe asks whether the LAST line
    is the end token, so a head-only excerpt invents a different ending and
    the judge grades a reply that never existed. That produced `mech=ok
    judge=FAIL` on answers which did end correctly [OBSERVED 2026-10-14, B60
    sweep]. `judge.py` already truncates in the middle for the same reason.

    In: the reply. Out: the reply unchanged when short enough, else head and
    tail with an elision marker between them.
    """
    if len(text) <= max_chars:
        return text
    head, tail = max_chars // 3, max_chars - max_chars // 3
    marker = "\n\n...[%d chars elided]...\n\n" % (len(text) - max_chars)
    return text[:head] + marker + text[-tail:]


def grade(probe, text):
    """Score one answer mechanically, then ask the judge, and keep both.

    Why keep both rather than trusting one: the mechanical check cannot see a
    model that emits the right token inside an otherwise degenerate wall of
    text, and the judge cannot be trusted to spot a missing end-token it was
    not told to look for. Disagreement between them is the signal worth
    reading, exactly as `judge.py` argues for the health rubric.

    In: a PROBES entry and the model's reply. Out: a dict with `mechanical`
    (bool), `judged` (bool or None) and `why`. `judged` is None when the
    judge was unreachable -- callers must not read that as a pass.
    """
    mechanical = bool(probe["check"](text)) if text and not text.startswith("[error:") else False
    verdict = ask(CORRECTNESS_RUBRIC,
                  f"QUESTION:\n{probe['ask']}\n\n"
                  f"REFERENCE ANSWER:\n{probe['reference']}\n\n"
                  f"MODEL ANSWER:\n{text[:4000] or '(empty)'}")
    judged, why = None, ""
    if verdict and "correct" in verdict:
        judged = bool(verdict["correct"])
        why = verdict.get("why", "")
    elif verdict:
        why = verdict.get("_error", "judge unavailable")
    return {"mechanical": mechanical, "judged": judged, "why": why}


def flush(out, model, base, results):
    """Rewrite the results file after every probe, not once at the end.

    Why: the first real sweep (LFM2.5-24B-A2B on CPU, 2026-10-14) ran past
    90 minutes with 129 s prefills, and a crash at turn 20 would have thrown
    away every row before it while showing nothing on the way. Rows are a few
    hundred bytes, so rewriting the whole file each time costs nothing next to
    one generate and removes the need to watch stdout.

    In: output path, model id, server URL, the rows so far. Out: None; the
    file is replaced in place and is valid JSON after every probe.
    """
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": model, "server": base, "rows": results}, f, indent=2)


def run_probes(base, model, messages, arm, turn, tokens, results, max_tokens=700, out=None):
    """Ask every probe against the given history and record a row each.

    Why the probes do not accumulate into the history: each one would then be
    context for the next, and the recall probe would hand the secret back to
    the model right before the format probe asks for it. The conversation
    grows only through padding turns, so the three probes at one checkpoint
    all see the same prefix.

    In: transport details, the history to probe against, arm label, turn
    number, prompt size, and the list to append to. Out: None; rows land in
    `results` and a line is printed per probe.
    """
    for probe in PROBES:
        text, ptokens, secs, finish = post_chat(
            base, model, messages + [{"role": "user", "content": probe["ask"]}],
            max_tokens=max_tokens)
        score = grade(probe, text)
        # Keep the reply, truncated. Without it a mech/judge disagreement is
        # unresolvable after the fact and the run has to be repeated -- the
        # same blind spot `brain/proposed/2026-09-24-native-tool-template-as-
        # default.md` records, where nobody could say whether a tool call was
        # written inside <think> because no raw output was kept.
        row = {"arm": arm, "turn": turn, "probe": probe["key"],
               "prompt_tokens": ptokens or tokens, "seconds": round(secs, 1),
               "chars": len(text), "finish": finish,
               "truncated": finish == "length",
               "text": text[:1200], **score}
        results.append(row)
        if out:
            flush(out, model, base, results)
        jm = {True: "ok", False: "FAIL", None: "?"}[score["judged"]]
        cut = " CUT" if row["truncated"] else ""
        print(f"  {arm:12} turn {turn:>3}  {probe['key']:7} "
              f"{row['prompt_tokens']:>7} tok  "
              f"mech={'ok' if score['mechanical'] else 'FAIL':4} judge={jm:4}"
              f"{cut:4} {secs:6.1f}s  {score['why'][:44]}")


def main():
    """Run both arms over the checkpoints and dump the rows."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8000",
                    help="NoLlama base URL (use 127.0.0.1, not localhost)")
    ap.add_argument("--checkpoints", default="1,5,10,20",
                    help="Turn numbers to probe at (default 1,5,10,20)")
    ap.add_argument("--pad-chars", type=int, default=4000,
                    help="Real repo text added per padding turn (default 4000)")
    ap.add_argument("--max-tokens", type=int, default=700,
                    help="Token cap per probe answer (default 700). A "
                         "reasoning-only model spends budget on chain of "
                         "thought before it answers, so too low a cap reads "
                         "as EMPTY rather than as decay -- raise it for one.")
    ap.add_argument("--pad-tokens", type=int, default=200,
                    help="Token cap per padding reply (default 200). Same "
                         "trap: a reasoning model given 200 returns nothing "
                         "but thinking, and the history fills with empty "
                         "assistant turns.")
    ap.add_argument("--no-control", action="store_true",
                    help="Skip the cold-at-length arm (halves the runtime, "
                         "and makes a positive result uninterpretable)")
    ap.add_argument("--out", default="bench/decay-probe.json")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    checkpoints = sorted({int(x) for x in args.checkpoints.split(",") if x.strip()})
    model = first_model(base)
    print(f"model: {model}\nserver: {base}\ncheckpoints: {checkpoints}\n")

    results = []

    # --- conversational arm: one history, growing -------------------------
    print("conversational arm (history grows, turn count grows):")
    messages = [{"role": "user", "content": BRIEFING}]
    ack, tokens, _, _ = post_chat(base, model, messages, max_tokens=args.pad_tokens)
    messages.append({"role": "assistant", "content": ack})

    for turn in range(1, max(checkpoints) + 1):
        if turn in checkpoints:
            run_probes(base, model, messages, "conversation", turn, tokens,
                       results, args.max_tokens, args.out)
        pad = long_corpus(args.pad_chars, offset=turn * 37_501)
        messages.append({"role": "user", "content":
                         f"Summarise the following in ONE line.\n\n{pad}"})
        reply, tokens, _, _ = post_chat(base, model, messages,
                                        max_tokens=args.pad_tokens)
        messages.append({"role": "assistant", "content": reply})

    # --- control arm: same size, one turn ---------------------------------
    if not args.no_control:
        print("\ncontrol arm (same prompt size, no conversational history):")
        for turn in checkpoints:
            # Same text the conversational arm had absorbed by this turn,
            # delivered as a single block, so length matches and turns do not.
            bulk = "\n\n".join(long_corpus(args.pad_chars, offset=i * 37_501)
                               for i in range(1, turn))
            cold = [{"role": "user", "content":
                     BRIEFING + ("\n\nReference material:\n\n" + bulk if bulk else "")},
                    {"role": "assistant", "content": "Understood. " + END_TOKEN}]
            run_probes(base, model, cold, "control", turn, 0, results,
                       args.max_tokens, args.out)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": model, "server": base, "rows": results}, f, indent=2)
    print(f"\nSaved: {args.out}")

    print("\nsummary (mechanical pass rate):")
    for arm in ("conversation", "control"):
        rows = [r for r in results if r["arm"] == arm]
        if not rows:
            continue
        for turn in checkpoints:
            at = [r for r in rows if r["turn"] == turn]
            if at:
                scorable = [r for r in at if not r["truncated"]]
                ok = sum(1 for r in scorable if r["mechanical"])
                cut = len(at) - len(scorable)
                note = f"  ({cut} truncated, not scored)" if cut else ""
                print(f"  {arm:12} turn {turn:>3}  {ok}/{len(scorable)}  "
                      f"{at[0]['prompt_tokens']:>7} tok{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
