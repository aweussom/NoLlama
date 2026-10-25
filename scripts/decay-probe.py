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

from benchmark import collect_provenance, long_corpus  # noqa: E402
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

def _digits_only(text):
    """Strip every separator a model might put inside a number.

    Why not `.replace(",", "").replace(" ", "")`: LFM2.5-24B-A2B writes
    `14 400` using U+202F NARROW NO-BREAK SPACE, which an ordinary-space strip
    leaves alone, so a CORRECT answer scored wrong -- 0/5 on an arm where
    every run was right [OBSERVED 2026-10-14, B60]. The same model wrote
    `14,400` at temperature 0 and passed, so the bug only appeared when the
    formatting changed, which is the worst way for a checker to be wrong.

    Unicode separators seen or plausible: U+202F, U+00A0 (nbsp), U+2009 (thin
    space), U+2007 (figure space), U+066C (Arabic thousands), plus ASCII
    comma, apostrophe, underscore and space -- all legal thousands separators
    somewhere.

    In: any text. Out: the text with those removed, so a digit run can be
    matched whatever the model chose to put between the thousands.
    """
    for ch in (",", " ", " ", " ", " ", " ", "٬", "'", "_", "."):
        text = text.replace(ch, "")
    return text


def _last_line(text):
    """The reply's last non-empty line, stripped; '' when there is none.

    Why the format check is not `endswith(END_TOKEN)`: the briefing asks for
    the token ON ITS OWN LINE, and `endswith` also passes when the model tacks
    it onto the end of the answer's last sentence -- "...or caching. ###END".
    The judge, told the actual criterion, called those FAIL while the
    mechanical check called them ok, so the two were measuring different
    things and their disagreement meant nothing [OBSERVED 2026-10-14,
    Qwen3-8B on the B60]. A loose checker is worse than none: it reports a
    pass rate for a rule nobody is enforcing.

    In: the reply, possibly empty. Out: the last non-empty line, stripped.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


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
        "check": lambda text: "14400" in _digits_only(text),
    },
    {
        "key": "format",
        "ask": "In one sentence, what does a reverse proxy do?",
        "reference": f"IGNORE whether the content is correct. Judge ONLY "
                     f"whether the reply's last non-empty line is exactly "
                     f"{END_TOKEN}, the standing instruction from the briefing.",
        "check": lambda text: _last_line(text) == END_TOKEN,
    },
)

def ticket_code(k):
    """The build number planted for ticket k: four digits, distinct per k.

    Why arithmetic instead of random: both arms and every rerun must plant the
    same value, and 7919 is prime to 9000, so k = 1..9000 never collide -- a
    model that answers a NEIGHBOUR's number is visibly wrong, not lucky.

    In: ticket index >= 1. Out: int in 1000..9999.
    """
    return 1000 + (k * 7919) % 9000


def ticket_note(k):
    """The sentence that plants ticket k in a padding turn."""
    return (f"Unrelated, keep it for later: ticket T{k} was assigned build "
            f"number {ticket_code(k)}.")


def ticket_probe(k):
    """A recall probe for a fact planted MID-conversation, among look-alikes.

    Why it exists: `recall` asks for a secret in the FIRST message, which is
    the easiest place in a context to retrieve from, and Qwen3-8B,
    LFM2.5-24B-A2B and Qwen3-Coder-30B-A3B all passed it flat to 21-42k
    tokens [OBSERVED 2026-10-14..19, B60]. A probe that never fails cannot
    tell "no decay" from "cannot see decay". The failure that started T-047
    -- a reply built from the wrong turn -- is a MIDDLE-of-history,
    look-alike-interference failure, so this plants one near-identical note
    per padding turn and asks for the one halfway back.

    In: the ticket to ask about. Out: a PROBES-shaped dict, key `ticket`.
    """
    code = ticket_code(k)
    return {
        "key": "ticket",
        "ask": f"What build number was ticket T{k} assigned? Answer with the "
               f"number only.",
        "reference": f"The answer must be {code}, the build number for ticket "
                     f"T{k}. Any other ticket's number is wrong.",
        "check": lambda text: str(code) in _digits_only(text),
    }


def ticket_for(checkpoint):
    """Which ticket to ask about at a checkpoint: halfway back, None at turn 1.

    In: checkpoint turn. Out: k = turn // 2, or None when no padding turn has
    planted a note yet (turn 1 is probed before the first pad).
    """
    return checkpoint // 2 if checkpoint >= 2 else None


# Sentinel entry: resolved per checkpoint by `ticket_probe`, never asked as-is.
PROBES = PROBES + ({"key": "ticket", "per_ticket": ticket_probe},)


CORRECTNESS_RUBRIC = """You are grading one answer from a local language model \
against a known-correct reference. Judge ONLY correctness of the substance.

Reply with ONLY a JSON object, no prose, no code fence:
{"correct": true|false, "why": "one short sentence"}

Rules:
  - Extra explanation around a correct answer is still correct.
  - A correct answer in a different format is still correct.
  - If the answer contradicts the reference, or omits it entirely, it is false.
  - If the answer is empty or degenerate, it is false."""


# Client-side wait per request, set from --timeout. A cold CPU prefill of
# 42k tokens on the 285K outran the old fixed 1800 s [OBSERVED 2026-10-20].
REQUEST_TIMEOUT = 1800


def post_chat(base, model, messages, max_tokens, timeout=None):
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

    In: server base URL, model id, OpenAI-shaped messages, token cap, and a
    timeout that defaults to REQUEST_TIMEOUT (--timeout). Out: the tuple;
    text is "[error: ...]", tokens 0 and finish "error" when the turn failed
    or timed out, so one dead turn records a row instead of killing the
    sweep -- and callers must not score that row.
    """
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0,
                       "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout or REQUEST_TIMEOUT) as r:
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
                  f"MODEL ANSWER:\n{_excerpt(text) or '(empty)'}")
    judged, why = None, ""
    if verdict and "correct" in verdict:
        judged = bool(verdict["correct"])
        why = verdict.get("why", "")
    elif verdict:
        why = verdict.get("_error", "judge unavailable")
    return {"mechanical": mechanical, "judged": judged, "why": why}


def flush(out, model, base, results, meta=None):
    """Rewrite the results file after every probe, not once at the end.

    Why: the first real sweep (LFM2.5-24B-A2B on CPU, 2026-10-14) ran past
    90 minutes with 129 s prefills, and a crash at turn 20 would have thrown
    away every row before it while showing nothing on the way. Rows are a few
    hundred bytes, so rewriting the whole file each time costs nothing next to
    one generate and removes the need to watch stdout.

    `meta` carries what makes a file citable on its own: the calibration
    record and the server's provenance (runtime, driver, flags). The 2026-10-14
    files had neither, so which probes had been dropped, and on which driver,
    survived only in a terminal scrollback.

    In: output path, model id, server URL, the rows so far, optional meta
    dict merged in at top level. Out: None; the file is replaced in place and
    is valid JSON after every probe.
    """
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": model, "server": base, **(meta or {}),
                   "rows": results}, f, indent=2, default=str)


def run_probes(base, model, messages, arm, turn, tokens, results, max_tokens=700,
               out=None, probes=None, target=None, meta=None, ticket=None):
    """Ask every probe against the given history and record a row each.

    Why the probes do not accumulate into the history: each one would then be
    context for the next, and the recall probe would hand the secret back to
    the model right before the format probe asks for it. The conversation
    grows only through padding turns, so the three probes at one checkpoint
    all see the same prefix.

    The stored reply is head-and-tail (`_excerpt`), not a head slice: the
    format probe's verdict lives in the last line, and a head slice would keep
    exactly the part that cannot settle a disagreement.

    In: transport details, the history to probe against, arm label, turn
    number, prompt size, the list to append to, the meta block for `flush`,
    and the ticket the `ticket` probe asks about -- None skips that probe,
    since nothing was planted yet. Out: None; rows land in `results` and a
    line is printed per probe.
    """
    for probe in (probes if probes is not None else PROBES):
        if "per_ticket" in probe:
            if ticket is None:
                continue
            probe = probe["per_ticket"](ticket)
        text, ptokens, secs, finish = post_chat(
            base, model, messages + [{"role": "user", "content": probe["ask"]}],
            max_tokens=max_tokens)
        if finish == "error":
            # A request that failed measured nothing. Scoring it made a CPU
            # control arm whose 42k-token cold prefills outran the client
            # timeout print "0/3" at turn 40 -- read as decay -- and had the
            # judge grade the error message [OBSERVED 2026-10-20, 285K CPU].
            score = {"mechanical": None, "judged": None,
                     "why": "not scored: " + text[:80]}
        else:
            score = grade(probe, text)
        # Keep the reply, truncated. Without it a mech/judge disagreement is
        # unresolvable after the fact and the run has to be repeated -- the
        # same blind spot `brain/proposed/2026-09-24-native-tool-template-as-
        # default.md` records, where nobody could say whether a tool call was
        # written inside <think> because no raw output was kept.
        row = {"arm": arm, "turn": turn, "probe": probe["key"],
               "prompt_tokens": ptokens or tokens, "seconds": round(secs, 1),
               "chars": len(text), "finish": finish,
               "truncated": finish == "length", "target_tokens": target,
               "ticket": ticket if probe["key"] == "ticket" else None,
               "text": _excerpt(text, 2000), **score}
        results.append(row)
        if out:
            flush(out, model, base, results, meta)
        mark = {True: "ok", False: "FAIL", None: "?"}
        jm = mark[score["judged"]]
        cut = " CUT" if row["truncated"] else ""
        print(f"  {arm:12} turn {turn:>3}  {probe['key']:7} "
              f"{row['prompt_tokens']:>7} tok  "
              f"mech={mark[score['mechanical']]:4} judge={jm:4}"
              f"{cut:4} {secs:6.1f}s  {score['why'][:44]}")


def calibrate(base, model, max_tokens, runs):
    """Ask every probe cold, `runs` times, and keep only the ones it always passes.

    Why this gates the sweep: a probe the model fails COLD has no headroom to
    decay, and its failures at turn 20 are indistinguishable from its failures
    at turn 1. The arithmetic probe did exactly that on LFM2.5-24B-A2B -- it
    answered 1200, then 3600, then 3600, at every context size including a
    152-token prompt -- and three runs were scored before anyone noticed it
    had never worked [OBSERVED 2026-10-14]. A probe set is per-model, not
    universal.

    Cold means briefing plus the one question: the shortest prompt the probe
    can be asked in, so a failure here is the model, never the context. The
    `ticket` probe gets one short padding turn carrying ticket T1, because
    without a planted note there is nothing for it to recall.

    In: transport details, the probe token cap, how many cold runs each probe
    must win. Out: (kept, report) -- the usable PROBES entries, and a row per
    probe with its pass count for the output file. An empty `kept` is a real
    answer: this model cannot be measured with these probes.
    """
    kept, report = [], []
    print(f"calibration ({runs} cold runs per probe):")
    for entry in PROBES:
        probe = entry["per_ticket"](1) if "per_ticket" in entry else entry
        passes, whys = 0, []
        for _ in range(runs):
            cold = [{"role": "user", "content": BRIEFING},
                    {"role": "assistant", "content": "Understood. " + END_TOKEN}]
            if "per_ticket" in entry:
                cold += [{"role": "user", "content":
                          "Summarise the following in ONE line.\n\n"
                          + long_corpus(1000, offset=37_501) + "\n\n" + ticket_note(1)},
                         {"role": "assistant", "content": "Noted. " + END_TOKEN}]
            text, _, _, finish = post_chat(
                base, model, cold + [{"role": "user", "content": probe["ask"]}],
                max_tokens=max_tokens)
            if finish == "length":
                whys.append("truncated")
                continue
            if probe["check"](text):
                passes += 1
            else:
                # Both ends: the format probe fails on the LAST line, and a
                # head-only example hid exactly that on Qwen3-Coder-30B.
                whys.append(_excerpt(text, 120).replace("\n", " "))
        ok = passes == runs
        if ok:
            kept.append(entry)
        report.append({"probe": probe["key"], "cold_passes": passes,
                       "of": runs, "usable": ok, "examples": whys[:2]})
        print(f"  {probe['key']:7} {passes}/{runs} cold  "
              f"{'USABLE' if ok else 'DROPPED'}"
              f"{'' if ok else '  e.g. ' + (whys[0] if whys else '')}")
    if not kept:
        print("  no probe survived calibration — nothing here can measure decay")
    print()
    return kept, report


def measure_chars_per_token(base, model, pad_chars):
    """Chars of filler per prompt token, measured on this model's tokenizer.

    Why measured rather than assumed: the control arm has to match the
    conversational arm on PROMPT TOKENS, and a hardcoded 4-chars-per-token
    guess is wrong by enough to defeat the comparison. The first sweep padded
    the control by turn index instead, which put the two arms at 19,512 and
    31,869 tokens at the same nominal "turn 20" -- so it compared two prompt
    lengths and called the difference turn count [OBSERVED 2026-10-14].

    Two one-token calls: the bare briefing, then the briefing plus one filler
    chunk. The difference is what the filler costs.

    In: transport details and the filler chunk size. Out: (chars_per_token,
    base_tokens) -- base_tokens is what the briefing alone costs, so the
    caller can solve for the filler it needs. Falls back to (4.0, 0) if the
    server reports no usage, which keeps the sweep running on a device that
    does not count tokens (the NPU).
    """
    bare = [{"role": "user", "content": BRIEFING},
            {"role": "assistant", "content": "Understood. " + END_TOKEN},
            {"role": "user", "content": PROBES[0]["ask"]}]
    _, t_bare, _, _ = post_chat(base, model, bare, max_tokens=1)
    chunk = long_corpus(pad_chars, offset=11)
    padded = [{"role": "user", "content": BRIEFING + "\n\nReference material:\n\n" + chunk},
              bare[1], bare[2]]
    _, t_padded, _, _ = post_chat(base, model, padded, max_tokens=1)
    if t_bare <= 0 or t_padded <= t_bare:
        return 4.0, max(t_bare, 0)
    return pad_chars / (t_padded - t_bare), t_bare


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
    ap.add_argument("--calibrate", type=int, default=3,
                    help="Cold runs each probe must win before it is used "
                         "(default 3). A probe the model fails cold has no "
                         "headroom to decay; 0 skips the check and makes any "
                         "result arguable.")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Seconds to wait per request (default 1800). Raise "
                         "it on CPU: a 42k-token cold prefill there takes "
                         "longer, and a timed-out probe is recorded but not "
                         "scored. The server keeps prefilling after the "
                         "client gives up, so the NEXT probe may then look "
                         "fast -- it is reading that cached prefix.")
    ap.add_argument("--out", default="bench/decay-probe.json")
    args = ap.parse_args()
    global REQUEST_TIMEOUT
    REQUEST_TIMEOUT = args.timeout

    base = args.base.rstrip("/")
    checkpoints = sorted({int(x) for x in args.checkpoints.split(",") if x.strip()})
    model = first_model(base)
    print(f"model: {model}\nserver: {base}\ncheckpoints: {checkpoints}\n")

    try:
        with urllib.request.urlopen(base + "/health", timeout=30) as r:
            health = json.load(r)
    except Exception:
        health = None  # not NoLlama, or too old to serve provenance
    prov = collect_provenance(health)
    srv = prov.get("server") or {}
    print(f"provenance: openvino {srv.get('openvino')}, genai "
          f"{srv.get('openvino_genai')}, gpu driver {srv.get('gpu_driver')}\n")

    results = []
    meta = {"provenance": prov, "params": vars(args), "calibration": []}
    probes = PROBES
    if args.calibrate > 0:
        probes, meta["calibration"] = calibrate(base, model, args.max_tokens,
                                                args.calibrate)
        flush(args.out, model, base, results, meta)
        if not probes:
            return 0

    targets = {}  # checkpoint -> the prompt size the control arm must match

    # --- conversational arm: one history, growing -------------------------
    print("conversational arm (history grows, turn count grows):")
    messages = [{"role": "user", "content": BRIEFING}]
    ack, tokens, _, _ = post_chat(base, model, messages, max_tokens=args.pad_tokens)
    messages.append({"role": "assistant", "content": ack})

    for turn in range(1, max(checkpoints) + 1):
        if turn in checkpoints:
            run_probes(base, model, messages, "conversation", turn, tokens,
                       results, args.max_tokens, args.out, probes=probes,
                       meta=meta, ticket=ticket_for(turn))
            seen = [r["prompt_tokens"] for r in results
                    if r["arm"] == "conversation" and r["turn"] == turn]
            if seen:
                targets[turn] = max(seen)
        pad = long_corpus(args.pad_chars, offset=turn * 37_501)
        messages.append({"role": "user", "content":
                         f"Summarise the following in ONE line.\n\n{pad}\n\n"
                         f"{ticket_note(turn)}"})
        reply, tokens, _, _ = post_chat(base, model, messages,
                                        max_tokens=args.pad_tokens)
        messages.append({"role": "assistant", "content": reply})

    # --- control arm: same size, one turn ---------------------------------
    if not args.no_control:
        print("\ncontrol arm (same prompt size, no conversational history):")
        cpt, base_tokens = measure_chars_per_token(base, model, args.pad_chars)
        print(f"  (matched on tokens: {cpt:.2f} chars/token measured, "
              f"briefing costs {base_tokens})")
        for turn in checkpoints:
            # Match the conversational arm's MEASURED prompt size, not its turn
            # index. Padding by index put the arms at 19,512 and 31,869 tokens
            # at the same nominal turn 20, which compares two prompt lengths
            # and calls the difference turn count. See T-047.
            target = targets.get(turn)
            need = int(max(0, target - base_tokens) * cpt) if target else 0
            bulk = long_corpus(need, offset=37_501) if need > 0 else ""
            # The same ticket notes the conversation planted, one per
            # pad-sized piece, so the `ticket` probe asks the control arm the
            # same question at roughly the same depth with the same
            # look-alikes around it. Only the turn structure differs.
            pieces = [bulk[i:i + args.pad_chars]
                      for i in range(0, len(bulk), args.pad_chars)]
            bulk = "\n\n".join(p + "\n\n" + ticket_note(i + 1)
                               for i, p in enumerate(pieces))
            ticket = ticket_for(turn)
            if ticket is not None and ticket > len(pieces):
                ticket = None  # filler too short to hold that note
            cold = [{"role": "user", "content":
                     BRIEFING + ("\n\nReference material:\n\n" + bulk if bulk else "")},
                    {"role": "assistant", "content": "Understood. " + END_TOKEN}]
            run_probes(base, model, cold, "control", turn, 0, results,
                       args.max_tokens, args.out, probes=probes, target=target,
                       meta=meta, ticket=ticket)

    flush(args.out, model, base, results, meta)
    print(f"\nSaved: {args.out}")

    print("\nsummary (mechanical pass rate):")
    for arm in ("conversation", "control"):
        rows = [r for r in results if r["arm"] == arm]
        if not rows:
            continue
        for turn in checkpoints:
            at = [r for r in rows if r["turn"] == turn]
            if at:
                scorable = [r for r in at
                            if not r["truncated"] and r["finish"] != "error"]
                ok = sum(1 for r in scorable if r["mechanical"])
                cut = sum(1 for r in at if r["truncated"])
                err = sum(1 for r in at if r["finish"] == "error")
                note = "".join([f"  ({cut} truncated, not scored)" if cut else "",
                                f"  ({err} failed requests, not scored)" if err else ""])
                print(f"  {arm:12} turn {turn:>3}  {ok}/{len(scorable)}  "
                      f"{at[0]['prompt_tokens']:>7} tok{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
