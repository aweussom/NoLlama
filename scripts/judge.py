r"""Ask a capable model whether a completion is healthy. LLM-as-judge.

Why: "is this output degenerate?" is a judgement, and hand-tuned heuristics
approximate it badly. On 2026-09-13 three separate thresholds in
model-test.py were miscalibrated in one afternoon — a token cap set from one
prompt and applied to a longer one (twice), and a unique-word ratio that
cannot tell Python from looping, because real code legitimately repeats `def`,
`return` and `wrapper`. A reader would not make any of those mistakes.

It also sees failures no heuristic here covers: one Phi-3.5-mini run switched
to Norwegian mid-answer, and another answered the previous turn's question
using this turn's research. Nothing mechanical flagged either.

NOT a replacement for the heuristics. They stay as the cheap local pass with
no network and no key; this is the deep pass for our own sweeps, and the
INTERESTING signal is where the two disagree — a heuristic gap in one
direction, a false positive in the other.

Transport is the **local Ollama daemon** first, which proxies `:cloud` models
using the account the daemon is already signed in to — so a run needs no
OLLAMA_API_KEY on any box where `ollama list` shows the cloud tags [OBSERVED
2026-10-14, 285K: glm-5.3-flash 0.9 s, glm-5.2 2.2 s, keyless]. Ollama Cloud
over https://ollama.com/v1 is the fallback, and that one does need the key.

Use `ask()` for a different rubric against the same transport; `judge()` is
the health rubric built on it.

    from judge import judge
    verdict = judge(prompt, completion)
"""
import json
import os
import sys
import urllib.error
import urllib.request

# OLLAMA_HOST is Ollama's own variable, so setting it points both this and the
# `ollama` CLI at the same daemon. It earns its keep across boxes: the B60 runs
# no Ollama at all, so a sweep there borrows the 285K's daemon over Tailscale
# (`OLLAMA_HOST=http://100.98.33.88:11434`), which answers because that box
# binds more than loopback [OBSERVED 2026-10-14, Ollama 0.40.1].
#
# `127.0.0.1`, never `localhost`, for the default: on the B60 box localhost
# resolves to ::1 first and the refused IPv6 connect costs a flat 2 s per call
# [OBSERVED 2026-10-07, docs/dev/machines.md].
LOCAL_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not LOCAL_HOST.startswith("http"):
    LOCAL_HOST = "http://" + LOCAL_HOST  # Ollama accepts a bare host:port
LOCAL_ENDPOINT = LOCAL_HOST + "/v1/chat/completions"
CLOUD_ENDPOINT = "https://ollama.com/v1/chat/completions"

# None until probed once: whether LOCAL_HOST is a real Ollama daemon.
_LOCAL_OK = None
#
# deepseek-v4.1-flash first: fastest measured and, per Tommy, not heavily
# loaded — which matters because a decay sweep makes dozens of calls, not one.
# glm-5.3-flash second, rated the better judge of the two. Round trips on the
# 285K, keyless through the local daemon [OBSERVED 2026-10-14]:
#   deepseek-v4.1-flash 0.4 s · glm-5.3-flash 0.9-1.7 s · glm-5.2 2.2 s
#
# The fallback this replaced, deepseek-v4-flash:0731, had been **retired since
# 2026-09-25** (HTTP 410) and nothing noticed, because a fallback is only
# exercised on the day the primary fails. Re-check these against `ollama list`
# before trusting a sweep.
# nemotron-3-ultra stays out: secondreader's table records it at 13 min a call.
MODELS = ("deepseek-v4.1-flash:cloud", "glm-5.3-flash:cloud")
# Generous on purpose. BOTH candidates are thinking models and return EMPTY
# content when the budget runs out mid-reasoning — measured 2026-09-13, glm-5.2
# spent all 16 tokens of a small cap and answered nothing. That is the very
# failure this judge exists to detect, and it would have silently judged
# everything as unparseable.
MAX_TOKENS = 1200

RUBRIC = """You are grading the OUTPUT of a small local language model. Decide \
whether the output is HEALTHY or FAULTY.

FAULTY means one of:
  "empty"      - no substantive answer at all
  "runaway"    - never stops; decays into word salad, concatenated words, \
alphabetical or alliterative marches, or a wall of repeated characters
  "looping"    - repeats the same content over and over
  "off_topic"  - a fluent answer to a DIFFERENT question than the one asked
  "wrong_lang" - switches away from the language of the prompt mid-answer

HEALTHY ("ok") means it answers the prompt and stops sensibly. IMPORTANT:
  - Code that repeats `def`, `return`, `print` or a variable name is NORMAL, \
not looping.
  - An answer cut off mid-sentence because it ran out of budget is \
"truncated", NOT faulty - only say truncated if the text is otherwise coherent.
  - Long is not faulty. Judge the TEXT, not the length.

Reply with ONLY a JSON object, no prose, no code fence:
{"verdict": "ok"|"empty"|"runaway"|"looping"|"off_topic"|"wrong_lang"|\
"truncated", "confidence": 0.0-1.0, "why": "one short sentence"}"""


def _is_nollama(host, timeout=5):
    """True when `host` is a NoLlama server wearing Ollama's clothes.

    Why this check exists at all: NoLlama serves its Ollama-compatible surface
    on 11434, so on any box running it the judge's default endpoint is the
    server under test. The judge would hand the model its own output to grade
    and report a verdict with nothing in the output saying so -- a silently
    wrong answer, which is worse than a failed call. The B60 answered
    `{"version": "nollama-2026-09-18-72d0fc9"}` on that port [OBSERVED
    2026-10-14].

    In: a base URL. Out: True only on a positive identification; False on any
    error, because an unreachable host is the caller's problem to report, not
    this function's to guess at.
    """
    try:
        with urllib.request.urlopen(host + "/api/version", timeout=timeout) as r:
            return "nollama" in (json.load(r).get("version") or "").lower()
    except Exception:
        return False


def _call(model, messages, timeout=240, max_tokens=None):
    """One OpenAI-compatible chat call: local Ollama daemon, else Ollama Cloud.

    Why local first: the daemon is signed in to the account already, so a
    `:cloud` tag resolves with no OLLAMA_API_KEY anywhere in the environment.
    That removes the one setup step a sweep on a fresh box would otherwise
    need, and it is the configuration every one of these machines is in.

    Why hand-rolled urllib rather than the openai package: this script runs
    from NoLlama's venv, which has no reason to carry an SDK for one endpoint.

    In: a model id, the message list, optional per-call token cap. Out: the
    assistant's content string. Raises after BOTH transports fail, so the
    caller can move to the next model — a local daemon that is simply not
    running must not look like a dead model.
    """
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens or MAX_TOKENS, "temperature": 0}
    body = json.dumps(payload).encode()

    global _LOCAL_OK
    if _LOCAL_OK is None:
        _LOCAL_OK = not _is_nollama(LOCAL_HOST)
        if not _LOCAL_OK:
            print(f"judge: {LOCAL_HOST} is a NoLlama server, not Ollama — "
                  f"refusing to let the model grade itself. Set OLLAMA_HOST to "
                  f"a real Ollama daemon, or export OLLAMA_API_KEY for cloud.",
                  file=sys.stderr)
    if not _LOCAL_OK:
        key = os.environ.get("OLLAMA_API_KEY", "")
        if not key:
            raise RuntimeError(
                f"{LOCAL_HOST} is NoLlama, not Ollama, and OLLAMA_API_KEY is unset")
        return _cloud_call(model, payload, key, timeout)

    try:
        req = urllib.request.Request(
            LOCAL_ENDPOINT, data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        return d["choices"][0]["message"].get("content") or ""
    except Exception as local_err:
        key = os.environ.get("OLLAMA_API_KEY", "")
        if not key:
            raise RuntimeError(
                f"local Ollama failed ({type(local_err).__name__}: "
                f"{str(local_err)[:120]}) and OLLAMA_API_KEY is not set")

    return _cloud_call(model, payload, key, timeout)


def _cloud_call(model, payload, key, timeout):
    """The Ollama Cloud leg of `_call`, split out so both paths reach it.

    Why separate: the NoLlama guard above has to skip the local leg entirely
    rather than fail into it, and duplicating the request building is how the
    two legs drift apart.

    In: model id, the already-built payload, the API key, timeout. Out: the
    assistant's content string; raises on transport failure.
    """
    # Cloud ids carry no `:cloud` suffix; the local daemon requires it.
    payload = dict(payload)
    payload["model"] = model[:-6] if model.endswith(":cloud") else model
    req = urllib.request.Request(
        CLOUD_ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or ""


def ask(system, user, max_tokens=None):
    """Run any rubric through the judge transport and parse its JSON verdict.

    Why this is public and `judge()` is built on it: a second rubric (is this
    answer still CORRECT, as against is it degenerate) needs the same model
    list, the same local-first transport and the same tolerant parse, and a
    copy of that plumbing is a copy that drifts. `decay_probe.py` is the
    first other caller.

    In: the rubric as a system prompt, the material as a user prompt. Out:
    the parsed dict with `judged_by` added, or None when every model failed
    or replied unparseably — None means NO JUDGEMENT, never "fine".
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    last = ""
    for model in MODELS:
        try:
            reply = _call(model, messages, max_tokens=max_tokens)
            got = _parse(reply)
            if got:
                got["judged_by"] = model
                return got
            last = f"{model}: unparseable reply {reply[:80]!r}"
        except Exception as e:
            last = f"{model}: {type(e).__name__} {str(e)[:80]}"
    return {"_error": last} if last else None


def _parse(text):
    """Pull the verdict object out of a reply that may carry extra prose.

    Why not json.loads directly: the rubric asks for bare JSON and the models
    mostly comply, but a thinking model sometimes wraps it in a fence or adds
    a sentence. Being strict here would turn a good judgement into a parse
    error, so we take the first balanced {...} span instead.

    In: the raw reply. Out: the parsed dict, or None when nothing parses —
    None is the caller's signal to try the fallback model rather than to
    trust a guess.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def judge(prompt, completion, max_chars=6000):
    """Grade one completion. Returns a dict, never raises.

    Why the completion is truncated in the MIDDLE rather than the end: decay
    shows up at the END of a runaway answer, so the tail is the evidence. A
    head-only excerpt would hide the exact thing being looked for.

    In: the prompt that produced it and the completion. Out: a dict with
    verdict / confidence / why / judged_by. On total failure the verdict is
    "unavailable" — that must never be mistaken for "ok", so callers should
    treat it as "no judgement" and fall back to the heuristics.
    """
    body = completion
    if len(body) > max_chars:
        head, tail = max_chars // 3, max_chars - max_chars // 3
        body = (body[:head] + f"\n\n...[{len(completion) - max_chars} chars elided]...\n\n"
                + body[-tail:])

    got = ask(RUBRIC,
              f"PROMPT GIVEN TO THE MODEL:\n{prompt}\n\n"
              f"OUTPUT TO GRADE ({len(completion)} chars total):\n{body}")
    if got and "verdict" in got:
        return got
    return {"verdict": "unavailable", "confidence": 0.0,
            "why": (got or {}).get("_error", "no reply"), "judged_by": None}
