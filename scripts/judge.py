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

Transport is Ollama Cloud (OpenAI-compatible, https://ollama.com/v1), reusing
the account secondreader already uses. The key comes from OLLAMA_API_KEY, the
same environment fallback llm.py documents.

    from judge import judge
    verdict = judge(prompt, completion)
"""
import json
import os
import urllib.error
import urllib.request

ENDPOINT = "https://ollama.com/v1/chat/completions"
# glm-5.2 first: 976k context and secondreader's table rates it the best
# non-MiMo model it has measured. deepseek-v4-flash is the fallback — both are
# flat-rate on this account. nemotron-3-ultra is deliberately not used: the
# same table records it at 13 minutes a call.
MODELS = ("glm-5.2", "deepseek-v4-flash:0731-cloud")
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


def _call(model, messages, timeout=240):
    """One OpenAI-compatible chat call against Ollama Cloud.

    Why hand-rolled urllib rather than the openai package: this script is run
    from NoLlama's venv, which has no reason to carry an SDK for one endpoint.

    In: a model id and the message list. Out: the assistant's content string.
    Raises on transport failure so the caller can try the next model.
    """
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": MAX_TOKENS, "temperature": 0}).encode()
    key = os.environ.get("OLLAMA_API_KEY", "")
    if not key:
        raise RuntimeError("OLLAMA_API_KEY is not set")
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or ""


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

    messages = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content":
            f"PROMPT GIVEN TO THE MODEL:\n{prompt}\n\n"
            f"OUTPUT TO GRADE ({len(completion)} chars total):\n{body}"},
    ]
    last = ""
    for model in MODELS:
        try:
            reply = _call(model, messages)
            got = _parse(reply)
            if got and "verdict" in got:
                got["judged_by"] = model
                return got
            last = f"{model}: unparseable reply {reply[:80]!r}"
        except Exception as e:
            last = f"{model}: {type(e).__name__} {str(e)[:80]}"
    return {"verdict": "unavailable", "confidence": 0.0, "why": last,
            "judged_by": None}
