# T-048 — `finish_reason` is always "stop", never "length"

**Effort:** 1 h · **Produces:** a fix in `nollama.py` + a test ·
**Area:** `nollama.py` (`_assistant_message`, `stream_tokens`, `_sse_tool_stream`)

A generation that ran out of `max_tokens` is reported to the client as though
the model finished on its own. The OpenAI spec reserves `"length"` for exactly
this, and agent clients branch on it: a tool call or a file edit cut off
mid-write currently arrives labelled "done".

## Repro

Against any LLM slot (B60, `LFM2.5-24B-A2B@GPU`, build `2026-09-18-72d0fc9`,
and the same code is on `main` at 6e9ab31) [OBSERVED 2026-10-14]:

```
stream=False max_tokens=16   -> finish_reason='stop'  completion_tokens=16
stream=False max_tokens=64   -> finish_reason='stop'  completion_tokens=64
stream=True  max_tokens=16   -> finish_reason='stop'
stream=True  max_tokens=64   -> finish_reason='stop'
```

`completion_tokens` lands exactly on the cap every time, so the server has the
evidence and discards it.

## Where

Three sites, all hardcoded:

- `_assistant_message` — `finish = "stop"` in the no-tool-calls branch
- `stream_tokens` — `yield frame({}, "cancelled" if was_cancelled else "stop")`
- `_sse_tool_stream` — the same line again

Each already counts tokens (`token_count`, and `usage.completion_tokens` on
the seam), so the fix is to compare against the request's cap rather than to
plumb anything new. Mind that the two streaming sites are **cross-file twins
in the same file**: fix both, and say in the docstrings why they stay separate.

## Why it is worth an hour

It is not cosmetic. It silently defeats any attempt to tell a truncated answer
from a bad one, which is not hypothetical: it wasted the first two runs of
`scripts/decay-probe.py` on 2026-10-14. A reasoning-only model loops inside an
unclosed `<think>` until the cap, the rescue path in `_assistant_message`
hands that text back as content, and the client is told `stop`. From the
outside it looks exactly like a model that chose to answer with 7,500
characters of repetition.

Agent clients are the real cost. `docs/dev/tool-calling.md` already records
clients that mis-handle turn shape; this hands them a wrong one for free.

## Done when

`max_tokens=16` against a prompt that would run long returns
`finish_reason: "length"` on both the streaming and non-streaming OpenAI
paths, with a test that asserts it, and the Ollama surface left consistent
with whatever it reports today.
