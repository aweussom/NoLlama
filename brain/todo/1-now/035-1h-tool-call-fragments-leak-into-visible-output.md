# T-035 — Bare `<tool_call>` and token fragments reach the user's screen

**Effort:** 1 h · **Produces:** a fix in `_ToolCallGate` / `parse_tool_calls` ·
**Area:** `nollama.py`

[OBSERVED 2026-09-23, 140V, `Qwen3-14B-int4` under OpenCode 1.18.30] A real
session showed this in the *rendered assistant output*, between two tool calls:

```
→ Read calc.py

<tool_call>

<tool_call>

← Edit calc.py
```

and later a bare `TokenName` on its own line.

`<tool_call>` is an opener the gate knows (`_ToolCallGate._OPENERS`), so it
should never be visible. Two candidate causes, and the first job is to tell
them apart:

- the model emitted an EMPTY `<tool_call>` with no JSON, `parse_tool_calls`
  found nothing to parse, and the gate released the held text verbatim — the
  designed fallback, but it prints markup at the user
- the opener arrived split across chunks in a way the one-chunk lookahead does
  not cover

- [ ] reproduce from a captured stream rather than a live model (the transcript
      above is enough to build the fixture)
- [ ] decide what an unparseable opener should render as — dropping it silently
      hides a real model failure, so prefer a marker the user can report
- [ ] a `tests/test_stream_tools.py` case, since that file already owns the gate

Cosmetic, but it is the kind of thing a user screenshots into an issue, and it
makes a working model look broken.
