# T-019 — `benchmark.py --llm-only` skips text tests on a VLM-only server (#40)

**Effort:** 30 min · **Produces:** a fallback in `benchmark.py` ·
**Area:** `benchmark.py`

It assigns text tests from `/health` `type == "llm"` only, so a **text-capable
model that landed on the VLM slot** (Qwen3.6-35B-A3B) prints "No LLM model
found" and runs nothing.

- [ ] fall back to the VLM slot for text when no LLM slot is ready

Workaround until then: `--model Qwen3.6-35B-A3B@GPU`.

## Done when

`--llm-only` against a VLM-only server runs the text tests instead of
reporting an empty result as success.
