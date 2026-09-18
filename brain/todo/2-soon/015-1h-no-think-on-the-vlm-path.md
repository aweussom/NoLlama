# T-015 — `enable_thinking` is wired for LLM, still prose for VLM

**Effort:** 1 h · **Produces:** `extra_context` on the VLM generate paths ·
**Area:** `nollama.py`

**The LLM half landed 2026-09-13.** `_apply_thinking_switch` sets
`enable_thinking=false` via `ChatHistory.set_extra_context` in both
`generate_llm` and `stream_llm` when the UI's prose marker is present.

The VLM path still passes a **flattened string with no `extra_context` hook**,
so "Just answer me, dammit!" on a VLM slot is still a sentence in the prompt
asking politely rather than the model's own switch.

- [ ] find the VLM equivalent of the `set_extra_context` seam, or establish
      that `VLMPipeline` has none and record that instead
- [ ] if none exists, say so in `docs/dev/tool-calling.md` and leave the prose
      marker as the documented behaviour rather than a silent no-op

Why it matters beyond tidiness: a thinking model that cannot be switched off
can spend an entire side-request budget in `<think>` and return empty content —
that is exactly how SmolLM3-3B lost the small-model slot (3/3 empty on NPU 4,
and the same on CPU once bare genai was asked).

## Done when

A VLM slot honours the toggle, or `docs/dev/tool-calling.md` records why it
cannot and the UI stops implying it does.
