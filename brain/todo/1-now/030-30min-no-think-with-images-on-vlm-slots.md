# T-030 — No-think on a VLM slot with an image in the turn

**Effort:** 30 min · **Produces:** one guard removed in `render_nothink_prompt`,
or a recorded reason it stays · **Area:** `nollama.py`

T-015 closed on 2026-09-18 for **text-only** turns: the handler renders the
chat template itself with `enable_thinking=false` and passes the string with
`apply_chat_template=False`. Image turns were left on the old prose-only
behaviour because nobody has checked that `<ov_genai_image_N>` tags in a
pre-rendered prompt still reach the embedder. [INFERRED] they should — the
tag scan runs on the prompt string either way — but it is untested, and the
guard `if not self._nothink_ok or images: return None` says so.

- [ ] one image + the web UI's no-think checkbox against a VLM slot
      (Qwen3.8-27B or Qwen2.5-VL-3B), streaming and blocking
- [ ] confirm the image is *seen* (ask what is in it), not just that the
      request succeeds — a dropped tag fails silently as a text-only answer
- [ ] then drop the `images` guard and the `[INFERRED]` note, or record the
      failure in `TODONT.md`

## Done when

A VLM turn with an image honours the toggle, or `render_nothink_prompt`'s
docstring says exactly why it cannot.
