# T-036 — An agent verdict for LFM2.5-8B-A1B, in its own dialect

**Effort:** 2 h · **Produces:** an `agent` verdict for the only model that
fits a 16 GB box, and an answer on whether our Qwen-XML tool rendering biases
every non-Coder verdict · **Area:** `prepare_messages_for_tools`,
`models.json`, `OPENCODE-PLAN.md`

Why it matters: it is the **only candidate that fits a 16 GB machine with a
usable cache**, and that tier has no agent-capable model at all.

What is established [OBSERVED 2026-09-23, 140V, OpenVINO 2026.4]:

| | |
|---|---|
| on disk | 4.2 GB (int4), 32 experts top-4, ~1B active |
| KV | **48 KB/token** — a 3 GB pool holds ~65k tokens |
| load / bare-probe / pelican | 5 s / 6/6 / valid SVG in 18.1 s |
| agent probe after the parser fix (`b950e75`) | **0/2**, but it *does* call tools: 5 calls parsed server-side |
| fix task | ran the tests correctly, then invented `workdir='/workspace/CondaEnvironment/app'`; OpenCode auto-rejected it |
| feature task | 3 globs found the files; next turn said "didn't find any files" and wrote a name-less JSON "tool_calls" blob as prose |

Transcripts: `bench/agent-probe/20260923-221320-LFM2.5-8B-A1B_GPU-*` (local,
gitignored), server side in `bench/lfm2-8b-a1b.log`.

## Why that 0/2 is not yet a verdict

`prepare_messages_for_tools` renders **every** model's tool prompt, its prior
calls and its tool results in **Qwen3-Coder XML** (results as `user` turns in
`<tool_response>`). LFM2.5's template wants `List of tools: [...]`, Pythonic
`<|tool_call_start|>[...]` history and a native `tool` role. From turn 2 on,
the model reads its own history in a foreign dialect — and the feature-turn
failure looks like format confusion. Qwen3-Coder, the one model that passes,
is the one whose dialect we speak. [INFERRED] — the A/B below confirms or kills it.

genai's `Tokenizer.apply_chat_template(..., tools=)` is **not** a faithful
native render for this model: it put tools in the system prompt natively but
rendered history and results as generic JSON, not the jinja's
`<|tool_call_start|>` / `<|im_start|>tool` [OBSERVED 2026-09-23, genai 2026.4].
So the A/B renders with HF `transformers` from `chat_template.jinja`.

- [ ] A/B: the same two probe tasks, native render vs ours, bare genai on the
      140V — a scratch loop is enough, no server change yet
- [ ] if native wins: this is structural — `TODONT.md` check, then a
      per-model native-render path, then **re-probe Qwen3-8B and Qwen3-14B**,
      whose FAIL/PARTIAL came through the same XML
- [ ] if native loses too: LFM2.5 FAILs as it stands; record it in
      `models.json` and say in `docs/MODELS.md` that nothing qualifies at 16 GB

## Done when

LFM2.5 carries a measured verdict from a run in its own dialect, and MODELS.md
says what a 16 GB laptop should install for agentic coding — or that nothing
qualifies.
