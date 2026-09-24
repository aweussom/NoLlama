# `--tool-template native` as the default for non-Coder models

**Noticed:** 2026-09-24 · **Status:** proposed, not trusted

## Claim

Rendering tool turns in the model's own template helps some models and has
hurt none tried: Qwen3-14B's fix task went from 18 min plus a syntax error
(Qwen-XML) to a clean one-line edit in 3 of 4 runs (native). LFM2.5-8B-A1B
and Qwen3-8B failed either way. Qwen3-Coder-30B-A3B, the one passing model,
went **4/4 native** on the B60, every diff minimal and correct against the
original tests, 27-73 s a task (2026-09-24, `bench/coder30b-native-b60.log`).

## Why not trusted yet

- One model improved, and that one is still not an agent, so no verdict
  flipped. The case for switching is "no worse, sometimes better", on four
  models and two devices.
- Qwen3-Coder's 4/4 native has no same-day Qwen-XML control on the B60 under
  the stricter verifier; its XML passes predate it.
- Scope today: OpenAI endpoint, genai LLM slots. VLM, optimum and the Ollama
  surface still speak Qwen-XML.

## Also seen

A B60 feature-task turn produced 382 tokens with no parsed call and no
content, and OpenCode ended the run. Either the turn was all `<think>`, or a
call was written *inside* `<think>`, where `_ToolCallGate` never looks. The
server keeps no raw output, so which one is unknown. Capturing the raw text of
tool turns (behind `--debug`) would answer it.

## Suggested home

A structural change, so `TODONT.md` first, then a Qwen3-Coder native run on
the B60, then `docs/dev/tool-calling.md`.
