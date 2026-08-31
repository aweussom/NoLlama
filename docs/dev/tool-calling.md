# Tool calling / agent turns

Read this before touching `tools` handling, `render_tools_prompt`,
`parse_tool_calls`, or the SSE heartbeat.

## Where it works

**GPU/iGPU + CPU**, on both LLM and VLM slots. VLM tool turns — including
images alongside tools — landed 2026-08-18; they are buffered like LLM tool
turns and generated through the VLM pipeline.

Gated by `_tools_supported`, i.e. `device_name in ("GPU","CPU")`. The
**NPU is excluded**: it has a hard prompt cap and small NPU-class models
can't drive agent loops, so when the NPU serves the request we ignore
`tools` and answer as plain chat. `/api/show` advertises the `tools`
capability only for GPU/CPU slots, so Copilot won't offer NPU models for
agent mode.

CPU is viable for agents on strong desktops (e.g. Core Ultra 9, many
cores) where prefill can beat a weak iGPU.

## How a turn works

Tool specs from the request's `tools` array are rendered into a system
prompt (Qwen3-Coder native format); the model's emitted call is parsed back
into OpenAI/Ollama `tool_calls`.

`parse_tool_calls` recognizes several native formats, because a model often
ignores our prompt and falls back to whatever it was trained on:

- Qwen3-Coder XML
- Hermes JSON-in-`<tool_call>`
- bare `<function=>` with no wrapper (Qwen2.5-Coder native)
- Mistral `[TOOL_CALLS]`
- Llama `<|python_tag|>`
- DeepSeek `<｜tool▁calls▁begin｜>` blocks
- bare-JSON fallback

See `render_tools_prompt` / `parse_tool_calls`.

Client surfaces: Copilot Chat 0.53+, OpenCode and Zed hit
`/v1/chat/completions` (delegates to `chat_completions`); `/api/chat` is
also handled (pre-0.53 Copilot, Open WebUI).

## Token-streamed, gated at the tool-call opener (since 2026-08-30)

Tool-enabled turns on `/v1/chat/completions` **stream**. `_sse_tool_stream`
consumes the token seam (`stream_tokens`, or `stream_vlm_tokens` on a VLM
slot) and runs each chunk through two filters:

- `_ThinkSplitter` — `<think>…</think>` spans go out as
  `delta.reasoning_content`, everything else as `delta.content`. Tags can
  straddle chunks, so a possible tag prefix is held for one chunk. Some
  templates (Qwen3.5, Qwen3.8) open the block *in the prompt*, so the model
  emits only `</think>`; `_prompt_preseeds_think` renders one dummy turn
  through the loaded pipeline's tokenizer at load time and the splitter then
  starts inside a block (`slot.think_preseeded`). It must use the pipeline's
  own tokenizer — a fresh `ovg.Tokenizer` on an IR newer than the runtime
  segfaults (2026.3.0 + Qwen3.8 main-branch IR, 2026-08-30).
- `_ToolCallGate` — answer text passes through until the first opener of
  any syntax `parse_tool_calls` understands (`<tool_call>`, `<function=`,
  `<atem:function_calls>`, `[TOOL_CALLS]`, `<|python_tag|>`, the DeepSeek
  marker); from there the turn is held. An answer whose first non-blank
  character is `{`/`[` is held whole (the bare-JSON fallback has no opener).

When generation ends, `parse_tool_calls(gate.held, tools)` produces the
structured `tool_calls` deltas and `finish_reason: "tool_calls"`; an opener
that never became a call is released as content. Only the held text can
contain a call — nothing past an opener was ever released — so the
pre-opener text is never re-sent.

Why it was buffered before, and why that hurt: a structured `tool_calls`
delta needs the whole block parsed, and the simple way to guarantee that was
to buffer the turn. But agent clients send `tools` on **every** request, so
they never saw a live token — OpenCode showed a "Thinking" spinner until the
turn ended (issue #36). Reasoning also has to travel as `reasoning_content`
for those clients to render it live (OpenCode decodes exactly that field,
anomalyco/opencode#35283); the web UI folds the field back into `<think>`
tags (`appendDelta` in `app.js`) so its renderer is unchanged.

`--think-in-content` restores the old wire shape (tags inside `content`) for
a client that turns out to depend on it. The **Ollama surface is
unchanged**: `/api/chat` keeps `<think>` in content and still emits a
tool-enabled turn as one buffered ndjson line.

Two layers of test, and they cover different halves. `scripts/agent-loop-test.py`
runs the **round trip against real weights**: stream a turn that ends in
`tool_calls`, send the conversation back with a tool-call-only assistant
message (`content: null`) plus a `tool` result, stream the second turn, and
check the model actually used the result. That second-turn history is the
shape that broke Zed (#24) and the one `prepare_messages_for_tools` re-renders
every turn — nothing else exercises it. Point it at a running GPU/CPU server:
`python scripts/agent-loop-test.py http://127.0.0.1:8000` (verified 2026-08-31,
Qwen3-8B on an Arc 140V).

Tests: `tests/test_stream_tools.py` drives the real `_sse_tool_stream` with
a fake seam (fuzzed chunkings, keep-alives, false alarms, bare JSON, errors,
legacy flag) — no model needed.

## Heartbeat — why it exists

A slow prefill on a big agent prompt trips client idle watchdogs
(Copilot/OpenClaw abort with no output after ~120s). So:

- every SSE consumer (`_sse_stream` behind `stream_llm`/`stream_vlm`, and
  `_sse_tool_stream`) turns the seam's `None` marker — `HEARTBEAT_SECS` of
  silence — into an empty-content delta, which resets content- and
  byte-based client watchdogs alike and is a no-op for message assembly.

Big agent prompts (OpenClaw ships ~21k-token system prompts) prefill slowly
on weak iGPUs — ~6 min TTFT on the desktop 285K Xe-LPG. Mitigations: a
smaller coder model, CPU on strong desktops, trimming the client's tool
set, and the keep-alive above so turns complete instead of aborting.

OpenVINO **can't cancel a blocked prefill**, so an aborted client leaves
the generation churning — another reason to keep clients connected via
heartbeat. (Same root cause as the `/v1/cancel` caveat: cancel relies on
OpenVINO invoking the streamer callback.)

Note the "minutes of prefill" worry does not generalize: on the B60 class,
33k tokens prefill in ~9s on the plain pipeline.
