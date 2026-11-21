# T-058 — Anthropic `/v1/messages` endpoint, so Claude Code can drive NoLlama

**Effort:** 2 d · **Produces:** `/v1/messages` (non-streaming and SSE) on the OpenAI port, an entry in
`docs/API.md`, a pass on the Claude Code agent probe · **Area:** `nollama.py` API layer, tool-call parsing

NoLlama speaks OpenAI and Ollama. Claude Code speaks only Anthropic Messages, to one `ANTHROPIC_BASE_URL`. With
this endpoint, a laptop iGPU can serve as Claude Code's subagent host, or as the whole model when the laptop is
all a user has. Nobody expects it to be fast. A slow local model beats no model. Prefix caching plus prewarm is
the reason to try it on NoLlama rather than on plain Ollama: every Claude Code subagent starts with the same
~17k-token prefix [OBSERVED 2026-10-29, cloud-and-local-coding-harness Explore test], about 65 s cold at the 140V's
260 tok/s prefill [INFERRED], and the cached-prefix TTFT here is 1.9 s (docs/BENCHMARKS.md).

**Checked:** `nollama.py` routes (no `/v1/messages`, nothing Anthropic); `git log --all` (only Co-Authored-By
hits); `brain/proposed/2026-05-26-claude-code-cli-as-an-ollama-backend.md` is the opposite direction and does
not overlap. Requester side: `cloud-and-local-coding-harness` T-006.

## Where Anthropic's tool calling differs from OpenAI's

The traps, each one a way a naive OpenAI-to-Anthropic mapping breaks Claude Code
[DOCUMENTED platform.claude.com/docs/en/build-with-claude/streaming for the stream; the rest GUESS from memory of
the Messages API, verify against platform.claude.com/docs/en/api/messages before coding]:

- **Content is a list of typed blocks, never a plain string.** One assistant turn can hold `thinking`, `text` and
  several `tool_use` blocks, interleaved. Incoming user content can be a string OR a block list; accept both.
- **`tool_use.input` is a JSON object.** OpenAI's `arguments` is a JSON string. Parse it here; never emit a string.
- **`tool_result` arrives in a `user` message**, as blocks with `tool_use_id`, optional `is_error`, and content
  that is a string or a list of text/image blocks. Several results share one user message, often with a text
  block after them. OpenAI wants one `role: tool` message each. Chat templates differ again (Qwen wraps results
  in `<tool_response>`).
- **`system` is a top-level field**, a string or a list of text blocks, not a message.
- **Tools are `{name, description, input_schema}`**, no `function` wrapper.
- **`cache_control` appears on system, tools and messages.** Ignore it, never reject it. Our prefix cache does
  the work regardless, as long as the rendered prompt prefix is byte-stable across turns: check that, because a
  template that re-renders history differently each turn kills the cache.
- **`thinking`:** Claude Code may send a `thinking` parameter and replays earlier `thinking` blocks (with a
  `signature`) in history. Accept them, keep or drop per template, never 400 on them. Qwen3.x thinking output
  maps to a `thinking` block.
- **`stop_reason`** must be `tool_use` when the turn ends in a tool call, else `end_turn` or `max_tokens`. Claude
  Code decides whether to run tools from this, not from the blocks.
- **`usage`:** `input_tokens`, `output_tokens`, and `cache_read_input_tokens`. Report real prefix-cache hits in
  the last one: then the probe measures our cache for free.
- **Streaming events**, in order: `message_start` → per block `content_block_start` (a `tool_use` block starts
  with `id`, `name`, `input: {}`) → `content_block_delta` (`text_delta`, `thinking_delta`, or `input_json_delta`
  with `partial_json`) → `content_block_stop` → `message_delta` (`stop_reason`, usage) → `message_stop`. Any
  number of `ping` events; errors as `event: error`. Each SSE event carries `event: <type>` AND the same `type`
  in its data. Our parser only knows a tool call once it is complete, so emit its whole `input` as ONE
  `input_json_delta`; that is valid.
- Auth arrives as `x-api-key` or `Authorization: Bearer` (`ANTHROPIC_AUTH_TOKEN`); accept and ignore both.
  `anthropic-version` header: ignore.
- `tool_choice`, `/v1/messages/count_tokens`, `metadata`: Ollama does not support them either
  [DOCUMENTED docs.ollama.com/api/anthropic-compatibility], and Claude Code 2.1.312 ran 15/15 probe tasks
  against Ollama anyway [OBSERVED 2026-10-29]. Skip them; return a clear 4xx if they turn out to matter.

## Jump start: who already does this

- **Ollama** (Go, open source): the behaviour Claude Code is known to accept; feature list at
  docs.ollama.com/api/anthropic-compatibility. Source location not found yet [GUESS: an Anthropic package in
  ollama/ollama]. Best used as a differential oracle: same request to Ollama and to NoLlama, diff the block
  structure and the event sequence (doctrine D3: share test data, never code).
- **LiteLLM** `litellm/adapters/anthropic_adapter.py` (Python): Anthropic request → OpenAI and back, including
  `tool_use` → `tool_calls` with JSON-serialized input. Seen on a mirror (git.kvant.cloud/phoenix/litellm-mirror);
  check the path in upstream BerriAI/litellm.
- **vllm-mlx** (github.com/waybarrios/vllm-mlx) and **Rapid-MLX** (github.com/raullenchai/Rapid-MLX): Python
  servers that speak Anthropic natively. Rapid-MLX claims to verify Claude Code against itself every release:
  read their test fixtures for the edge cases they hit.
- **claude-code-router**: a proxy translating Anthropic to OpenAI-compatible backends, so the streaming
  translation exists there. Not read yet.

## Verification tools already built

Both in `..\cloud-and-local-coding-harness` (github.com/aweussom/cloud-and-local-coding-harness). Both take a base
URL and a model id, so they point at NoLlama unchanged. `probe-agent.ps1` is a port of our own
`scripts/agent-probe.ps1` to `claude -p`, so the verdict means the same thing here as it does for OpenCode.

```powershell
cd ..\cloud-and-local-coding-harness

# Raw endpoint: N tool calls, each must come back as a tool_use block with stop_reason=tool_use
.\probe-toolcall.ps1 -BaseUrl http://127.0.0.1:8000 -Model <nollama-model-id> -Runs 5

# Claude Code itself: fix, feature, delegate, hunt, rename; verdict from running the tests
.\probe-agent.ps1 -BaseUrl http://127.0.0.1:8000 -Config N-nollama -MainModel <id> -SubagentModel <id>
```

`probe-agent.ps1` writes one JSONL row per task to that repo's `results/`, with tool calls, permission
denials and tokens per model, plus transcripts and diffs. `results/agent-probe-*.jsonl` there already holds
the same five tasks against Ollama (cloud, hybrid, qwen3.8:27b on an RTX 5090) [OBSERVED 2026-10-29]: the
baseline to compare against. Its Ollama-quota column (only with `OLLAMA_API_KEY` set) should read 0 against NoLlama; anything else means
some other session spent Ollama Cloud quota during the run.

## How to run it

- [ ] Translate at the edge: Anthropic request → the internal chat/tools shape the OpenAI route already uses →
      Anthropic response. One parser for tool calls, no second one.
- [ ] Non-streaming first; `probe-toolcall.ps1` above, 5/5.
- [ ] SSE second; diff the event sequence against Ollama's for the same request.
- [ ] Replay a captured Claude Code subagent request twice: cold, then warm. Record TTFT and
      `cache_read_input_tokens`.
- [ ] `probe-agent.ps1` from that repo against NoLlama, Qwen3-Coder-30B-A3B on the B60, then on the 140V.
- [ ] Record the driver with every measurement.

## Done when

`probe-agent.ps1` passes 5/5 against NoLlama on the B60, warm subagent TTFT is measured on the 140V, and
`docs/API.md` lists the endpoint and what it does not support.
