# Coding agents (OpenCode, Goose, VS Code Copilot)

NoLlama drives tool-calling coding agents: the model emits function calls,
NoLlama parses them into OpenAI/Ollama `tool_calls`, and the agent acts on the
results. Any client that speaks the OpenAI chat-completions API works; the
ones people actually run against it are **OpenCode**, **Goose**, **VS Code
Copilot Chat** and **Continue**.

> **Tool calling runs on GPU/iGPU and CPU — not the NPU.** The NPU has a hard
> prompt cap and small NPU-class models can't reliably drive agent loops, so
> NoLlama ignores `tools` there and answers as plain chat; `/api/show` advertises
> the `tools` capability only for GPU/CPU slots. Load a coder LLM on the GPU, or
> on a strong desktop CPU (many-core Core Ultra) where prefill can beat a weak
> iGPU. The NPU still has a job in an agent session — see the two-server recipe
> below.
>
> Tool turns **stream** on `/v1/chat/completions`: reasoning arrives as
> `reasoning_content`, prose as `content`, and only the tool-call block itself
> is held until it can be parsed into structured `tool_calls` (OpenCode and Zed
> show the model thinking live). The server also emits SSE keep-alive pings
> during a long prefill so agent clients don't hit their idle timeout and
> abort. On the Ollama API (`/api/chat`) a tool turn is still one buffered
> reply. Big agent system prompts (OpenCode ~8k tokens, some clients ~20k)
> prefill slowly on weak iGPUs — a smaller model, the CPU, or trimming the
> client's tool set all help. And **prefix caching is on by default**, so that
> big system prompt is prefilled once, not every turn — after the first turn,
> agent turns are fast (~47x on the cached prefix). Disable with
> `--no-prompt-cache`.
>
> **Size the KV pool for your sessions.** With caching on, blocks are never
> released, so a long coding session eventually owns the whole pool and then
> *every* turn misses — TTFT jumps from well under a second to tens of seconds
> and grows linearly with the prompt (82k chars → 58 s, 199k → 175 s on a
> Panther Lake iGPU, issue #32). The auto-sizer caps the pool around 64k
> tokens of the model's KV geometry; agent sessions that carry 50k+ tokens of
> context want more: `--cache-size-gb 12` (16 if you routinely exceed 200k
> chars) on a 64 GB machine. The startup line `prefix caching on (N GB KV
> pool …)` shows what you got; if a *repeated* prompt's TTFT stays under a
> second as the context grows, the pool is big enough. OpenCode's own
> `compaction` settings are the other half: let the client shrink its context
> before it outgrows the pool.

The tool prompt is rendered in Qwen3-Coder native format, and `parse_tool_calls`
also understands Hermes, Mistral `[TOOL_CALLS]`, Llama `<|python_tag|>`, DeepSeek,
and bare-JSON outputs — so most instruct/coder models work.

## OpenCode

**`install.ps1` writes this file for you.** Pick the "Coding agent" use-case
and it writes `opencode.json` for one server, with the model id NoLlama will
actually advertise, an honest context limit, the raised timeouts and the
tool-output caps. The "Chat +
Coding agent" combo writes the dual-mode form instead (one process,
`<name>@GPU` / `<name>@NPU`). Manual installs can run
`scripts/New-OpenCodeConfig.ps1` with the same arguments. What follows is
what that file contains and why.

**You do not have to copy it anywhere.** `.\launch-agent.ps1 -Path <project>`
starts the server if it is down, waits for every slot to finish loading, and
opens OpenCode with `OPENCODE_CONFIG` pointed at the generated file.

That env var **layers** on top of the user's own config rather than replacing
it [OBSERVED 2026-09-13, opencode 1.18.30: with a project `opencode.json`
defining provider `testmarker` and `OPENCODE_CONFIG` pointing at ours,
`opencode models` listed both `testmarker/*` and `nollama/*`], so nothing is
written to the machine and existing providers, agents and keybinds survive.
Copying the file into a project root still works if you prefer it.

It waits for **every** slot, not `/health`'s top-level status, which reports
ready as soon as any one slot is. That is right for a liveness probe and wrong
here: `small_model` traffic is routed to the NPU/CPU slot, so handing over
early sends the first title request at a model that is still loading.

OpenCode speaks the OpenAI API and needs only a provider block in
`opencode.json` (project root). The minimal, one-server form:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "nollama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8000/v1" },
      "models": { "Qwen3-Coder-30B-A3B-Instruct": {} }
    }
  },
  "model": "nollama/Qwen3-Coder-30B-A3B-Instruct"
}
```

The model key is the name NoLlama prints in its banner (`/v1/models` lists
`<name>@<DEVICE>`; either form is accepted). Start the server with
`--idle-timeout 0` so the prefix cache survives between turns, and size the
pool as above.

### Two servers: optional, and no longer the installer's default

OpenCode sends a small request (~2k chars, its `title` agent) beside the
turn, and `small_model` can point it at a second server. **The installer no
longer does this** (2026-09-24): under real OpenCode traffic the side lane
gets one such request per task, and on a single server the title took 0.7 s
on the coder with no queue behind it. The long queue that once justified a
second server was a *thinking* model reasoning about a title, which NoLlama
now prevents. A second server costs a second model's RAM and buys nothing
measurable. Measurements in `OPENCODE-PLAN.md`; the recipe below still works
if you want it.

```powershell
# terminal 1 — the coder, on the GPU
python nollama.py --port 8000 --device GPU --model-dir <coder-model> --idle-timeout 0
# terminal 2 — a small chat model, on the NPU (int4-cw export, see docs/MODELS.md)
python nollama.py --port 8002 --ollama-port 0 --device NPU --model-dir <small-npu-model> --idle-timeout 0
```

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "nollama-gpu": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8000/v1" },
      "models": { "Qwen3-Coder-30B-A3B-Instruct": { "limit": { "context": 120000, "output": 32000 } } }
    },
    "nollama-npu": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8002/v1" },
      "models": { "SmolLM3-3B-int4-cw": { "limit": { "context": 4096, "output": 1024 } } }
    }
  },
  "model": "nollama-gpu/Qwen3-Coder-30B-A3B-Instruct",
  "small_model": "nollama-npu/SmolLM3-3B-int4-cw"
}
```

Verified 2026-09-11 (OpenCode 1.18.30, Core Ultra 9 285K: Qwen3-8B on the
iGPU, SmolLM3-3B on the NPU): the title request hit the NPU server and was
answered in 2.8 s; the turn reached the GPU in the same second and started
prefilling immediately instead of queueing behind it. And 2026-09-12 on real
coder hardware: `Qwen3-Coder-30B-A3B-Instruct-int4` on an Arc Pro B60 with
SmolLM3-3B on the box's CPU standing in for the NPU — a read-and-write
task over a 4,700-line file completed in **52–56 s wall clock**, seven to
eight turns, 13–20 s cold first token and 1–4 s after that, correct answer
(`OPENCODE-PLAN.md`, "First results"). Two processes rather
than NoLlama's dual mode on purpose — each has its own lock, KV pool and
crash domain, so a GPU-side failure (#37, #38) leaves the NPU server up.
Dual mode (`--gpu-model-dir`) gives one port instead, addressed as
`<name>@GPU` / `<name>@NPU`.

The `context` limit on the NPU model matters: the NPU prompt cap is 4096
tokens, and OpenCode uses the limit to decide what it may send there.

**Two OpenCode timeouts to know about** [DOCUMENTED:
`packages/opencode/src/provider/provider.ts`, v1.18.30]: a **5-minute
per-chunk timeout** on the raw SSE stream (`chunkTimeout`, default
300 000 ms, reset by any received bytes — NoLlama's 15-second keep-alives do
reset it) and a 5-minute **header timeout**. Both are per-provider `options`.
A slow iGPU that can spend more than five minutes in one cold prefill wants
them raised:

```json
"options": { "baseURL": "http://localhost:8000/v1", "chunkTimeout": 1800000, "headerTimeout": 1800000 }
```

## Goose

Goose (Block's agent, CLI and Desktop) takes an OpenAI-compatible provider:
base URL `http://localhost:8000/v1`, any API key, model name as printed in
the banner. Its system prompt and tool schemas are large, so on an iGPU
pin the KV pool (`--cache-size-gb`) before pointing it at a 26B-class MoE —
gemma-4-26b-a4b has a 240 KB/token KV and lands on the 2 GB floor otherwise
(issue #38).

## VS Code Copilot Chat

Copilot Chat (0.53+) uses the Ollama API. Start with `--vscode-compat` so
VS Code accepts the version handshake:

```powershell
python nollama.py --gpu-model-dir gpu-coder-model --vscode-compat
```

Then set the Ollama base URL to `http://localhost:11434` and pick the GPU
model. (Add `--debug` while wiring it up to see exactly what Copilot sends.)

## Pre-warming

With `--idle-timeout 0` NoLlama captures the largest system prompt it sees
into `prewarm-<port>.json` and prefills it at the next startup, so the first
real turn of a session is a cache hit rather than a cold prefill. It is
client-agnostic — OpenCode's ~8k-token prompt shows up in the startup log as
`pre-warmed prompt cache from prewarm-8000.json`. Disable with `--no-prewarm`.
