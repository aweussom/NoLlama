# API and clients

## Usage

```powershell
# Auto-detect (picks best device)
python nollama.py

# Force a specific device
python nollama.py --device NPU
python nollama.py --device GPU
python nollama.py --device CPU

# Dual mode: NPU chat + GPU vision
python nollama.py --model-dir model --gpu-model-dir gpu-model

# Different port
python nollama.py --port 9000

# Change the default idle-unload timeout (default is 1800 = 30 min)
python nollama.py --idle-timeout 600     # unload after 10 min idle
python nollama.py --idle-timeout 0       # never unload — keep models loaded forever

# Log every inbound API request (method, path, User-Agent, body) — handy when
# wiring up a new agent client and you need to see exactly what it sends
python nollama.py --debug

# Report a real Ollama version on /api/version so VS Code's Ollama client
# accepts the server (needed for VS Code Copilot Chat in Ollama mode)
python nollama.py --vscode-compat

# Prefix (KV) caching is ON by default for GPU/CPU LLM slots — a repeated prompt
# prefix is prefilled once, not every turn (big win for agent loops, ~47x on a
# cached turn). The pool is auto-sized per device from its memory budget and
# the model's KV geometry (a third of what the weights leave free, floor 2 GB,
# cap ~64k tokens' worth) — the startup log shows the chosen size and its
# token capacity. Pin the size, or disable caching:
python nollama.py --cache-size-gb 4     # pin the KV-cache pool (skips auto-sizing)
python nollama.py --no-prompt-cache     # disable prefix caching

# Pre-warm the cache at startup so the FIRST agent turn is fast too (not just
# turn 2+). The file auto-populates from the first big prompt served, so the
# workflow is: run once, then restart with --prewarm to skip the cold prefill.
# With --idle-timeout 0 this is automatic (as prewarm-<port>.json; opt out
# with --no-prewarm) — combining --prewarm with idle unload gets a warning,
# because the warmed cache is thrown away when the model idle-unloads.
python nollama.py --prewarm prewarm.json

# What models do I actually have? Reports each model directory's real
# contents — no server started, no model loaded.
python nollama.py --scan
python nollama.py --scan D:\models      # search somewhere else
```

### What have I got? (`--scan`)

`--scan` answers that from the files on disk rather than from what a folder
is called:

```
  C:\Users\you\models\Qwen3-30B-A3B-int4-ov
    Name in API/UI : Qwen3-30B-A3B      (from directory name)
    Kind           : LLM (text)
    Architecture   : Qwen3MoeForCausalLM / qwen3_moe
    Weights        : INT4 (asymmetric, group size 128)   15.2 GB on disk
    MoE            : 128 experts, 8 active per token
    Geometry       : 48 layers, 40,960-token context, 96 KB/token KV
    Exported with  : OpenVINO 2026.0.0, optimum-intel 1.27.0.dev0, transformers 4.57.6
    Prefix caching : yes — 48 fused SDPA ops
    Agent mode     : tool calling on GPU/CPU; never on NPU (hard prompt cap)
    Integrity      : weights complete
```

The precision comes from the IR's own `nncf` record, not the directory name
— a folder called `-int4-ov` can contain anything, and `--scan` reports what
the weights actually are (including partial quantization and AWQ). It also
runs the truncation check, so it's the quickest way to tell a bad download
from a bad model.

**`Prefix caching` is the one line that can save you a download.** Caching is
built by rewriting the fused `ScaledDotProductAttention` nodes in the
language model's IR, so an export that traced attention decomposed into
matmul+softmax cannot cache at all, and says so:

```
    Prefix caching : NO — no fused SDPA op in the language model, so the
                     caching backend cannot be built and this IR runs on
                     the plain pipeline. An export defect — re-export fixes it.
```

Nothing in a model's name, size, precision or geometry reveals this, and
until you see it you have no way to know: Intel's own
`OpenVINO/gemma-4-E4B-it-int8-ov` has the defect while its two siblings do
not (openvino.genai#4343). On a `NO` model every turn re-prefills the whole
prompt — fine for one-shot vision, wrong for an agent loop.

Don't read the count as "one per layer". That holds for dense models, but
hybrids emit one node per *attention* layer: Qwen3.5-4B shows 8 for 32
layers because it interleaves linear attention every fourth layer, and it is
perfectly healthy. **Any count above zero can cache.** NPU slots never
cache regardless — they keep the plain pipeline by design.

**To rename a model,** rename its directory: that name is what the web UI
shows and what clients request as the model ID. There's deliberately no
`--model-name` flag — see `TODONT.md`.

**The KV pool sizes itself.** The pool must hold the whole conversation:
bytes-per-token scale with the model's layer/head geometry (~56 KB/token
for a 7B coder, ~96 KB/token for Qwen3-Coder-30B). Too small doesn't just
evict cache — generation on a big agent prompt **fails outright**
(`Got unfinished GenerationStatus`, see issue #21). So NoLlama sizes the
pool per device at load: a third of what the weights leave free in the
device's memory budget, floored at 2 GB and capped at ~64k tokens of the
model's geometry (enough for a big agent system prompt plus a long
session). It's a ceiling the cache grows into, not an upfront allocation
— and the fraction leaves RAM for the compilers and tests an agent runs
on the same machine. The startup log shows the chosen size and its token
capacity; `--cache-size-gb N` pins it when you know better (e.g. whole-book
contexts). The preflight still warns when agent prompts would exhaust the
pool, and when model + pool exceed the device budget entirely. On Core
Ultra iGPUs that budget is ~half of system RAM by default — raise it with
Intel Graphics Software's "Shared GPU Memory Override" (driver 101.6987+).
Per-request log lines include TTFT, so a prefix-cache hit (sub-second) vs
a cold prefill (seconds-to-minutes) is visible directly; `/health` reports
the cache config (`prompt_cache_info.auto`, per-slot `kv_pool_gb`) and
each slot's last TTFT.

### Idle unload

NoLlama frees model memory after **30 minutes of inactivity by default**
(an 8B INT4 model holds ~5 GB of RAM; a VLM another ~3 GB). The next
request automatically reloads the model — the client just sees a slow
first response (~30-60s for an 8B model on NPU). The web UI shows
"Reloading model..." while it waits.

Change with `--idle-timeout <seconds>`. Use `0` to keep models loaded
forever (the old behavior) — recommended for agent use: it also
auto-enables `--prewarm`, and the warmed prefix cache survives (an idle
unload discards it until the next restart, which is why mixing
`--prewarm` with idle unload prints a warning).

`/health` reports `idle_unloaded` slots; the overall status stays
`ready` because requests can still be served (with a reload).

### Repetition penalty (`nollama.ini`)

NoLlama applies a **repetition penalty of 1.05** to every request unless the
client sends its own. It is the one generation setting with a non-neutral
default, so it is worth knowing it exists.

The penalty down-weights tokens that have already appeared, making the model
less likely to pick them again. It is the main defence against degeneration
— a phrase repeating verbatim, a list that never ends, or a thinking model
spiralling inside `<think>` until it burns the whole token budget.

The cost is that **some text is supposed to repeat.** Code is the clearest
case: a variable named eight times must be named eight times, and `return`,
`if` and `self` recur constantly. Penalising them nudges the model toward a
synonym or a subtly altered identifier. JSON keys, tables and any format
with required repetition have the same problem. That is why NoLlama uses
1.05 rather than Ollama's 1.1, which is known to degrade code output.

| Value | Effect |
|---|---|
| `1.0` | off — nothing discouraged, loops possible |
| **`1.05`** | NoLlama's default: loop insurance with little distortion |
| `1.1` | Ollama's default; noticeable, hurts code |
| `1.2`+ | heavy; visibly avoids correct repeated words |

**It scores the prompt, not just the output** [OBSERVED 2026-09-01 — the
bounds assertion below fires on a *prompt* token before any generation].
With a large agent system prompt, every word already in the prompt starts
slightly disfavoured in the answer — which is the same setup where the
answer most needs to reuse identifiers from the prompt verbatim. If you are
driving a coding agent and the model keeps renaming things, this is the
first knob to try at `1.0`.

To change the default, copy `nollama.ini.example` to `nollama.ini` (beside
`nollama.py`, gitignored) and edit `[generation] repetition_penalty`.

Per-request overrides always win: OpenAI clients send `repetition_penalty`,
Ollama clients `options.repeat_penalty`. Note the real OpenAI API has no
`repetition_penalty` field — it has `frequency_penalty`/`presence_penalty` —
so most agent tooling never sends one and quietly runs on the default.

**Automatic exception on some vision models.** A few VLM exports encode
image placeholders as token ids outside the model's vocabulary, and the
repetition-penalty transformer asserts when it walks them
(`input_ids token out of bounds`). NoLlama detects this on the first image
turn, warns once, and serves that slot's image turns with the penalty off;
text turns on the same slot keep it. The practical effect is that image
descriptions from such a model have a slightly higher chance of repeating
themselves. `OpenVINO/Phi-3.5-vision-instruct-int4-ov` is the known case,
filed upstream as openvino.genai#4405.

## API

Standard OpenAI `/v1/chat/completions`. Works with any OpenAI client.

### Text chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### Image (VLM, requires GPU with vision model)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":[
      {"type":"text","text":"What is in this image?"},
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
    ]}]
  }'
```

### Local file shortcut

When client and server are on the same machine, skip base64:

```python
{"type": "image_url", "image_url": {"url": "file:///C:/path/to/image.jpg"}}
```

**Note:** `file://` URIs only work locally. Remote clients must use base64.

### Streaming

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a story"}],"stream":true}'
```

#### Reasoning comes back as `reasoning_content`

Thinking models' `<think>…</think>` spans are **not** in `content`. Streaming
chunks carry them in `delta.reasoning_content` (the field OpenCode, Zed and
other OpenAI-compatible agent clients render as live thinking), and the
non-streaming reply in `message.reasoning_content`; `content` is the answer
only. An empty think block (a `/no_think` turn) is dropped.

```json
{"choices":[{"delta":{"reasoning_content":"The user wants"},"finish_reason":null}]}
{"choices":[{"delta":{"content":"Oslo"},"finish_reason":null}]}
```

Tool-enabled turns stream too: reasoning and any prose before the call arrive
token by token, then the parsed `tool_calls` delta and
`finish_reason: "tool_calls"`. Start the server with `--think-in-content` to
get the pre-2026-08-30 shape (tags inside `content`) for a client that
depends on it. The Ollama API (`/api/chat`) is unaffected either way.

### Other endpoints

- `GET /health` — device status, model names, readiness
- `GET /v1/models` — list loaded models (OpenAI format)

### Response headers

Every response includes `X-Device` and `X-Model` headers so you can
see which device handled it:

```
X-Device: NPU
X-Model: qwen3-8b
```

## Using with the openai Python package

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

## Ollama API

NoLlama also serves a full Ollama-compatible API on port 11434 (the
Ollama default). Any tool or client that talks to Ollama works without
modification — it thinks it's talking to a real Ollama instance.

Supported endpoints:

- `POST /api/chat` — chat with streaming (newline-delimited JSON)
- `POST /api/generate` — single-turn completion
- `GET /api/tags` — list models
- `POST /api/show` — model info

```bash
curl http://localhost:11434/api/chat \
  -d '{"model":"qwen3-8b-int4-cw","messages":[{"role":"user","content":"Hello!"}]}'
```

Disable with `--ollama-port 0` if you don't need it or port 11434 is taken.

## Using with OpenWebUI

OpenWebUI can connect via either API:

**OpenAI mode** (recommended):

| Field | Value |
|---|---|
| Base URL | `http://host.docker.internal:8000/v1` |
| API Key | `not-needed` |

**Ollama mode** (no config needed if NoLlama runs on default port):

| Field | Value |
|---|---|
| Ollama Base URL | `http://host.docker.internal:11434` |

## Web UI

The server includes a built-in chat interface at http://localhost:8000.
No separate install, no Docker, no Node.js.

![NoLlama chat UI](docs/images/nollama-chat.gif)

A native Windows GUI is planned to replace the browser-based UI.

Features:
- Streaming chat with tokens appearing in real-time
- Collapsible "Thinking..." blocks (Qwen3 reasoning models)
- Drag-and-drop / paste images for VLM queries
- Model selector showing loaded models and their devices
- Device badge on each response (`[NPU 1.2s]`, `[GPU 2.8s]`)
- Dark theme
- Keyboard shortcuts: Enter to send, Shift+Enter for newline,
  Ctrl+V to paste images, Ctrl+N for new chat, Escape to cancel
