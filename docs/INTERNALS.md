# How it works, and what it doesn't

## What it does

- **OpenAI API** (`/v1/chat/completions`) — works with any OpenAI client, OpenWebUI, etc.
- **Ollama API** (`/api/chat`, `/api/generate`) — works with Ollama clients, OpenWebUI Ollama mode, etc.
- **Auto-detects** NPU, ARC iGPU, ARC discrete, CPU — picks the best available
- **VLM support** — send images via base64 or `file://` URIs for vision models
- **Streaming** — token-by-token for text chat, with collapsible thinking blocks
- **Dual device** — NPU for chat + GPU for vision, simultaneously
- **Tool calling / agents** (GPU/iGPU + CPU, not NPU) — works with OpenCode, Goose and VS Code Copilot Chat; the model drives tools on the ARC GPU or a strong CPU
- **Prefix caching** (on by default) — a repeated prompt prefix (e.g. an agent's fixed system prompt) is prefilled once, not every turn — ~47× faster on cached turns
- **MoE disk offload** (`--offload-ratio`) — run 30B-class MoE models on 16 GB-class XMX GPUs by streaming expert weights from disk (verified: 2.35 GB resident for a 15.2 GB model)
- **Built-in web UI** — chat, image drop zone, model selector, dark theme
- **Model menu** — curated list of verified models, no conversion nightmares

## How it works

The server auto-detects your model type (VLM or LLM) from
`config.json` and loads the right OpenVINO GenAI pipeline:

- **VLMPipeline** for vision models — handles images + text
- **LLMPipeline** for text models — handles chat with streaming

In dual mode, both pipelines run on separate devices with separate
locks. They don't interfere with each other.

> **Future simplification:** OpenVINO GenAI may unify VLMPipeline and
> LLMPipeline into a single pipeline that handles both text and images.
> When that lands, the dual-pipeline detection and routing logic in
> NoLlama can be collapsed into one code path.

## Files

```
nollama.py              The server
install-windows.bat     Windows entry point — double-click me (installs
                        PowerShell 7 + Python if missing, runs install.ps1)
install-linux.sh        Linux entry point — checks pwsh/python3, prints the
                        install command for your distro, runs install.ps1
install.ps1             Setup wizard (cross-platform; the shims above call it)
download-model.ps1      Download/convert any HuggingFace model
benchmark.py            Device performance benchmark
start.ps1               Auto-generated launcher (after install)
models.json             Curated model registry
model/                  Primary model (NPU or GPU)
gpu-model/              Secondary GPU model (dual mode)
venv/                   Python virtual environment
```

`model/`, `gpu-model/`, `venv/`, and `start.ps1` are gitignored.
The repo is pure code.

## Known limitations

These are known and intentionally not fixed — either because the cause
is upstream, the fix would hurt simplicity, or it doesn't matter for a
local single-user tool.

- **Cancel may not interrupt mid-generation.** The cancel endpoint
  signals OpenVINO's streamer callback to stop. If OpenVINO is blocked
  inside a native call and not invoking the callback, there's no way
  to interrupt it from Python. Generation completes; lock releases
  when it does.
- **NPU prompt limit is 4096 tokens.** Long chat histories will
  eventually exceed this. The UI doesn't trim history — use Ctrl+N to
  start fresh if you hit the limit.
- **Vision runs on the GPU, not the NPU — by design.** The NPU *can*
  load a VLM (Qwen2.5-VL-3B compiles and runs via VLMPipeline; Qwen3.5
  and MiniCPM-V don't compile at all), but the NPU caps the prompt at
  ~1024 tokens *including image tokens*, and Qwen2.5-VL spends one token
  per 28×28 px. That leaves a usable ceiling around **768×768 (~784
  image tokens)**: at that size — or smaller — it answers correctly, so
  NPU vision works **well-ish on very small images** (a 256–512px crop is
  fine). But prefill already takes ~17s at the ceiling, and a plain
  1024×768 photo overflows the cap and fails outright (720p/1080p never
  stand a chance). So vision stays on the GPU, which has no such cap,
  runs at full resolution, and is faster. Measured with
  `test_npu_vlm_imagesize.py`.
- **Ollama management endpoints are stubs.** `/api/pull`, `/api/delete`,
  `/api/copy` return success but don't do anything. Model management is
  via `install.ps1` or `download-model.ps1`, not the API.
- **No graceful shutdown.** Ctrl+C is abrupt. If you hit it mid-load,
  NPU/GPU resources may not free cleanly — usually resolves on next
  launch, occasionally needs a reboot.
- **Flask dev server, not production.** Single-user local tool. Don't
  put it on the internet without a reverse proxy.
