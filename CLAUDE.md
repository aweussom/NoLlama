# NoLlama

OpenAI-compatible LLM/VLM server for Intel hardware. NPU-first.

## Architecture

- `nollama.py` — Flask server, DeviceSlot class per device, auto-detects VLM/LLM from config.json
- NPU: LLMPipeline with MAX_PROMPT_LEN=4096, streaming via SSE
- GPU: VLMPipeline (images) or LLMPipeline (text). Both stream as of openvino-genai 2026.1 — verified on Arc 140V iGPU.
- Prefix (KV) caching: **default on** for GPU/CPU **LLM and VLM** slots — they load via the
  continuous-batching backend (`LLMPipeline/VLMPipeline(..., scheduler_config=SchedulerConfig(
  enable_prefix_caching=True, cache_size=slot.kv_pool_gb))`). VLMPipeline honoring
  scheduler_config was verified 2026-08-18 on 2026.3 release (140V: ~9k-token prefix
  21.7s→3.9s TTFT) and 2026.4 nightly (B60: 33k tokens 54.5s→1.3s) — the earlier
  "CB backend is LLM-only" note was stale. A repeated prompt prefix (an
  agent's fixed system prompt + tool schemas, identical every turn) is prefilled once, not
  every turn — measured ~47× faster on a cached turn (24.4s→0.5s for a ~2k-token prefix on
  the 285K CPU). Auto-invalidated by any prefix change (no staleness). `--no-prompt-cache`
  disables it. The pool is **auto-sized per slot** (`_resolve_kv_pool`): a third of what the
  weights leave free in the device budget, floor 2 GB (`AUTO_KV_MIN_GB`), cap ~64k tokens of
  the model's KV geometry (`AUTO_KV_TOKENS`) — sized from the *total* budget (not free RAM)
  so it's stable across restarts/reloads; the CB backend grows into it rather than
  allocating upfront, but prefix-cached blocks are never released, hence the fraction.
  `--cache-size-gb N` pins it (skips auto). NPU slots keep the
  plain pipeline (no CB path; NPU keeps MAX_PROMPT_LEN). Falls back to the plain
  pipeline with a warning if a device or runtime can't build the CB backend.
  Prewarm/prompt-capture is still LLM-only (VLM prewarm untested — see NEXT-STEPS). `--prewarm <file>`
  prefills a saved agent prompt at startup (the file auto-captures the first big prompt
  served via `_maybe_capture_prewarm` — on both the OpenAI and Ollama chat paths — so: run
  once → restart with `--prewarm`) so even the first turn is a cache hit instead of a cold
  prefill that can trip a client's idle watchdog. Prewarm is auto-enabled as
  `prewarm-<port>.json` when `--idle-timeout 0` (opt out: `--no-prewarm`); combining
  `--prewarm` with idle unload warns, since unload discards the warmed cache and the reload
  path deliberately does NOT re-warm (a synchronous re-warm would stall the triggering
  request pre-SSE and trip the client watchdogs the heartbeat exists to defeat).
- Observability: per-request log lines include TTFT (streaming: wall-clock to first token;
  non-streaming: `perf_metrics` via `extract_perf`) — a prefix-cache hit is sub-second vs a
  cold multi-second/minute prefill, so hits/misses are visible without instrumentation.
  `/health` has `prompt_cache_info` (pinned `pool_gb` or null + `auto`, prewarm file) +
  per-slot `kv_pool_gb` (resolved size), `last_ttft_ms` and `prewarmed`; `prompt_cache`
  stays a bare bool (start-openclaw.ps1 truth-tests it).
- Memory preflight at load (`_preflight_memory`): warns (never blocks) when weights + KV
  pool exceed the device budget (GPU: `GPU_DEVICE_TOTAL_MEM_SIZE`, which reflects Windows'
  ~half-RAM iGPU policy and Intel's "Shared GPU Memory Override" driver setting; CPU: total
  RAM) and logs the KV pool's token capacity from `config.json` geometry (~56 KB/token for a
  7B coder, ~96 KB for 30B — a too-small pool hard-fails generation with
  `Got unfinished GenerationStatus`, issue #21; `explain_genai_error` annotates that error
  with a `--cache-size-gb` hint wherever it surfaces).
- Whisper: WhisperSlot + WhisperPipeline for STT, `POST /v1/audio/transcriptions`, CPU or GPU
- OpenVINO GenAI may unify VLM/LLMPipeline — when that happens, simplify the dual-pipeline routing
- Routing: images go to GPU, text goes to NPU (or GPU if no NPU)
- Web UI: `templates/index.html` + `static/css/style.css` + `static/js/app.js`
- Collapsible `<think>` blocks, "Just answer me, dammit!" button, temperature slider
- Markdown rendering of responses/thinking (`mdEscapeAndRender`, PR #23) — hand-rolled, no
  library; attribute values go through `escapeAttr` + `safeUrl` (scheme whitelist), never
  `escapeHtml` alone (XSS fixed in review). Streaming scroll is pinned-but-releasable
  (`streamState`/`updateStreamBubble`): follows the stream until the user scrolls up.
- `threaded=True` on Flask, concurrency via per-device locks
- `models.json` — curated model registry (npu, gpu_vlm, gpu_llm, whisper categories)
- `install.ps1` detects devices, shows model menu, generates `start.ps1`. Agent setups get
  `--idle-timeout 0` (keeps the prefix cache alive; auto-enables prewarm). Local/cached
  models are validated before being offered or linked: the `.bin`+`.xml` pair must exist
  and the `.bin` must not be truncated (#17) — the IR `.xml` records each weight blob's
  offset+size, so max(offset+size) is the exact minimum `.bin` size (the IR has no
  checksum; truncation is the realistic failure, corruption-in-place is out of scope).
  `nollama.py` re-checks the same invariant at load (`_verify_weights_integrity`) since
  models can arrive without install.ps1 — a truncated/missing model fails with a
  plain-English error, and the "Is another process using the NPU?" hint is suppressed
  for that class of failure.
- Model naming: the **directory name is authoritative** — it's the web-UI label and the
  model ID clients request. `resolve_display_name` uses the name as given and only follows
  a symlink/junction when that name is generic (`model/`, `gpu-model/`, which is what
  install.ps1 links). It previously called `realpath` unconditionally, which silently
  discarded a deliberate rename (#19). There is deliberately **no `--model-name` flag** —
  see `TODONT.md` for why the rename is the interface.
- `--scan` reports what each model directory actually holds — display name (and where it
  came from), LLM/VLM/Whisper, architecture, MoE shape, geometry, integrity, and the real
  weight precision read from the IR's model-level `<rt_info>` (`nncf/weight_compression/
  mode` + `group_size` + `ratio` + `awq`) rather than from the folder name, which can lie.
  `read_ir_rt_info` seeks the **tail** of the `.xml` (the graph is tens of MB on a large
  model; the model-level block is the last `<rt_info>`, after `<edges>`). No server, no
  device init, no model load. Note VLM configs nest geometry under `text_config` —
  `_text_config` handles that, which also fixed the KV half of the memory preflight
  silently no-op'ing on every VLM.
- `download-model.ps1` — fetch/convert any HF model. PowerShell-style flags
  (`-Convert -Weight int4 -Trust`), NOT GNU `--convert` (#19: the docs once showed
  `--` syntax and users copy-pasted it; a catch-all param now prints the corrected
  command when someone tries). **Conversion is RAM-bound, not disk-bound**: optimum-intel's
  Qwen3-Next patcher builds an fp32 copy of every expert weight (workaround for OpenVINO
  CVS-181449) — for Qwen3-Coder-Next that's `512 experts × 2048 × 512 × 4 B` = 2 GB per
  projection stack, ~288 GB across 48 layers × 3. Measured: **400 GB of Windows pagefile
  (on 128 GB RAM) succeeded**, 200 GB did not (#19, Dmitriy Teteruk). Weight format is
  irrelevant to this stage — the blowup is before quantization.
- Tool calling: **GPU/iGPU + CPU**, on both LLM and VLM slots (VLM tool turns —
  including images alongside tools — added 2026-08-18; buffered like LLM tool turns,
  generated through the VLM pipeline). Gated by `_tools_supported`, i.e.
  `device_name in ("GPU","CPU")`; the **NPU is excluded** — it has a hard prompt cap and
  small NPU-class models can't drive agent loops, so when the NPU serves the request we
  ignore `tools` and answer as plain chat. `/api/show` advertises the `tools` capability only
  for GPU/CPU slots (so Copilot won't offer NPU models for agent mode). CPU is viable for
  agents on strong desktops (e.g. Core Ultra 9, many cores) where prefill can beat a weak
  iGPU. Tool specs from the request `tools` array are rendered into a system prompt
  (Qwen3-Coder native format); the model's emitted call is parsed back into OpenAI/Ollama
  `tool_calls`. `parse_tool_calls` recognizes several native formats, since a model
  often ignores our prompt and falls back to what it was trained on: Qwen3-Coder XML, Hermes
  JSON-in-`<tool_call>`, **bare `<function=>` with no wrapper (Qwen2.5-Coder native)**, Mistral
  `[TOOL_CALLS]`, Llama `<|python_tag|>`, DeepSeek `<｜tool▁calls▁begin｜>` blocks, plus a
  bare-JSON fallback. See `render_tools_prompt` / `parse_tool_calls`. Copilot Chat 0.53+ hits
  `/v1/chat/completions` (delegates to `chat_completions`); `/api/chat` also handled.

## Environment

- Primary: Windows 11, Python 3.10+
- Cross-platform: scripts use `#requires -Version 7.0` and branch on
  `$IsWindows`. Linux + PowerShell 7 is confirmed working (user-reported
  on Core Ultra 7 258V with NPU + GPU, issue #6). There is no install.sh —
  Linux runs the same install.ps1 via pwsh. On Linux, NPU/GPU need the
  Intel userspace drivers installed or only CPU is detected; the Linux NPU
  stack (`intel-npu-driver`) is less mature than Windows.
- Intel Core Ultra (NPU) + Intel ARC 140V 16GB (GPU)
- OpenVINO 2026.1+ with openvino_genai
- venv in `venv/`, activate before running
- `venv-2026.3/` — OpenVINO 2026.3 runtime **plus the modern export stack**
  (optimum-intel 2.1.0, transformers pinned to 5.4 — the LFM2 exporter's cap —
  einops, nncf 3.3). Use it for exporting architectures optimum-intel 1.27
  can't load (EAGLE-3 drafts, new-family models); `venv/` keeps the old
  1.27/tf-4.57 stack that some exporters still need. 2026.3 passed the
  regression suite 2026-08-06.

## Development preferences

- Read `TODONT.md` before proposing anything structural — it records approaches
  already rejected, with the reason. Add an entry whenever one is abandoned.
- Keep it simple. One file (`nollama.py`) is fine. Don't split into modules unless it gets unwieldy.
- PowerShell for install/launch scripts (Windows-native users).
- Runtime flags over hardcoded config (e.g. `--port`, `--device`).
- When testing, use small payloads / short prompts. Don't run full model loads unless needed.
- VLM prompts must be dead simple for small models (3B). One question, one answer, minimal JSON. All logic in Python, not in the prompt.
- Qwen3-VL is now pre-exported by Intel (OpenVINO/Qwen3-VL-8B-Instruct-int4-ov, May 2026) — not yet tested here. Earlier note about optimum-intel support is obsolete.

## Known issues

- NPU default prompt limit is 1024 tokens — we override to MAX_PROMPT_LEN=4096
- (resolved 2026-05-25) VLMPipeline gained streaming support in openvino-genai 2026.1; verified on Arc 140V iGPU at ~11 tok/s decode.
- Qwen3 thinking models can exhaust token budget on `<think>` before producing an answer
- Cancel (`/v1/cancel`) relies on OpenVINO invoking the streamer callback. If the native code blocks without yielding, cancel won't take effect — generation completes naturally.
- Chat history unbounded in web UI — user clears with Ctrl+N when long sessions approach MAX_PROMPT_LEN
- Tool-enabled turns are buffered, not token-streamed: we must see the whole tool-call block
  before emitting a structured `tool_calls` delta, so the full generation is collected before
  the result is sent (no incremental tokens that turn). To stop a slow prefill on a big agent
  prompt from tripping client idle watchdogs (Copilot/OpenClaw abort with no output after
  ~120s), the streaming tool path runs generation in a background thread and emits SSE
  keep-alive pings every `HEARTBEAT_SECS` (`_sse_tool_stream`); the plain stream path
  (`stream_llm`) pings the same way during a long prefill. True token streaming on tool turns
  (stream until a tool-call prefix appears) is still TODO.
- Big agent prompts (OpenClaw ships ~21k-token system prompts) prefill slowly on weak iGPUs
  (~6 min TTFT on the desktop 285K Xe-LPG). Mitigations: smaller coder model, CPU on strong
  desktops, trimming the client's tool set, and the keep-alive above so turns complete instead
  of aborting. OpenVINO can't cancel a blocked prefill, so an aborted client leaves the
  generation churning — another reason to keep clients connected via heartbeat.

## MoE disk offload (2026-08-06)

`--offload-ratio PCT` streams PCT% of MoE expert weights from disk on GPU
slots (OpenVINO 2026.3 `OFFLOAD_RATIO`). **Requires XMX** (Arc/Lunar Lake;
`GPU_HW_MATMUL` in OPTIMIZATION_CAPABILITIES) — silent no-op without, and
NoLlama warns at startup. Verified on Arc 140V, Qwen3-30B-A3B int4
steady-state: ratio 30 → 10.8 GB resident @ 25.3 tok/s (interactive!);
90 → 2.35 GB @ 5.1. Pick the smallest ratio that fits. The expert LRU
needs ~60 tokens to warm — benchmark steady-state, not first-sentence.
Known upstream bug: a SECOND generate() on an offload-active PLAIN
pipeline hangs in native code (uninterruptible). NoLlama's serving path is
unaffected — the CB backend it uses was verified with sequential requests
(140V, ratio 30: 12.5 then 15.9 tok/s, prefix cache TTFT 8.0s→1.9s). Non-XMX iGPUs can't
load big MoE at all (USM staging OOM) — full story in TODONT.md.

## NPU export rule (2026-08-06)

Models converted for the NPU **must be channel-wise** (`download-model.ps1
-Weight int4-cw` or `int8-cw`): default group-quantized int4 IRs crash the
NPU driver compiler ("Found N duplicated names", known vpux bug). int8-cw
halves decode vs int4-cw but keeps more quality — except on LFM2-family,
where no good int8 NPU variant exists (see TODONT.md). OFFLOAD_RATIO (2026.3
MoE disk offload, GPU-only) could not be validated on the desktop iGPU —
see TODONT.md before recommending it.

## Verified models

- Qwen3-8B (INT4-CW) on NPU — recommended, needs MAX_PROMPT_LEN=4096
- SmolLM3-3B (INT4-CW 23 tok/s, INT8-CW 12 tok/s) on 285K NPU — 2026.3, our export
- LFM2-1.2B / LFM2.5-1.2B-Instruct (INT4-CW, ~37-39 tok/s) on 285K NPU — NPU-only
  builds, old-stack exports fail CPU/GPU (see TODONT.md)
- MiniCPM5-1B (INT4) on GPU/CPU — 2026.3, no NPU support upstream
- Phi 3.5 Mini (INT4-CW) on NPU — smaller, faster
- DeepSeek-R1-1.5B (INT4-CW) on NPU — works but terrible quality (testing only)
- Gemma 3 4B Vision (INT4) on GPU — fast VLM
- Qwen2.5-VL-3B/7B (INT4/INT8) on GPU — proven for image tasks
- Qwen3-30B-A3B on GPU — needs >16GB VRAM, falls back to CPU silently on 16GB cards
