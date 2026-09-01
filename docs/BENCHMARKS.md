# Benchmarks

How to reproduce, and every measured number. Run `benchmark.py` against a
running NoLlama (or Ollama with `--backend ollama`); it does 1 warmup + N runs
and discards IQR outliers. The `count 1-100` test is the steady-state decode
metric used throughout.

**Cross-backend rows need care.** `benchmark.py` pins `temperature: 0` so both
servers decode greedily — omit it and NoLlama defaults to 0.0 while Ollama
defaults to 0.8, so one side samples. The same prompt also doesn't buy the same
work: on `count 1-100`, NoLlama's qwen3-8b emits 293 tokens where Ollama's emits
~1755 for an identical 291-character answer, because its build ignores
`/no_think` and spends the rest on hidden reasoning. tok/s is still tok/s — read
those rows as throughput, not as time to finish the task.

Ollama's OpenAI-compatible `/v1/chat/completions` and its native `/api/chat`
agree exactly once temperature is pinned (1755 tokens either way), so the
endpoint choice isn't a variable.

## Big MoE models on small GPUs (disk offload)

OpenVINO 2026.3 can stream Mixture-of-Experts weights from disk instead of
keeping them GPU-resident. NoLlama exposes it as `--offload-ratio PCT`
(GPU slots). Measured on an Arc 140V (16 GB) laptop, Qwen3-30B-A3B INT4 —
a 15.2 GB model that doesn't fit resident at all:

| `--offload-ratio` | Resident GPU memory | Steady-state decode |
|---|---|---|
| 30 | 10.8 GB | **25.3 tok/s** |
| 50 | 8.1 GB | 22.1 tok/s |
| 90 | **2.35 GB** | 5.1 tok/s |

(Steady-state, measured after the expert LRU warms up — the first ~60
tokens run 2-5× slower while the cache fills, so don't judge offload by
its first sentence. `scripts/offload-test.py` measures this properly.)

Pick the **smallest ratio that fits** your memory. At moderate ratios this
is genuinely interactive: 25 tok/s from a 15.2 GB model on a 16 GB-class
laptop iGPU matches a 24-core desktop CPU running the same model resident.
High ratios (90) trade speed for extreme footprint — batch/overnight
territory. **Requires an XMX-capable GPU** (Arc, Lunar Lake and newer —
`install.ps1` tells you at device detection); on iGPUs without XMX the
feature silently does nothing, and NoLlama warns at startup instead of
letting you believe your model got smaller.

**XMX confirmed on** (`GPU_HW_MATMUL` in `OPTIMIZATION_CAPABILITIES`): Arc 140V
iGPU, Arc Pro B60, Arc B390 Xe3 iGPU (Panther Lake, issue #32). **Not** on
the Xe-LPG iGPUs — desktop 285K and the laptop 140T (285H) alike. The flag only means
offload will engage — not that a model fits, and nothing at all for dense
models, which have no experts to stream.

**Don't use offload on a discrete card.** It exists to run models that don't
fit. If the model fits, every offloaded byte crosses PCIe instead of VRAM.
Qwen3-30B-A3B on a B60 (2026-08-18): **50.8 tok/s resident, ~10.5 at
`--offload-ratio 30`.** 5× slower, copy engine at 97%, 3.2 GB in VRAM against
10.2 GB in host RAM, both disks idle. It streams across the bus, not from disk.

That's why an iGPU does better: there the offloaded weights sit in RAM the GPU
reads directly (~136 GB/s on LPDDR5X), no bus hop. **Memory topology decides
this, not XMX.**

Two cautions. Greedy output stopped being reproducible under offload — 87 to
2040 tokens for the same prompt, where resident gave 478 every time. And nobody
has compared offload against `--device CPU` on a card that *can't* fit the
model, so don't assume offload wins there.

### Where does your hardware land? (big-MoE routes, measured 2026-08)

Same model family (Qwen3 MoE, A3B-class), steady-state decode, best route
per hardware class — including a CUDA flagship for perspective. Mixed
quants and sizes, so read it as *routes*, not a controlled A/B:

| Hardware | Stack & route | Model | tok/s |
|---|---|---|---|
| RTX 5090 32 GB + CPU (hybrid auto-split) | Ollama/CUDA | Coder-Next Q4, 53 GB | **~73** |
| **Arc Pro B60 24 GB dGPU, model fits resident** | NoLlama/OpenVINO | 30B-A3B int4, 15 GB | **50.8** |
| Arc Pro B60, same model, `--offload-ratio 30` | NoLlama/OpenVINO | 30B-A3B int4, 15 GB | ~10.5 — don't |
| Arc 140V laptop iGPU, `--offload-ratio 30` | NoLlama/OpenVINO | 30B-A3B int4, 15 GB | 25.3 |
| 24-core desktop CPU (64 GB RAM), model fits | NoLlama/OpenVINO | 30B-A3B int4 | 23.7 |
| 24-core desktop CPU, model **bigger than RAM** | NoLlama/OpenVINO | Coder-Next int8, **74 GB** | 9-11.5 |
| **Arc B390 Xe3 laptop iGPU (Panther Lake, 64 GB LPDDR5X-8533), resident** | NoLlama/OpenVINO | 30B-A3B int4, 15 GB | **52.7** |
| Arc 140T Xe-LPG laptop iGPU (Arrow Lake-H, 64 GB shared budget), resident, **no XMX** | NoLlama/OpenVINO | Coder-Next int4, 80B-A3B, ~40 GB | 14.8 |
| 8-core laptop CPU (LPDDR5X) | NoLlama/OpenVINO | 30B-A3B int4 | 9.1 |
| Non-XMX iGPU, model bigger than its shared-memory budget | — | any big MoE | won't load — offload needs XMX, so there is no fallback |

The two B60 rows differ by one flag. Offload isn't a speed feature — it's a way
to run what otherwise won't, and on a card with room it costs 5×.

Takeaways: a dedicated CUDA card is now ~1.4× the best Intel route — but
every Intel row above is *usable*, runs on hardware you may already own,
and two of them (offload, bigger-than-RAM CPU) were impossible before
OpenVINO 2026.3 and the MoE era. Decode is the whole story here; on
thinking models multiply by your patience.

### Benchmark (Core Ultra 7 258V, ARC 140V 16 GB) — laptop, LPDDR5X

Tested with `benchmark.py` — 1 warmup + 5 runs, outliers discarded.

```powershell
# Text-only (no images required)
python benchmark.py --llm-only

# With VLM tests — provide 4 images: two "same vehicle" + two "different"
python benchmark.py --images-dir C:\path\to\images
python benchmark.py --same-1 a.jpg --same-2 b.jpg --diff-1 c.jpg --diff-2 d.jpg
```

**LLM text (Qwen3 8B INT4-CW, same model on NPU and CPU):**

| Test | NPU | CPU |
|---|---|---|
| "Say hello" (thinking) | 11.7s, 5.2 tok/s | 8.1s, 7.4 tok/s |
| "Say hello" (no-think) | 10.6s, 4.6 tok/s | 8.6s, 7.3 tok/s |
| "What is 2+2?" (thinking) | 11.7s, 5.3 tok/s | 9.0s, 7.0 tok/s |
| "What is 2+2?" (no-think) | 5.5s, 0.7 tok/s | 2.7s, 1.5 tok/s |

**GPU (Qwen2.5-VL 3B on ARC 140V, non-streaming):**

| Test | Time |
|---|---|
| "Say hello" (thinking) | 2.6s |
| "Say hello" (no-think) | 2.6s |
| "What is 2+2?" (thinking) | 2.6s |
| "What is 2+2?" (no-think) | 2.4s |
| Same vehicle? (2 images) | 3.8s |
| Different vehicles? (2 images) | 3.8s |

Above benchmarks were captured before VLMPipeline gained streaming
support (openvino-genai 2026.1). VLM now streams on Arc 140V at
roughly 11 tok/s decode after prefill — see
`benchmark.py --backend vlm` for fresh numbers.

CPU beats NPU on throughput (~7.4 vs ~5.2 tok/s) for this model.
GPU text is fast but runs a smaller 3B model (not directly comparable).
VLM image responses take ~3-4s regardless of answer length.

### Panther Lake (Core Ultra X7 358H, Arc B390 Xe3 iGPU, 64 GB LPDDR5X-8533) — community, Linux

Reported by ktecho in issue #32 (2026-08-26, Ubuntu 26.04, NoLlama
`2026-08-24-1f89a63`, `benchmark.py --runs 5`), Qwen3-30B-A3B-Instruct-2507
int4, fully resident:

| Test | Xe3 iGPU decode | CPU decode |
|---|---|---|
| count 1-100 (steady state) | **52.7 tok/s** | 21.1 tok/s |
| say hello (thinking) | 53.4 | 25.3 |
| TTFT, short prompt | 0.09 s | 0.5–0.9 s |

**An integrated GPU matching a discrete Arc Pro B60 (50.8) on the same
model.** Xe3 has XMX, 8533 MT/s memory, and 64 GB to hold the whole MoE
resident — the three things the 140V's offload route lacks. Same tester,
same model, Qwen3.8-27B: not yet run.

Also from that thread, the agent-session failure mode that is *not* the
hardware: once a coding session's context outgrew the KV pool, every turn
re-prefilled the whole prompt — 82k chars → 58 s TTFT, 140k → 108 s,
199k → 175 s, linear in prompt size, after earlier repeat turns had hit
the cache at 0.35 s. Fix: `--cache-size-gb 12` (or more) on a machine with
64 GB. See [Agent tools](AGENTS.md).

### Arrow Lake-H (Core Ultra 9 285H, Arc 140T Xe-LPG iGPU, no XMX) — community, Windows

Reported by Dmitriy Teteruk in issue #24 (2026-08-28, `benchmark.py --runs
5`), 64 GB shared-memory budget on the iGPU, everything resident:

| Model | Decode tok/s (count 1-100) | TTFT | Note |
|---|---|---|---|
| Qwen3-8B int4-cw | 14.0 | 0.21 s | between the 285K desktop Xe-LPG (15.4) and the 140V (21.7) |
| Qwen3-Coder-Next int4 (80B-A3B) | **14.8** | 1.2 s | resident, no offload — a non-XMX iGPU runs a big MoE fine if the memory is there |
| Qwen3-Coder-Next int8 (74 GB) | 8.6 | 1.7 s | matches the 9.1 measured earlier (TODONT) |
| Qwen3.8-27B int4 (dense, VLM path) | 2.4–3.0 | 7–11 s text, 77–91 s with images | confirms the "dense 28B ≈ 2–3 tok/s on this path" prediction |
| Qwen3.8-27B int8 | 1.2–1.5 | 137–224 s with images | unusable |
| Qwen3-VL-8B (int8) | 5.3–6.5 | 7.2 s with images | int8-vs-int4 explains the gap to Qwen3-8B |
| LFM2.5-1.2B int4-cw **on the NPU** | **32.5** (41 on short prompts) | 1.0 s | 285K desktop NPU: 38.8. His first run gave 16–21 tok/s on a 100 W USB-dock supply; the laptop's own 140 W adapter restored it — **NPU throughput follows the power budget**, so benchmark on the real adapter (driver 5540, genai 2026.3.1) |

**Everything above except the LFM2.5 row was measured on the 100 W dock
supply.** A second batch on 2026-08-31 re-ran two of them on the laptop's own
140 W adapter, and the power effect is not an NPU quirk — it costs the
**iGPU** as much or more [OBSERVED 2026-08-31, issue #24]:

| Model | 100 W dock | 140 W adapter | |
|---|---|---|---|
| Qwen3-Coder-Next int4 (80B-A3B) | 14.8 | **18.0** | +22% |
| Qwen3-Coder-Next int8 (74 GB) | 8.6 | **11.3** | +31% |

So the rule generalises: **benchmark on the machine's real power adapter**,
and treat any laptop number taken through a dock as a lower bound. The
tables below are all 140 W.

#### Arrow Lake-H, second batch (2026-08-31, 140 W) — text models

| Model | Decode tok/s (count 1-100) | TTFT |
|---|---|---|
| Qwen2.5-Coder-1.5B int4 | **57.0** | 0.09 s |
| Qwen2.5-Coder-7B int4 | 16.3 | 0.21 s |
| Qwen3-Coder-Next int4 (80B-A3B) | 18.8 | 0.70 s |
| Qwen3-Coder-Next int8 | 11.8 | 1.04 s |
| DeepSeek-R1-Distill-Qwen-7B int4-cw **(NPU)** | 8.6 | 4.30 s |
| Mistral-7B-Instruct-v0.3 int4-cw **(NPU)** | 10.5 | 5.05 s |

Both NPU entries load and answer correctly — the two longest-standing
"Untested" rows in issue #24. Note the NPU's ~4–5 s TTFT against the iGPU's
0.1–1.0 s: that is the NPU's fixed prompt-compile cost, and it is why the
NPU suits short prompts and the GPU suits agent loops.

#### Arrow Lake-H, second batch (2026-08-31, 140 W) — vision models

Best decode tok/s seen across the two image questions and the text-only
questions on the same slot. VLM slots have no `count 1-100` test, so these
are not directly comparable with the text table above.

| Model | Image | Text | Note |
|---|---|---|---|
| Qwen3-VL-4B-Instruct int4 | **17.2** | 16.8 | best vision throughput of the batch |
| gemma-4-E2B-it int4 | 14.5 | 18.8 | |
| InternVL2-4B int4 | 12.3 | 18.9 | first image answer was 4 tokens — see below |
| gemma-3-4b-it int4-cw | 10.6 | 14.6 | |
| gemma-4-26b-a4b-it int4 (MoE) | 8.4 | 11.1 | 4B active; pin `--cache-size-gb`, its KV is 240 KB/token |
| gemma-4-E4B-it int8 | 8.1 | 8.0 | **Intel's build — no prefix caching** (see below) |
| Qwen3.5-9B int4 | 8.9 | 6.3 | |
| Qwen3-VL-8B-Instruct int8 | 7.0 | 6.9 | |
| Qwen3-VL-4B-Instruct fp16 | 6.0 | 5.0 | ~3x slower than the int4 of the same model |
| Qwen3.5-9B int8 | 5.7 | 3.7 | |
| gemma-3-12b-it int4 | 5.6 | 6.0 | |
| Qwen3.8-27B int4 | 2.9 | 2.6 | |
| Qwen3.8-27B int8 | 1.5 | 1.6 | unusable |
| Qwen3.5-9B fp16 | 2.4 | 1.6 | unusable; int4 is ~4x faster |
| Phi-3.5-vision int4 | **FAILED** | 18.7 | both image questions failed; text generation fine |

Three things in that table are worth more than their row:

- **`Phi-3.5-vision-instruct-int4-ov` fails on images — and it is our bug,
  not the model's.** Every image request returns

  ```
  Check '(prompt_id >= 0) && (prompt_id < vocab_size)' failed at
  .../sampling/logit_transformers.hpp:412: input_ids token out of bounds
  ```

  The trigger is **NoLlama's default `repetition_penalty` of 1.05**
  [OBSERVED 2026-09-01, Arc 140V, genai 2026.3.0.0-3277 and 2026.5.0.0-3402]:
  driving `VLMPipeline` directly, the same model and the same images answer
  correctly with the penalty at 1.0 and fail at 1.05, while `presence_penalty`
  and `frequency_penalty` change nothing. Only the repetition-penalty
  transformer walks the *prompt* ids, and Phi-3 vision's image placeholders
  sit outside `[0, vocab_size)`.

  So the model is fine and the images are fine — a bare `VLMPipeline` reads
  a screenshot correctly at every size from 336x336 to 2048x2048. What is
  broken is that we apply a repetition penalty to a prompt containing
  placeholder ids. Fix belongs in NoLlama (and arguably in genai, which
  should skip placeholders rather than assert). See `TODONT.md`.
- **`gemma-4-E4B-it-int8` was Intel's published build**, whose IR has no
  fused SDPA op and therefore gets no prefix caching at all — a defect Intel
  confirmed on 2026-08-31 (openvino.genai#4343). The number above is honest
  for that artifact, and `models.json` ships our re-export instead. See
  `docs/dev/prefix-cache.md`.
- **fp16 is never worth it here.** Both fp16 entries are 3–4x slower than the
  int4 of the same weights, on a memory-bound iGPU where the extra precision
  buys nothing measurable.

### NoLlama vs Ollama on the Arc 140V iGPU

Ollama now runs on Intel iGPUs via its Vulkan backend, so this is the
direct apples-to-apples question: **same Qwen3-8B, same 4-bit, same
Arc 140V iGPU.** Measured 2026-06-16 with `benchmark.py` (3 runs), using
the `count 1-100` test as the steady-state decode metric.

| | NoLlama (OpenVINO INT4-CW) | Ollama 0.30.8 (Vulkan GGUF Q4) |
|---|---|---|
| **Decode tok/s** (count 1-100) | **21.7** | 13.4 |
| Decode tok/s (2+2, thinking) | 18.6 | 11.2 |

**NoLlama's OpenVINO GPU path is ~1.6× faster on decode.** Prefill isn't
compared — the two were measured at different times. Two caveats that matter in
practice:

- **Ollama drops the iGPU by default** — it needs `OLLAMA_IGPU_ENABLE=1`,
  or it silently runs on CPU. The out-of-the-box Ollama experience on
  this laptop is *CPU*, not GPU.
- Ollama can't use the **NPU** at all, and has no local **vision** model
  on Intel — both are NoLlama-only.

> **Roadmap note — GPU/CPU support is here to stay** *(updated 2026-08:
> this reverses the earlier "provisional" stance)*. NoLlama's original
> reason to exist is the Intel **NPU** (which Ollama doesn't support), and
> the plan was to drop GPU/CPU once Ollama's Intel performance caught up.
> That hasn't happened and isn't on the horizon: Ollama's Intel path runs
> through a non-OpenVINO shim and remains much slower, while most real
> NoLlama users drive coding agents (OpenClaw, Copilot) on the GPU/CPU
> path. So GPU/CPU — and with them tool calling, prefix caching, and
> prewarm — are supported for the foreseeable future. If you outgrow a
> single-user local server (multi-user, production serving of 30B+
> models), the step up is [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
> — same runtime underneath, built for that job.

### Benchmark (Core Ultra 9 285K, RTX 5090) — desktop, DDR5

Same Qwen3 8B INT4-CW model on every Intel device, plus the same model
served via Ollama (GGUF Q4_K_M) on the RTX 5090 for context. 1 warmup +
3 runs. The "count 1-100" test (`max_tokens=4096`, no-think) is the
cleanest cross-stack number — long output, steady-state, no thinking confound.

```powershell
# Each NoLlama device — restart the server with --device <name> first
python benchmark.py --label npu --runs 3 --llm-only
python benchmark.py --label igpu --runs 3 --llm-only
python benchmark.py --label cpu --runs 3 --llm-only

# Ollama (any backend it's running on — CUDA, ROCm, CPU)
python benchmark.py --backend ollama --model qwen3:8b --label rtx5090 --runs 3 --llm-only
```

**Decode throughput, count-1-100 test:**

| Backend | Device | Decode tok/s | Speed vs CPU |
|---|---|---|---|
| Ollama (GGUF/CUDA) | RTX 5090 | ~230 | 12.9× |
| NoLlama (OpenVINO) | CPU (8P + 16E @ DDR5) | 17.8 | 1.0× |
| NoLlama (OpenVINO) | iGPU (Xe-LPG, 4 cores) | 15.4 | 0.87× |
| NoLlama (OpenVINO) | NPU 3 (Intel AI Boost) | 10.0 | 0.56× |

Prefill isn't the story here — all these devices hit first token in ~0.2 s on a
short prompt. Long agent prompts are another matter: see
[Agent tools](#agent-tools--coding-assistants-vs-code-copilot-openclaw).

**Surprises on this hardware:**

- **CPU beats iGPU.** Arrow Lake's 285K (8P + 16E at high clocks) plus
  OpenVINO's tuned INT4 CPU kernels add up to more decode throughput
  than the small Xe-LPG iGPU (only 4 Xe cores on the desktop part —
  the laptop's ARC 140V has 8). Both share the same DDR5 pool, so the
  iGPU has no bandwidth advantage, only a compute disadvantage.
- **NPU is the slowest Intel device on desktop**, opposite of the laptop
  story. NPU's value is power efficiency (laptop on battery), not
  throughput on mains.
- **It's a decode gap.** The 5090 leads the NPU ~23× on decode. Short prompts
  reach first token fast everywhere, so what you feel is throughput.
- **The dGPU dominates** — if you have one, use it. NoLlama's CPU
  fallback is good for "Intel-only laptop on battery", not for
  competing with a discrete card.

**Why the desktop iGPU/NPU are slower than the laptop's:**
LPDDR5X-8533 (laptop, ~136 GB/s) vs DDR5-6400 dual-channel (desktop,
~100 GB/s). Decode throughput on INT4 LLMs is memory-bandwidth-bound,
so the laptop's faster system memory closes some of the gap that
silicon size alone would suggest. (The Core Ultra 7 258V Lunar Lake
NPU also has more compute units than the 285K Arrow Lake NPU.)

**Practical guidance:**

| Hardware | Best NoLlama device |
|---|---|
| Intel Core Ultra laptop (Lunar Lake) | NPU (efficiency) or ARC 140V iGPU |
| Intel Arrow Lake desktop, no dGPU | **CPU** — surprisingly best |
| Intel + ARC discrete (A770, B580) | ARC discrete |
| Intel + NVIDIA discrete | Use Ollama for the dGPU; NoLlama on CPU/NPU/iGPU as fallback |

### Dual mode (NPU + GPU)

When you have both, text requests go to the NPU (streaming) and image
requests go to the GPU (VLM). Or put a bigger LLM on the GPU for
smarter chat. The routing is automatic — send a request and the right
device handles it.

```
POST /v1/chat/completions
  "What is the capital of Norway?"  --> NPU (streaming)
  [image + "What vehicle is this?"] --> GPU (VLM)
```

## Why not OpenVINO Model Server (OVMS)?

Intel already ships OVMS — a production-grade OpenVINO inference server.
If you're deploying LLMs in a datacenter or on Kubernetes, use OVMS.
NoLlama is a different target: your laptop.

| | OVMS | NoLlama |
|---|---|---|
| Target | Production, datacenter, K8s | Laptop, desktop, local |
| Runtime | C++ | Python (Flask) |
| OpenAI API | Yes (recent versions) | Yes |
| Ollama API | No | **Yes** |
| Built-in web UI | No (add OpenWebUI) | **Yes** |
| Auto device detection | No | **Yes** |
| Dual-device routing | One model per instance | **NPU chat + GPU vision, simultaneously** |
| Config | JSON, manual | Zero — `install.ps1` and go |

OVMS is a proper inference server. NoLlama is the thing that makes
your Core Ultra feel like Ollama already ran on it.

### ...and why not llm-scaler-vllm?

Same answer, different Intel stack. [`intel/llm-scaler`](https://github.com/intel/llm-scaler)
(vLLM + IPEX, the Battlematrix software) is Intel's official serving
path for **Arc Pro B-series** cards — and if you're building a
dedicated Linux inference box around them, use it: multi-card
tensor-parallel serving is its home game. It's also Ubuntu-with-a-
specific-kernel, Docker, and Linux-only for the vLLM path.

The axis that actually decides is streams × precision. LLM decode is
memory-bandwidth-bound, and 4-bit weights move roughly a quarter of
the bytes per token — INT4 IR is openvino-genai's native format, so
**single-user quantized decode on Intel silicon is NoLlama's tier**:
one to a few streams, the machine you sit at. Moderate shared
concurrency is OVMS's tier (continuous batching, same INT4 IR).
Multi-GPU tensor-parallel on Linux is llm-scaler's. Different jobs,
all three real.
