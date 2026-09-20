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

**Kill servers by port owner, not by pid.** A venv built from the Microsoft
Store Python has a redirector at `venv\Scripts\python.exe`, so
`Start-Process -PassThru` returns the *launcher's* pid and the real server
survives being stopped. The next server then fails to bind, and the benchmark
quietly keeps talking to the previous model — a wrong number that looks
perfectly healthy. `scripts/bench-b60.ps1` kills by port and asserts `/health`
reports the expected model; copy both.

**Use the 285K or the B60 box, not the laptop.** A busy 140V reads about 30%
low (Qwen3-8B int4-cw: 14.8 tok/s with a browser and chat apps running, 19.4
quiet). Machine rules and SSH gotchas: `docs/dev/machines.md`.

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
iGPU, Arc Pro B60, Arc B390 Xe3 iGPU (Panther Lake, issue #32), **and the Arc
140T** (Arrow Lake-H, 285H): the mobile Arrow Lake iGPU is Xe-LPG+, which added
XMX/DPAS over the desktop Xe-LPG [DOCUMENTED: Intel Arrow Lake-H launch
material; installer prints `XMX: yes` on a 140T in issue #38]. **Not** on the
desktop 285K's Xe-LPG — the only no-XMX GPU this project has measured on. Earlier
versions of these docs called the 140T a no-XMX part; every conclusion drawn
from that is corrected as of 2026-09-11. The flag only means
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
| Arc 140T Xe-LPG+ laptop iGPU (Arrow Lake-H, **has XMX**, 64 GB shared budget), resident | NoLlama/OpenVINO | Coder-Next int4, 80B-A3B, ~40 GB | 14.8 |
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

Same tester, same box, **Qwen3.6-35B-A3B int4** (`OpenVINO/Qwen3.6-35B-A3B-int4-ov`,
lands on a VLM slot, `--cache-size-gb 12`), issue #40, 2026-09-11, NoLlama
`2026-08-24-da4e19e`, `benchmark.py --llm-only --runs 5`:

| Test | Xe3 iGPU decode | CPU decode |
|---|---|---|
| count 1-100 (steady state) | **41.1 tok/s** | 16.1 tok/s |
| say hello (thinking) | 40.9 | 16.6 |
| TTFT, short prompt | 0.13–0.24 s | 0.6–1.9 s |

About 0.78x the 30B-A3B on the same GPU and 0.76x on the CPU — a newer,
multimodal generation with a bit more per-token work, not a different class.
The `no-think` rows produced *more* tokens than the thinking ones (354 vs
322 on "say hello"): the Qwen3.5-MoE family honours neither no-think lever,
so `benchmark.py`'s no-think system prompt is a no-op there and those rows
measure the same thing twice.

Also from that thread, the agent-session failure mode that is *not* the
hardware: once a coding session's context outgrew the KV pool, every turn
re-prefilled the whole prompt — 82k chars → 58 s TTFT, 140k → 108 s,
199k → 175 s, linear in prompt size, after earlier repeat turns had hit
the cache at 0.35 s. Fix: `--cache-size-gb 12` (or more) on a machine with
64 GB. See [Agent tools](AGENTS.md).

### Arrow Lake-H (Core Ultra 9 285H, Arc 140T Xe-LPG+ iGPU, XMX) — community, Windows

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
  placeholder ids.

  **Hardware-independent** [OBSERVED 2026-09-01]: identical on **CPU**, Arc
  140T (community), Arc 140V, and Arc **Pro B60 discrete** — so no Intel GPU
  is needed to see it — and on genai 2026.3.0.0-3277 and 2026.5.0.0-3402. **Fixed in NoLlama 2026-09-01**: an image turn that hits
  the assertion retries once without the penalty and the slot remembers, so
  the model serves vision normally from then on and text turns keep their
  penalty. **Filed upstream as openvino.genai#4405** (2026-09-01) — a
  penalty over a VLM prompt should skip placeholder ids rather than assert.
  See `TODONT.md`.
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
> NoLlama users drive coding agents (OpenCode, Copilot) on the GPU/CPU
> path. So GPU/CPU — and with them tool calling, prefix caching, and
> prewarm — are supported for the foreseeable future. If you outgrow a
> single-user local server (multi-user, production serving of 30B+
> models), the step up is [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
> — same runtime underneath, built for that job.

### NoLlama vs llama.cpp's OpenVINO backend (Arc Pro B60)

llama.cpp took an **OpenVINO backend upstream** (`-DGGML_OPENVINO=ON`, preview
alongside OpenVINO 2026.1), so on Intel hardware the runtime is no longer what
separates us from it — it is the same OpenVINO underneath, reached through a
different front-end. That makes this the sharper comparison than the Ollama one
above, which still runs through Vulkan.

Measured **2026-09-07** on the B60 box: Ryzen 9 5950X + **Arc Pro B60 24 GB**,
native Windows (not WSL — see the note below), GPU driver `32.0.101.8805`.
**OpenVINO 2026.3.0-22451-bd8d6542e3c on every row**, llama.cpp at commit
`465e49b9c` built with MSVC 19.44. Model: Qwen3-8B, GGUF **Q4_K_M** (4.68 GiB)
for llama.cpp against OV **int4** (4.52 GiB) for ours — within 3.5% on weight
bytes. `llama-bench -p 512 -n 128 -r 3`; our side greedy, `ignore_eos`,
`min_new_tokens == max_new_tokens`, warmup discarded, medians of 3.

| Arc Pro B60 | pp512 | tg128 | pp2048 | tg128 |
|---|---|---|---|---|
| llama.cpp OpenVINO, stateless (default) | **3967** | 27.2 | — | — |
| llama.cpp OpenVINO, `GGML_OPENVINO_STATEFUL_EXECUTION=1` | 3339 | 38.7 | 3143 | 38.8 |
| NoLlama, bare `openvino_genai` (no server) | 3849 | **67.3** | **5497** | **62.5** |
| **NoLlama server** (HTTP/SSE, prefix cache on) | 2100 | 65.7 | 3189 | 61.2 |

**Cold prefill is a tie — the same runtime showing through two front-ends —
and decode is not.** NoLlama decodes at 65.7 tok/s against their best 38.7
(**1.7x**) and their default 27.2 (**2.4x**). Their stateless/stateful switch
is a trade with no good corner: stateful buys +42% decode and costs 16%
prefill.

#### The number that decides agent use

Prefill throughput is the wrong metric for an agent, which re-sends a growing
prefix every turn. 8192-token prompt, 8 tokens generated, so the request is
almost entirely prefill:

| NoLlama server, B60 | wall time |
|---|---|
| unique prompt each run (cache miss) | 3.98 s |
| same prompt repeated (cache **hit**) | **0.205 s** |

**19x.** The prefill itself drops from ~3.86 s to under 0.1 s. `llama-server`
on the OpenVINO backend is **stateless-only with no context shifting** (their
own docs), so it has no counterpart to this at all. On agent traffic that gap
dwarfs the decode one — which is the honest summary of where NoLlama wins:
not raw tokens per second, but being usable as an agent backend on hardware
these models otherwise run badly on.

#### CPU: OpenVINO is the wrong choice either way

Same box, native Windows, same models:

| Ryzen 9 5950X | pp512 | tg128 |
|---|---|---|
| llama.cpp, its own CPU backend | **91.5** | **9.74** |
| NoLlama, bare `openvino_genai` | 78.3 | 9.08 |
| llama.cpp OpenVINO backend | 60.1 | 4.35 |
| llama.cpp OpenVINO + stateful | 59.5 | 5.77 |

llama.cpp's hand-written CPU kernels beat **every** OpenVINO row, ours
included. This is the measurement behind the README's advice to use Ollama if
you are CPU-only. Between the two OpenVINO front-ends ours is the faster one
(78.3/9.08 vs 60.1/4.35), which is a narrow thing to win.

#### Do not benchmark this under WSL

Same commit, same model, same box, WSL 2 (Ubuntu 24.04) vs native Windows:

| pp512 | WSL | native | penalty |
|---|---|---|---|
| NoLlama bare GenAI, CPU | 46.7 | 78.3 | **-40%** |
| llama.cpp OpenVINO, CPU | 47.9 | 60.1 | -20% |
| llama.cpp own CPU backend | 87.6 | 91.5 | -4% |

**The WSL penalty is not a constant** — it lands hardest on OpenVINO's CPU
path and barely touches llama.cpp's own kernels, so a WSL-measured comparison
against native numbers ranks the layer, not the stack. The 2-4% figure in
`DOCKER-INSTALL.md` is a GPU result and does not carry over to CPU. The B60
also cannot be reached from WSL without replacing Ubuntu's compute-runtime
(its packaged NEO predates BMG-G31), which is the other reason every row above
is native.

#### Caveats

- Not the same 4-bit: Q4_K_M vs OV int4. Close on bytes, not identical in
  format, so decode differences are runtime *and* quantisation.
- The server row is not the bare row's equal by construction: chat template,
  HTTP/SSE, and the continuous-batching path. Its ~1.8x slower cold prefill vs
  bare is the known CB cold-prefill cost — you pay once per new prefix and win
  19x on every turn that reuses one.
- `llama-bench` feeds raw tokens with no chat template, which flatters it
  slightly on the like-for-like rows.
- **Stock Ollama has no OpenVINO path** as of 2026-09; only third-party forks
  (`zhaohb/ollama_openvino`). The Ollama comparison above is therefore still
  current — this section is about llama.cpp built from source.

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
[Agent tools](AGENTS.md).

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

## Bonsai 2 27B (ternary) vs its base model, Qwen3.8-27B (2026-09-18)

PrismML's [Ternary Bonsai 2 27B](https://prismml.com/news/bonsai-2-27b) is
Qwen3.8-27B distilled to ternary weights: 7.2 GB as `PQ2_0` (2.13 bpw)
against ~15-17 GB for a 4-bit quant of the base, with a claimed 98% of the
base's aggregate score. It runs only on PrismML's llama.cpp fork
(`prism-b10685` here), which has PQ2_0 kernels for CUDA, Metal, HIP and CPU —
**not Vulkan, not SYCL**, so there is no Intel GPU route for it yet. The base
model runs everywhere. So this is a cross-stack comparison by necessity:
Bonsai on the fork, Qwen3.8 on Ollama (RTX 5090) and on NoLlama (Arc Pro
B60, CPU). Harness: `scripts/bonsai-bench.py`, raw JSON in `bench-results/`.

Every row: greedy (`temperature 0`), thinking **off** for the speed tests
(each stack's own switch: `chat_template_kwargs` on llama-server, `think:
false` on Ollama's native API, the system-prompt marker on NoLlama), 1
warmup + 3 runs, median. Prefill uses a *different* ~4.4k-token document
per run because every stack caches the previous prompt's KV — the first
version of the harness reported 70,000-100,000 tok/s "prefill" that was a
cache lookup.

### The first Bonsai number was 27x too low, and nothing said so

The demo's `setup.ps1` picked the **Vulkan** build on the 285K: it greps
`nvidia-smi` for `CUDA Version:` and driver 616.92 prints `CUDA UMD
Version: 13.4`. Vulkan has no PQ2_0 kernels, so the server loaded, answered
correctly, and decoded at **4.9 tok/s** (prompt processing 4-7 tok/s). The
CUDA 13.3 build of the same release, same file: **132 tok/s** and 348
tok/s. Reported on PrismML-Eng/Bonsai-demo#176, which already carried three
PRs for the regex. If a Bonsai number looks wrong, check which
`bin\<backend>` the launcher printed before anything else.

### RTX 5090 (285K box, driver 616.92)

| Arm | Decode, free text | Decode, count 1-100 | Prefill (4,411 tok) | TTFT short | Probes no-think | Probes think | Think tokens / probe |
|---|---|---|---|---|---|---|---|
| **Bonsai 2 PQ2_0**, fork CUDA build, no drafter | **131** | 131 | **3,135 tok/s** (1.4 s) | 0.11 s | 20/23 | **23/23** | 103 |
| Qwen3.8 Q4_K_M, **same fork CUDA build** (Ollama's GGUF blob), no drafter | 79 | 79 | 2,860 tok/s (1.5 s) | 0.10 s | 21/23 | **23/23** | 105 |
| Qwen3.8 Q4_K_M, Ollama 0.34, `draft_num_predict=0` | 77 | 77 | 3,021 tok/s (1.5 s) | 0.09 s | 21/23 | 22/23 | 94 |
| Qwen3.8 Q4_K_M, Ollama 0.34, **MTP drafter on (default)** | 95 | 201 | 2,703 tok/s (1.6 s) | 0.10 s | 21/23 | 22/23 | 94 |

**The like-for-like pair is the first two rows: same binary, same flags,
no speculation on either side, 131 vs 79.** The ternary model decodes
1.66x faster than a 4-bit quant of its base on the same card. The second
Qwen3.8 row is the same GGUF through Ollama, and the gap between the two
Qwen3.8 rows (79 vs 77) is what the runtime is worth — about 3%, so the
1.7x was not "half ternary, half Ollama tax" (the objection was raised,
and this row is the answer). The fork is mainline llama.cpp plus extra
types, so any standard GGUF runs in it; that is what makes the same-binary
row possible.

Read the two Ollama rows together. Ollama runs Qwen3.8 with the model's
multi-token-prediction drafter by default (`ollama show` lists
`draft_num_predict 4`; the server log reports ~4.3 accepted tokens per
step). On a maximally predictable output (counting to 100) that is 2.6x;
on free prose it is 1.23x. Bonsai has its own drafter
(`BONSAI_SPECULATIVE=1`, needs a converted dspark GGUF) that was not
enabled — an open item, `brain/todo/3-someday/032-1h-bonsai-dspark-drafter-arm.md`.

Against the bandwidth ceiling (1.8 TB/s): 17 GB of q4_K_M weights allow
~105 tok/s and Ollama gets 77 (73%); 7.2 GB of PQ2_0 allow ~250 and Bonsai
gets 131 (52%). Unpacking 2-bit weights costs compute, which is why the
byte ratio (2.4x) does not turn into the speed ratio (1.7x).

### Arc Pro B60 (NoLlama, OpenVINO 2026.3.1, driver 32.0.101.8805)

| Arm | Decode, free text | Decode, count 1-100 | Prefill (~4.4k tok) | TTFT short | Probes no-think | Probes think | Think tokens / probe |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B int4-ov (Intel export, VLM slot, `--cache-size-gb 3`) | **22.9** | 23.3 | ~1,150 tok/s (3.9 s) | 0.38 s | 21/23 | **23/23** | 96 |

That is 75% of the card's ~30 tok/s ceiling for 15 GB of weights at ~456
GB/s — the B60 is not underperforming, it has a quarter of the 5090's
bandwidth and twice Bonsai's bytes to move. Fine for chat (faster than
reading), slow for thinking mode and agents: the identical thinking probe
pass took 326 s here and 46 s on the 5090. For agent work on this card the
MoE models remain the pick (Qwen3-30B-A3B, 50.8 tok/s above). The 7.2 GB
Bonsai file would fit with 17 GB to spare; it needs the fork's SYCL port.

Two things this arm surfaced, both fixed or recorded the same day:

- **The no-think switch was dead on VLM slots.** The marker only reached
  `LLMPipeline`'s `ChatHistory`; a VLM turn kept reasoning, and a 600-token
  "no-think" story budget came back as empty content. The pre-fix run is
  kept as `bench-results/*PREFIX-nothink-bug*` (14/23 no-think, then every
  thinking probe 503'd). Fix: `render_nothink_prompt` in `nollama.py`;
  `TODONT.md` has the approach that did *not* work.
- **The auto-sized 5 GB KV pool died with `CL_OUT_OF_RESOURCES`** on the
  34th request; 3 GB ran two full passes clean. `docs/dev/machines.md`.

### CPU, same models — these rows measure the fork's kernels, not the format

**Read the CPU rows as "what PrismML's CPU build does on this instruction
set", never as "ternary is slow on CPU".** The fork's PQ2_0 CPU path is
unpack-then-dot, not a lookup-table kernel of the T-MAC / bitnet.cpp kind
[DOCUMENTED: `ggml/src/ggml-cpu/arch/x86/quants.c` and `repack.cpp` at tag
`prism-b10685-7dffb15`]: `ggml_vec_dot_pq2_0_q8_0` spreads the 2-bit codes
to bytes and runs VNNI `dpbusd` against Q8_0 activations, guarded by
`__AVXVNNI__` or AVX-512 VNNI with a scalar loop otherwise; and the
repacked GEMM/GEMV that does batched prefill (`ggml_gemm_pq2_0_4x8_q8_0`)
is guarded by **AVX-512F + BW + DQ + VNNI**, falling back to a scalar
generic loop on anything less. Consequences on the two CPUs below: the
Core Ultra 9 285K (Arrow Lake: AVX-VNNI, **no AVX-512**) gets the 256-bit
dot for decode and scalar prefill; the Ryzen 9 5950X (Zen 3: AVX2 only)
gets scalar everything. Mainline llama.cpp's `TQ2_0` CPU path has an AVX2
kernel with no 512 gate, but Bonsai 2 cannot use it until its activation
transform is upstream — which is the argument for getting it there.
Someone is already on it: PrismML-Eng/Bonsai-demo#196 (2026-09-19) offers
AVX2/AVX-VNNI PQ2_0 kernels as PR PrismML-Eng/llama.cpp#206, measured
bit-exact on an i7-13620H (decode 1.3 → 4.2 tok/s, prompt processing
6.7 → 16 tok/s); our two CPUs are posted there as corroboration.

Same binary on both sides of each pair (the fork's `bin\cpu`, `-c 8192`,
`-np 1`, text-only), 2 runs, no thinking:

| CPU | Model | Decode, free text | Prefill (~4.4k tok) | Probes no-think |
|---|---|---|---|---|
| **Core Ultra 9 285K** (Arrow Lake: AVX-VNNI, no AVX-512; 24 threads, DDR5) | Bonsai 2 PQ2_0 | **7.2** | 11.2 tok/s (393 s) | 20/23 |
| | Qwen3.8 Q4_K_M | 4.3 | 42 tok/s (104 s) | 21/23 |
| **Ryzen 9 5950X** (Zen 3: AVX2 only; 16 threads, DDR4) | Bonsai 2 PQ2_0 | 2.7 | **3.3 tok/s (1,342 s)** | 20/23 |
| | Qwen3.8 Q4_K_M | 2.4 | 27 tok/s (163 s) | 21/23 |
| 285K, for reference | Qwen3.8 Q4_K_M via Ollama `num_gpu 0`, **MTP drafter on** | 5.2 | 29 tok/s | 21/23 |

Read down each pair. **Decode:** on Arrow Lake the ternary model is 1.65x
the 4-bit base in the same binary (the 256-bit VNNI dot doing its job);
on Zen 3 the two are within 10% of each other, because both are on
scalar-or-memory-bound paths and Q4_K_M's 15.7 GB sits at that box's
~50 GB/s DDR4 ceiling anyway. **Prefill is where the gate shows:** Q4_K_M
prefills the 4.4k-token document 3.8x faster than Bonsai on the 285K
(42 vs 11 tok/s) and 8x faster on the 5950X (27 vs 3.3), because
mainline's Q4_K GEMM has an AVX2 path and the fork's PQ2_0 GEMM has only
the scalar fallback below AVX-512. That is six and a half minutes to first
token for a 4.4k prompt on a 24-core desktop, twenty-two on the Ryzen; CPU
Bonsai is a short-chat tool until PR #206 or the AVX-512 tier applies.

Two cautions. Ollama's CPU row runs its MTP drafter and is therefore not a
runtime comparison; the fork's own Q4_K_M row (4.3) is. And the fork's CPU
build is itself slower than Ollama's runner on the same GGUF, so absolute
fork-CPU numbers understate llama.cpp on CPU; the within-binary ratios are
what this table is for. The base model was also loaded under NoLlama on the
5950X CPU and stopped before measuring (15 GB of OpenVINO weights on a 32 GB
box in use left 1.5 GB free); at ~3 tok/s it would not have changed the
picture.

A Zen 5 box (9950X3D, full AVX-512 VNNI) is the missing row: the only
consumer CPU class where the fork's *fast* PQ2_0 path applies at all —
and where upstream reports it crashing at load (Bonsai-demo#182,
llama.cpp#219). Planned.

The base model on the 5950X CPU was loaded but not measured: 15 GB of
weights on a 32 GB box in use left 1.5 GB free and Claude Code's own
memory reaper started killing background shells, so the run was stopped.
Expect ~3 tok/s from the 285K CPU row scaled by DDR4 bandwidth; measure it
on an idle box if the number ever matters.

### What the probes say about the distillation

23 short tasks with deterministic checkers (arithmetic, a word problem,
logic, dates, string reversal, JSON, three code functions run against
asserts, a regex, SQL, a Norwegian translation, instruction constraints,
and one tool call). It is a smoke test, not MMLU:

- **With thinking on, both models pass everything in the same binary
  (23/23 each, fork CUDA build).** Friday's "Bonsai 23/23 vs base 22/23"
  was Ollama missing the string reversal, not the model — the same GGUF in
  the fork gets it. Twenty-three probes cannot measure retention: PrismML's
  published 98.2% is a claim this set is far too small to confirm or
  contradict, and a Q4_K_M baseline is itself a lossy stand-in for the
  base. What the probes do show is **no obvious collapse** on the shapes a
  coding agent produces: tool calls, JSON, code, instruction constraints
  all pass. A real retention number needs a Q8_0 baseline and a standard
  suite (GSM8K, EvalPlus, MMLU-Redux), which nobody has run here.
- **With thinking off the two are within one probe** (20-21/23). Both fail
  string reversal and one multi-step arithmetic item without reasoning;
  Bonsai additionally miscomputes the 09:40→13:15 duration (235 vs 215) on
  both CUDA and CPU, so that is the model, not the backend.
- **Thinking is cheap on these prompts**: ~100 tokens per probe median for
  both models. The token-count parity is itself a result — a distillation
  that had learned to ramble would show up here first.

### Reproduce

```powershell
# Bonsai on the fork (CUDA build; check the launcher prints bin\cuda):
python scripts\bonsai-bench.py --url http://127.0.0.1:8080 --model bonsai --nothink template_kwargs --label bonsai2-cuda
# Base model in Ollama, drafter off for the like-for-like decode row:
python scripts\bonsai-bench.py --url http://127.0.0.1:11434 --model qwen3.8:27b --transport ollama --ollama-opt draft_num_predict=0 --label qwen38-nodraft
# Base model under NoLlama (B60):
python scripts\bonsai-bench.py --url http://127.0.0.1:8000 --model auto --nothink nollama --label qwen38-b60
python scripts\bonsai-bench.py --compare bench-results\bonsai-*.json
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
