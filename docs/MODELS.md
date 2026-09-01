# Models

## Recommended models

New here, or re-running `install.ps1`? Pick a **use-case** in the menu — here are
the proven models per role on a Core Ultra laptop (NPU + ARC iGPU):

| Use-case | Role | Pick in the menu | HuggingFace | Size |
|---|---|---|---|---|
| Chat | **NPU chat** | Qwen3 8B (INT4-CW) | `OpenVINO/Qwen3-8B-int4-cw-ov` | ~5 GB |
| Vision | **GPU vision** | Qwen3-VL 8B (INT8) | `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov` | ~9 GB |
| Coding agent | **GPU/CPU coder** | Qwen2.5-Coder 7B (INT4) | `OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov` | ~5 GB |

Qwen3 8B is the best-quality text model verified on the NPU. Qwen3-VL 8B
is the matching vision model — the INT8 build keeps fine detail (OCR,
small numbers) and fits a 16 GB ARC; drop to the ~6 GB INT4 build
(`…-int4-ov`) if you're tight on VRAM. For **coding agents** (VS Code Copilot
Chat, OpenClaw), pick the "Coding agent" use-case and a **Qwen2.5-Coder** model —
7B for snappy turns, 14B for stronger multi-step work; it runs on the GPU, or on
the CPU (which beats a weak iGPU on strong desktops). All are pre-exported — **no
conversion step**, though the multi-GB download still takes a while — and returning
users see them flagged **"Already on disk"** (those link instantly).

## Models we publish

Where a model the ecosystem needs doesn't exist in OpenVINO form, we build,
verify and [publish it on HuggingFace](https://huggingface.co/aweussom) —
several of these are the only OpenVINO builds of their model in existence.
All are in the `install.ps1` menu, with measured numbers on real hardware:

| Model | NPU (285K) | Notes |
|---|---|---|
| [`SmolLM3-3B-int4-cw-ov`](https://huggingface.co/aweussom/SmolLM3-3B-int4-cw-ov) | 23.3 tok/s | New in OpenVINO 2026.3; also runs GPU (29.7) / CPU (37.5) |
| [`SmolLM3-3B-int8-cw-ov`](https://huggingface.co/aweussom/SmolLM3-3B-int8-cw-ov) | 12.3 tok/s | Quality-first variant; ~half the speed of int4-cw |
| [`LFM2.5-1.2B-Instruct-int4-cw-ov`](https://huggingface.co/aweussom/LFM2.5-1.2B-Instruct-int4-cw-ov) | **38.8 tok/s** | Fastest model we've verified on an NPU. NPU-only build. **NPU 3 only** — see below |
| [`LFM2-1.2B-int4-cw-ov`](https://huggingface.co/aweussom/LFM2-1.2B-int4-cw-ov) | 36.5 tok/s | NPU-only build. **NPU 3 only** — see below |
| [`Qwen2.5-VL-3B-Instruct-int8-ov`](https://huggingface.co/aweussom/Qwen2.5-VL-3B-Instruct-int8-ov) | — (GPU VLM) | The proven small vision model, now a download instead of a 10-min conversion. Research license |
| [`LFM2-8B-A1B-int4-ov`](https://huggingface.co/aweussom/LFM2-8B-A1B-int4-ov) | — (GPU MoE) | 87 tok/s resident on an Arc 140V; the disk-offload test model |

The NPU builds are **channel-wise** exports (`-cw`) on purpose: the default
group-quantized int4 that `optimum-cli` produces crashes the Intel NPU
driver compiler (a known vpux bug — `"Found N duplicated names"`). If you
convert your own models for the NPU, use `download-model.ps1 -Weight
int4-cw` (or `int8-cw`), which encodes the working recipe.

**The two LFM builds are NPU 3 only (Arrow Lake / Meteor Lake).** On a
Lunar Lake NPU 4 (Core Ultra 7 258V) the same files run at 46–48 tok/s and
emit word salad — `Say hello.` → `cohclclclcl…`, byte-identical across
OpenVINO 2026.3.0, 2026.3.1 and the 2026.5 nightly, with the plugin
compiler and the driver compiler alike; Intel's own
`OpenVINO/LFM2.5-350M-int8-ov` fails the same way, and the same file on
CPU/GPU in the same venv answers correctly. The 285K's NPU 3 runs them
correctly with identical software.

**Updating the NPU driver does not fix it** (checked 2026-09-01): output
is byte-identical on 32.0.100.4778 and 32.0.100.5540, while other models
stay correct on the same NPU and driver. Upstream:
openvinotoolkit/openvino#37322. Details in `TODONT.md`.

## Models

`install.ps1` shows a curated menu of models known to work on Intel
hardware. All pre-exported models are download-only (no conversion).
The menu is defined in `models.json` — add entries when new models
are verified.

### Gated or private models (HuggingFace token)

The curated `OpenVINO/…` models are public and download anonymously — no
token needed. You only need a [HuggingFace
token](https://huggingface.co/settings/tokens) (the `hf_…` string) for
**gated** models (ones that make you accept a license, e.g. Llama) or
**private** repos. Pass it with `-HfToken`:

```powershell
.\install.ps1 -HfToken hf_xxxxxxxxxxxxxxxxxxxxx
.\download-model.ps1 some-org/gated-model -HfToken hf_xxxxxxxxxxxxxxxxxxxxx
```

Note: `hf auth login` won't help on a first run — `install.ps1` is what
installs the `hf` CLI in the first place, so there's no `hf` to log in
with yet. `-HfToken` works on a clean machine because it sets `HF_TOKEN`
before the download (which `huggingface_hub` reads automatically). If you
already have an `hf auth login` token stored from elsewhere, that's used
too — `-HfToken` is just the bootstrap-proof way.

### Adding models outside the menu

Use `download-model.ps1` to grab any HuggingFace model:

```powershell
# Pre-exported OpenVINO model (just download)
.\download-model.ps1 OpenVINO/Qwen3-8B-int4-cw-ov

# Convert a HuggingFace model to OpenVINO (PowerShell flags: single dash)
.\download-model.ps1 Qwen/Qwen2.5-VL-3B-Instruct -Convert -Weight int8

# With trust-remote-code (some models require this)
.\download-model.ps1 Qwen/Qwen2.5-VL-3B-Instruct -Convert -Weight int4 -Trust
```

Models download to `~/models/<name>/`. Point NoLlama at them:

```powershell
python nollama.py --model-dir ~/models/my-model --device GPU
python nollama.py --gpu-model-dir ~/models/my-vlm
```

Model folders are sanity-checked both at install time and at server start:
the `openvino_model.bin` + `.xml` pair must be present, and the `.bin` must
be at least as large as the `.xml` says it should be (byte-exact — catches
interrupted downloads and half-synced copies). A broken folder is re-fetched
by the installer, or refused at load with an error that says exactly what's
missing — so if you assembled a model directory by hand and NoLlama rejects
it, trust the message, not the folder listing.

### Model won't load? Run the canary first

Before debugging anything else, establish whether the problem is **your
model** or **your stack**. The registry's smallest model is a ~1 GB
known-good canary — output quality is terrible, that's not the point;
it loads everywhere:

```powershell
.\download-model.ps1 OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov
python nollama.py --model-dir "~/models/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov" --device NPU   # or GPU / CPU
```

- **Canary loads, your model doesn't** → the stack is healthy; the model
  is the problem. NPU limits are real: INT4-**CW** quantization, ≤ 8B
  params (~6 GB on disk). Bigger or group-quantized INT4 models die in
  the NPU compiler ("Compilation failed", `vpux-compiler` errors). Run
  that model on GPU/CPU instead, or pick an NPU model from the menu.
- **Canary fails too** → driver/stack problem, not the model. Windows:
  update the Intel NPU driver. Linux: the `intel-npu-driver` and
  `intel-npu-compiler` versions must match each other and OpenVINO —
  distro-repo packages often lag; use the
  [intel/linux-npu-driver releases](https://github.com/intel/linux-npu-driver/releases)
  compatibility table.

The install-menu models are the proven set; anything you bring via
`download-model.ps1 -Convert` is best-effort territory — the server's
startup log now says explicitly what's wrong (missing/truncated files,
memory that won't fit, KV pool too small, NPU compiler rejection)
instead of generic errors, so read it before opening an issue. 🙂

### Finding newer/better models

The model menus rot fast — new architectures appear monthly. The
authoritative place to look is the OpenVINO org on HuggingFace:

**[huggingface.co/OpenVINO](https://huggingface.co/OpenVINO)**

These are pre-exported by Intel, so there's **no conversion step** — just a
download (still slow for multi-GB models, but no 5-20 min `optimum-cli` export).
What to look for:

| Suffix | Where it runs | Notes |
|---|---|---|
| `-int4-cw-ov` | NPU + GPU | Channel-wise INT4. NPU's preferred format. |
| `-int4-ov` | GPU only | Standard INT4. Not always NPU-compatible. |
| `-int8-ov` | GPU + CPU | Better fine-detail retention than INT4 (OCR, numbers). |
| `-fp16-ov` | GPU + CPU | Full precision. Largest, slowest, sharpest. |

Quick rules of thumb:
- **NPU chat:** must be `-int4-cw-ov` and ≤ 8B params (~6 GB on disk) —
  a 14B INT4 fits the old size advice but fails in the NPU compiler (#20).
- **GPU vision (VLM):** any `-int4-ov` or `-int8-ov` model marked
  "Image-Text-to-Text" on HF.
- **GPU LLM (smarter than NPU):** any `-int4-ov` model up to your
  VRAM. Above ~16 GB falls back to CPU silently.
- **Whisper (STT):** OpenVINO ships pre-quantized whisper variants
  (`whisper-{tiny,base,small,medium,large-v3}-{int4,int8,fp16}-ov`).

Once a model proves itself, add it to `models.json` so it appears in
the install menu. Keep "Untested" tags on entries that haven't been
verified yet — be honest about what's measured vs. assumed.

> **Recommended VLM:** OpenVINO ships
> [Qwen3-VL-8B](https://huggingface.co/OpenVINO/Qwen3-VL-8B-Instruct-int8-ov)
> pre-exported in INT4/INT8/FP16 — the natural vision sibling to the
> proven Qwen3-8B NPU chat model. The INT8 build is verified here on the
> Arc 140V in dual mode (2026-06-16) and is the default GPU vision pick
> (see [Recommended models](#recommended-models)); INT4 is the lighter
> ~6 GB option.

### NPU models (chat)

| Model | Size | Notes |
|---|---|---|
| Qwen3 8B (INT4-CW) | ~5 GB | Recommended. Best quality. |
| Phi 3.5 Mini (INT4-CW) | ~2 GB | Smaller, faster. |
| DeepSeek R1 Distill 7B (INT4-CW) | ~4 GB | Reasoning. |
| DeepSeek R1 Distill 1.5B (INT4-CW) | ~1 GB | Testing only. |
| Mistral 7B v0.3 (INT4-CW) | ~4 GB | General purpose. |

### GPU vision models

| Model | Size | Notes |
|---|---|---|
| Qwen3-VL 8B (INT8) | ~9 GB | Recommended pairing for 16 GB ARC. Keeps fine detail (OCR, numbers). |
| Qwen3-VL 8B (INT4) | ~6 GB | Lighter alternative. Newer Qwen-VL generation; verified on Xe-LPG. |
| Qwen2.5-VL 3B (INT8, convert) | ~4 GB | Proven. INT8 better at fine detail (OCR, numbers). |
| Gemma 3 4B Vision (INT4) | ~3 GB | Untested. |
| Gemma 3 12B Vision (INT4) | ~7 GB | Untested. Needs ~12 GB RAM with KV cache. |
| InternVL2 4B (INT4) | ~3 GB | Untested. |
| Phi 3.5 Vision (INT4) | ~3 GB | Untested. |

### GPU large LLMs (smarter than NPU)

| Model | Size | Notes |
|---|---|---|
| Qwen3 14B (INT4) | ~8 GB | Great reasoning. |
| Qwen3 30B-A3B MoE (INT4) | ~17 GB | 30B brain, 3B speed. |
| Phi 4 (INT4) | ~8 GB | Strong reasoning. |
| Phi 4 Reasoning (INT4) | ~8 GB | Chain-of-thought. |

### Gemma 4 on OpenVINO: measured 2026-08-21

Intel publishes three Gemma 4 exports. All are `image-text-to-text`, so they
all load as **VLM slots** (tool calling and prefix caching apply). Tested on
an Arc Pro B60 with the 2026.3 release:

| Model | Size | Notes |
|---|---|---|
| `gemma-4-E2B-it-int4-ov` | ~4 GB | Works, prefix caching works. Weakest: misreads small text and, when it cannot resolve an image, replies *"the image is missing"* instead of saying so. Nothing is wrong with your setup. |
| `gemma-4-E4B-it-int8-ov` | ~8 GB | Reads detail E2B cannot, but **gets no prefix caching** on this runtime — its IR has no SDPA op, so the caching backend cannot be built. Fine for one-shot vision (it is ~2.2x faster on a cold turn); for agents use our re-export below. |
| [`aweussom/gemma-4-E4B-it-int8-ov`](https://huggingface.co/aweussom/gemma-4-E4B-it-int8-ov) | ~8 GB | Same weights re-exported with fused attention so **prefix caching works** — ~2.6x faster per turn on a repeated prefix, byte-identical answers to Intel's export. Published 2026-08-21; reported upstream ([discussion](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int8-ov/discussions/1), [optimum-intel#1948](https://github.com/huggingface/optimum-intel/issues/1948)). |
| `gemma-4-26b-a4b-it-int4-ov` | ~15 GB | Best of the three; prefix caching works. Its KV is **240 KB/token**, so the auto-sized pool lands on its 2 GB floor and holds only ~8k tokens — **pin `--cache-size-gb`** before running an agent against it. NoLlama warns at load. Loads in ~40s. |

Two things that apply to every VLM this size, not just Gemma:

- **They cannot count objects.** Shown 17 dots, six different
  model/stack/quantization combinations answered 15, 15, 16, 18, 20 and 20 —
  including a 25.8B model. Verified against Ollama/llama.cpp on an RTX 5090
  as well, so it is the model class, not OpenVINO.
- **A model claiming it received no image is usually just failing the task.**
  Same weights on llama.cpp answer confidently but wrongly where the OpenVINO
  build says "the image is missing". Check with a simpler question on the same
  image before suspecting your plumbing.

Loading a 15 GB model stages through host RAM, so **peak system memory during
load is roughly model-sized** even on a discrete card. Two such loads at once
on a 32 GB box will thrash the pagefile for tens of minutes; start one server
at a time.

## A note about small models

During initial NPU testing with DeepSeek R1 1.5B, we asked:
"What is the capital of Norway?"

The model's response:

> "I need to figure out the capital of Norway. I know it's a country
> in Norway. I remember that Norway is a small island..."

Norway is, in fact, not a small island.

Or *is* it? To paraphrase the greatest detective of all time, Ford
Fairlane: "...an island in an ocean of diarrhea."

The point: 1.5B parameter models are for testing the plumbing, not
for geography. Use Qwen3-8B or larger for actual chat. The small
models will catch up — they're getting smarter every month.

## Muse Glimmer: now on the GenAI path

Muse Glimmer graduated off the optimum backend on 2026-08-18. Intel
published an official **VLM-shaped** export —
[`OpenVINO/Muse-Glimmer-30B-int4-ov`](https://huggingface.co/OpenVINO/Muse-Glimmer-30B-int4-ov)
(2026-08-12, 17 GB, INT4_ASYM g64) — and a VLM-shaped export runs on
GenAI's `VLMPipeline`, which feeds the language model from embeddings by
design. NoLlama routes any VLM-shaped `muse_glimmer` export there
automatically (our own
[`aweussom/Muse-Glimmer-30B-int4-ov`](https://huggingface.co/aweussom/Muse-Glimmer-30B-int4-ov)
is VLM-shaped too and reroutes the same way — it is **superseded** by
Intel's, staying published as the repro for openvino#37419). Verified raw
on the Arc Pro B60 (2026-08-18): loads, quotes the instruction back
verbatim, ~14 tok/s against 8–11 on the optimum path — and the GPU
corruption story below **never applied to the GenAI path**. Glimmer's ATEM
reasoning channels are translated to `<think>` blocks on this path too
(plain-text translation — the pipeline strips the channel markers).

Confirmed on an **integrated** GPU as well — the datapoint that was
missing, since every corrupted Glimmer result had come from the optimum
path. Arc B390 iGPU (Core Ultra X7 358H, Fedora, nightly runtime
`2026.4.0-22828`, Intel's export, community report in issue #29,
2026-08-21): loads as a VLM slot, answers correctly, reasoning framed in a
`<think>` block with no channel-marker leak, **4.5 tok/s** decode (66
tokens in 14.6 s). Same box, same model, optimum path on 2026.3 garbled —
so the split is the *backend*, not the GPU generation. Budget roughly a
quarter of discrete-Battlemage decode on an iGPU: chat-grade, not
agent-loop-grade.

What to know before switching:

- Needs **OpenVINO 2026.3.1 or newer** — the 2026.3.1 release (2026-08-26)
  is where Intel lists it as functionally enabled, and `requirements.txt`
  now floors there. Verified on the Arc 140V with the release wheels
  (2026-08-30): quotes the instruction back verbatim, ~2.5 tok/s with 17 GB
  of weights in shared memory — chat-grade on an iGPU, agent-grade on a
  B60 (~14 tok/s). It is in the `install.ps1` menu as of the same day.
- It lands on a **VLM slot** — which since 2026-08-18 means **tool calling
  works** (structured `tool_calls` on both API surfaces, verified with
  Glimmer on the B60) and **prefix caching works** (VLMPipeline honors
  `scheduler_config` on 2026.3+; measured on the B60: a 33k-token prefix
  went 54.5s → 1.3s TTFT on the repeat turn — what an agent's fixed system
  prompt does every turn). Prewarm covers VLM slots too, so a restart
  prefills the captured agent prompt before the first request (Glimmer/B60:
  first turn 12.4s → 0.65s TTFT). Images are untested on Glimmer.
- `--backend optimum` still forces the old path (needs `venv-optimum/`).

## Brand-new architectures: the optimum backend

Some architectures land in optimum-intel (export) before openvino_genai
(serving) learns to run them — as of 2026-08 that's **NVIDIA Nemotron 3.5
Lightning** (`nemotron_h`); Muse Glimmer lived here until Intel's official
export (above), and its history fills the rest of this section. NoLlama
serves these through optimum-intel's python runtime instead: detection is
automatic (`--scan` shows a `Backend` line; `--backend` overrides), tool
calling works, and both API surfaces behave identically. Differences from
GenAI slots: **text-only for now** (images get a clean 400), no prefix
cache / prewarm (a GenAI feature), no `--offload-ratio`, and no NPU. GPU
support also depends on the OpenVINO GPU plugin executing the model's
dynamic-shape graph. On OpenVINO **2026.3 and earlier, no Intel GPU runs
Glimmer correctly on this backend — integrated or discrete** — so on a
release runtime, use `--device CPU`. **Fixed in 2026.4** (verified on the
nightly, see below):

- **2026.4.0.dev20260814** (Arc Pro B60, 2026-08-15): the issue's own repro
  script now quotes the prompt **verbatim** on GPU and reasons coherently to
  the right answer, where 2026.3 misquoted it and answered a question that
  was never asked. Decode also went from ~2 to **8–9 tok/s** — nine times the
  CPU control on the same box, which turns Glimmer-on-GPU from
  verification-only into something usable. NoLlama checks the runtime version
  at load and downgrades its GPU warning to a note on 2026.4+.

  **So Glimmer on an Intel GPU is coming, and we know it works** — but the
  fix is only in a nightly today, and NoLlama stays leading edge rather
  than bleeding edge. It moves into `install.ps1`/`models.json` when 2026.4
  ships as a *release* (and the stack gate closes too — see
  `NEXT-STEPS.md`); until then the manual path below is the honest
  offering. Sanity-check your first reply regardless: the failure mode was
  always silent.

The 2026.3 evidence, kept because it's what the version check is protecting
you from:

- **Xe-LPG** (desktop Arrow Lake iGPU): fails loudly at warmup
  (`Count is called for dynamic shape`).
- **Xe2** (Arc 140V, Windows, verified 2026-08-13): loads and warms up
  fine, then **silently computes garbage** — the model half-perceives the
  prompt (drops words) and greedy decoding degenerates into a two-word
  loop inside the think channel. The same IR with the same sampling params
  comprehends and complies perfectly on CPU. There is no error to catch:
  the only symptom is a model that seems drunk.
- **Xe3** (Arc B390 iGPU in Core Ultra X7 358H, **Linux**, community
  report in issue #24, 2026-08-13): identical corruption fingerprint —
  same "the user message is garbled" half-perception, same think-loop
  hang under greedy. Three iGPU generations and two OSes rule out any
  Windows-driver or Xe2-specific theory. (Runtime confirmed after the fact
  as `2026.3.0-22451` — a release, not the 2026.4 nightly, so it is *this*
  bug and not a hole in Intel's fix. The same device runs Glimmer correctly
  on the GenAI path — see above.)
- **Discrete Battlemage** (Arc Pro B60 24 GB, Windows, verified
  2026-08-15): **same corruption.** Dedicated VRAM does not save it, which
  kills the shared-memory theory the iGPU-only evidence had suggested.
  "Say hi" returned `Respond directly.` in the think channel — the system
  prompt restated with most of its words missing — then the `HELLO!`
  prompt looped until cancelled, GPU pegged near 100% the whole time. It
  is a runaway generation, not a deadlock: it would have ground on to
  `max_tokens` and returned garbage.

  Controlled on the same machine, same venv, same IR, same session: CPU
  quoted the instruction back verbatim and answered `HELLO!` correctly.
  That control matters more than it looks — `install-optimum.ps1` tracks
  transformers `main`, so without it "the GPU is broken" and "transformers
  regressed this week" fit the evidence equally well.

  Read the *think channel*, not the answer — but note that restating the
  system prompt there is normal Glimmer behaviour, on CPU too. The tell is
  words going *missing* from that restatement, not the restatement itself.
  (Distinct from the Xe2 case, where it quoted a system prompt that was
  never sent at all — that one really is hallucination.)

Four device classes across two OSes, so on 2026.3 this was plugin-wide rather
than any one generation; tracked upstream as
[openvinotoolkit/openvino#37419](https://github.com/openvinotoolkit/openvino/issues/37419),
and **fixed in 2026.4**. Re-run the comprehension test on each new OpenVINO
release anyway — and note the GenAI path was never affected: Qwen3.8-27B runs
correctly on the same B60 on both runtimes.

Test it yourself with `.\install-optimum.ps1 -Nightly`, which builds a second
`venv-optimum-nightly/` and leaves the release venv intact as a control. Keep
that control: without a same-venv, same-session CPU run, "the GPU plugin
changed" and "transformers main moved" are indistinguishable, because
`install-optimum.ps1` tracks git main for both.

The catch is the python stack: these models need transformers **from git
main** plus optimum-intel **from git main**, which no NoLlama venv pins.
`install-optimum.ps1` (Windows and Linux, needs git on PATH) builds a
dedicated `venv-optimum/` with the right stack in the right order — the
order matters: optimum-intel pins `transformers<5.6`, so the git
transformers goes in last to override it:

```powershell
.\install-optimum.ps1
venv-optimum\Scripts\python.exe nollama.py --model-dir ~\models\Muse-Glimmer-30B-int4-ov --backend optimum --device CPU --idle-timeout 0
```

(`--device CPU` is deliberate — see the iGPU verdicts above. `--backend
optimum` is now required for Glimmer: auto-detection routes its VLM-shaped
export to the GenAI path, which this venv's release runtime can't run. If
upstream main breaks, pin with `-TransformersRef <commit>` /
`-OptimumIntelRef <commit>`.)

Running a plain install against such a model exits immediately with an
error naming this section instead of failing minutes into the load. When
openvino_genai gains these architectures, `--backend genai` (or just
re-exporting) moves them onto the faster path with prefix caching.

Measured on this backend (Muse Glimmer 30B int4, short chat prompts,
2026-08-13 — historical; Glimmer's current path is GenAI, see above):
Core Ultra 7 258V laptop CPU 1.4 tok/s / TTFT 12.9 s; Core Ultra 9 285K
desktop CPU 2.6 tok/s / TTFT 9.6 s. Dense-30B bandwidth physics — fine
for verification, not agent loops; a 24 GB Arc-class card is the real
host. Note Glimmer *always* reasons by default (`reasoning_strength`
defaults to high in its template); the web UI's no-think toggle sends its
native `Reasoning strength: low.` directive.

## Qwen3.8-27B: a release model since 2026.3.1 — pinned to a repo branch

[`OpenVINO/Qwen3.8-27B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov)
spent two weeks needing a nightly runtime; OpenVINO **2026.3.1**
(2026-08-26) lists it as functionally enabled and it is back in the
`install.ps1` menu. One trap the menu handles for you: the repo's **`main`
branch is a 2026.4-toolchain export that segfaults the 2026.3.x runtime at
load** (no error, the process just dies); the IR that matches the release
lives on the **`2026.3.1` branch**, so the registry entry carries
`"revision": "2026.3.1"` and `download-model.ps1` takes `-Revision`. Verified
2026-08-30 on the Arc 140V with the release wheels: correct answers,
**3.6–4.8 tok/s** — a dense 27B on an iGPU is quality-over-speed. Its chat
template opens the `<think>` block itself; NoLlama detects that at load so
reasoning still arrives as `reasoning_content`.

## The OpenVINO nightly runtime (`-Nightly`) — a test harness, not a model gate

Sometimes Intel publishes a working OpenVINO IR *before* the runtime that
can read it ships (Qwen3.8 above was the 2026-08 case). Nothing in the
registry needs a nightly today; `-Nightly` remains the way to ask "does the
next runtime fix X?" without touching the venv you serve from.

```powershell
.\install.ps1 -Nightly     # builds venv-nightly/, leaves venv/ alone
.\start.ps1                # generated pointing at venv-nightly
```

On a fresh Windows box that has neither PowerShell 7 nor a relaxed
execution policy, go through the shim instead — it forwards its arguments:

```bat
install-windows.bat -Nightly
```

`-Nightly` never touches your stable `venv/`. It builds a second, complete
runtime in `venv-nightly/` and bakes `-VenvName venv-nightly` into the
generated `start.ps1`, so a machine can hold both and launch either. The
install prints the exact wheel versions it landed on — quote those in any
bug report, because "openvino nightly" is not a reproducible statement.

A registry model that ever needs this stack again carries
`"requires_nightly": true` in `models.json` and is **hidden from the normal
menus** (the menu says how many, and how to see them). Offering a 16 GB
download that then fails to load is worse than not offering it.

Know what you are signing up for before you pull Qwen3.8's 15 GB:

- It is a **VLM**, so it lands on `VLMPipeline`. Prefix caching works on
  VLM slots (verified 2026-08-18 on 2026.3 release and the 2026.4
  nightly), so agent turns reuse the prefilled system prompt — and prewarm
  covers VLM slots too, so after the first captured run even a restart's
  first turn is a cache hit.
- It is **dense**, not MoE, so `--offload-ratio` does nothing for it. The
  15 GB of weights must be resident.
- That rules out a stock 16 GB Arc 140V. It wants a 24 GB card (Arc B60)
  or a raised shared-memory budget (the 140V run above used a 25 GB
  override).
- `venv-nightly/` pins `transformers==5.2` per Intel's model card, which
  is the opposite of `requirements.txt`'s `<5` cap — so **Qwen3-Next
  conversions must still run from `venv/`**.

