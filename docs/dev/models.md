# Model plumbing: naming, integrity, `--scan`, export

Read this before touching model discovery, display names, `--scan`,
weight-integrity checks, or `download-model.ps1`.

## Naming — the directory name is authoritative

The directory name is the web-UI label **and** the model ID clients
request. `resolve_display_name` uses the name as given and only follows a
symlink/junction when that name is generic (`model/`, `gpu-model/` — what
`install.ps1` links). It previously called `realpath` unconditionally,
which silently discarded a deliberate rename (#19).

There is deliberately **no `--model-name` flag** — see `TODONT.md` for why
the rename *is* the interface.

## Weight integrity

`install.ps1` validates local/cached models before offering or linking
them: the `.bin`+`.xml` pair must exist and the `.bin` must not be
truncated (#17). The IR `.xml` records each weight blob's offset+size, so
**max(offset+size) is the exact minimum `.bin` size**. The IR has no
checksum; truncation is the realistic failure, corruption-in-place is out
of scope.

`nollama.py` re-checks the same invariant at load
(`_verify_weights_integrity`) since models can arrive without
`install.ps1`. A truncated/missing model fails with a plain-English error,
and the "Is another process using the NPU?" hint is suppressed for that
class of failure.

## `--scan`

Reports what each model directory actually holds: display name (and where
it came from), LLM/VLM/Whisper, architecture, MoE shape, geometry,
integrity, and the **real weight precision read from the IR's model-level
`<rt_info>`** (`nncf/weight_compression/mode` + `group_size` + `ratio` +
`awq`) rather than from the folder name, which can lie. `--scan` also shows
a `Backend` line.

`read_ir_rt_info` seeks the **tail** of the `.xml`: the graph is tens of MB
on a large model, and the model-level block is the last `<rt_info>`, after
`<edges>`.

No server, no device init, no model load.

## `download-model.ps1`

Fetch/convert any HF model. **PowerShell-style flags** (`-Convert -Weight
int4 -Trust`), NOT GNU `--convert` — #19: the docs once showed `--` syntax
and users copy-pasted it, so a catch-all param now prints the corrected
command when someone tries.

**Conversion is RAM-bound, not disk-bound.** optimum-intel's Qwen3-Next
patcher builds an fp32 copy of every expert weight (workaround for OpenVINO
CVS-181449) — for Qwen3-Coder-Next that's `512 experts × 2048 × 512 × 4 B`
= 2 GB per projection stack, ~288 GB across 48 layers × 3. Measured: **400
GB of Windows pagefile (on 128 GB RAM) succeeded**, 200 GB did not (#19,
Dmitriy Teteruk). Weight format is irrelevant to this stage — the blowup
happens before quantization.

## Two probes, and which question each answers

`bare-probe.py` establishes that the runtime can run the model at all, and is
the standing order before anything else. After that there are two, and using
the cheap one to answer the expensive question is the mistake to avoid.

| Probe | Cost | Answers | Cannot answer |
|---|---|---|---|
| `scripts/pelican-probe.py` | one request, seconds | is this model any good | whether it can drive a tool loop |
| `scripts/agent-probe.ps1` | two OpenCode tasks, minutes | can it drive OpenCode, on this hardware | how good the prose or code is |

Tier 1 is Simon Willison's pelican-on-a-bicycle prompt, verbatim, because the
value is the published corpus to compare against. It runs against any
OpenAI-compatible URL, so the whole registry can go through it in minutes —
including a server on another box.

Tier 2 builds a throwaway project with two failing tests and decides pass/fail
by **running the tests**, never by reading the transcript. That matters because
the failure it catches reads like success: a model that narrates tool use and
hands the work back sounds exactly like one that did it.

**Only tier 2 promotes a model to `"agent": true` in `models.json`.**
Qwen2.5-Coder-14B writes valid Python at 6-8 tok/s and cannot call a tool
[OBSERVED 2026-09-23]; tier 1 would have passed it.

**A tier-2 FAIL counts only once the model has run in its own tool dialect.**
Our default rendering speaks Qwen3-Coder XML to every model;
`--tool-template native` renders the model's own template. Neither dialect
rescued the 16 GB candidates [OBSERVED 2026-09-23, 140V, OpenVINO 2026.4,
transcripts in `bench/agent-probe/`]:

| model | Qwen-XML | native | how it fails, both ways |
|---|---|---|---|
| `LFM2.5-8B-A1B-int4` | 0/2 | 0/2 | invents paths (`/workspace/...`); native also sends `file_path` against a `filePath` schema, repeatedly after the error names the key |
| `Qwen3-8B-int4-cw` | 0/2 | 0/2 | invents `src/calc.py` in every run; native also overwrote the test file |
| `Qwen3-14B-int4` | fix in 18 min, syntax error on the way | before the fixes below: fix 3/4, feature 0 verified in 6. **After them: 7 of 8 tasks** -- B60 4/4 (50-166 s a task), 140V 3/4 (fix 484 s / 188 s, feature 1034 s / FAIL) | the failures were two of OURS: calls written inside a `<think>` it never closed were dropped (08bf596; the recovery fired 3+ times per run on both GPUs), and the Windows fixture went CRLF under the reset (cc26d26). The one remaining FAIL is the model's: discount bug left in, test file mangled until its own runner printed nothing -- a false pass the original-tests check caught |

Qwen3-14B's rows ran at `--cache-size-gb 3`, which is the 16 GB configuration
(~12 GB on the iGPU with Shared GPU Memory Override raised). Native is
clearly better *for it*, and it is still not an agent: it lands a single
focused fix, not a two-part task -- **superseded 2026-09-24**: with the unclosed-`<think>` recovery and an LF fixture it passes 4/4 on the B60. The earlier B60 runs [OBSERVED 2026-09-24, Arc Pro
B60, driver 32.0.101.8805, OpenVINO 2026.4] fail the same way the 140V's do,
three times faster, so the laptop iGPU was never the variable. Since 2026-09-24 the probe verifies
against the fixture's original tests and keeps each task's diff, so a
rewrite-the-tests pass can no longer happen unseen.

## Two traps when re-exporting a model yourself (2026-08-21)

Both found while re-exporting `google/gemma-4-E4B-it` to check an Intel IR.

1. **Attention must come out fused, or you silently lose prefix caching.**
   The CB backend is built by rewriting `ScaledDotProductAttention` nodes, so
   an export that traced decomposed matmul+softmax attention cannot use it.
   optimum-intel only pins the attention implementation for models listed in
   `FORCE_ATTN_MODEL_CLASSES`; everything else takes whatever the export
   environment resolves to. Check the result, don't assume:

   ```bash
   python nollama.py --scan <dir>   # Prefix caching : yes — N fused SDPA ops
   ```

   Any count above zero can cache; `> 0` is the predicate, **not** one per
   layer — hybrids legitimately carry fewer (`prefix-cache.md` has the
   measured table).

2. **The chat template is baked into `openvino_tokenizer.xml` at export
   time.** Editing `chat_template.jinja` in a finished export changes
   nothing. And a template that works in Python Jinja2 may not parse in
   openvino_genai's C++ Jinja — Google's Gemma 4 template uses implicit
   string concatenation inside `raise_exception(...)`, which fails with
   "Expected closing parenthesis in call args" at warmup. Patch the template
   in the *source* directory before exporting.

Failure mode 1 is invisible until you read the load log; failure mode 2
loads and caches fine and then dies at warmup. See `prefix-cache.md`.

## NPU export rule (2026-08-06)

Models converted for the NPU **must be channel-wise**
(`download-model.ps1 -Weight int4-cw` or `int8-cw`): default
group-quantized int4 IRs crash the NPU driver compiler ("Found N
duplicated names", known vpux bug).

`int8-cw` halves decode against `int4-cw` but keeps more quality — except
on the LFM2 family, where no good int8 NPU variant exists (see
`TODONT.md`).

## Verified models

### Embedding models (issue #43)

Served through `--embed-model-dir`, not a chat slot. Both probed
bare through `TextEmbeddingPipeline` first, then through the server
[OBSERVED 2026-09-17, this laptop: Core Ultra 7 258V + Arc 140V,
OpenVINO 2026.3.1, GPU driver 32.0.101.8991].

| model | CPU | GPU (140V) | NPU |
|---|---|---|---|
| nomic-embed-text v1.5 (fp16, ONNX route) | 768 dims, 0.20s/3 docs | 0.59s/3 docs | **refused** |
| all-MiniLM-L6-v2 (int8) | 384 dims, 0.03s/2 docs | 0.63s/2 docs | **refused** |

**Tested against a real RAG client, not just curl** [OBSERVED 2026-09-17]:
LangChain (`langchain-ollama` / `langchain-openai`, i.e. the official `ollama`
and `openai` SDKs) over FAISS, indexing this repo's own docs — 25
documents, 483 chunks — then answering questions with Qwen3-8B on the iGPU
from the same NoLlama process. Index built in 297s; retrieval returned the
right source files; 3 of 3 answers were correct and grounded. The same text
through `/v1/embeddings` and `/api/embed` agrees to cosine 1.0000.

**That test is what found the batching defect.** A RAG client sends the
whole corpus in ONE request; passing it straight to the pipeline measured
431s and a peak RSS of **18.52 GB** for 375 chunks, on a 32 GB laptop.
Sliced, the same work is 224s and ~5 GB. Throughput is flat-to-better at
small slices, so the default is 16:

| slice | total (375 chunks) | ms/chunk | worst query wait during indexing |
|---|---|---|---|
| 8 | 247.8s | 661 | 10.1s |
| **16 (default)** | **223.9s** | **597** | 15.4s |
| 32 | 236.4s | 631 | 25.0s |
| 64 | 289.7s | 773 | 55.3s |
| unsliced | 431.0s | 1149 | the whole corpus |

The lock is per slice, so a query waits one slice, not the corpus. None of
this was visible from single-request testing — every endpoint answered in
milliseconds by hand.

CPU and GPU vectors are interchangeable in one index: cosine **0.999999**
(nomic) and **0.999781** (MiniLM int8) between the two devices. Retrieval
behaves — a query matched its own document at 0.71-0.81 against 0.34-0.43
for an unrelated one, through the server on the Ollama route.

**The NPU refuses both, and that is the result, not a skip:**

```
Check 'check_sdpa_nodes(model)' failed at
src\plugins\intel_npu\src\plugin\npuw\embedding\prepare_embedding_model.cpp:345
```

The NPU's embedding path wants SDPA nodes in the graph; a standard encoder
export has plain attention, so it never compiles. This is not the NPU
prompt cap or an export mistake — it is the plugin declining the
architecture, identically for an int8 optimum export and an fp16 ONNX
conversion. Do not spend another afternoon on it without an upstream
change; `--embed-device NPU` falls back to CPU with a warning.

Small is not slow here: CPU beat the iGPU on both models, because an
encoder pass of a few hundred tokens is dominated by dispatch rather than
compute. Hence `--embed-device CPU` is the default. MyrkoF measured the
reverse on a loaded box (9.29s Ollama CPU vs 0.02s OpenVINO iGPU, issue
#43) — that machine was under load ~19 and the comparison was against
Ollama, so both readings can be true. Measure on the box you will serve on.

### Whisper (speech-to-text) on CPU, iGPU and NPU 4 (2026-10-11)

Intel's `OpenVINO/whisper-{base,small,medium}-int8-ov` as shipped (stateful),
bare `WhisperPipeline`, no NoLlama, 258V laptop: OpenVINO/genai 2026.4.1,
NPU driver 1005540 (`32.0.100.5540`), GPU driver 32.0.101.8991. 64 clips:
`jfk.wav`, OpenVINO's `how_are_you_doing_today.wav`, 20 each from FLEURS
`nb_no` and `en_us` test (read speech), 20 from NPSC session 20170207
(Storting, spontaneous), one 62 s concatenation of FLEURS nb, 10 s of
silence. `language` set per clip, except on the silence. WER on lowercased,
punctuation-stripped words [OBSERVED 2026-10-11]:

| model | device | compile | RTF nb / en / NPSC | WER nb / en / NPSC / 62 s |
|---|---|---|---|---|
| base | CPU | 1 s | 0.105 / 0.081 / 0.176 | 0.46 / 0.12 / 0.81 / 0.50 |
| base | iGPU | 20 s | 0.044 / 0.041 / 0.062 | 0.46 / 0.12 / 0.88 / 0.50 |
| base | **NPU** | 46 s | **0.019 / 0.023 / 0.043** | 0.46 / 0.12 / 0.82 / 0.50 |
| small | CPU | 3 s | 0.238 / 0.220 / 0.355 | 0.26 / 0.08 / 0.62 / 0.30 |
| small | iGPU | 16 s | 0.059 / 0.066 / 0.077 | 0.26 / 0.07 / 0.65 / 0.30 |
| small | **NPU** | 102 s | **0.036 / 0.048 / 0.102** | 0.26 / 0.08 / 0.61 / 0.33 |
| medium | CPU | 6 s | 0.715 / 1.158 / 1.710 | 0.15 / 0.06 / 0.52 / 0.13 |
| medium | iGPU | 15 s | 0.113 / 0.123 / 0.215 | 0.15 / 0.05 / 0.51 / 0.13 |
| medium | NPU | 156-246 s | -- | **refuses**, see below |

RTF is wall time over audio time; 0.02 is fifty times faster than real time.

- **base and small work on NPU 4 as shipped.** No stateless re-export
  needed; the earlier guess that the stateful export would not compile was
  wrong. The NPU is the fastest device for both, and its WER is the CPU's
  within noise. Transcripts are not byte-identical across devices (base:
  NPU matches CPU on 41 of 64 clips, iGPU matches CPU on 50); the
  differences are scattered single words, in both directions.
- **medium compiles on NPU 4 and dies on the first inference**:
  `L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST, code
  0x70000001 - device hung, reset, was removed, or driver update occurred`.
  Twice, compile 246 s and 156 s. The process then dies; the NPU itself
  recovers (base transcribed correctly straight after). **NPU 3 runs the
  same files** on the same driver and runtime (below), so this is an NPU 4
  fault. Not tried: a stateless re-export.
  Re-confirmed [OBSERVED 2026-10-20] with `scripts/whisper-npu-repro.py`
  on `jfk.wav`: identical DEVICE_LOST on the 2026.4.1 release (compile
  130 s) and on the 2026.5.0 nightly (23394 / genai 3521, compile 131 s);
  one earlier release run that day died with exit 139 (segfault) before
  printing the compile line. Base on the NPU (0.18 s) and medium on CPU
  (3.50 s) passed in between. 5540 is Intel's newest NPU driver as of
  2026-10-20, so there is no newer driver to try; reported upstream as
  [genai#4677](https://github.com/openvinotoolkit/openvino.genai/issues/4677)
  (2026-10-20).
- **Silence without a language hallucinates everywhere, worst on the NPU.**
  CPU and iGPU return `"you"`. NPU base returns a Norwegian phrase repeated
  to `max_new_tokens` (440 words); NPU small returns a six-word Norwegian
  sentence. The language detector is guessing on nothing, and the NPU's
  arithmetic tips it differently [INFERRED].
- **Norwegian accuracy is a model-size question, not a device one.** NPSC
  is parliamentary speech with names and dialect; base gets four words in
  five wrong, medium one in two. Read Bokmal (FLEURS) is far easier. None of
  this covers Northern Norwegian.
- Clips over 30 s chunk correctly on every device.

**NPU 3, and the 285K's CPU and Xe-LPG iGPU.** Same three models and files,
same 64 clips rebuilt from the sources by `scripts/whisper-build-set.py`
(group counts identical), same `scripts/whisper-probe.py`, one process per
model x device. Core Ultra 9 285K, OpenVINO/genai 2026.4.1, NPU driver
1005540 (`32.0.100.5540`, **the laptop's driver too**), GPU driver
32.0.101.8991. Run over SSH from the B60 box [OBSERVED 2026-10-19]:

| model | device | compile | RTF nb / en / NPSC | WER nb / en / NPSC / 62 s |
|---|---|---|---|---|
| base | CPU | 0.6 s | 0.027 / 0.051 / 0.091 | 0.46 / 0.12 / 0.81 / 0.50 |
| base | iGPU | 25 s | 0.028 / 0.019 / 0.032 | 0.46 / 0.12 / 0.86 / 0.50 |
| base | **NPU 3** | 13 s | **0.019 / 0.018 / 0.031** | 0.45 / 0.12 / 0.86 / 0.50 |
| small | CPU | 0.9 s | 0.137 / 0.174 / 0.309 | 0.26 / 0.08 / 0.61 / 0.30 |
| small | iGPU | 27 s | 0.053 / 0.051 / 0.087 | 0.26 / 0.08 / 0.63 / 0.30 |
| small | **NPU 3** | 36 s | **0.050 / 0.048 / 0.081** | 0.26 / 0.08 / 0.61 / 0.30 |
| medium | CPU | 1.8 s | 0.446 / 0.527 / 0.945 | 0.15 / 0.06 / 0.52 / 0.13 |
| medium | iGPU | 30 s | 0.114 / 0.119 / 0.210 | 0.15 / 0.05 / 0.52 / 0.12 |
| medium | **NPU 3** | 88 s | 0.147 / 0.142 / 0.249 | 0.15 / 0.06 / 0.52 / 0.13 |

- **medium runs on NPU 3**: 64 of 64 clips, no errors, WER equal to CPU.
  NPU 4 dies on the same files with driver and runtime held constant, so
  the NPU generation is the whole variable — the LFM2 pattern again.
- **WER is the CPU's on every device**, within a word or two per group.
  Transcripts identical to CPU on 40 / 52 / 55 of 64 clips (NPU 3, base /
  small / medium); iGPU 45 / 51 / 52.
- **Speed is closer than on the laptop.** NPU 3 is fastest for base and
  ties the Xe-LPG iGPU for small; for medium the iGPU is ahead. The 285K's
  CPU is 2-4x the 258V's, so the NPU's lead over CPU shrinks on a desktop.
- **Compile is 2-4x faster than NPU 4** (13 / 36 / 88 s against 46 / 102 /
  156-246 s) [INFERRED: desktop CPU doing the compile; not separated].
- **Silence hallucinates on NPU 3 exactly as on NPU 4**: base repeats a
  Norwegian phrase to 440 words, small gives a six-word Norwegian sentence,
  medium gives `Teksting av Amara.org` plus two CJK characters (Whisper's
  subtitle-credit hallucination). CPU and iGPU say `"you"`.
  `audio_level_spread` keeps such clips away from Whisper in the server.
- Raw JSONL on the 285K under `C:\devel\t042-whisper\out\`, outside the repo
  like the audio.

**Beside a chat model, through the server.** Qwen3-8B int4-cw streaming a
~670-token answer on the 140V iGPU while whisper-small transcribes 8 short
clips (FLEURS nb + smoke) back to back, for the whole stream; chat speed
alone measured before and after each arm and averaged
(`scripts/whisper-concurrency.py`). Two passes, opposite order
[OBSERVED 2026-10-11, same laptop and stack as above]:

| Whisper on | clips done during the stream | Whisper slower by | chat slower by |
|---|---|---|---|
| NPU | 57-72 | 1.6-2.1x | 32-47% (3 runs) |
| CPU | 14-21 | 1.9-2.2x | 8-32% |
| iGPU (same as chat) | 27 | 3.8-5.9x | 0-11% |

- **The NPU does not leave the GPU alone on Lunar Lake.** Run continuously,
  NPU Whisper takes a third or more off the chat's decode speed. Same
  package, same LPDDR5X, same power budget [INFERRED: bandwidth or power,
  not separated]. Per clip transcribed, NPU and CPU cost the chat about the
  same; the NPU just gets through five times as many.
- On the GPU it is the other way round: the chat wins the contention and
  Whisper waits.
- Error bars are wide: chat speed alone ranged 12.3-20.2 chunks/s across
  the runs, once 13.1 before and 18.0 after the same arm. A laptop on
  battery-tuned power management; only the direction of each row is solid.
- For push-to-talk none of this bites: a spoken sentence costs the NPU
  ~0.5 s, so the chat slows for half a second. Sustained transcription
  beside a busy chat is where the device choice matters.

### Chat and vision models

- **LFM2.5-24B-A2B (INT4) on GPU — converts and runs, but read the caveat**
  [OBSERVED 2026-10-14]. Released 2026-10-13; `lfm2_moe`, 24B total / 2B
  active, 40 layers, 64 experts, 131k context. Converted here the day it
  landed, on the current *release* stack — no nightly, no branch.

  | | |
  |---|---|
  | IR | 11.75 GiB, `INT4_ASYM` group_size 128 ratio 1.0 |
  | B60 Arc Pro 24 GB | **99 tok/s**, TTFT 151 ms, KV pool 5 GB auto, driver 32.0.101.8805 |
  | 285K CPU | 27 tok/s |
  | bare probe | 6/6 on GPU and on CPU, full penalty sweep |
  | tool dialect | `<|tool_call_start|>`, already handled by `_ToolCallGate` |
  | reasoning | `<think>`, splits cleanly to `reasoning_content` |

  **Export recipe — transformers 5.4.0 EXACTLY.** 4.57.6 does not know
  `lfm2_moe`; 5.5.4 is refused by optimum-intel's own cap
  (`Maximum required is 5.4.0`). Two stages, because the one-shot
  `--weight-format int4` writes an uncompressed intermediate and died in
  `openvino.save_model` with `ios_base::badbit` *after* compression had
  succeeded: `--weight-format fp16`, then
  `nncf.compress_weights(INT4_ASYM, group_size=128, ratio=1.0)`.

  **Converting it needs ~171 GB of commit**, not RAM. A 32 GB box with the
  stock 66 GB pagefile dies at 62% of the weight load with `0xC0000005` in
  `c10.dll` — torch's allocator taking an access violation instead of raising
  `MemoryError`. With a 260 GB pagefile the same box finishes in 1h51m, peak
  139 GB of pagefile, and is disk-bound at 13% CPU throughout. The 63 GB box
  does the same work in ~25 minutes.

  **The caveat: a short instruction preamble costs it the answer**
  [OBSERVED 2026-10-14, B60, n=1 per arm — see T-049, sampling here is
  deterministic]. Same arithmetic question, same 16k cap:

  | | tokens | correct |
  |---|---|---|
  | question alone | 549 | **5/5** |
  | 125-token standing-instruction preamble first | 5073 | **0/5** |

  9.2x the tokens and a wrong answer (`3 x 1 = 3` against 14,400), stopping
  cleanly both times, so it is not truncation. It is sold as an agent model,
  and an agent always has a system prompt — so this wants repeating with a
  real tool-calling prompt before the model is recommended for that.
  `scripts/decay-probe.py` could only measure `recall` on it: `reason` and
  `format` both fail calibration cold.

- Qwen3-8B (INT4-CW) on NPU — recommended, needs MAX_PROMPT_LEN=4096
- SmolLM3-3B (INT4-CW 23 tok/s, INT8-CW 12 tok/s) on 285K NPU — 2026.3,
  our export
- LFM2-1.2B / LFM2.5-1.2B-Instruct (INT4-CW, ~37-39 tok/s) on 285K NPU —
  NPU-only builds, old-stack exports fail CPU/GPU (see `TODONT.md`). On
  NPU 4 word salad below OpenVINO 2026.4.1, correct from 2026.4.1 on
  (openvino#38100, verified 2026-10-23 bare and through the server)
- MiniCPM5-1B (INT4) on GPU/CPU — 2026.3, no NPU support upstream
- **Phi 3.5 Mini on OpenVINO collapses past ~4-5k prompt tokens, on every
  device and every export** [OBSERVED 2026-10-19, B60 driver 32.0.101.8805
  + 5950X; found by `scripts/decay-probe.py`, T-047]. Fine on the NPU,
  where NoLlama caps prompts at 4096 anyway, and for short side requests.
  Not fine for anything that hands it a long document.

  Same prompt (a planted code, real repo text as filler, "what is the
  code?"), greedy, bare genai with NoLlama absent:

  | prompt tokens | transformers bf16, CPU | Ollama q8_0, RTX 5090 | OV `int4-cw` | OV `int4` | OV `fp16` (our export) |
  |---|---|---|---|---|---|
  | ~2.1k | ok | ok | ok | ok | ok |
  | ~4.1k | — | — | wrong code | ok | — |
  | ~4.6k | — | — | garbage | ok | — |
  | ~4.85k | ok | ok | garbage | **empty** | **garbage** |
  | ~6.8k | ok | ok | garbage | garbage | garbage |
  | ~12.7k | — | ok | garbage | garbage | garbage |
  | ~30.8k | — | ok | — | — | — |

  Held across GPU and CPU, release 2026.4.1 and the 2026.5 nightly
  (`int4`), Intel's two exports and our own fp16 one (optimum-intel 2.3.0,
  transformers 5.4.0). Microsoft's weights through transformers and the
  GGUF through llama.cpp read the same prompts correctly, so it is not
  the model and not quantisation. `original_max_position_embeddings` is
  4096 and the RoPE type is `longrope`; the export's Phi-3 patch
  (`optimum/exporters/openvino/model_patcher.py`, the short/long factor
  switch) is where to look first [INFERRED — the cliff sits near the
  switch, but `int4` passing at 4.6k says it is not simply "> 4096"].
  Through the server it shows as a decay curve that is really a wall: the
  probe's control arm is correct at 351 tokens and degenerate at 6,230.
  Upstream report drafted in `brain/next/`, for Tommy to file.

  Possibly the anomaly that started T-047 — the Phi-3.5-mini run that
  "answered the previous turn's question" — was this cliff, not turn
  confusion [GUESS: that run's prompt size was not recorded].
- DeepSeek-R1-1.5B (INT4-CW) on NPU — works but terrible quality (testing
  only)
- Gemma 3 4B Vision (INT4) on GPU — **untested here.** The "fast VLM" claim
  this list used to carry traced back to a drive-by line in an unrelated
  commit with no measurement behind it, and `docs/MODELS.md` has always said
  "Untested". Believe the latter.
- Gemma 4 (Intel's `OpenVINO/gemma-4-*`) on GPU — measured 2026-08-21 on the
  Arc Pro B60, 2026.3 release. All three exports are `image-text-to-text`, so
  they land on **VLM slots** whatever `models.json` files them under:
  - `E2B-it-int4-ov` (4.1 GB, 35 layers, 35 KB/token KV) — prefix caching
    works. Weakest of the three: misreads a 3x3 letter grid at 360x360 and
    answers *"the image is missing"* rather than admitting it cannot resolve
    the glyphs. That phrasing is a **model** quirk, not a lost image — see
    the cross-stack note below.
  - `E4B-it-int8-ov` (7.8 GB, 42 layers, 84 KB/token KV) — reads what E2B
    cannot. The build described below, up to 2026-09-17, got **no prefix
    caching** (its IR had no SDPA op; see `prefix-cache.md`), so for a while
    we shipped our own re-export. Intel's 2026-09-17 re-export fixed it and
    needs OpenVINO 2026.4+. On the B60 under 2026.4.1 the two match on cold
    turn (2.73-2.76 s) and on cached turns, and Intel's loads faster (17 s vs
    23-26 s) [OBSERVED 2026-10-11], so NoLlama uses Intel's only; ours is
    no longer referenced and is due for deletion (N-043).
    History of the defect: Reconfirmed on second hardware [OBSERVED 2026-08-24, Xe-LPG iGPU on
    the 285K, OpenVINO 2026.3]: Intel's build logs `prefix caching
    unavailable (has_op_with_type<ScaledDotProductAttention> failed at
    sdpa_to_paged_attention.cpp:82)` and falls back to the plain pipeline,
    while the re-export logs `prefix caching on` — same prompt, both answer
    correctly. The defect travels with the IR, not with the GPU.
  - `26b-a4b-it-int4-ov` (14.3 GB, INT4-AWQ, MoE 128 experts, 30 layers,
    262k context) — the best of the three and prefix caching works, but its
    KV is **240 KB/token**, so the auto-sizer hits its 2 GB floor and buys
    only ~8k tokens. The preflight says so at load; **pin
    `--cache-size-gb`** before pointing an agent at it. Loads in ~40s.

Two caveats that apply to every VLM this size, both measured across two
inference stacks (OpenVINO on the B60, Ollama/llama.cpp on an RTX 5090):

- **They cannot count.** Asked how many dots were in an image of 17, six
  model/stack/quantization combinations answered 15, 15, 16, 18, 20 and 20.
  Not one correct, including a 25.8B model. Don't build on it.
- **Failure style differs by stack, capability does not.** Ollama and
  OpenVINO produced byte-identical OCR transcriptions (three lines including
  an arbitrary serial) and the identical wrong count at matching model size —
  but where a model is at its limit, OpenVINO's E2B says "the image is
  missing" while llama.cpp's E2B confidently names the wrong row. **When a
  small Gemma claims the image is missing, the image is not missing.**
- Qwen2.5-VL-3B/7B (INT4/INT8) on GPU — proven for image tasks
- Qwen3-30B-A3B on GPU — needs >16 GB VRAM, falls back to CPU silently on
  16 GB cards
- Qwen3-Coder-30B-A3B **int8** on an **Arc 140T** (Xe-LPG+, XMX; 64 GB
  shared) — the #33 story, closed 2026-10-07. On OpenVINO 2026.3.1 and the
  driver before 9033, long real-code prompts (~100k chars) died in the
  fused grouped MoE gemm ("could not create a primitive descriptor for the
  matmul primitive", oneDNN "insufficient registers"); short or synthetic
  prompts and bare `LLMPipeline` passed, and none of our three GPUs
  reproduced it (upstream openvino#38211, closed by Intel 2026-10-08 with no
  comment on whether the runtime or the driver carried the fix). [OBSERVED 2026-10-06/07, reporter,
  `benchmark.py --long`, JSON in #33]: passes 6/6 on the 2026.5.0 nightly
  and 3/3 on the 2026.4.0 release, both on driver 32.0.101.9033; TTFT 134-138 s
  for 100k chars with the cache on, on the release. Fix is in 2026.4.0 or
  in the driver, not split. User-facing answer: update both. The int4 export
  never failed and stays the default entry.
- Muse-Glimmer-30B (Intel's `OpenVINO/Muse-Glimmer-30B-int4-ov`) on GPU —
  VLM slot on the GenAI path, needs the **nightly** runtime until 2026.4
  releases. Arc Pro B60 ~14 tok/s raw / 18.5 through the serving path; Arc
  B390 iGPU (Xe3, Linux, community report, issue #29, 2026-08-21) 4.5
  tok/s. The optimum-path GPU corruption (openvino#37419) never applied
  here. `_AtemPlainFilter` translates Glimmer's surviving ATEM channel
  markers into `<think>` blocks on both `generate_vlm` and `stream_vlm`.
  Full story in `docs/MODELS.md`.

- Qwen3.8-27B (Intel's `OpenVINO/Qwen3.8-27B-int4-ov`, rev `2026.3.1`) on
  the **Arc Pro B60** (the 5950X CPU load succeeded but was stopped before
  measuring — 1.5 GB of RAM left on the 32 GB box) — measured 2026-09-18 as the
  base-model arm of the Bonsai 2 comparison (`docs/BENCHMARKS.md`). VLM
  slot; bare `bare-probe.py` passes every case on the GPU. **~23 tok/s
  decode, ~1,150 tok/s prefill** on the B60 with `--cache-size-gb 3`. Two
  things it surfaced: the auto-sized 5 GB pool died with
  `CL_OUT_OF_RESOURCES` on the 34th request of a run (`docs/dev/machines.md`,
  B60 section), and the no-think switch did nothing on VLM slots until
  2026-09-18 (`TODONT.md`, "VLMPipeline.set_chat_template"). Its template
  pre-seeds `<think>`; 3.6-4.8 tok/s on the 140V per `docs/MODELS.md`.

- Qwen3.5-4B (Intel's `OpenVINO/Qwen3.5-4B-int4-ov`) — **re-exported by
  Intel 2026-09-25** (`d481b7f` → `f237729`: every IR rewritten, an MTP draft
  model added). Re-checked 2026-10-19 bare, both revisions side by side, and
  **they behave identically**, including one fault that predates the
  re-export [OBSERVED 2026-10-19, B60 GPU driver 32.0.101.8805, 285K iGPU
  32.0.101.8991]:
  - **On the B60 with OpenVINO 2026.4.1, alternating text-only and image
    turns on one `VLMPipeline` dies on the third image turn** (turn 6 of
    t,i,t,i,t,i) with `[GPU] CL_OUT_OF_RESOURCES`. Every later call fails
    and the process exits `0xC0000409`. Deterministic across runs and both
    revisions. Eight image turns in a row pass, so do t,i and t,i,i.
    `repetition_penalty` is irrelevant: it showed up in `bare-probe.py` only
    because that probe interleaves, and penalty-only sequences pass.
  - **Fixed in the 2026.5 nightly** (`2026.5.0-23394`, same box, same
    driver): eight interleaved turns pass, and every `bare-probe.py` row.
  - **Not on the 285K's Xe-LPG iGPU** (2026.4.1, eight interleaved turns
    pass), **not on CPU**, and **not other VLMs on the B60** on 2026.4.1:
    Qwen3-VL-8B, gemma-4-E2B, and Qwen3.8-27B (the same `qwen3_5`
    architecture) all pass eight interleaved turns.
  - **Fixed by the driver.** On `32.0.101.8976` (installed 2026-10-19
    evening), same release venv, `bare-probe.py` passes all twelve rows,
    and through the server the decay probe's calibration (twelve cold
    generates to the 2000-token cap, the sequence that killed the slot on
    the third) ends with the slot still `ready` [OBSERVED 2026-10-25, B60,
    OpenVINO/genai 2026.4.1]. Drivers between `.8805` and `.8976` are not
    tested. Users on 2026.4.x and a B60 need the current driver.
  - On the old driver this landed in `_note_poisoned` through NoLlama: the
    slot went out of service with "restart NoLlama". Possibly the same
    class as the Qwen3.8 `CL_OUT_OF_RESOURCES` on its 34th B60 request
    above [INFERRED: same error, same card, same family; not shown].

Not yet tested here: Qwen3-VL, pre-exported by Intel as
`OpenVINO/Qwen3-VL-8B-Instruct-int4-ov` (May 2026).

## Prompting small models

VLM prompts must be dead simple for small models (3B): one question, one
answer, minimal JSON. **All logic in Python, not in the prompt.**
