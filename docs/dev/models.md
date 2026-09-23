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

### Two probes, and which question each answers

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

- Qwen3-8B (INT4-CW) on NPU — recommended, needs MAX_PROMPT_LEN=4096
- SmolLM3-3B (INT4-CW 23 tok/s, INT8-CW 12 tok/s) on 285K NPU — 2026.3,
  our export
- LFM2-1.2B / LFM2.5-1.2B-Instruct (INT4-CW, ~37-39 tok/s) on 285K NPU —
  NPU-only builds, old-stack exports fail CPU/GPU (see `TODONT.md`)
- MiniCPM5-1B (INT4) on GPU/CPU — 2026.3, no NPU support upstream
- Phi 3.5 Mini (INT4-CW) on NPU — smaller, faster
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
    cannot, but **gets no prefix caching on this runtime** (its IR has no
    SDPA op; see `prefix-cache.md`). Use
    [`aweussom/gemma-4-E4B-it-int8-ov`](https://huggingface.co/aweussom/gemma-4-E4B-it-int8-ov)
    instead for agent work — our re-export of the same weights, fused
    attention, prefix caching working, byte-identical answers. Intel's is
    still the better pick for one-shot vision, being ~2.2x faster on a cold
    turn. Reconfirmed on second hardware [OBSERVED 2026-08-24, Xe-LPG iGPU on
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

Not yet tested here: Qwen3-VL, pre-exported by Intel as
`OpenVINO/Qwen3-VL-8B-Instruct-int4-ov` (May 2026).

## Prompting small models

VLM prompts must be dead simple for small models (3B): one question, one
answer, minimal JSON. **All logic in Python, not in the prompt.**
