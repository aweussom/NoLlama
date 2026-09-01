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

Not yet tested here: Qwen3-VL, pre-exported by Intel as
`OpenVINO/Qwen3-VL-8B-Instruct-int4-ov` (May 2026).

## Prompting small models

VLM prompts must be dead simple for small models (3B): one question, one
answer, minimal JSON. **All logic in Python, not in the prompt.**
