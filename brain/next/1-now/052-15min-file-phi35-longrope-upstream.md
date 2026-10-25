# N-052 — File the Phi-3.5-mini long-prompt collapse upstream?

**Cost:** 15 min · **Asks:** Tommy (files it) · **Area:** upstream,
`docs/dev/models.md` (Phi 3.5 Mini entry)

## What

Phi-3.5-mini under OpenVINO answers in gibberish once a prompt passes
~4-5k tokens. Every OpenVINO configuration tried fails; transformers and
llama.cpp read the same prompts correctly to 30k. Full table in
`docs/dev/models.md`. Found by T-047's decay probe on 2026-10-19.

## Where to file

**optimum-intel** first: the IR is where the LongRoPE short/long switch
gets baked in (`model_patcher.py`, Phi-3 patch), and our own fp16 export
from Microsoft's weights fails like Intel's int4 ones. If they bounce it as
a runtime problem, openvino.genai. Not split further here: telling export
from runtime would need the IR's RoPE subgraph read by hand.

## Draft

> **Phi-3.5-mini-instruct: OpenVINO IR produces garbage past ~4.8k prompt
> tokens (LongRoPE?)**
>
> A recall prompt (a code planted at the start, ~N tokens of filler, "what
> is the code?"), greedy, `openvino_genai.LLMPipeline`:
>
> | prompt tokens | transformers 5.4.0 bf16 | llama.cpp (Ollama 0.40.1) q8_0 | OV int4-cw (OpenVINO/…-int4-cw-ov) | OV int4 (OpenVINO/…-int4-ov) | OV fp16 (own export) |
> |---|---|---|---|---|---|
> | 2,041 | ok | ok | ok | ok | ok |
> | 4,848 | ok | ok | garbage | empty | garbage |
> | 6,833 | ok | ok | garbage | garbage | garbage |
> | 12,685 | — | ok | garbage | garbage | garbage |
>
> Same result on GPU (Arc Pro B60, driver 32.0.101.8805) and CPU (Ryzen 9
> 5950X); OpenVINO/genai 2026.4.1 and 2026.5.0 nightly (2026.5.0.0-3521).
> fp16 export: `optimum-cli export openvino --task
> text-generation-with-past --weight-format fp16`, optimum 2.3.0,
> transformers 5.4.0, nncf 3.4.0. `original_max_position_embeddings` 4096,
> `rope_scaling.type` longrope. With and without
> `SchedulerConfig(enable_prefix_caching=True)`.
>
> Repro script: attach `scripts/longrope-recall-probe.py`.

## Decide

File it, or leave it documented and move on? The model is a registry
entry (NPU canary, OpenCode `small_model`), and the NPU path is safe
behind the 4096 cap, so nothing user-facing waits on the answer.
