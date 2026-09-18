# T-011 — Test the Norwegian translation models: the NPU's real workload

**Effort:** 1 d · **Produces:** a verified-model verdict in `docs/dev/models.md`,
maybe a `models.json` entry · **Area:** NPU, `models.json`

The NPU is not primarily meant for coding LLMs. The intended workload is
**translating internal documents on-device**, so company material never leaves
the laptop. Privacy is the point; throughput is secondary.

Every documented NPU limitation stops mattering for this workload, which is why
it is the right one:

- `MAX_PROMPT_LEN=4096` is a hard ceiling for agent prompts. Translation is
  chunked per paragraph and never approaches it.
- The NPU keeps the plain pipeline with **no prefix cache**. Translation would
  miss that cache on every chunk anyway — each chunk is different text. The one
  feature the NPU lacks is the one this workload does not want.
- Tool calling never works on the NPU. Translation needs no tools.
- Sustained low-power throughput over a long document is the NPU's design
  point, and it leaves the GPU free.

**Hard requirement: Bokmal and Nynorsk.** That is what disqualifies the
otherwise ideal candidate — Tencent's `HY-MT1.5-1.8B` is exactly the right
shape and its 33 languages contain **no Nordic language at all** (checked
2026-09-04 against the HF tag list). Worth recording that the *export* path is
fine: `model_type` is `hunyuan_v1_dense`, and optimum-intel already registers
`HunyuanV1DenseOpenVINOConfig` as a `LlamaOpenVINOConfig` subclass
(`optimum/exporters/openvino/model_configs.py:6219`, verified in our venv). A
future Nordic-capable HY-MT converts with no work.

## Candidates

| Model | Size | Licence | Note |
|---|---|---|---|
| `NbAiLab/borealis-1b` | 1B | NB-licence (Apache-derived) | Gemma 3 based; smallest sane NPU target |
| `NbAiLab/borealis-4b` | 4B | NB-licence | the one to beat |
| `norallm/normistral-7b-warm` | 7B | Apache-2.0 | NB's own writeup calls NorMistral stronger *on translation* |

Borealis is Nasjonalbiblioteket's AI-lab family, released 2026-05-26, covering
Bokmal + Nynorsk + English, commercial use permitted, sizes
270m/1b/4b/12b/27b. `gemma3_text` is already registered for OpenVINO
text-generation, so the export path exists. Skip
`normistral-11b-thinking` — reasoning tokens are pure waste on a translation
turn, and we already know a thinking model can burn the whole budget before
answering (that is how SmolLM3-3B lost the small-model slot).

Caveat: the Borealis instruct variants are still `-instruct-preview` and carry
the **Gemma licence** rather than NB's own. Check the terms before an installer
entry, not after.

## How to run it — the standing orders apply in full

- [ ] **Bare `openvino_genai` first**, no NoLlama:
      `.\venv\Scripts\python scripts\bare-probe.py <model-dir>`
- [ ] Then under the server.
- [ ] Then **every device** — CPU, iGPU, B60, and the NPU — each bare *and*
      served. A device that refuses the model is a result, not a skip; record
      the error. -> `docs/dev/machines.md`, `docs/dev/models.md`
- [ ] NPU export must be channel-wise (`-Weight int4-cw` or `int8-cw`); default
      group-quantised int4 IRs crash the NPU driver compiler.
- [ ] Record the driver with every measurement.

**What "tested" means here.** Translation quality is not a tok/s number, and
our benchmarks will not catch a model that is fluent and wrong. Fix a small
held-out set of real internal-doc paragraphs, both directions, and compare
candidates on the same set.

## Done when

`docs/dev/models.md` carries a verdict per candidate per device, including the
refusals, and either a `models.json` entry or a recorded reason there is none.
