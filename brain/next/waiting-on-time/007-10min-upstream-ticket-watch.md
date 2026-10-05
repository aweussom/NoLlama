# N-007 — Upstream tickets: the watch list

**Cost:** 10 min · **Asks:** Tommy · **Waiting on:** Intel. Check when a new
OpenVINO release lands, and otherwise monthly — next sweep due 2026-10-18.

One item deliberately, not six: the action is identical for all of them (look,
and do nothing unless something moved). A `Ref. <number>` in a thread is how
Intel marks acceptance into engineering triage.

| ticket | what | status |
|---|---|---|
| [openvino#38100](https://github.com/openvinotoolkit/openvino/issues/38100) | LFM2 350M/1.2B word salad on NPU 4, correct on NPU 3 | filed 2026-09-13, scoped to 350M + 1.2B with DavidDohmen's working 2.6B as the boundary. **No `Ref.` yet** |
| [openvino#37501](https://github.com/openvinotoolkit/openvino/issues/37501) | USM OOM: a full-sequence logits allocation, `vocab_size x seq_len x width` | **closed 2026-09-24, fixed by genai#4511** (not in 2026.4). Verified on the B60 2026-09-29: 2026.4 release still requests 32,095,728,896 B on attempt 1, nightly `dev20260928` passes 3/3. Drop the row once a release carries it |
| [genai#4405](https://github.com/openvinotoolkit/openvino.genai/issues/4405) | Phi-3.5-vision asserts under `repetition_penalty` | `Ref. 194483`, accepted |
| [genai#4343](https://github.com/openvinotoolkit/openvino.genai/issues/4343) | VLMPipeline's `scheduler_config` / prefix caching is undocumented | **the documentation ask is still unanswered.** Restated so it does not close as "E4B fixed" with the docs untouched |
| [openvino#38211](https://github.com/openvinotoolkit/openvino/issues/38211) | MoE `matmul primitive` on Xe-LPG+ with XMX | `Ref. 196134` (2026-09-30), accepted. Same day Intel asked for a nightly run; only the #33 reporter can run it, see N-006. Acknowledged upstream 2026-10-05, result still pending |
| optimum-intel PR #1789 | `nemotron_h` exporter — merged descoped | see N-003 |

## Two things that are ours, not theirs

- **#37501 carries a retraction.** We read a x1.1 ratio into the allocation
  sizes and sent Intel down that path. `202,048 = 11 x 18,368`, so *every*
  allocation of the form `vocab_size x n x width` divides by 1.1 for any n. Two
  data points fitting a ratio is not evidence when the ratio's factors sit in
  the operands. Worth remembering as a method failure, not just a wrong answer.
- **Intel's suggested workaround for #37501 does not work** — a dummy short
  generate first, measured 2026-08-25, fails identically. It tightens the
  predicted length from prompt+55 to prompt+7, i.e. 0.12% against a 7.0 GB gap.

## Also on each release: re-walk TODONT's "Last checked" lines

Folded in from T-029 on 2026-10-02. Every runtime-dependent `TODONT.md` entry
carries a **Last checked** line. On a new OpenVINO release, re-run the cheap
ones (the lines name the probe) and re-date the rest. First walk done against
2026.4.0: seven re-run on the laptop, all held; one trigger fired (gemma-4 E4B
prefix cache on Intel's IR, registry flip pending); three need the B60 or the
285K.

## Saying yes means

Nothing, until one of them moves. Quote the `Ref.` number if anyone asks.
