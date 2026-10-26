# N-007 — Upstream tickets: the watch list

**Cost:** 10 min · **Asks:** Tommy · **Waiting on:** Intel. Check when a new
OpenVINO release lands, and otherwise monthly — last swept 2026-10-19, next
due 2026-11-19.

One item deliberately, not six: the action is identical for all of them (look,
and do nothing unless something moved). A `Ref. <number>` in a thread is how
Intel marks acceptance into engineering triage.

| ticket | what | status |
|---|---|---|
| [openvino#39053](https://github.com/openvinotoolkit/openvino/issues/39053) (was genai#4677) | whisper-medium-int8-ov: `ZE_RESULT_ERROR_DEVICE_LOST` on first inference on NPU 4, runs on NPU 3, driver 5540 on both, release and 2026.5.0 nightly | filed 2026-10-20 (T-050), moved to `openvino` by 2026-10-22. No `Ref.` yet. #38100 turned out to be a runtime fix, so re-probe this one on each OpenVINO release too |
| [openvino#37501](https://github.com/openvinotoolkit/openvino/issues/37501) | USM OOM: a full-sequence logits allocation, `vocab_size x seq_len x width` | **closed 2026-09-24, fixed by genai#4511** (not in 2026.4). Verified on the B60 2026-09-29: 2026.4 release still requests 32,095,728,896 B on attempt 1, nightly `dev20260928` passes 3/3. Drop the row once a release carries it |
| [genai#4405](https://github.com/openvinotoolkit/openvino.genai/issues/4405) | Phi-3.5-vision asserts under `repetition_penalty` | `Ref. 194483`, accepted |
| [genai#4343](https://github.com/openvinotoolkit/openvino.genai/issues/4343) | VLMPipeline's `scheduler_config` / prefix caching is undocumented | **the documentation ask is still unanswered.** Restated so it does not close as "E4B fixed" with the docs untouched |
| [optimum-intel#2080](https://github.com/huggingface/optimum-intel/issues/2080) | `nemotron_h` exporter request (Lightning 30B-A3B; Bolt and Super-VL are `nemotron_h_omni` on top of it). Also flags the transformers `5.16.0.dev0` text-only `generate()` break (T-017). #1789 merged descoped, #1724/#1844 closed unmerged | filed 2026-10-24 (was N-003), offered testing on 140V/B60/CPU. When a PR appears: export Lightning, bare-probe, then `NEEDS_OPTIMUM` slot |
| [genai#4734](https://github.com/openvinotoolkit/openvino.genai/issues/4734) | SLIDING_WINDOW cache type for CB: sliding-attention layers keep only their window. Glimmer 64k context 3.25 GiB here vs 949 MiB in llama.cpp. #4146 checked, not related (RoPE fp16) | filed 2026-10-26 (was N-035), offered testing on B60/140V/285K. When a branch appears: Glimmer on the B60 at 64k+ with a smaller pool, then revisit the auto pool sizing |
| [optimum-intel#2086](https://github.com/huggingface/optimum-intel/issues/2086) | Phi-3.5-mini OpenVINO IR collapses past ~4.5k prompt tokens (LongRoPE short/long switch suspected); transformers and llama.cpp fine to 30k | filed 2026-10-27 (was N-052), on current drivers (B60 8976, 285K iGPU 9150), offered testing on B60/140V/285K. If bounced as runtime: refile on openvino.genai. When fixed: rerun `scripts/longrope-recall-probe.py`, then drop the warning in `docs/dev/models.md` |
| [openvino#37419](https://github.com/openvinotoolkit/openvino/issues/37419) | GPU plugin computes wrong for muse_glimmer int4 on the 140V; same IR correct on CPU | **no response at all since 2026-08-15**, no `Ref.`. Added to this list 2026-10-19; it had none |
| [optimum-intel#1948](https://github.com/huggingface/optimum-intel/issues/1948) | export can silently produce an IR with no fused SDPA op, so no PagedAttention | **no response since 2026-09-02**. Added 2026-10-19 |

openvino#38211 (MoE matmul on Xe-LPG+) dropped 2026-10-19: closed by Intel
2026-10-08 without saying whether the runtime or the driver carried the fix
(`STATUS.md`).

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
prefix cache on Intel's IR; registry flipped 2026-10-11, see N-043); three need the B60 or the
285K.

## Saying yes means

Nothing, until one of them moves. Quote the `Ref.` number if anyone asks.
