# T-056 — Nemotron on optimum-intel main: re-bench after #2085 merged

**Effort:** 1 d · **Produces:** a verified-list entry in `docs/dev/models.md`,
and a follow-up on #2085 if Lightning still fails · **Area:** export, N-007
(#2080)

A contributor's optimum-intel#2085 (NemotronH export, fixes our #2080)
**merged 2026-11-13** as `0ee909b` (final head `22de222`). It ships in the
next optimum-intel release. The merged version runs the experts through
grouped MatMul (rkazants pointed him at optimum-intel#1947 on 2026-11-03).
It also uses int4 group size 64 for these models because Nano-4B's hidden
size, 3136, is not a multiple of 128. His announcement credits our testing
and says Lightning 30B now loads on a CPU "where it used to be killed at
113 GB", which is our watchdog's kill point. Treat that as his claim until we
reproduce it here.

## What we measured on the pre-merge head, 2026-11-02

285K, `venv-nemotron`: PR head `dbab784`, OpenVINO/genai 2026.4.1,
transformers 5.15.1, nncf 3.4.0, torch 2.15.0. IRs in `C:\models\nemotron\`,
JSONs in `C:\devel\nemotron\results\`, bench `scripts/nemotron-bench.py`.
Reply posted on #2085 (issuecomment-6589414327).

- **Nano-4B int4** (2.33 GB, 2.9 min export): all correct, needle found at
  1k/4k/16k. Bare genai decode 26 tok/s CPU, 29 iGPU; 16k TTFT 101 s CPU,
  52 s iGPU. Ollama Q4_K_M on the 5090: 375 tok/s, 16k-ish in 0.6 s.
- **Lightning int4** exported (43 min, 15.9 GB) but **did not load**:
  LLMPipeline passed 100 GB private memory on CPU and iGPU before a token.
  A 100 GB watchdog killed it at 105.5 GB on the iGPU and 113.2 GB on CPU.
- **Cause, CPU, confirmed** (`scripts/moe-fold-probe.py`): the non-gated
  experts missed the MoE fusion and stayed one batched 3D u4 constant per
  projection, which CPU compile folded to f32 (+2.92 GB for 0.32 GB of u4).
  The same bytes as a 2D weight stayed compressed (+0.38). GPU was not
  reproduced synthetically.
- **Lightning on the 5090** (Ollama Q4_K_M): all correct, 360-440 tok/s.

## Round 3, grouped MatMul, 2026-11-03/04

Exported on the 285K from PR commit `f294ba2` (grouped experts, still int4
gs128, **before** the dt clamp `ef105b7` and the gs64 defaults
`2379328`..`22de222`). OpenVINO/genai nightly `2026.5.0-23715`. Logs:
285K `C:\devel\nemotron\run3.log` + `mem3.log`, B60 box same paths.

- **Lightning grouped, 285K CPU: loads** (62 s, ~31 GB private), all
  correct, needle 1k/4k/16k, 21-25 tok/s, 16k TTFT 175 s. The 113 GB kill
  is gone, which matches the contributor's claim.
- **Lightning grouped, 285K iGPU: still does not load.** Watchdog killed it
  at 102.6 GB private after 4 min (commit 180 GB, RAM free 0). The GPU half
  of the round-2 failure is not fixed by grouped MatMul.
- **Lightning grouped, B60 (24 GB): loads** (53 s), all correct, needle
  ok, ~43 tok/s, 16k TTFT 4.8 s. So the blowup is the iGPU path, not GPU.
- **Nano-4B grouped vs old**, 285K, nightly: identical answers, same
  2.34 GB, CPU 26 tok/s both (16k TTFT 83 vs 92 s), iGPU 28.5 both.
- **Nano-4B old export, B60 box**, 2026.4.1 release: CPU 16 tok/s (16k TTFT
  124 s), **B60 ~112 tok/s** (16k TTFT 3.9 s), all correct. Through NoLlama
  answers are correct but **the bench's server timing is broken** (ttft
  0.00 s, 39e9 tok/s) — a `nemotron-bench.py` bug, not a server number.

## Round 4, optimum-intel main `0ee909b`, 2026-11-16/17

285K, iGPU driver 32.0.101.9150, venvs `venv-nemotron-main-nightly`
(2026.5.1-23881) and `venv-nemotron-main-2026.4.1`, transformers 5.15.1.
Script `C:\devel\nemotron\nemotron-285k.round4.ps1`, logs `run4.log` /
`mem4.log` (`run4.noop.log` is a first run that launched nothing: a
PowerShell function returned its Step output along with the python path).

- **Lightning export**: the first try died on disk (`ios_base::badbit` in
  `save_model`, the ~60 GB uncompressed intermediate did not fit on C:).
  Re-run 2026-11-18 after Tommy cleaned C: (`nemotron-285k.round4b.ps1`,
  `run4b.log` / `mem4b.log`; pagefile now system-managed and it grew on
  demand): 25 min, 16.16 GB, 185/185 int4 at gs64.
- **Lightning main gs64, 285K CPU: loads and is correct on both runtimes**,
  needle 1k/4k/16k all `7341` — so the 16k regression below is Nano-only.
  Decode ~25 tok/s on both. Prefill is the difference: TTFT 1k/4k/16k
  6.5/28.6/165 s on nightly 2026.5.1, 12.3/51.7/271 s on 2026.4.1. Load
  23.5 s nightly, 15.1 s release (round 3: 62 s).
- **Lightning main gs64, 285K iGPU: still does not load.** Watchdog killed
  it at 110.5 GB private after ~2 min (nightly). Same as round 3's 102.6 GB:
  neither grouped MatMul nor main fixes the iGPU path.
- **Lightning main gs64, B60, 2026.4.1 release: word salad.** Loads (55 s),
  runs at ~40 tok/s, every answer garbage ("up up upnxuel foryy-ytic…"),
  same IR (MD5 `cef22705…`, matches the 285K) that is correct on 285K CPU.
  Driver 32.0.101.8976. B60 box `C:\devel\nemotron\run4.log`,
  `lightning-main-release-b60.out`. The nightly leg was killed by the
  harness (box memory pressure), not by the model. Round 3's grouped IR was
  correct on the B60, but only ever ran on the nightly — so this is either
  2026.4.1's GPU plugin or the main export on GPU. Controls, 2026-11-18:

  | Lightning IR, B60 | 2026.4.1 release | nightly 2026.5.0-23715 |
  |---|---|---|
  | main `0ee909b` gs64 | word salad, ~40 tok/s | **all correct**, 16k ok, 29-35 tok/s, 16k TTFT 4.1 s |
  | round 3 `f294ba2` gs128 | word salad, ~37 tok/s | all correct (round 3), ~43 tok/s |

  **It is the 2026.4.1 GPU plugin, not the export**, and the nightly has it
  fixed. 2026.4.1 CPU is fine (285K, above), so GPU-plugin only. Lightning on
  GPU is therefore nightly-only: no installer entry until a release has the
  fix. Check the next release before filing — it may already be in.
- **Nano-4B main gs64, B60** (IR MD5 `cb7dd127…`, re-exported 2026-11-18
  after the cleanup): same as the 285K — correct, needle 1k/4k ok, **16k
  fails on both** (release '0'; nightly 2048 tokens of thinking), "Hmm"
  opening. 2026.4.1 is fine on Nano's GPU path. ~80-95 tok/s, below the old
  gs128 export's ~112 on the same box; 16k TTFT 3.6-4.0 s.
- **Nano-4B gs64 (main's default, plain `--weight-format int4`)**: 2.35 GB,
  92/92 int4 layers at gs64. Correct on the short questions, needle 1k/4k
  ok, **needle 16k fails on all four runs** (CPU/iGPU × nightly/2026.4.1:
  '0', or 2048 tokens without an answer). Decode now opens "Hmm, the user
  is asking..." where every earlier export wrote the story.
- **The 16k failure is in the export, between `f294ba2` and `0ee909b`**:

  | IR (all CPU) | runtime | opens | 16k |
  |---|---|---|---|
  | round 3 `f294ba2` gs128+adjust | nightly 2026.5.1 | story | ok |
  | main gs128+adjust | nightly 2026.5.1 | story | 'The\nThe' |
  | main gs128+adjust | 2026.4.1 | story | 'The\nThe' |
  | main gs64 | both | "Hmm" | fail |

  Same flags, transformers and runtime: the "Hmm" opening tracks gs64, the
  16k failure tracks optimum-intel.
- **Bisected to the dt clamp `ef105b7`**, 2026-11-18 (285K CPU, nightly
  2026.5.1, gs128+adjust, 16k needle; upstream `refs/pull/2085/head`, the
  PR was squash-merged; `C:\devel\nemotron\bisect.log`):
  `a4fc4bb` pass '7341' · `ef105b7` fail 'The\nThe' · `902b825` fail
  'The\nThe'. Decode is the same story on all three. **Confirmed**: main
  `0ee909b` with only that line reverted (`tuple(config.time_step_limit)`)
  passes, '7341', 61 tokens, same as `a4fc4bb`. `ef105b7` changed the
  NemotronH clamp from `config.time_step_limit` (0, inf; NVIDIA, mamba_ssm,
  vLLM) to `time_step_min` (transformers' PyTorch path), asked for in the
  review. **transformers itself fails the same way** (2026-11-19, 285K CPU,
  transformers 5.15.1, torch 2.15 CPU, fp32, BF16 checkpoint, greedy, the
  same haystack, prefill in 1024-token pieces through the cache because the
  CPU fallback scan wants ~100 GB at 16k in one call; `C:\devel\nemotron\
  tf-needle-*.out`): as shipped, `time_step_limit=(0.001, inf)` → "We don't
  have context", no code; every mixer set to `(0, inf)` → '7341'. So the
  bug is upstream, in transformers, and transformers#48989 (approved, open
  since 2026-09-21) is the fix; optimum-intel follows it once merged.
  Both models have `time_step_min` 0.001, yet Lightning passes 16k
  on main. Why only Nano: heads trained to dt < 0.001 for long memory decay
  faster once clamped up [INFERRED — dumping Nano's dt_bias / per-head dt
  on a 16k prompt would confirm].

## Done when

- [x] Reported 2026-11-20: transformers#48989 comment
      (issuecomment-7031987340) with the A/B, and optimum-intel#2136 with the
      bisect and revert. Once either lands, re-export Nano and rerun the 16k
      needle.

- [ ] Lightning on the 285K iGPU: main still blows past 100 GB, so report
      it on optimum-intel (or openvino, if a bare-IR repro points there) —
      the B60 and CPU loading makes it iGPU-specific.
- [x] Server-path timing fixed in `nemotron-bench.py` (streams; client
      timing when the server sends no durations). 2026-11-16, B60, old
      Nano-4B IR, 2026.4.1, driver 32.0.101.8976, `--backend genai`: all
      correct, 107-118 tok/s, needle 16k TTFT 3.6 s — the same as bare genai.
- [ ] `NEEDS_OPTIMUM = {"nemotron_h"}` in `nollama.py` is stale: auto picks
      the optimum backend, which fails to load on this venv ("No module named
      'transformers.core_model_loading'"), while every number above is bare
      genai. Drop it once the main export is confirmed on genai, and check
      `docs/dev/runtime-stacks.md` for the reason it went in.
- [ ] Nano-4B on the 140V, bare genai, then the server.
- [ ] Results in `docs/dev/models.md`'s verified list, driver recorded.
