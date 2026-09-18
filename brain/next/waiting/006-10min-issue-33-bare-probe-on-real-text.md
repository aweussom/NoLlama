# N-006 — Issue #33: one probe would settle the MoE matmul failure

**Cost:** 10 min · **Asks:** Tommy · **Waiting on:** the #33 reporter — filed
upstream as openvinotoolkit/openvino#38211 on 2026-09-17 with his logs and our
three-GPU negatives.

## Where it stands

`matmul primitive` on his Arc Pro 140T, and we have ruled out everything we own.

- **Not zero-points, not XMX alone**: asymmetric int8 dense at 60k chars passes
  on a non-XMX Xe-LPG (285K iGPU), and his symmetric `int8-cw` re-convert fails
  identically.
- **Not the runtime version**: a scratch `venv-2026.3.1` run on the 285K passed
  (60k chars, scheduler path, 699 s).
- **Not the scheduler path** [OBSERVED 2026-09-16]: `--no-prompt-cache` in a
  real OpenCode session failed one run of two, so the plain pipeline fails
  under the server on weights bare genai accepts.
- **Our MoE control is clean**: `LFM2-8B-A1B` INT8 asymmetric (7.8 GB, 32
  experts) passes on the B60 at 60k and 100k, and on the 285K iGPU the
  scheduler path passes at 60k (657 s) and 100k (475 s). Prefill ran at ~21
  tok/s — the *unfused* expert path, so TODONT's XMX gate is holding.

**The 140T has XMX** (Xe-LPG+; our docs were wrong about it until 2026-09-11),
so the fused grouped gemm engages on his box and never on ours.

[INFERRED] The remaining axis is **expert routing breadth**: a repeated
sentence lands on a handful of experts, real agent text spreads across all 128,
and the wide grouped gemm is the one with no kernel. His 120k-char *synthetic*
probe passed; a bare probe fed a **real** 100k-char document on his box would
confirm or kill it. That is the ask.

The `CL_PROFILING_INFO_NOT_AVAILABLE` lines in his logs are verbose-profiler
noise — present on passing runs too, so do not chase them.

## Saying yes means

One comment asking for that one run. Nothing here is actionable without it —
the hardware is a device the project does not own.
