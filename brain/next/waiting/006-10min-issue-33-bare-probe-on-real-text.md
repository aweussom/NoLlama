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

## Intel's ask (2026-09-30): run the nightly

`Ref. 196134` landed, then Zulkifli asked for a nightly run. We promised in
#33 to ping the reporter if Intel asked for something; this is that.

Expectation, so the result is read right [INFERRED, from upstream git on
2026-10-02]: the nightly (2026.5.0.dev) pins the **same oneDNN** as 2026.4.1
(`a3d45972`, rls-v3.13), so the gemm generator that reports "Insufficient
registers" is byte-identical. Three MoE commits are master-only: #38147 adds
`fpmath:f16` to `moe_gemm_onednn.cpp`, but the reporter's failing descriptor
already carried `attr-fpmath:f16:true` via `grouped_matmul_helper.hpp` on
2026.3.1, so it cannot be the fix; #37637 (scatter-reduce row LUT) and #37800
(dGPU offload USM) do not touch dispatch. A clean nightly run would be a
surprise worth a bisect; a failing one is the expected result and still moves
the ticket, because Intel asked.

Recipe for the reporter (a second venv, stable one untouched):
`.\install.ps1 -Nightly`, then bare `probe.py` on the real prompt and one
OpenCode session, both with `ONEDNN_VERBOSE=1`. 2026.4.1 (released
2026-10-01) is worth one run too, as the nearest release.

Asked in #33 on 2026-10-02 (comment 5948051083), recipe and expectation
included. Told Intel upstream on 2026-10-05 (comment 5987730865) that the
run is delegated to the reporter and that I expect the same failure.
Waiting on the reporter.

## Saying yes means

Nothing until he answers. Then one reply upstream either way: a fail is the
expected result and still moves the ticket; a pass means a bisect across the
three master-only MoE commits.
