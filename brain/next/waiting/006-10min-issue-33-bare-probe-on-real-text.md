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

## His answer (2026-10-05): the nightly passed, once

[OBSERVED 2026-10-05, reporter in #33] OpenVINO `2026.5.0-23326-6ce8fccc044`,
genai `2026.5.0.0-3501-ff212f97cf5`, GPU driver 32.0.101.9033, Windows build
26300. Through NoLlama (not bare `probe.py`), `--no-prompt-cache`, his usual
prompt: no matmul error, ran until the context window was full. No
`ONEDNN_VERBOSE` output posted.

That contradicts the expectation above, and upstream was told to expect a fail.
Two reasons it is not settled: the 2026-09-16 failure under the same flag was
one of two, so a single pass is weak, and without the verbose log we cannot
tell a new kernel from a dispatch that no longer asks for the grouped gemm.
His driver in September was never recorded, so a driver change is not ruled
out.

Asked in #33 on 2026-10-05 (comment 5998179892): `python benchmark.py
--long` (added for this, 59d9364; 3 runs of 100k chars of real code, pass/fail
plus versions and driver in a paste block) on the nightly with and without
`--no-prompt-cache`, the same on his stable venv if he has time, and his
September driver. The verbose log was dropped from the ask to keep it short;
ask for it if the repeats pass.

## The repeats (2026-10-06): 6 of 6 pass on the nightly

[OBSERVED 2026-10-06, reporter in #33, JSON attached there] Nightly
`6ce8fccc044`, driver 32.0.101.9033, `benchmark.py --long`: 3/3 with
`--no-prompt-cache` (TTFT 185–194 s), 3/3 with the cache (287–469 s; that gap
went to T-039). His September driver is "the previous one", not known
exactly; he updates through Intel's assistant as releases appear.

So it is fixed on nightly + 9033, but the fix could be either one. He did
not run his stable venv. That run is the one that tells them apart: stable
(2026.4.x) on 9033 fails, the runtime fixed it; it passes, the driver (or
2026.4.x) did.

## Saying yes means

Post the #33 thanks plus the one stable-venv ask, and tell Intel upstream
it passes 6/6 on nightly + 9033, with the stable run to follow. Close #33
when the stable run is in, whichever way it goes, since either way the
user-facing fix is "update". Repeats pass and 2026.4.1 fails: tell Intel it
is fixed on master and ask which change did it. Both pass: fixed in 2026.4.1,
check the driver. Nightly fails again: it is intermittent and the ticket
stays open.
