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

## Done when

- [ ] Lightning and Nano-4B re-exported on the 285K from optimum-intel main
      (≥ `0ee909b`), same 100 GB watchdog, same bench. First find out
      whether main still needs openvino-nightly (#1947 was to be tested
      there) or runs on the 2026.4.1 release.
- [ ] Nano-4B old vs new compared: size (gs64 vs our 2.33 GB), speed,
      needle.
- [ ] Nano-4B on the B60 and the 140V, bare genai, then the server.
- [ ] Results in `docs/dev/models.md`'s verified list, driver recorded.
