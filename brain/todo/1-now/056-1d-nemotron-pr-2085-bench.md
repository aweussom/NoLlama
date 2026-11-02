# T-056 — Nemotron on optimum-intel#2085: bench it and reply

**Effort:** 1 d · **Produces:** a reply on optimum-intel#2085 with numbers,
a verified-list entry in `docs/dev/models.md` · **Area:** export, N-007
watch (#2080)

A contributor opened optimum-intel#2085 (NemotronH export, fixes our #2080)
and pinged us on #2080 2026-10-26. He covered Nano-4B on CPU and a 140T
iGPU; he could not run Lightning 30B-A3B (32 GB box).

## Where it stands, 2026-11-02

285K, `venv-nemotron`: PR head `dbab784`, OpenVINO/genai 2026.4.1,
transformers 5.15.1, nncf 3.4.0, torch 2.15.0. IRs in `C:\models\nemotron\`,
JSONs in `C:\devel\nemotron\results\`, bench `scripts/nemotron-bench.py`.

- **Nano-4B int4** (2.33 GB, 2.9 min export): all correct, needle found at
  1k/4k/16k. Bare genai decode 26 tok/s CPU, 29 iGPU; 16k TTFT 101 s CPU,
  52 s iGPU. Ollama Q4_K_M on the 5090: 375 tok/s, 16k-ish in 0.6 s.
- **Lightning int4** exports (43 min, 15.9 GB) but **does not load**:
  LLMPipeline passed 100 GB private memory on CPU and iGPU before a token.
  Round 1's CPU run took the box to 359 GB commit for ~2 days; round 2 had
  a 100 GB watchdog (kills at 105.5 GB iGPU, 113.2 GB CPU).
- **Cause, CPU, confirmed** (`scripts/moe-fold-probe.py`): the non-gated
  experts miss the MoE fusion and stay one batched 3D u4 constant per
  projection; CPU compile folds that to f32 (+2.92 GB for 0.32 GB of u4),
  the same bytes as a 2D weight stay compressed (+0.38). The trailing f32
  Convert is not it (variant B). GPU not reproduced synthetically.
- **Lightning on the 5090** (Ollama Q4_K_M): all correct, 360-440 tok/s.

## Done when

- [x] Reply posted on #2085 2026-11-02 (issuecomment-6589414327).
- [ ] Nano-4B on the B60 and the 140V, bare genai, then the server.
- [ ] Lightning again once OpenVINO has a non-gated expert pattern (the
      contributor says that needs a new OpenVINO pattern); onto N-007.
