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
- **Nano-4B on the B60 box** (2026-11-03, `venv` genai 2026.4.1, driver
  .8976, IR copied from the 285K): all correct, needle 3/3 on both.
  5950X CPU 16 tok/s, TTFT 7.5/29.9/124 s; B60 97-118 tok/s, TTFT
  0.75/1.3/3.9 s. Through the server on the B60: all correct, ~119 tok/s,
  3.6 s TTFT at 13k, prefix-cache repeats at 48 ms. Needed
  `NEEDS_OPTIMUM` emptied first (1f300df): auto-detect sent nemotron_h to
  the optimum backend, which died on the B60 venv's transformers.
- **#2085 head f294ba2** (2026-11-03): merged main; from transformers
  5.13 + torch 2.10 the experts export as grouped MatMul (#1947) and become
  GatherMatmul, int4 kept. Contributor's synthetic 1-layer test: CPU
  5.46 -> 1.35 GB after load, decode 11 -> 96 tok/s. He asked us to retry
  Lightning. Round 3, nightly 2026.5.0.dev20261102 [OBSERVED 2026-11-03/04]:
  grouped export 15.7 min (was 43), `GroupedMatMul` x46. 285K CPU loads
  it: 31 GB private, all correct, 21-25 tok/s, TTFT 16k 175 s. 285K iGPU
  still killed at 102.6 GB. B60 (`venv-nightly-1102` on the 5950X box):
  loads in 53 s, ~23 GB host commit, all correct, 43-44 tok/s, TTFT 16k
  4.8 s. Nano-4B (no MoE layers) bmm vs grouped identical within noise.
  Reply 2 posted on #2085 2026-11-04 (issuecomment-6629397783).
  Caveat for re-runs: the B60 box's watchdog read 0 GB; Store Python's
  process is not named `python` there. Match on the venv path.

## Done when

- [x] Reply posted on #2085 2026-11-02 (issuecomment-6589414327).
- [x] Nano-4B on the B60, bare genai, then the server (2026-11-03).
- [x] Nano-4B on the 140V laptop, bare genai, then the server
      [OBSERVED 2026-11-08]: grouped IR copied from the 285K, nightly
      2026.5.0.dev20261102, driver .9150. All correct, needle 3/3 on every
      arm. iGPU 35-38 tok/s, TTFT 1k/4k/16k 1.4/3.2/17.8 s; CPU (258V)
      14-22 tok/s, 12.6/59.8/252.5 s. Server (7621ea4, same nightly venv,
      flask/pillow/soundfile added to `venv-nightly-1102`): all correct,
      ~37 tok/s; its TTFTs are not comparable, the needle prompts share a
      filler prefix and the cache serves it. The Ollama surface reports no
      durations, so the bench's server timings are junk; read the server
      log. The 2026.4.1 `venv` refuses this IR (our IR-version guard).
- [x] Lightning on the grouped-MatMul export: CPU and B60 run it, the
      285K iGPU does not (2026-11-04). Was: Lightning again on the
      grouped-MatMul MoE export. rkazants
      (2026-11-03) pointed the contributor at optimum-intel#1947 (merged
      2026-10-30, `torch._grouped_mm`, already handles non-gated experts),
      to be tested on openvino-nightly. On a new #2085 head: re-export
      Lightning and Nano-4B on the 285K against the nightly, same 100 GB
      watchdog, compare Nano-4B current vs grouped. `upstream-watch` opens
      an issue on a new head, or once on 2026-11-16 if none came.
- [x] 285K iGPU blowup: no XMX, so no MoE on GPU; Intel says so in
      openvino#34415 and docs/dev/moe-offload.md had it since 2026-09-11.
      The contributor pointed it out on #2085 (2026-11-04).
- [x] Lightning on the 140V laptop iGPU, bare genai [OBSERVED 2026-11-08]:
      nightly 2026.5.0.dev20261102, driver .9150, budget 24.7 GB (override
      85), 31.5 GB RAM. Loads in 29 s, all correct, needle 3/3, 23-28
      tok/s, TTFT 1k/4k/16k 2.1/4.9/21.0 s. Bench process peaks 22.1 GB
      private at the 16k needle; free RAM bottoms at 0.7 GB, commit 33.9
      GB, so it leans on the pagefile briefly. A first run with a
      free-RAM < 1.5 GB kill died 42 s into the load (started with 16.6 GB
      free vs 20.9 the second time): on a 32 GB laptop, close things first.
      JSON: `C:\devel\nemotron\results\` on the laptop. Reply 3 posted on
      #2085 2026-11-08 (issuecomment-6722201082).
- [ ] Lightning through the NoLlama server on the B60 once a release
      OpenVINO carries GroupedMatMul (nightly-only today: no installer entry).
