# T-056 — Nemotron on optimum-intel#2085: bench it and reply

**Effort:** 1 d · **Produces:** numbers for a reply on optimum-intel#2080 /
#2085, a verified-list entry in `docs/dev/models.md` · **Area:** export,
N-007 watch (#2080)

A contributor opened optimum-intel#2085 (NemotronH export, fixes our #2080)
and pinged us on #2080 2026-10-26. He covered Nano-4B on CPU and a 140T iGPU
in depth; he could not run Lightning 30B-A3B (32 GB box). His notes: the
experts have no gate, so OpenVINO's MoE fusion does not apply and they run
dense; Mamba-2 only gets `SelectiveSSM` in genai's paged path. Our value is
Lightning, the B60, and an RTX 5090 llama.cpp (Ollama) reference.

**Running since 2026-10-30 16:14** on the 285K as scheduled task
`nemotron-285k` (`C:\devel\nemotron\nemotron-285k.ps1`): `venv-nemotron`
with the PR branch, int4 exports to `C:\models\nemotron\`, bare genai CPU
and iGPU, then Ollama on the 5090, Nano-4B then Lightning. Log
`C:\devel\nemotron\run.log`, commit charge `mem.log`, JSONs in
`C:\devel\nemotron\results\`. Bench: `scripts/nemotron-bench.py`.

## Done when

- [ ] 285K run finished: read `run.log` (export time, peak commit in
      `mem.log`), all six JSONs present.
- [ ] Lightning and Nano IRs copied to the B60 and the laptop; bench there
      (B60 GPU, 140V iGPU), bare genai first.
- [ ] Through the NoLlama server once bare passes (standing order).
- [ ] Reply on #2085 with the table, detuned, shown to Tommy before posting.
