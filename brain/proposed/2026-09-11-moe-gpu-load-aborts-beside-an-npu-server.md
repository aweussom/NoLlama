# A genai GPU load of a MoE aborts while a NoLlama NPU server runs on the same box

**Noticed:** 2026-09-11 · **Status:** proposed, not trusted

## Claim

[OBSERVED 2026-09-11, 285K, OpenVINO 2026.3.0, three for three]
`LLMPipeline(<LFM2-8B-A1B int4 or int8>, "GPU.0")` dies with "Fatal Python
error: Aborted" (exit 3) — no exception, no event-log entry — while
`nollama.py --device NPU` (SmolLM3-3B) is up. With the NPU server stopped, the
same load compiles (212 s core, 60-69 s genai) and runs.

Dense Qwen3-8B loaded fine beside the NPU server all afternoon, so it is not
"two processes" on its own.

## Evidence

Three for three is the whole of it. No counter-test with a different MoE, and
no instrumentation of the failing process.

[INFERRED] The MoE compile's ~25 GB of shared-memory staging plus the NPU
process exceeds the box's 32.9 GB shared budget, and the driver aborts rather
than throws. Unverified. Raising this box's shared-memory override and
retrying would confirm or kill it.

## Suggested home

`docs/AGENTS.md` if it holds — the two-server recipe would need "start the GPU
server first, then the NPU one", which is itself untested. It is also another
argument for raising the 285K's shared-memory override.
