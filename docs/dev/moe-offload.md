# MoE disk offload (`--offload-ratio`, 2026-08-06)

`--offload-ratio PCT` streams PCT% of MoE expert weights from disk on GPU
slots (OpenVINO 2026.3 `OFFLOAD_RATIO`).

**Requires XMX** (Arc / Lunar Lake — `GPU_HW_MATMUL` in
`OPTIMIZATION_CAPABILITIES`). Without it the property is a **silent no-op**,
so NoLlama warns at startup — full story in `TODONT.md`, which also records
that OFFLOAD_RATIO could not be validated on the desktop iGPU.

**XMX gates the offload, not the model.** This note used to say non-XMX
iGPUs "can't load big MoE at all (USM staging OOM)". That is wrong when the
memory is actually there: on an Arc 140T (Xe-LPG, no XMX) with a 64 GB
shared budget, Qwen3-Coder-Next int4 (80B-A3B, ~40 GB) runs resident at 18.8
tok/s and gemma-4-26b-a4b at 8–11 tok/s [OBSERVED 2026-08-28 and 2026-08-31,
issue #24, two independent batches]. What a non-XMX GPU cannot do is *stream
experts from disk* — so it cannot trade residency for capacity, and a model
that does not fit simply does not load.

Verified on Arc 140V, Qwen3-30B-A3B int4, steady state:

| ratio | resident | decode |
|---|---|---|
| 30 | 10.8 GB | 25.3 tok/s (interactive) |
| 90 | 2.35 GB | 5.1 tok/s |

Pick the smallest ratio that fits. The expert LRU needs **~60 tokens to
warm** — benchmark steady state, not the first sentence.

Known upstream bug: a **second** `generate()` on an offload-active **plain**
pipeline hangs in native code, uninterruptible. NoLlama's serving path is
unaffected: the CB backend it uses was verified with sequential requests
(140V, ratio 30 — 12.5 then 15.9 tok/s, prefix-cache TTFT 8.0s→1.9s).
