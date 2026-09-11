# MoE disk offload (`--offload-ratio`, 2026-08-06)

`--offload-ratio PCT` streams PCT% of MoE expert weights from disk on GPU
slots (OpenVINO 2026.3 `OFFLOAD_RATIO`).

**Requires XMX** (Arc / Lunar Lake — `GPU_HW_MATMUL` in
`OPTIMIZATION_CAPABILITIES`). Without it the property is a **silent no-op**,
so NoLlama warns at startup — full story in `TODONT.md`, which also records
that OFFLOAD_RATIO could not be validated on the desktop iGPU.

**No XMX means no big MoE, full stop — and the datapoint that seemed to say
otherwise was misread.** On 2026-08-28 this note was rewritten to say a
non-XMX iGPU loads big MoE fine when the memory is there, on the strength of
an Arc 140T running Qwen3-Coder-Next int4 (80B-A3B, ~40 GB) resident at 18.8
tok/s and gemma-4-26b-a4b at 8–11 tok/s [OBSERVED 2026-08-28 and 2026-08-31,
issue #24]. Those numbers are real, but the 140T is **Xe-LPG+ and has XMX**
[DOCUMENTED: Intel Arrow Lake-H; `install.ps1` prints `XMX: yes` on a 140T in
issue #38] — it was never a no-XMX datapoint. The original claim stands and
is now measured on a genuinely XMX-less GPU: on the desktop 285K's Xe-LPG a
15 GB int4 MoE stages ~47 GB of shared memory and never finishes loading
[OBSERVED 2026-09-11, `TODONT.md`]. Without XMX the MoE fusion is off,
offload is a silent no-op, and the unfused expert constants blow up staging.

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
