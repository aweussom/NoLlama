# N-004 — Issue #38: the reporter's box is the only correct hardware

**Cost:** 10 min · **Asks:** Tommy · **Waiting on:** the #38 reporter, since
2026-09-16 — he has moved this issue three times already and was last asked to
update his GPU driver from `32.0.101.8508`.

## Where it stands

**Idle time alone does not poison an Intel GPU's OpenCL context** — three runs,
model and OpenVINO build held constant so the device was the only variable:

| device | box | longest idle | generate | warm baseline |
|---|---|---|---|---|
| Arc 140V (iGPU) | laptop | 210 min | 1.35 s | 1.42 s |
| Arc Pro B60 (dGPU) | `100.81.4.88` | 180 min | 3.02 s | 0.60 s |

Both on `Qwen3-8B-int4-ov`, OpenVINO 2026.3.1; drivers `32.0.101.8991` and
`32.0.101.8805`. Artefacts in each box's `bench-results/`.

**The dGPU penalty plateaus, and that is the new part.** A discrete card does
pay for idling: 0.60 s warm, 1.82 s at 5 min, then 3.68 / 3.13 / 3.02 s at
45 / 120 / 180 min. Eviction completes between 5 and 45 minutes, and three
hours costs no more than forty-five. The 5-minute rung looks cheap because it
caught the eviction partway through. Recorded in `docs/dev/machines.md`.

Caveat on the mechanism: the probe samples **host** counters only and never the
GPU, deliberately — a VRAM query during the idle window would be the keepalive
touch whose absence is under test.

**We cannot close it from here.** His model is ~15 GB inside a stock 32 GB
shared ceiling; ours was 4.55 GB inside a 27.2 GB override. And **we own no
stock-cap iGPU with XMX** — the 285K is stock-cap but has no `GPU_HW_MATMUL`,
while his Arc Pro 140T has XMX. The model-against-budget axis is not testable
on our hardware at all.

## Saying yes means

Ask him once more, or close it as not-reproducible and say what we ran. He has
the probe path; `scripts/run-idle-probe.ps1` runs unattended.
