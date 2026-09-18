# N-005 — Issue #24: does `_reset_vlm_state` rescue an allocation-cap throw?

**Cost:** 10 min · **Asks:** Tommy · **Waiting on:** the #24 reporter (Arc Pro
140T, stock 4.29 GB per-allocation cap), asked 2026-09-01.

## Where it stands

Two different failure classes, and only one is answered.

- **`CL_OUT_OF_RESOURCES` — answered, and the answer is no.** The slot stays
  dead until NoLlama restarts, once until a reboot: a poisoned driver context,
  exactly as OpenVINO's own error text warns [OBSERVED 2026-09-12, Arc 140T].
  NoLlama now takes the slot out of service (`_note_poisoned`: status "error",
  reason in `/health`) and the explainer names the remedy.
- **The allocation-cap throw from #24 — still unverified.** Different class.
  His 4.29 GB cap is the only place it can be checked, and we own no stock-cap
  iGPU with XMX.

Also seen on both boxes the same day: GPU misbehaviour that clears with a
reboot after long uptime under heavy load. Keep asking for driver versions.

## Saying yes means

One comment on #24 asking him to confirm whether the slot reset works on the
cap throw, or accepting that this stays unverified and saying so where
`/health` is documented.
