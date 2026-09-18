# T-010 — Memory preflight cries wolf on a discrete card

**Effort:** 30 min · **Produces:** two fixes in `_preflight_memory` ·
**Area:** `nollama.py`

[OBSERVED 2026-09-12, B60] "model (~15.2 GB) + KV pool (6 GB) needs ~23.3 GB
but the device budget is 23.3 GB — this will likely NOT work (raise the iGPU
budget ...)". It loaded and ran fine.

Two separate defects in one line:

- [ ] **Equality is not failure.** `needs > budget` is the test; it is
      currently `>=`, or the rounding makes it so.
- [ ] **The hint is iGPU-only prose.** "Shared GPU Memory Override" is a
      Windows setting for integrated graphics and means nothing on a dGPU.
      Word the remedy by device type.

A preflight that is wrong in the scary direction gets ignored, which costs more
than not having one.

## Done when

The B60 loads that pair with no warning, and a genuinely-too-big model on the
same card gets a hint that names something a dGPU owner can actually do.
