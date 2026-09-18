# T-027 — The offload split does not track the ratio on a discrete card

**Effort:** 2 h · **Produces:** an explanation, or a `TODONT.md` entry ·
**Area:** `--offload-ratio`

At `--offload-ratio 30` on the B60, **3.2 GB of 15.2 GB stayed resident
(~24%)** where the 140V measured 10.8 GB (~71%) — with 21 GB of VRAM free.

Either the ratio is a ceiling that a demand-driven expert LRU never fills, or
it behaves differently on discrete hardware. Runs at 50 and 90 would tell.

- [ ] ratio 50 and 90 on the B60, resident bytes recorded at each
- [ ] if it is an LRU ceiling, say so in `docs/dev/moe-offload.md` — the flag
      currently reads like a split, not a cap

Related, and deliberately filed separately as a `proposed/` entry because
nobody has checked whether the output is *wrong* or merely *different*: the
non-determinism at the same ratio.

## Done when

`docs/dev/moe-offload.md` predicts the resident fraction on both a dGPU and an
iGPU, or records that it cannot.
