# T-016 — B60 numbers for Glimmer and Qwen3.8 on 2026.3.1

**Effort:** 2 h · **Produces:** replaced figures in `docs/MODELS.md` ·
**Area:** benchmarks

Both models came back into the menu on 2026-08-30 once OpenVINO shipped them in
a *release* (2026.3.1, 2026-08-26). Both were verified on the Arc 140V with the
release wheels — and those figures are **iGPU-bound and say nothing about the
card users will actually buy for these models**: Qwen3.8 at 3.6-4.8 tok/s,
Glimmer at ~2.5.

- [ ] both models on the B60 with the release stack
- [ ] replace the 140V numbers in `docs/MODELS.md`, or keep both and label the
      device on each

Do this before T-008 lands if you want the provenance header on the runs; do it
after if you would rather have the numbers sooner. Note Qwen3.8 is pinned to
its `2026.3.1` repo branch (see `TODONT.md`, "nightly stack in the default
install").

## Done when

`docs/MODELS.md` quotes a discrete-GPU number for the two models people buy a
discrete GPU to run.
