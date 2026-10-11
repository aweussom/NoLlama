# T-016 — B60 numbers for Glimmer and Qwen3.8 on 2026.3.1

**Effort:** 2 h · **Produces:** replaced figures in `docs/MODELS.md` ·
**Area:** benchmarks

Both models came back into the menu on 2026-08-30 once OpenVINO shipped them in
a *release* (2026.3.1, 2026-08-26). Both were verified on the Arc 140V with the
release wheels — and those figures are **iGPU-bound and say nothing about the
card users will actually buy for these models**: Qwen3.8 at 3.6-4.8 tok/s,
Glimmer at ~2.5.

- [x] **Qwen3.8 on the B60, 2026-09-18**: 23 tok/s decode, ~1,150 tok/s
      prefill, `--cache-size-gb 3` (the auto-sized 5 GB pool crashed the slot —
      T-031). Recorded in `docs/dev/models.md` and `docs/BENCHMARKS.md`
      (Bonsai 2 section); `docs/MODELS.md` still quotes only the 140V figure.
- [x] **Glimmer on the B60, 2026-09-29** (2026.4.0 release, not 2026.3.1): a
      ~35k-token cold prefill in 51–53 s (~680 tok/s), 400-token turns in
      21.1–21.9 s (~19 tok/s if every turn reached `max_tokens`, which the
      constant time suggests; `usage` is -1, T-034). **Only at
      `--cache-size-gb 2`**: the auto 3 GB pool spills (T-031). Recorded in
      `docs/dev/prefix-cache.md`
- [ ] replace the 140V numbers in `docs/MODELS.md`, or keep both and label the
      device on each

Do this before T-008 lands if you want the provenance header on the runs; do it
after if you would rather have the numbers sooner. Note Qwen3.8 is pinned to
its `2026.3.1` repo branch (see `TODONT.md`, "nightly stack in the default
install").

## Done when

`docs/MODELS.md` quotes a discrete-GPU number for the two models people buy a
discrete GPU to run.
