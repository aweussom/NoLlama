# T-021 — File the `ShapePredictor` ticket, deliberately deferred

**Effort:** 1 h · **Produces:** an upstream issue · **Area:** upstream

Found while testing the USM OOM: setting `OV_GPU_SHAPE_PREDICTOR_SETTINGS` (a
`RELEASE_INTERNAL` option) **crashes pipeline construction** —
`ShapePredictor::Settings` has no string parser ("Bad as from std::string"), so
the env knob is unusable and a bad value kills the load.

**Held back on purpose (2026-08-25).** It is independent of the
logits-allocation bug in #37501, and now *more* orphaned than before —
ShapePredictor is no longer implicated in that bug at all, so it will never get
attention buried as a "bonus" in a ticket about something else. Two open
tickets from us on the same subsystem, one of which we already had to retract a
theory in, is a good way to get both triaged slowly.

- [ ] **re-run the repro first** — it has not been retested since 2026-08-18
- [ ] minimal repro: set the env var, construct any pipeline, it dies
- [ ] state the ask plainly: either a string parser for
      `ShapePredictor::Settings`, or the option rejects bad input without
      killing construction

Gate: file it once #37501 is resolved.

## Done when

The issue is filed, or re-running the repro shows it fixed and this file is
`git rm`'d.
