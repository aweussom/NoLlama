# T-029 — Re-run the TODONT comprehension test on each new OpenVINO release

**Effort:** 1 h · **Produces:** confirmation, or a reopened entry ·
**Area:** `TODONT.md`

Standing chore, not a one-off. `TODONT.md` is full of entries whose verdict was
"the runtime cannot do this" — and runtimes change. An entry that has silently
become wrong is worse than no entry, because it stops people trying the thing
that now works. The VLM prefix-cache belief ("CB backend is LLM-only") was
exactly this: stale, load-bearing, and wrong.

- [ ] on each OpenVINO release, walk the entries whose verdict rests on a
      runtime limitation rather than a design decision
- [ ] re-run the cheap ones; for the expensive ones, note the version last
      checked against so the staleness is visible

## Done when

Every runtime-dependent `TODONT.md` entry carries the version it was last
checked against.
