# T-032 — Is the Qwen3.8 revision pin dead now that the floor is 2026.4?

**Effort:** 1 h (mostly a 15 GB download) · **Produces:** one field removed from
`models.json`, or a recorded reason to keep it · **Area:** `models.json`,
`install.ps1`

`models.json` pins `OpenVINO/Qwen3.8-27B-int4-ov` to the repo's `2026.3.1`
branch, because Intel keeps `main` on the *next* runtime and main segfaulted
2026.3.x at load.

That gap may have closed. The IR stamps say so [OBSERVED 2026-09-23, read from
each branch's `openvino_language_model.xml`]:

| branch | `Runtime_version` |
|---|---|
| `2026.3.1` (what we ship) | `2026.3.1-22476` |
| `main` | **`2026.4.0-22768`** — our floor since 2026-09-23 |

A version stamp is not a run, and this repo has been burned by exactly that
distance before, so it needs the download.

- [ ] `download-model.ps1 OpenVINO/Qwen3.8-27B-int4-ov` (no `-Revision`) into a
      separate directory, so the pinned copy survives for comparison
- [ ] `bare-probe.py` on it, GPU, on the B60 — the laptop has 27B + vision
      working but its margins are thin (see `docs/dev/machines.md`)
- [ ] if clean: drop `"revision"` from the entry and say in the commit that the
      pin outlived the runtime gap
- [ ] if it fails: keep the pin and write the failure into the entry's note, so
      the next session does not re-run this

## Done when

The entry either has no `revision` field, or has one with a dated reason.
