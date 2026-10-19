# T-051 — Qwen3.5-4B on the B60 dies on interleaved text/image turns (2026.4.1)

**Effort:** 1 h · **Produces:** a driver verdict, then either a user-facing
note or nothing · **Area:** GPU, VLM, B60 · **Machine:** the B60 box

Found 2026-10-19 while verifying Intel's re-upload of the model (#48). The
re-upload is innocent: both revisions fail identically. Full matrix in
`docs/dev/models.md`, the Qwen3.5-4B entry.

Short version: `VLMPipeline` on the B60, OpenVINO 2026.4.1, turns
t,i,t,i,t,i → the sixth (third image) throws `[GPU] CL_OUT_OF_RESOURCES`, the
context is gone, the process exits `0xC0000409`. Passes on the 2026.5 nightly,
on the 285K's iGPU, on CPU, and for three other VLMs on the B60 (including
Qwen3.8-27B, same architecture).

Repro (no NoLlama): `scripts/bare-probe.py ~/models/Qwen3.5-4B-int4-ov
--device GPU` on the release venv. It interleaves text and image cases, so
it dies at the `repetition_penalty=1.05 image` row — the penalty in that
label is a coincidence of position, not the cause.

- [ ] **Driver.** The B60 is on `32.0.101.8805` (2026-07); the other boxes on
      `.8991`, and `.9033` exists. **Ask Tommy before updating** (he reboots).
      Re-run bare-probe on the release venv after the update. Record the
      driver either way.
- [ ] If it still fails on a current driver: it is a 2026.4.x bug the
      nightly already fixed. No upstream report (nothing to fix); a line in
      `docs/MODELS.md` for Qwen3.5-4B ("B60 + 2026.4: avoid mixing text and
      image turns, or use 2026.5") and re-check when 2026.5 ships.
- [ ] If the driver fixes it: say so in the models.md entry, delete this.
- [ ] Optional, cheap: does it reproduce through the server? Expected yes
      (slot taken out of service by `_note_poisoned`), which is the
      user-visible symptom worth naming in `docs/MODELS.md`.

## Done when

The models.md entry says which of driver or runtime the fix rides on, and
`docs/MODELS.md` warns B60 users if they need warning.
