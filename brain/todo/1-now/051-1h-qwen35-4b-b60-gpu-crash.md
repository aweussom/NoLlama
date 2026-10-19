# T-051 — Qwen3.5-4B on the B60 dies after a few generations (2026.4.1)

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

**Images are not needed either** [OBSERVED 2026-10-19, B60, driver 8805,
2026.4.1, through the server]. T-047's decay probe killed it text-only: three
cold ~150-token prompts, each generating to the 2000-token cap (the model
thinks without closing `<think>` on "what is the access code?" — 8051 chars
every time), and the third threw the same `CL_OUT_OF_RESOURCES`; the slot
went out of service. Log: `bench/decay-qwen35-4b-b60-gpu.json`
(calibration rows).

**But bare genai does not die text-only** [OBSERVED 2026-10-19, same box,
driver, model]: a single-prompt loop of 2000-token generates on
`VLMPipeline` survived 8 of 8 on release and on nightly, 8 of 8 with
`SchedulerConfig` prefix caching at 4 GB, 6 of 6 at 8 GB (the server's
auto-sized pool), and 6 of 6 with `repetition_penalty=1.05`. So "N long
generates" alone is not it, and the server is now the suspect — the common
case per `CLAUDE.md`. Remaining difference: the server sends a **chat
history** (briefing, assistant ack, question), the loop a single prompt.
Next: replay the probe's exact calibration sequence bare, with that
history, then through the server with `--no-prefix-cache`. Loop scripts
were scratch; the shape is three lines around `VLMPipeline.generate`.

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
