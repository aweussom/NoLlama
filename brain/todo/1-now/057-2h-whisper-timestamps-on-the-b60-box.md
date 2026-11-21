# T-057 — Whisper timestamps: large-v3 on the B60 box's CPU

**Effort:** 2 h · **Produces:** verdict on a bigger Whisper for timed lyrics,
and whether f32 input beats s16 · **Area:** `WhisperSlot`, `/v1/audio/transcriptions`

`verbose_json` with `segment`/`word` timestamps landed 2026-11-07 (docs/API.md),
tested on the laptop with whisper-small int8 on NPU, CPU and GPU, bare and
through the server. Consumer: `extract-lyrics-and-timing` (Tommy's karaoke
tool), which wants one OpenAI client for OpenRouter and NoLlama.

## Open

- [ ] B60 box (5950X, no NPU): `OpenVINO/whisper-large-v3-int8-ov` and
      `whisper-large-v3-turbo-int8-ov` on CPU, bare genai with
      `word_timestamps=True`, then the server. Does the export carry
      `alignment_heads`? Word timing quality against small on the same song.
- [ ] f32 vs s16 WAV: one song gave different segment boundaries (15 vs 14).
      Check two or three more before it becomes advice beyond "if you can".
- [ ] First word of a song absorbs the intro (0.00-7.10 s). Check whether
      large-v3 does the same; if so it is the client's job (clip to segment start).
