# T-028 — Text-to-speech, `/v1/audio/speech`

**Effort:** 2 h · **Produces:** a verdict · **Area:** API surface

`openvino_genai.Text2SpeechPipeline` exists, and the OpenAI API has a shape for
it we already half-implement (`/v1/audio/transcriptions` is served by
`WhisperSlot`, so the audio half of the surface is not new ground).

The blocker is content, not code: **only SpeechT5 is supported** so far, and
nobody has asked for TTS. Revisit when either changes.

- [ ] check whether the supported-model list has grown
- [ ] if it has, one slot class mirroring `WhisperSlot` is most of the work

## Done when

Either `/v1/audio/speech` answers, or a `TODONT.md` line records that the model
support is not there.
