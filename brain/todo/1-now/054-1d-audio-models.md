# T-054 — Audio models: what runs next

**Effort:** 1 d · **Produces:** a verdict per audio model class, and the
first new audio slot · **Area:** API surface, `WhisperSlot`, web UI, NPU

Tommy, 2026-10-27: audio models are "our next thing". Whisper is the NPU's
best workload and the installer now leads with it on NPU machines (T-042);
the next step is what else audio can do here.

Absorbs T-028 (text-to-speech, filed 2026-09-18): `openvino_genai.Text2SpeechPipeline`
exists and the OpenAI API has `/v1/audio/speech`, but at the time only
SpeechT5 was supported and nobody had asked for TTS. That list has not been
re-checked since.

## Look at

- **Text-to-speech**: has `Text2SpeechPipeline`'s model list grown past
  SpeechT5? Norwegian voices? Which devices, NPU included?
- **Better speech-to-text** than Whisper medium: large-v3 / turbo exports,
  and anything genai supports besides Whisper.
- **Audio-in chat models** (audio tokens into an LLM/VLM): does any run under
  genai or optimum-intel yet?

Each through bare genai first, then the server, on every device: the
standing orders apply (`CLAUDE.md`).

## Rules already settled

- **An audio model is not a chat model.** It reports outside `/health`'s
  `devices` block (as `whisper` and `embed` do), so the web UI's chat picker
  never offers it; `loadModels` keeps only the generative slots (2026-10-27,
  after `whisper (NPU)` showed up in the picker).
- One slot class per pipeline, mirroring `WhisperSlot`.
- The installer offers only what fits the hardware, NPU first where the NPU
  is the best device (T-042).

## Done when

A verdict per class above in `docs/dev/models.md` (runs, or the error, per
device), and at least one new audio endpoint served, or a `TODONT.md` entry
saying why not.
