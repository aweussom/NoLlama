# T-042 — Whisper on the NPU

**Effort:** 2 h · **Produces:** an NPU verdict per Whisper entry in
`docs/dev/models.md`; maybe `NPU` in `--whisper-device` · **Area:** NPU,
`WhisperSlot`, `models.json`

Speech-to-text on the NPU, so the GPU keeps the chat model. Measured: on
Lunar Lake it is not free under sustained load (shared memory and power),
but a dictated sentence costs the chat half a second.

The usual NPU limits don't apply here. Whisper's decoder output is short, it
needs no prefix cache and no tools, and the encoder input has a fixed size
(30 s of mel frames).

What we know (checked 2026-10-11):

- OpenVINO GenAI docs list NPU as a `WhisperPipeline` device, and the 2026.0.0
  release notes add word timestamps "on CPUs, GPUs, and NPUs" [DOCUMENTED].
- `--whisper-device` help says "CPU or GPU", but the flag has no `choices=`,
  so `NPU` goes straight through to `WhisperPipeline`. Never run.
- **NPU 4 done, bare (2026-10-11):** base and small run as shipped, fastest
  of the three devices, WER equal to CPU. medium compiles and then hits
  `ZE_RESULT_ERROR_DEVICE_LOST` on the first inference. Silence without a
  language hallucinates hardest on the NPU. Numbers and clip set:
  `docs/dev/models.md`, "Whisper (speech-to-text)". The clip set is
  rebuilt from FLEURS/NPSC/whisper.cpp by `scripts/whisper-build-set.py`
  (sources in its docstring); the audio stays outside the repo.

Prompted by CynipsNet (7M-param CTC ASR on the STM32N6 NPU). Rejected as a
model: it ships only an ST LL_ATON blob and a static lib, no ONNX, no
checkpoint, no training code. The weights are CC BY-NC-SA, and it is English
only. Whisper already covers Norwegian.

## How to run it — the standing orders apply

- [x] Bare probe: `scripts/whisper-probe.py`, set from `scripts/whisper-build-set.py`.
- [ ] Bare genai on **NPU 3 (285K)**; NPU 4 (laptop) done. Different
      generations, and that has been the whole answer before (LFM2).
      -> `docs/dev/machines.md`. Note the 285K NPU driver is older; record both.
- [x] Base, small, medium int8 as shipped on NPU 4. Medium refuses; a
      stateless re-export of medium is the one thing left to try for it.
- [x] Under the server: `--whisper-device NPU` serves (laptop, 2026-10-11).
- [x] Measured against CPU and iGPU on the same clips (laptop).
- [x] Silence guard: `audio_level_spread` + `collapse_repeat_loop`
      (1d7f279).
- [x] Concurrency measured: the claim does **not** hold under sustained
      load (chat -32..-47% with NPU Whisper running non-stop). Fine for
      push-to-talk. `docs/dev/models.md`.
- [ ] Install menu offers no Whisper at all; add a step (device default
      NPU where present, small) or decide it stays manual.

## Done when

`docs/dev/models.md` carries an NPU verdict (works, or the error) per Whisper
entry on both NPU generations. If it works, the `--whisper-device` help, the
installer and `docs/DEVICES.md` say NPU. If not, the reason goes in `TODONT.md`.
