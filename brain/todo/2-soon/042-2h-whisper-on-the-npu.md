# T-042 — Whisper on the NPU

**Effort:** 2 h · **Produces:** an NPU verdict per Whisper entry in
`docs/dev/models.md`; maybe `NPU` in `--whisper-device` · **Area:** NPU,
`WhisperSlot`, `models.json`

Speech-to-text on the NPU leaves the GPU free for the LLM. That is the whole
point: transcription that costs the GPU nothing while it keeps generating.

The usual NPU limits don't apply here. Whisper's decoder output is short, it
needs no prefix cache and no tools, and the encoder input has a fixed size
(30 s of mel frames).

What we know (checked 2026-10-11):

- OpenVINO GenAI docs list NPU as a `WhisperPipeline` device, and the 2026.0.0
  release notes add word timestamps "on CPUs, GPUs, and NPUs" [DOCUMENTED].
- `--whisper-device` help says "CPU or GPU", but the flag has no `choices=`,
  so `NPU` goes straight through to `WhisperPipeline`. Never run.
- The community NPU export `FluidInference/whisper-large-v3-turbo-int4-ov-npu`
  was exported with stateful mode **off**. That suggests Intel's stateful
  `OpenVINO/whisper-*-int8-ov` exports may not compile on the NPU as they are
  [INFERRED]. Confirm by trying them first; the fallback is a re-export with
  `--disable-stateful`.

Prompted by CynipsNet (7M-param CTC ASR on the STM32N6 NPU). Rejected as a
model: it ships only an ST LL_ATON blob and a static lib, no ONNX, no
checkpoint, no training code. The weights are CC BY-NC-SA, and it is English
only. Whisper already covers Norwegian.

## How to run it — the standing orders apply

- [ ] `scripts/bare-probe.py` has no Whisper path (LLM/VLM only). Add one, or
      write a ten-line bare `WhisperPipeline(model_dir, "NPU")` probe with a
      short Norwegian and English clip.
- [ ] Bare genai on **NPU 3 (285K) and NPU 4 (laptop)**. Different
      generations, and that has been the whole answer before (LFM2).
      -> `docs/dev/machines.md`. Note the 285K NPU driver is older; record both.
- [ ] Base, small, medium int8 as shipped. If they refuse, record the error,
      then try a stateless re-export.
- [ ] Then under the server: `--whisper-device NPU`, hit
      `POST /v1/audio/transcriptions`.
- [ ] Measure against CPU and iGPU on the same clips: wall clock per second of
      audio, compile/cold-load time (NPU compile can be slow), and whether the
      transcript matches.
- [ ] Run a transcription **while** the GPU slot generates. The concurrency
      claim is the reason for doing this, so measure it rather than assume it.

## Done when

`docs/dev/models.md` carries an NPU verdict (works, or the error) per Whisper
entry on both NPU generations. If it works, the `--whisper-device` help, the
installer and `docs/DEVICES.md` say NPU. If not, the reason goes in `TODONT.md`.
