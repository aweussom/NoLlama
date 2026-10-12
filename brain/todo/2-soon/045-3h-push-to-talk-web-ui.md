# T-045 — Push-to-talk in the web UI

**Effort:** 3 h · **Produces:** a mic button in `templates/index.html` /
`static/js/app.js` · **Area:** web UI, Whisper · **Depends on:** T-042's
server step and its silence guard

Talk to the laptop, the NPU transcribes, the GPU answers, nothing leaves the
machine. It is the concrete answer to "why can't a notebook do what the
STM32N6 does" (Facebook thread, 2026-10-11), and it gives the NPU a job on
boxes where the chat model lives on the GPU.

## Shape

- Button shown only when `/health` reports a Whisper slot `ready`. Its
  tooltip names the device.
- Hold to talk (mouse and Space-when-input-empty), release to send. The
  transcript lands in the input box for editing; it is **not** auto-sent.
- Browsers record WebM/Opus, which `soundfile` cannot read. Decode in the
  browser (`AudioContext.decodeAudioData`), downmix, resample to 16 kHz,
  encode 16-bit WAV, POST to `/v1/audio/transcriptions`. No server change.
- `getUserMedia` needs a secure context: `localhost` works, `http://<lan-ip>`
  does not. Say so in the button's disabled state instead of failing
  silently.
- Clips under ~0.3 s are dropped client-side; the server's silence guard
  covers the rest (T-042).
- Language: a small selector next to the button, remembered in
  `localStorage`, default from `navigator.language`. Sending none is what
  makes Whisper hallucinate.

## Done when

- [x] Edge, real microphone, laptop: "Testing push to talk." word-perfect,
      2.0 s clip transcribed in 0.4 s on the NPU with Qwen3-8B on the iGPU
      beside it. GPU driver 32.0.101.8991, NPU driver 1005540, OpenVINO
      2026.4.1 (2026-10-12).
- [ ] Firefox, real microphone — **needs Tommy**. First run after the
      2026-10-12 driver update, so it doubles as the new-driver check.
      Built on branch `t-045-push-to-talk`; merge when this box is ticked.
- [x] Headless Chrome with `jfk.wav` as a fake mic (`scripts/ptt-e2e.mjs`):
      word-perfect through the NPU, 630 ms release-to-text; silence comes
      back empty and shows "Nothing heard" (2026-10-11).
- [x] Disabled with its reason over plain-HTTP LAN (same script, LAN URL).
      Hidden with no Whisper slot: by code reading only.
- [x] `docs/API.md` web UI list, CLAUDE.md web UI line, `docs/webui-ptt.mmd`.
