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

- [ ] Works in Edge and Firefox on the laptop with a real microphone —
      **needs Tommy**, Claude cannot speak into it.
- [ ] Button absent with no Whisper slot; disabled with a reason over
      plain-HTTP LAN.
- [ ] `docs/` mention it next to the web UI description; CLAUDE.md's web UI
      line gets one clause.
