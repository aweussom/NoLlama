# T-042 — Offer Whisper in the install menu

**Effort:** 1 h · **Produces:** a Whisper choice in `install.ps1`, written into
`start.ps1` as `--whisper-dir` / `--whisper-device` · **Area:**
installer, `models.json` (`whisper`)

Whisper is served (`POST /v1/audio/transcriptions`, the web UI's
push-to-talk), on CPU, GPU and NPU, but the installer never offers it, so a
user only gets it by editing `start.ps1` by hand.

**Decided 2026-10-19 (Tommy): offer it.** It is in the API and the web UI,
and it is the NPU's best workload.

- Default: **small on the NPU** where there is one, CPU otherwise.
- **Medium stays off the NPU choice on NPU 4** (Lunar Lake): it dies with
  `ZE_RESULT_ERROR_DEVICE_LOST` on first inference. Fine on NPU 3, CPU and
  GPU. Tracked as T-050 / openvino#39053.
- NPU on Lunar Lake is not free under sustained load (chat -32..-47% with
  Whisper running non-stop), fine for push-to-talk. Say nothing in the menu;
  it is in `docs/dev/models.md`.

Measured verdicts per device and NPU generation: `docs/dev/models.md`,
"Whisper (speech-to-text)".

## Done when

- [ ] `install.ps1` offers Whisper (none / base / small / medium), defaulting
      to small on the NPU or CPU, and hides medium-on-NPU on arch 4000.
- [ ] The generated `start.ps1` carries the flags; tested on the laptop
      (NPU 4) and the 285K (NPU 3).
- [ ] `tests/test_menu.ps1` covers the new choice.
