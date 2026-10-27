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

- [x] `install.ps1` offers Whisper, filtered by hardware: base withdrawn,
      medium off NPU 4, one fit taken without a menu. With an NPU it is asked
      before the use-case menu, as the NPU's job; without one, after the
      models on CPU/GPU (Tommy, 2026-10-27). `Test-ModelCacheValid` now knows
      Whisper's encoder/decoder pair; before, a cached Whisper was deleted
      and re-downloaded on every install.
- [x] `tests/test_install_whisper.ps1`: NPU 4, NPU 3, CPU, GPU, declined,
      skipped, ordering, cache check. Function-level, no real install.
- [x] Chat + Whisper on NPU 4 in one server, sequential and parallel
      (`docs/dev/models.md`).
- [ ] A real `install.ps1` run on the laptop (NPU 4) and the 285K (NPU 3):
      the generated `start.ps1` starts with Whisper and the mic shows. The
      285K checkout has local edits to `scripts/decay-probe.py` and an
      untracked bench file that block a pull; Tommy's call.
