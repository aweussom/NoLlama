# T-050 — Whisper medium on NPU 4: re-test on the new driver, then report upstream

**Effort:** 1 h · **Produces:** a verdict on the new NPU driver; an upstream
issue *drafted* (Tommy files it) or a closed question · **Area:** NPU,
Whisper · **Machine:** the **258V laptop** (NPU 4) — nowhere else has one

Split out of T-042. `OpenVINO/whisper-medium-int8-ov` as shipped:

- **NPU 4 (laptop), driver 32.0.100.5540**: compiles, then the first
  inference dies with `ZE_RESULT_ERROR_DEVICE_LOST` (2026-10-11, twice).
- **NPU 3 (285K), driver 32.0.100.5540**, same OpenVINO/genai 2026.4.1,
  same files: runs, 64/64 clips, WER equal to CPU (2026-10-19).

So the NPU generation is the variable, as in openvino#38100 (LFM2). Tommy
installed a new NPU driver on the laptop on 2026-10-18; a report on an old
driver gets closed on sight, so the new one goes first.

## Steps (laptop)

- [ ] Record the driver: `Get-CimInstance Win32_PnPSignedDriver | ? DeviceName -match 'AI Boost' | select DriverVersion, DriverDate`.
      Update the laptop row in `docs/dev/machines.md`.
- [ ] Have the model and clip. The model is probably already on disk from
      T-042; else `.\download-model.ps1 OpenVINO/whisper-medium-int8-ov`.
      Clip: `jfk.wav` from `https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav`,
      into its own directory.
- [ ] Release stack:
      `.\venv\Scripts\python scripts\whisper-npu-repro.py <medium-dir> NPU <jfk.wav>`
      — the process may simply die after `generate ...`; that is the failure.
- [ ] Same on `venv-nightly` if the laptop has one (Intel asked for a nightly
      run on #38211; offering it up front saves a round trip).
- [ ] Control on the same box: base on NPU, medium on CPU — both should pass.

## Then

**If medium now passes on NPU 4:** the driver fixed it. No issue. Update the
NPU 4 row in `docs/dev/models.md` (keep the old 5540 failure, add the new
driver's result with date), `docs/MODELS.md`, and the `--whisper-device` help
in `nollama.py`. Medium then runs on both NPU generations. Delete this item.

**If it still dies:** draft the issue below into this file, with the real
output pasted in, and stop — **Tommy files it**. Target
`openvinotoolkit/openvino.genai` (it is `WhisperPipeline`); Intel may move it
to `openvino`, as with #38100. Then add it to the watch list in
`brain/next/waiting-on-time/007-*` and move this item there. Once filed, a
short comment on openvino#38100 linking it is the nudge that thread has been
waiting for (N-007 says: no bare nag).

## Draft issue

> **[Whisper][NPU] whisper-medium-int8-ov: ZE_RESULT_ERROR_DEVICE_LOST on
> Lunar Lake (NPU 4), works on Arrow Lake (NPU 3) with the same driver**
>
> `OpenVINO/whisper-medium-int8-ov`, unmodified, in `WhisperPipeline` on
> `NPU`. Compiles, then the first `generate()` fails with
> `L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST, code
> 0x70000001`. The same files, driver and OpenVINO run correctly on an NPU 3.
> `whisper-base-int8-ov` and `whisper-small-int8-ov` work on both.
>
> | | Core Ultra 7 258V (NPU 4, arch 4000) | Core Ultra 9 285K (NPU 3, arch 3720) |
> |---|---|---|
> | NPU driver | `<new laptop driver>` | 32.0.100.5540 |
> | openvino / genai | 2026.4.1 / 2026.4.1.0 | same |
> | medium | DEVICE_LOST on first inference | OK |
> | base, small | OK | OK |
>
> Repro (attached `whisper-npu-repro.py`, 16 kHz `jfk.wav`):
> `python whisper-npu-repro.py whisper-medium-int8-ov NPU jfk.wav`
>
> NPU 3 output:
> ```
> openvino 2026.4.1-22982-e213a147257-releases/2026/4 | genai 2026.4.1.0-3408-ab01d1771e4
> device NPU Intel(R) AI Boost
> NPU_DRIVER_VERSION 1005540 | DEVICE_ARCHITECTURE 3720
> compiled in 44.8 s
> generate ...
> OK in 1.69 s: And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
> ```
> NPU 4 output: `<paste>`
> Nightly (`<version>`): `<paste>`
>
> Over 64 clips (FLEURS nb/en, NPSC, silence) medium on NPU 3 matches CPU's
> WER. Possibly related: #38100, also correct on NPU 3 and wrong on NPU 4
> with files and driver held constant.

## Done when

The new driver's NPU 4 result is in `docs/dev/models.md`, and either the docs
say medium runs on both NPU generations or a drafted issue is waiting for
Tommy to file.
