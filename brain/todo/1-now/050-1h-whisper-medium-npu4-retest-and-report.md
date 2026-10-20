# T-050 — Whisper medium on NPU 4: report upstream

**Effort:** 15 min left · **Produces:** an upstream issue, filed by Tommy ·
**Area:** NPU, Whisper · **Machine:** the **258V laptop** (NPU 4) — nowhere
else has one

Split out of T-042. `OpenVINO/whisper-medium-int8-ov` as shipped:

- **NPU 4 (laptop), driver 32.0.100.5540**: compiles, then the first
  inference dies with `ZE_RESULT_ERROR_DEVICE_LOST` (2026-10-11, twice;
  2026-10-20, release and nightly).
- **NPU 3 (285K), driver 32.0.100.5540**, same OpenVINO/genai 2026.4.1,
  same files: runs, 64/64 clips, WER equal to CPU (2026-10-19).

So the NPU generation is the variable, as in openvino#38100 (LFM2).

**The "new driver" premise was wrong.** What went on the laptop on
2026-10-18 was the Arc driver (8991 → 9150). The NPU driver is `32.0.100.x`,
a separate package, and 5540 is Intel's newest as of 2026-10-20 (Intel's
download page and the Series 2 support page both list it). So the crash is
on a current driver and there is no re-test to wait for. `machines.md` now
says so.

## Done on the laptop, 2026-10-20

- [x] Driver recorded: NPU `32.0.100.5540` (2026-08-20), GPU `32.0.101.9150`.
- [x] Release stack: DEVICE_LOST on first generate, compile 130 s. One
      earlier run that evening died with exit 139 (segfault) before
      printing the compile line; the retry with `-u` gave the DEVICE_LOST
      below. Say "twice on 2026-10-11, twice on 2026-10-20" if asked.
- [x] Nightly: identical DEVICE_LOST, compile 131 s.
- [x] Controls: base on NPU OK (0.18 s), medium on CPU OK (3.50 s).
- [x] `docs/dev/models.md` NPU 4 paragraph carries the re-confirmation.
- [ ] `docs/MODELS.md` and the `--whisper-device` help already say
      "crashes a Lunar Lake NPU", nothing to change.

## Tommy: file it

Target `openvinotoolkit/openvino.genai` (it is `WhisperPipeline`); Intel may
move it to `openvino`, as with #38100. Attach `scripts/whisper-npu-repro.py`
and `jfk.wav` (whisper.cpp's sample, 16 kHz mono). Run `detune` on the text
first (done 2026-10-20: stamped copy at the scratchpad `t050-issue.detuned.md`, fact-check half unchecked because the checker reply was unparseable; it has an `UNCHECKED` note appended that must not be posted). After filing: add the number to
`brain/next/waiting-on-time/007-*`, `git mv` this file there, and leave a
short comment on openvino#38100 linking it (N-007: no bare nag).

## Draft issue

> **[Whisper][NPU] whisper-medium-int8-ov: ZE_RESULT_ERROR_DEVICE_LOST on
> Lunar Lake (NPU 4), works on Arrow Lake (NPU 3) with the same driver**
>
> `OpenVINO/whisper-medium-int8-ov`, unmodified, in `WhisperPipeline` on
> `NPU`. Compiles, then the first `generate()` fails with
> `L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST, code
> 0x70000001`. Same files, driver and OpenVINO run fine on NPU 3.
> `whisper-base-int8-ov` and `whisper-small-int8-ov` work on both. Same
> result on 2026.5.0 nightly. 32.0.100.5540 is the current NPU driver.
>
> | | Core Ultra 7 258V (NPU 4, arch 4000) | Core Ultra 9 285K (NPU 3, arch 3720) |
> |---|---|---|
> | NPU driver | 32.0.100.5540 | 32.0.100.5540 |
> | openvino / genai | 2026.4.1 / 2026.4.1.0 | same |
> | nightly | 2026.5.0-23394 / 2026.5.0.0-3521 | not run |
> | medium | DEVICE_LOST on first inference, release and nightly | OK |
> | base, small | OK | OK |
> | medium on CPU | OK | OK |
>
> Repro (attached `whisper-npu-repro.py`, 16 kHz `jfk.wav` from whisper.cpp):
> `python whisper-npu-repro.py whisper-medium-int8-ov NPU jfk.wav`
>
> NPU 4, release:
> ```
> openvino 2026.4.1-22982-e213a147257-releases/2026/4 | genai 2026.4.1.0-3408-ab01d1771e4
> device NPU Intel(R) AI Boost
> NPU_DRIVER_VERSION 1005540 | DEVICE_ARCHITECTURE 4000
> compiled in 130.2 s
> generate ...
> FAILED: RuntimeError('Exception from src\inference\src\cpp\infer_request.cpp:246:
> Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:382:
> L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST, code 0x70000001 - device hung, reset, was removed, or driver update occurred
> ')
> ```
> NPU 4, nightly:
> ```
> openvino 2026.5.0-23394-972aa05b9b6 | genai 2026.5.0.0-3521-283994ee0a5
> device NPU Intel(R) AI Boost
> NPU_DRIVER_VERSION 1005540 | DEVICE_ARCHITECTURE 4000
> compiled in 131.3 s
> generate ...
> FAILED: RuntimeError('Exception from src\inference\src\cpp\infer_request.cpp:246:
> Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:384:
> L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST, code 0x70000001 - device hung, reset, was removed, or driver update occurred
> ')
> ```
> NPU 4 controls, same session: base on NPU `OK in 0.18 s`, medium on CPU
> `OK in 3.50 s`, both with the correct transcript.
>
> NPU 3, release:
> ```
> openvino 2026.4.1-22982-e213a147257-releases/2026/4 | genai 2026.4.1.0-3408-ab01d1771e4
> device NPU Intel(R) AI Boost
> NPU_DRIVER_VERSION 1005540 | DEVICE_ARCHITECTURE 3720
> compiled in 44.8 s
> generate ...
> OK in 1.69 s: And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
> ```
>
> Over 64 clips (FLEURS nb/en, NPSC, silence) medium on NPU 3 matches CPU's
> WER. Possibly related: #38100, also correct on NPU 3 and wrong on NPU 4
> with files and driver held constant.

## Done when

The issue is filed and this file is in `waiting-on-time`.
