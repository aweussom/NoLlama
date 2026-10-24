# T-050 — Whisper medium on NPU 4: filed as genai#4677, now openvino#39053, waiting on Intel

**Effort:** 0 · **Waiting on:** Intel, via the N-007 watch list · **Produces:** nothing until the ticket moves ·
**Area:** NPU, Whisper · **Machine:** the **258V laptop** (NPU 4) — nowhere
else has one

**Scope: one model on one NPU. Whisper itself works and ships.** base and
small run on NPU 4 (50x / 25x real time, WER equal to CPU) and that is what
`--whisper-device NPU` serves; all three sizes run on CPU, iGPU and NPU 3;
the web UI's push-to-talk (T-045) transcribes through
`/v1/audio/transcriptions`. Only **medium on NPU 4** crashes. Nothing here
blocks a user — they get base/small on the NPU or medium elsewhere.
→ `STATUS.md`, `docs/dev/models.md` "Whisper (speech-to-text)".

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

## Filed

[openvinotoolkit/openvino.genai#4677](https://github.com/openvinotoolkit/openvino.genai/issues/4677), moved by Intel to [openvino#39053](https://github.com/openvinotoolkit/openvino/issues/39053) by 2026-10-22,
2026-10-20, from the laptop, body detuned and stamped. Link comment left on
openvino#38100 the same day. Row added to N-007. The issue text is the
ticket; the draft below was deleted with the filing.

## Done when

Intel answers on #39053, or a release makes medium run on NPU 4. Either way
the NPU 4 row in `docs/dev/models.md` and the `--whisper-device` help move,
and this file is deleted.
