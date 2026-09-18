# T-013 — Benchmark Intel's pre-exported Whisper models

**Effort:** 2 h · **Produces:** replaced `models.json` whisper entries ·
**Area:** `models.json`, `benchmark.py`

Intel ships pre-exported, pre-quantized Whisper models on
[huggingface.co/OpenVINO](https://huggingface.co/OpenVINO). Our `models.json`
whisper entries still convert from `openai/whisper-*` to FP16 — slower install,
larger files, and a conversion step that can fail on a user's box.

(This item had lost its heading in `TODO.md` and was sitting inside the
CPU-non-goal section, which is part of why nobody picked it up.)

Candidates:

- [ ] **`OpenVINO/distil-whisper-large-v3-int8-ov`** — most-downloaded whisper
      variant in the OpenVINO org (9k+ downloads). Distilled large-v3 is
      reportedly ~6x faster at similar accuracy; INT8 on top should be a real
      win.
- [ ] **`OpenVINO/whisper-large-v3-int4-ov`** — best accuracy if the size fits.
      INT4 makes large-v3 viable on 16 GB GPUs.
- [ ] **`OpenVINO/whisper-medium-int8-ov`** and `-int4-ov` — the direct upgrade
      path for our current "Whisper Medium" FP16 entry.
- [ ] **`OpenVINO/whisper-small-int4-ov`** — smallest viable multi-language.

Measure (extend `benchmark.py --backend whisper`?):

- [ ] WER on Norwegian **and** English samples — local audio exists
- [ ] wallclock per second of audio
- [ ] cold-load time
- [ ] memory footprint

## Done when

`models.json` carries whatever benchmarked best, tagged as proven, and the rest
are dropped. Do not add untested ones to the install menu.
