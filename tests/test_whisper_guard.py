r"""The Whisper silence guard: audio_level_spread and collapse_repeat_loop.

Whisper turns silence into text, and on the NPU into a phrase looped to the
token limit. The guard skips clips whose loud-to-quiet spread is under 10 dB
and collapses back-to-back repeats after decoding.

The real-data case reads the T-042 clip set (scripts/whisper-build-set.py,
~/audio-samples/set) so the threshold is checked against the speech that
justified it: every speech clip must pass, the silence must not. Skips
cleanly when the set is not on this machine. The noise cases are synthetic:
there is no recorded noise in the set.

    venv\Scripts\python tests\test_whisper_guard.py
    venv\Scripts\python -m pytest tests\test_whisper_guard.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nollama import (WHISPER_MIN_SPREAD_DB, audio_level_spread,  # noqa: E402
                     collapse_repeat_loop)

SET = os.path.expanduser("~/audio-samples/set")
SR = 16000


def _level(x, db):
    return (x / np.sqrt((x ** 2).mean()) * 10 ** (db / 20)).astype(np.float32)


def test_digital_silence_and_tiny_clips_read_zero():
    assert audio_level_spread(np.zeros(10 * SR, np.float32)) == 0.0
    assert audio_level_spread(np.zeros(100, np.float32)) == 0.0
    assert audio_level_spread(np.ones(500, np.float32)) == 0.0


def test_steady_noise_is_below_threshold_at_any_level():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(5 * SR)
    t = np.arange(5 * SR) / SR
    for db in (-70, -45, -20):
        assert audio_level_spread(_level(white, db)) < WHISPER_MIN_SPREAD_DB
    hum = _level(np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 150 * t), -30)
    assert audio_level_spread(hum) < WHISPER_MIN_SPREAD_DB
    fan = _level(white * (1 + 0.2 * np.sin(2 * np.pi * 0.5 * t)), -45)
    assert audio_level_spread(fan) < WHISPER_MIN_SPREAD_DB


def test_bursty_signal_passes_even_when_quiet():
    # A stand-in for speech: bursts over a floor, peaking near -55 dBFS like
    # the quietest FLEURS en clips.
    rng = np.random.default_rng(1)
    a = _level(rng.standard_normal(4 * SR), -85)
    for s in range(0, 4 * SR, SR // 2):
        a[s:s + SR // 4] += _level(rng.standard_normal(SR // 4), -55)
    assert audio_level_spread(a) > WHISPER_MIN_SPREAD_DB


def test_real_set_speech_passes_silence_does_not():
    man = os.path.join(SET, "manifest.jsonl")
    if not os.path.exists(man):
        print("  SKIP real set: ~/audio-samples/set not present")
        return
    import soundfile as sf
    speech = silence = 0
    for line in open(man, encoding="utf-8"):
        c = json.loads(line)
        a, sr = sf.read(os.path.join(SET, c["path"]), dtype="float32")
        spread = audio_level_spread(a, sr)
        if c["group"] == "silence":
            assert spread < WHISPER_MIN_SPREAD_DB, (c["path"], spread)
            silence += 1
        else:
            assert spread >= WHISPER_MIN_SPREAD_DB, (c["path"], spread)
            speech += 1
    assert speech >= 60 and silence == 1, (speech, silence)


def test_npu_silence_loop_collapses():
    loop = "Vi har ikke en av de som er " + "en av de som er " * 60
    out = collapse_repeat_loop(loop.strip())
    assert out == "Vi har ikke en av de som er", out


def test_ordinary_repeats_survive():
    jfk = ("And so my fellow Americans, ask not what your country can do for "
           "you, ask what you can do for your country.")
    assert collapse_repeat_loop(jfk) == jfk
    assert collapse_repeat_loop("No, no, no, no. Not that one.") == "No, no, no, no. Not that one."
    assert collapse_repeat_loop("very very very good") == "very very very good"


def test_single_word_needs_eight():
    assert collapse_repeat_loop(" ".join(["ha"] * 7)) == " ".join(["ha"] * 7)
    assert collapse_repeat_loop("so " + " ".join(["ha"] * 9) + " end") == "so ha end"


def test_case_and_punctuation_do_not_hide_a_loop():
    loop = "Thank you. thank you, Thank you! thank you. THANK YOU."
    assert collapse_repeat_loop(loop) == "Thank you."


def test_empty_and_whitespace():
    assert collapse_repeat_loop("") == ""
    assert collapse_repeat_loop("  a   b  ") == "a b"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all passed")
