"""Regression test for the image-placeholder repetition-penalty guard.

Guards the one-shot retry that keeps a VLM usable when its export encodes
image placeholders outside `[0, vocab_size)`. genai's repetition-penalty
transformer walks the *prompt* ids and asserts on those, so every image turn
died while text turns worked — reproduced on three GPUs (Arc 140T, 140V and
Arc Pro B60) and two runtimes (genai 2026.3.0.0-3277, 2026.5.0.0-3402).

The costly half of that behaviour (a real model, a real GPU) lives in
scripts/bare-probe.py. What is worth pinning here is the cheap half that a
future edit could silently break: that the guard fires ONLY for image turns
on a model already proven bad, that text turns keep their penalty, and that
an unrelated RuntimeError is still re-raised rather than swallowed by a
retry.

    venv\\Scripts\\python tests\\test_vlm_penalty_guard.py
    venv\\Scripts\\python -m pytest tests\\test_vlm_penalty_guard.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nollama import DeviceSlot  # noqa: E402  (needs the path above)


class _Gen:
    """Stand-in for genai's GenerationConfig — only the field the guard touches."""

    def __init__(self, repetition_penalty=1.05):
        self.repetition_penalty = repetition_penalty


def _slot():
    """A DeviceSlot with no model, enough for the pure-logic methods."""
    return DeviceSlot.__new__(DeviceSlot)


BOUNDS_ERROR = RuntimeError(
    "Check '(prompt_id >= 0) && (prompt_id < vocab_size)' failed at "
    "src/cpp/src/sampling/logit_transformers.hpp:412:\ninput_ids token out of bounds"
)


def test_recognises_the_real_assertion():
    assert _slot()._is_placeholder_bounds_error(BOUNDS_ERROR)


def test_ignores_unrelated_errors():
    slot = _slot()
    assert not slot._is_placeholder_bounds_error(RuntimeError("out of memory"))
    # Half a match must not count: an OOM naming a vocab_size tensor is not
    # this bug, and retrying it without a penalty would hide a real failure.
    assert not slot._is_placeholder_bounds_error(RuntimeError("vocab_size mismatch"))
    assert not slot._is_placeholder_bounds_error(RuntimeError("prompt_id missing"))


def test_guard_does_nothing_before_the_model_proves_itself():
    slot = _slot()
    slot._rep_penalty_breaks_images = False
    gen = _Gen()
    assert slot._vlm_penalty_guard(gen, images=["img"]) is False
    assert gen.repetition_penalty == 1.05


def test_guard_clears_penalty_for_image_turns_once_flagged():
    slot = _slot()
    slot._rep_penalty_breaks_images = True
    gen = _Gen()
    assert slot._vlm_penalty_guard(gen, images=["img"]) is True
    assert gen.repetition_penalty == 1.0


def test_text_turns_keep_their_penalty_on_a_flagged_model():
    # The defect is in the *prompt* ids, so a text turn carries no
    # placeholders and has no reason to lose repetition control.
    slot = _slot()
    slot._rep_penalty_breaks_images = True
    gen = _Gen()
    assert slot._vlm_penalty_guard(gen, images=[]) is False
    assert gen.repetition_penalty == 1.05


def test_guard_is_idempotent_when_penalty_already_neutral():
    # A client that sent repetition_penalty=1.0 itself should not be
    # reported as "disabled" — nothing was taken away.
    slot = _slot()
    slot._rep_penalty_breaks_images = True
    gen = _Gen(repetition_penalty=1.0)
    assert slot._vlm_penalty_guard(gen, images=["img"]) is False
    assert gen.repetition_penalty == 1.0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL {name}: {e}")
    print("\nall passed" if not failures else f"\n{failures} failed")
    sys.exit(1 if failures else 0)
