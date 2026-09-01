r"""Contract test for the post-failure VLM state reset.

The reset exists because a VLM generate that throws part-way was reported to
wedge the slot — every later request dying on `Prompt ids size is less than
tokenized history size` (issue #24, Arc 140T). That wedge is NOT reproduced
locally and the reset is defensive; see `_reset_vlm_state`'s docstring for
exactly how much is evidence and how much is inference.

What a GPU cannot cheaply prove but a future edit can easily break is the
contract this thing runs under: it is called from inside `except` blocks, so
it must never raise, never mask the caller's original error, and must no-op
on a runtime that has no `finish_chat` at all — which is the shape genai is
heading for, since `finish_chat()` is deprecated as of 2026.3.

    venv\Scripts\python tests\test_vlm_state_reset.py
    venv\Scripts\python -m pytest tests\test_vlm_state_reset.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nollama import DeviceSlot  # noqa: E402  (needs the path above)


class _Pipe:
    """Pipe stand-in that records finish_chat() calls."""

    def __init__(self, raises=None):
        self.calls = 0
        self._raises = raises

    def finish_chat(self):
        self.calls += 1
        if self._raises:
            raise self._raises


class _PipeWithoutFinishChat:
    """A runtime predating (or postdating) the deprecated finish_chat API."""


def _slot(pipe):
    slot = DeviceSlot.__new__(DeviceSlot)
    slot.pipe = pipe
    return slot


def test_calls_finish_chat_when_available():
    pipe = _Pipe()
    _slot(pipe)._reset_vlm_state()
    assert pipe.calls == 1


def test_no_op_when_runtime_has_no_finish_chat():
    # finish_chat() is deprecated upstream; when it is finally removed this
    # must degrade to doing nothing, not to an AttributeError on every
    # failed image turn.
    _slot(_PipeWithoutFinishChat())._reset_vlm_state()


def test_never_raises_when_finish_chat_itself_fails():
    # It runs inside an exception handler. A throw here would replace the
    # caller's real error with a confusing one from the cleanup path.
    pipe = _Pipe(raises=RuntimeError("cache state already gone"))
    _slot(pipe)._reset_vlm_state()
    assert pipe.calls == 1


def test_returns_none_and_stays_silent_on_repeat_calls():
    # Both generate paths can reach it for a single failure; a second call
    # must be as harmless as the first.
    pipe = _Pipe()
    slot = _slot(pipe)
    assert slot._reset_vlm_state() is None
    assert slot._reset_vlm_state() is None
    assert pipe.calls == 2


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
