"""CL_OUT_OF_RESOURCES takes the slot out of service instead of failing per request.

Why this exists: a user provoked that error on an Arc 140T (issue #38,
2026-09-12) and the slot never recovered until NoLlama was restarted — once
until a reboot. OpenVINO's own error text warns the driver context is gone.
So the explainer names the remedy, and the seam marks the slot "error" with
a reason so routing skips it and /health shows why.

    venv\\Scripts\\python -m pytest tests\\test_poisoned_gpu.py -q
    venv\\Scripts\\python tests\\test_poisoned_gpu.py          # no pytest needed
"""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nollama  # noqa: E402
from nollama import DeviceSlot, explain_genai_error, _is_poisoned_context_error  # noqa: E402

REAL = (
    "Exception from src\\inference\\src\\cpp\\infer_request.cpp:224:\n"
    "Exception from src\\plugins\\intel_gpu\\src\\runtime\\ocl\\ocl_common.hpp:62:\n"
    "[GPU] CL_OUT_OF_RESOURCES exception.\n"
    "        Due to a driver bug, any subsequent OpenCL API call may cause the application to hang, "
    "so the GPU plugin may be unable to finish correctly."
)


class PoisonPipe:
    """generate() throws the real CL_OUT_OF_RESOURCES text on the first call."""

    def generate(self, history, gen, streamer):
        raise RuntimeError(REAL)


def _slot(pipe):
    slot = DeviceSlot.__new__(DeviceSlot)
    slot.lock = threading.Lock()
    slot._cancel = threading.Event()
    slot._stream_error = None
    slot.error_reason = None
    slot.status = "ready"
    slot.device_name = "GPU"
    slot.model_name = "fake"
    slot.pipe = pipe
    slot.last_used = 0
    slot.last_ttft_ms = None
    slot.think_preseeded = False
    # what the info property (the /health block) reads
    slot.model_type = "llm"; slot.device_full = "fake GPU"; slot.backend = "genai"
    slot.prewarmed = False; slot.kv_pool_gb = 0; slot.model_dir = "."
    return slot


def test_detector_only_fires_on_that_error():
    assert _is_poisoned_context_error(RuntimeError(REAL))
    assert not _is_poisoned_context_error(RuntimeError("Exceeded max size of memory object allocation: requested 5 bytes"))
    assert not _is_poisoned_context_error(RuntimeError("Got unfinished GenerationStatus"))


def test_explainer_names_restart_not_a_flag():
    msg = explain_genai_error(RuntimeError(REAL))
    assert "restart NoLlama" in msg and "reboot" in msg
    assert "poisoned" in msg


def test_seam_marks_slot_out_of_service():
    slot = _slot(PoisonPipe())
    frames = list(slot.stream_llm([{"role": "user", "content": "hi"}], None, "x", 0, time.perf_counter()))
    finish = [json.loads(f[6:])["choices"][0]["finish_reason"] for f in frames if f.startswith("data: {")]
    assert "error" in finish, finish
    assert slot.status == "error"
    assert slot.error_reason and "CL_OUT_OF_RESOURCES" in slot.error_reason
    assert not nollama._slot_serviceable(slot), "a poisoned slot must not be routed to"
    assert slot.info["reason"] == slot.error_reason


def test_ordinary_error_leaves_slot_in_service():
    class OrdinaryPipe:
        def generate(self, history, gen, streamer):
            raise RuntimeError("Got unfinished GenerationStatus")
    slot = _slot(OrdinaryPipe())
    list(slot.stream_llm([{"role": "user", "content": "hi"}], None, "x", 0, time.perf_counter()))
    assert slot.status == "ready" and slot.error_reason is None


# --------------------------------------------------------------------------
# The non-streaming paths. Until 2026-09-13 only the two streaming handlers
# called _note_poisoned, so a non-streaming request left the slot "ready" —
# and a ready slot is what the idle watchdog unloads, which aborts the
# process (issue #38, the reporter's second traceback).
# --------------------------------------------------------------------------

class _VlmPipe:
    """VLMPipeline stand-in: generate() throws, finish_chat() is recorded."""

    def __init__(self, exc):
        self.exc = exc
        self.finish_chat_calls = 0

    def generate(self, **kwargs):
        raise self.exc

    def finish_chat(self):
        self.finish_chat_calls += 1


def _vlm_slot(pipe):
    slot = _slot(pipe)
    slot.model_type = "vlm"
    slot._rep_penalty_breaks_images = False
    slot._atem = False
    return slot


class _Gen:
    repetition_penalty = 1.05


def _raises(fn):
    try:
        fn()
    except Exception as e:
        return e
    raise AssertionError("expected the original error to propagate")


def test_nonstreaming_llm_marks_slot_out_of_service():
    class Pipe:
        def generate(self, history, gen):
            raise RuntimeError(REAL)
    slot = _slot(Pipe())
    e = _raises(lambda: slot.generate_llm([{"role": "user", "content": "hi"}], None))
    assert "CL_OUT_OF_RESOURCES" in str(e), "the caller still sees the real error"
    assert slot.status == "error"
    assert not nollama._slot_serviceable(slot)


def test_nonstreaming_vlm_marks_slot_out_of_service():
    slot = _vlm_slot(_VlmPipe(RuntimeError(REAL)))
    e = _raises(lambda: slot.generate_vlm("hi", [], _Gen()))
    assert "CL_OUT_OF_RESOURCES" in str(e)
    assert slot.status == "error"
    assert not nollama._slot_serviceable(slot)


def test_poisoned_vlm_throw_does_not_touch_the_pipeline():
    """The regression guard: _reset_vlm_state calls finish_chat(), an OpenCL
    call, and OpenVINO says any OpenCL call after this error may hang the
    application. So the poison check must run BEFORE the reset, not after."""
    pipe = _VlmPipe(RuntimeError(REAL))
    slot = _vlm_slot(pipe)
    _raises(lambda: slot.generate_vlm("hi", [], _Gen()))
    assert pipe.finish_chat_calls == 0, "finish_chat() ran on a dead OpenCL context"


def test_ordinary_vlm_throw_still_resets_state():
    """The other half: an ordinary throw must keep the issue #24 reset."""
    pipe = _VlmPipe(RuntimeError("Got unfinished GenerationStatus"))
    slot = _vlm_slot(pipe)
    _raises(lambda: slot.generate_vlm("hi", [], _Gen()))
    assert pipe.finish_chat_calls == 1
    assert slot.status == "ready" and slot.error_reason is None


def test_idle_watchdog_never_unloads_a_poisoned_slot():
    """Unloading one killed the reporter's process; status "error" is what
    keeps it away from here, so the guard is asserted rather than assumed."""
    class _Stop(Exception):
        pass

    unloaded = []
    slot = _slot(PoisonPipe())
    slot.unload = lambda: unloaded.append(slot.device_name)
    slot.status = "error"        # what _note_poisoned leaves behind
    slot.error_reason = "poisoned"

    calls = []
    real_sleep = nollama.time.sleep

    def _sleep(_):
        calls.append(1)
        if len(calls) > 1:
            raise _Stop()

    nollama.time.sleep = _sleep
    try:
        nollama._idle_watchdog([slot], idle_timeout=0, check_interval=0)
    except _Stop:
        pass
    finally:
        nollama.time.sleep = real_sleep

    assert unloaded == [], "the watchdog unloaded a slot with a dead OpenCL context"

    # ...and the guard is the status, not the absence of a timeout: the same
    # slot marked ready IS unloaded, which is what makes the test meaningful.
    slot.status = "ready"
    calls.clear()
    nollama.time.sleep = _sleep
    try:
        nollama._idle_watchdog([slot], idle_timeout=0, check_interval=0)
    except _Stop:
        pass
    finally:
        nollama.time.sleep = real_sleep
    assert unloaded == ["GPU"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all passed")
