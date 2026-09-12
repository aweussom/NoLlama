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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all passed")
