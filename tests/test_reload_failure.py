"""A reload that fails leaves the slot retryable, with the reason in /health.

Why this exists: T-046. On 2026-10-12 an idle-unloaded GPU slot's reload
raised; status stayed 'loading', which routing never serves, so every request
after the first got 503 and the load ticker printed 'still loading...' until
the process was restarted. These tests force the failure with a stubbed
load(), so no model or device is needed.

    venv\\Scripts\\python -m pytest tests\\test_reload_failure.py -q
    venv\\Scripts\\python tests\\test_reload_failure.py          # no pytest needed
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nollama  # noqa: E402
from nollama import DeviceSlot, _LoadProgress, _slot_serviceable  # noqa: E402

USM = ("Exception from src\\inference\\src\\cpp\\core.cpp:112:\n"
       "[CL ext] Can not allocate 622329856 bytes for USM Host. "
       "ptr: 0000000000000000, error: -6")
POISON = ("Exception from src\\plugins\\intel_gpu\\src\\runtime\\ocl\\ocl_common.hpp:62:\n"
          "[GPU] CL_OUT_OF_RESOURCES exception.")


def _unloaded_slot():
    slot = DeviceSlot("GPU")
    slot.model_dir = "fake-model"
    slot.model_name = "fake"
    slot.unload()          # pipe is None: no-op, but keep the real path honest
    slot.status = "idle_unloaded"
    return slot


def _failing_load(err):
    def load(self, model_dir):
        self.status = "loading"   # what the real load() does first
        raise RuntimeError(err)
    return load


def test_failed_reload_is_retryable_with_reason(monkeypatch=None):
    slot = _unloaded_slot()
    slot.load = _failing_load(USM).__get__(slot)
    try:
        slot.ensure_loaded()
        assert False, "ensure_loaded should re-raise"
    except RuntimeError as e:
        assert "USM Host" in str(e)
    assert slot.status == "idle_unloaded"
    assert _slot_serviceable(slot), "routing must still send the next request here"
    assert slot.pipe is None
    assert "reload failed" in slot.info["reason"] and "USM Host" in slot.info["reason"]


def test_next_request_retries_and_clears_reason():
    slot = _unloaded_slot()
    slot.load = _failing_load(USM).__get__(slot)
    try:
        slot.ensure_loaded()
    except RuntimeError:
        pass
    calls = []

    def ok_load(self, model_dir):
        calls.append(model_dir)
        self.status = "loading"
        self.pipe = object()

    def ok_warmup(self):
        self.status = "ready"

    slot.load = ok_load.__get__(slot)
    slot.warmup = ok_warmup.__get__(slot)
    slot.ensure_loaded()
    assert calls == ["fake-model"]
    assert slot.status == "ready" and slot.error_reason is None


def test_warmup_failure_on_reload_counts_as_failure():
    slot = _unloaded_slot()

    def ok_load(self, model_dir):
        self.status = "loading"
        self.pipe = object()

    def bad_warmup(self):
        # DeviceSlot.warmup swallows its exception and sets these two
        self.status = "error"
        self.error_reason = "warmup failed: boom"

    slot.load = ok_load.__get__(slot)
    slot.warmup = bad_warmup.__get__(slot)
    try:
        slot.ensure_loaded()
        assert False, "a failed warmup must not return as if ready"
    except RuntimeError as e:
        assert "boom" in str(e)
    assert slot.status == "idle_unloaded" and _slot_serviceable(slot)
    assert "boom" in slot.error_reason


def test_poisoned_reload_is_not_retried():
    slot = _unloaded_slot()
    slot.load = _failing_load(POISON).__get__(slot)
    try:
        slot.ensure_loaded()
    except RuntimeError:
        pass
    assert slot.status == "error"
    assert not _slot_serviceable(slot)
    assert "poisoned" in slot.error_reason


def test_load_progress_ticker_stops_on_abort():
    """The ticker keys on still_loading(); abort() must end it even while the
    status still reads 'loading', which is the state a raising load leaves."""
    saved = nollama.LOAD_HEARTBEAT_SCHEDULE
    nollama.LOAD_HEARTBEAT_SCHEDULE = [0.05]
    try:
        before = threading.active_count()
        p = _LoadProgress("GPU", "fake-model", still_loading=lambda: True)
        p.start()
        time.sleep(0.12)
        p.abort()
        time.sleep(0.2)
        assert threading.active_count() <= before
    finally:
        nollama.LOAD_HEARTBEAT_SCHEDULE = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
