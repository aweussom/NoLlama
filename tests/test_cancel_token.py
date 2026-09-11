"""The cancel token is per request: one request's cleanup must not stop the next.

Runs without a model. A FakePipe stands in for openvino_genai and blocks in
"prefill" on an Event, so a test can hold request B inside the lock while
request A's consumer finally runs — the ordering that, with one shared
slot flag, cancelled B at its first token (issue #40: OpenCode fires a
title request beside every turn, so the two always overlap).

    venv\\Scripts\\python -m pytest tests\\test_cancel_token.py -q
    venv\\Scripts\\python tests\\test_cancel_token.py          # no pytest needed
"""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nollama  # noqa: E402  (imports the module; no server starts)
from nollama import DeviceSlot  # noqa: E402


class FakePipe:
    """generate() waits on `release` (the 'prefill'), then streams TOKENS until
    the streamer callback asks it to stop. `entered` says the lock is held."""

    TOKENS = ["Hel", "lo", "!"]

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = 0

    def generate(self, history, gen, streamer):
        self.calls += 1
        self.entered.set()
        self.release.wait(timeout=5)
        for t in self.TOKENS:
            if streamer(t):
                return
            time.sleep(0.005)


def _slot(pipe):
    slot = DeviceSlot.__new__(DeviceSlot)
    slot.lock = threading.Lock()
    slot._cancel = threading.Event()
    slot._stream_error = None
    slot.device_name = "GPU"
    slot.model_name = "fake"
    slot.pipe = pipe
    slot.last_used = 0
    slot.last_ttft_ms = None
    slot.think_preseeded = False
    return slot


MSGS = [{"role": "user", "content": "hi"}]


def _collect(frames):
    """SSE frames -> (content, finish_reason)."""
    content, finish = "", None
    for f in frames:
        body = f[6:].strip()
        if body == "[DONE]":
            continue
        ch = json.loads(body)["choices"][0]
        content += ch["delta"].get("content") or ""
        if ch["finish_reason"]:
            finish = ch["finish_reason"]
    return content, finish


def _drain_in_thread(gen):
    out = []

    def run():
        for item in gen:
            out.append(item)

    th = threading.Thread(target=run)
    th.start()
    return th, out


def _hold_b_in_prefill(slot, pipe, consumer):
    """Start request B through `consumer` and return once it holds the lock."""
    pipe.release.clear()
    pipe.entered.clear()
    th, out = _drain_in_thread(consumer())
    assert pipe.entered.wait(2), "B never reached generate"
    return th, out


def test_issue_40_previous_requests_finally_does_not_cancel_the_next():
    """The exact ordering from the OpenCode logs: A has streamed [DONE] but its
    generator is suspended (finally pending); B takes the lock and prefills;
    A's finally fires; B must still stream to completion with finish=stop."""
    pipe = FakePipe()
    slot = _slot(pipe)
    pipe.release.set()

    ga = slot.stream_llm(MSGS, None, "a", 0, time.perf_counter())
    frames_a = []
    for f in ga:
        frames_a.append(f)
        if f == "data: [DONE]\n\n":
            break  # A's worker is done; A's finally has NOT run yet
    assert _collect(frames_a) == ("Hello!", "stop")

    th, frames_b = _hold_b_in_prefill(
        slot, pipe, lambda: slot.stream_llm(MSGS, None, "b", 0, time.perf_counter()))

    ga.close()  # A's finally lands while B holds the lock — the stale set()
    assert slot._cancel.is_set() is False, "A's cleanup reached B's token"

    pipe.release.set()
    th.join(5)
    assert _collect(frames_b) == ("Hello!", "stop")


def test_own_finally_cancels_own_generation():
    pipe = FakePipe()
    slot = _slot(pipe)
    b = threading.Event()
    th, got = _hold_b_in_prefill(
        slot, pipe, lambda: slot.stream_tokens(MSGS, None, heartbeat=None, cancel=b))
    b.set()  # the consumer's finally: client disconnected during prefill
    pipe.release.set()
    th.join(5)
    assert [t for t in got if t] == []
    assert pipe.calls == 1


def test_queued_request_whose_client_left_never_generates():
    pipe = FakePipe()
    slot = _slot(pipe)
    dead = threading.Event()
    dead.set()
    got = list(slot.stream_tokens(MSGS, None, heartbeat=None, cancel=dead))
    assert [t for t in got if t] == []
    assert pipe.calls == 0, "prefilled for a client that had already gone"


def test_slot_cancel_reaches_the_active_request():
    pipe = FakePipe()
    slot = _slot(pipe)
    b = threading.Event()
    th, got = _hold_b_in_prefill(
        slot, pipe, lambda: slot.stream_tokens(MSGS, None, heartbeat=None, cancel=b))
    assert slot._cancel is b
    slot.cancel()  # /v1/cancel
    pipe.release.set()
    th.join(5)
    assert [t for t in got if t] == []
    assert b.is_set()


def test_sse_reports_cancelled_only_for_its_own_token():
    pipe = FakePipe()
    slot = _slot(pipe)
    th, frames = _hold_b_in_prefill(
        slot, pipe, lambda: slot.stream_llm(MSGS, None, "b", 0, time.perf_counter()))
    slot.cancel()
    pipe.release.set()
    th.join(5)
    assert _collect(frames) == ("", "cancelled")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all passed")
