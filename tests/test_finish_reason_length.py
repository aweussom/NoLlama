r"""T-048: a generation that exhausted max_tokens must report "length".

Before this, every finish_reason was "stop", so a turn cut off mid-write was
indistinguishable from one the model chose to end. Agent clients branch on
that field, and it also made truncation unmeasurable -- two runs of
scripts/decay-probe.py scored truncated answers as wrong ones.

    venv\Scripts\python -m pytest tests\test_finish_reason_length.py -q
    venv\Scripts\python tests\test_finish_reason_length.py       # no pytest needed
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nollama import _assistant_message, _hit_token_cap  # noqa: E402


def test_cap_hit_when_usage_reaches_it():
    assert _hit_token_cap({"completion_tokens": 64}, 0, 64) is True


def test_cap_hit_when_usage_exceeds_it():
    # The runtime occasionally reports one past the cap; >= not ==.
    assert _hit_token_cap({"completion_tokens": 65}, 0, 64) is True


def test_no_cap_when_model_stopped_early():
    assert _hit_token_cap({"completion_tokens": 12}, 0, 64) is False


def test_unknown_cap_is_never_length():
    assert _hit_token_cap({"completion_tokens": 999}, 0, None) is False
    assert _hit_token_cap({"completion_tokens": 999}, 0, 0) is False


def test_npu_sentinel_usage_falls_back_to_streamed_count():
    # usage_block sends -1 when the runtime reports nothing; -1 >= 64 would
    # be False anyway, but the fallback must be the frames actually sent.
    assert _hit_token_cap({"completion_tokens": -1}, 64, 64) is True
    assert _hit_token_cap({"completion_tokens": -1}, 3, 64) is False


def test_missing_usage_falls_back_to_streamed_count():
    assert _hit_token_cap({}, 64, 64) is True
    assert _hit_token_cap(None, 64, 64) is True


def test_zero_tokens_is_not_a_cap_hit():
    # An empty generation against max_tokens=0 must not read as truncation.
    assert _hit_token_cap({"completion_tokens": 0}, 0, 64) is False


def test_assistant_message_reports_length():
    _, finish = _assistant_message("some truncated text", [], False, True)
    assert finish == "length"


def test_assistant_message_reports_stop_without_cap():
    _, finish = _assistant_message("a complete answer", [], False, False)
    assert finish == "stop"


def test_tool_calls_win_over_cap():
    # Deliberate: parse_tool_calls produced structured calls, so the turn is
    # actionable and the clients in docs/dev/tool-calling.md branch on this.
    calls = [{"id": "c1", "type": "function",
              "function": {"name": "f", "arguments": "{}"}}]
    _, finish = _assistant_message("", calls, False, True)
    assert finish == "tool_calls"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all passed")
