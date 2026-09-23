"""Streaming tool turns and reasoning_content: pure-Python tests over fake slots.

Runs without a model: a FakeSlot replays a scripted token sequence through
the real _sse_tool_stream / _sse_stream code, and the splitter/gate are
fuzzed over every chunking of their inputs. Run:

    venv\\Scripts\\python -m pytest tests\\test_stream_tools.py -q
    venv\\Scripts\\python tests\\test_stream_tools.py          # no pytest needed
"""
import json
import os
import random
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nollama  # noqa: E402  (imports the module; no server starts)

from nollama import (_ThinkSplitter, _ToolCallGate, _split_think,  # noqa: E402
                     _sse_tool_stream, _assistant_message)


# --- helpers ---------------------------------------------------------------

def chunkings(text, n=40, seed=1):
    """The whole text as one chunk, one char at a time, and n random splits."""
    yield [text]
    yield list(text)
    rnd = random.Random(seed)
    for _ in range(n):
        cuts = sorted(rnd.sample(range(1, len(text)), min(len(text) - 1, rnd.randint(1, 12))))
        yield [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]


def split_all(chunks):
    sp = _ThinkSplitter()
    pieces = []
    for c in chunks:
        pieces += sp.feed(c)
    pieces += sp.close()
    return ("".join(t for k, t in pieces if k == "reasoning"),
            "".join(t for k, t in pieces if k == "content"))


class FakeSlot:
    """Just enough of DeviceSlot for the SSE consumers: a scripted token seam."""

    def __init__(self, tokens, error=None, preseeded=False):
        self._tokens = tokens
        self._error = error
        self._cancel = threading.Event()
        self._stream_error = None
        self.model_name = "fake"
        self.device_name = "GPU"
        self.last_ttft_ms = None
        self.think_preseeded = preseeded

    def stream_tokens(self, raw_messages, gen, heartbeat, tag="", cancel=None):
        self.last_cancel = cancel  # the consumer's own token; its finally sets THIS one
        for t in self._tokens:
            yield t
        self._stream_error = self._error

    def stream_vlm_tokens(self, text_prompt, images, gen, heartbeat, tag="", cancel=None,
                          raw_prompt=False):
        # Signature tracks DeviceSlot.stream_vlm_tokens; a kwarg added there
        # and not here fails every VLM case with a TypeError that reads like
        # a test bug rather than drift, which is how raw_prompt sat broken.
        self.last_raw_prompt = raw_prompt
        yield from self.stream_tokens(None, gen, heartbeat, tag, cancel)

    def preseeded_for(self, raw_prompt):
        # Twin of DeviceSlot.preseeded_for: a pre-rendered prompt closes its
        # own think block, so the splitter must not start inside one.
        return self.think_preseeded and not raw_prompt


def collect(frames):
    """Parse SSE frames -> (deltas, finish_reason)."""
    deltas, finish = [], None
    for f in frames:
        assert f.startswith("data: ") and f.endswith("\n\n"), f
        body = f[6:].strip()
        if body == "[DONE]":
            continue
        ch = json.loads(body)["choices"][0]
        deltas.append(ch["delta"])
        if ch["finish_reason"]:
            finish = ch["finish_reason"]
    return deltas, finish


def joined(deltas, key):
    return "".join(d.get(key) or "" for d in deltas)


TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]

CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"


# --- _ThinkSplitter / _split_think -------------------------------------------

def test_splitter_basic_every_chunking():
    text = "<think>\nlet me see\n</think>\n\nHello!"
    for chunks in chunkings(text):
        assert split_all(chunks) == ("let me see\n", "Hello!"), chunks


def test_splitter_empty_block_is_dropped():
    for chunks in chunkings("<think>\n\n</think>\n\n4"):
        assert split_all(chunks) == ("", "4"), chunks


def test_splitter_no_tags_passthrough():
    for chunks in chunkings("plain answer with <b>html</b> and < less-than"):
        assert split_all(chunks) == ("", "plain answer with <b>html</b> and < less-than")


def test_splitter_unclosed_block_flushes_as_reasoning():
    assert split_all(["<think>still thinking"]) == ("still thinking", "")


def test_splitter_legacy_flag_keeps_tags_in_content():
    nollama.THINK_IN_CONTENT = True
    try:
        assert _split_think("<think>x</think>\ny") == ("", "<think>x</think>\ny")
    finally:
        nollama.THINK_IN_CONTENT = False


# --- _ToolCallGate -----------------------------------------------------------

def test_gate_streams_prose_then_holds_call_every_chunking():
    text = "Let me check the weather.\n" + CALL
    for chunks in chunkings(text):
        g = _ToolCallGate()
        out = "".join(g.feed(c) for c in chunks)
        assert out + g.held == text, chunks
        assert out == "Let me check the weather.\n", (out, chunks)
        assert g.held == CALL


def test_gate_bare_json_is_held_whole():
    g = _ToolCallGate()
    out = "".join(g.feed(c) for c in ['{"name": ', '"get_weather", "arguments": {}}'])
    assert out == "" and g.held.startswith("{")


def test_gate_prose_with_angle_brackets_passes():
    text = "use <b>bold</b> and a [link](x) here"
    for chunks in chunkings(text, n=10):
        g = _ToolCallGate()
        out = "".join(g.feed(c) for c in chunks)
        assert out + g.held == text and g.held == "", chunks


# --- _sse_tool_stream ----------------------------------------------------------

def run_tool_stream(tokens, tools=TOOLS, error=None, vlm=None):
    slot = FakeSlot(tokens, error)
    frames = list(_sse_tool_stream(slot, [], None, tools, "id", 0, 0.0, vlm=vlm))
    return collect(frames), slot


def test_tool_turn_streams_reasoning_and_prose_then_tool_calls():
    tokens = ["<think>", "user wants", " weather", "</think>", "\n\nChecking ",
              "Oslo.\n", "<tool_", "call>\n<function=get_weather>\n<parameter=city>\nOslo\n",
              "</parameter>\n</function>\n</tool_call>"]
    (deltas, finish), slot = run_tool_stream(tokens)
    assert finish == "tool_calls"
    assert joined(deltas, "reasoning_content") == "user wants weather"
    assert joined(deltas, "content") == "Checking Oslo.\n"
    tcs = [tc for d in deltas for tc in (d.get("tool_calls") or [])]
    assert len(tcs) == 1 and tcs[0]["function"]["name"] == "get_weather"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"city": "Oslo"}
    assert slot.last_cancel.is_set()  # consumer's safety net ran, on its own token
    assert slot.last_ttft_ms is not None


def test_tool_turn_keepalive_and_plain_answer():
    (deltas, finish), _ = run_tool_stream([None, None, "Just ", "an answer."])
    assert finish == "stop"
    assert joined(deltas, "content") == "Just an answer."
    assert sum(1 for d in deltas if d.get("content") == "") == 2  # two keep-alives
    assert not any(d.get("tool_calls") for d in deltas)


def test_tool_turn_false_alarm_opener_is_released():
    (deltas, finish), _ = run_tool_stream(["The tag <function=", "foo> is not a call"])
    assert finish == "stop"
    assert joined(deltas, "content") == "The tag <function=foo> is not a call"


def test_tool_turn_bare_json_fallback():
    (deltas, finish), _ = run_tool_stream(['{"name": "get_weather", ', '"arguments": {"city": "Oslo"}}'])
    assert finish == "tool_calls"
    assert joined(deltas, "content") == ""
    tcs = [tc for d in deltas for tc in (d.get("tool_calls") or [])]
    assert tcs[0]["function"]["name"] == "get_weather"


def test_tool_turn_error_frame():
    (deltas, finish), _ = run_tool_stream(["partial"], error=RuntimeError("boom"))
    assert finish == "error"
    assert "[error: " in joined(deltas, "content")


def test_tool_turn_vlm_path_uses_vlm_seam():
    (deltas, finish), slot = run_tool_stream(["hi ", CALL], vlm=("prompt", []))
    assert finish == "tool_calls" and joined(deltas, "content") == "hi "
    assert slot.last_raw_prompt is False


def test_tool_turn_vlm_path_threads_raw_prompt():
    (_, finish), slot = run_tool_stream(["hi ", CALL], vlm=("prompt", [], True))
    assert finish == "tool_calls"
    assert slot.last_raw_prompt is True


def test_tool_turn_legacy_flag_keeps_think_in_content():
    nollama.THINK_IN_CONTENT = True
    try:
        (deltas, finish), _ = run_tool_stream(["<think>r</think>\nanswer"])
    finally:
        nollama.THINK_IN_CONTENT = False
    assert joined(deltas, "content") == "<think>r</think>\nanswer"
    assert joined(deltas, "reasoning_content") == ""


# --- pre-seeded <think> (Qwen3.5/3.8 templates open the block in the prompt) ----

def test_splitter_preseeded_orphan_closer_every_chunking():
    text = "We need to answer 4.\n</think>\n\n4"
    for chunks in chunkings(text):
        sp = _ThinkSplitter(preseeded=True)
        pieces = []
        for c in chunks:
            pieces += sp.feed(c)
        pieces += sp.close()
        r = "".join(t for k, t in pieces if k == "reasoning")
        c = "".join(t for k, t in pieces if k == "content")
        assert (r, c) == ("We need to answer 4.\n", "4"), chunks


def test_split_think_preseeded_vs_not():
    assert _split_think("thinking</think>\n\nOslo", preseeded=True) == ("thinking", "Oslo")
    assert _split_think("thinking</think>\n\nOslo") == ("", "thinking</think>\n\nOslo")


def test_tool_stream_preseeded_reasoning_goes_to_reasoning_content():
    slot = FakeSlot(["We should call ", "the tool.\n</think>\n\n", CALL], preseeded=True)
    (deltas, finish) = collect(_sse_tool_stream(slot, [], None, TOOLS, "id", 0, 0.0))
    assert finish == "tool_calls"
    assert joined(deltas, "reasoning_content") == "We should call the tool.\n"
    assert joined(deltas, "content") == ""


def test_prompt_preseeds_think_detection():
    class Tok:
        def __init__(self, tail): self.tail = tail
        def apply_chat_template(self, msgs, add_generation_prompt=True, **kw): return "…assistant\n" + self.tail
    assert nollama._prompt_preseeds_think(Tok("<think>\n")) is True
    assert nollama._prompt_preseeds_think(Tok("")) is False
    assert nollama._prompt_preseeds_think(None) is False

    class Broken:
        def apply_chat_template(self, *a, **kw): raise RuntimeError("no template")
    assert nollama._prompt_preseeds_think(Broken()) is False


# --- _assistant_message (non-streaming twin) ----------------------------------

def test_assistant_message_splits_reasoning():
    msg, finish = _assistant_message("<think>why</think>\n\nOslo", [])
    assert (msg["content"], msg["reasoning_content"], finish) == ("Oslo", "why", "stop")
    msg, finish = _assistant_message("", [{"id": "x", "type": "function",
                                           "function": {"name": "f", "arguments": "{}"}}])
    assert msg["content"] is None and finish == "tool_calls" and "reasoning_content" not in msg


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                failures += 1
                print("FAIL", name, "-", e)
    sys.exit(1 if failures else 0)
