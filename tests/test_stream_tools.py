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

    def stream_tokens(self, raw_messages, gen, heartbeat, tag="", cancel=None, prompt=None):
        self.last_cancel = cancel  # the consumer's own token; its finally sets THIS one
        self.last_prompt = prompt  # signature tracks DeviceSlot.stream_tokens
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

def run_tool_stream(tokens, tools=TOOLS, error=None, vlm=None, native_prompt=None,
                    preseeded=False):
    slot = FakeSlot(tokens, error, preseeded=preseeded)
    frames = list(_sse_tool_stream(slot, [], None, tools, "id", 0, 0.0, vlm=vlm,
                                   native_prompt=native_prompt))
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


def test_lfm2_pythonic_tool_call_is_parsed():
    raw = ("<|tool_call_start|>[bash(command='ls -la', workdir='C:\\devel')]"
           "<|tool_call_end|>Checking the directory.")
    content, calls = nollama.parse_tool_calls(raw, TOOLS)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"])["command"] == "ls -la"
    assert content == "Checking the directory."


def test_lfm2_call_with_no_arguments():
    _, calls = nollama.parse_tool_calls("<|tool_call_start|>[list_files()]<|tool_call_end|>", TOOLS)
    assert len(calls) == 1 and calls[0]["function"]["arguments"] == "{}"


def test_pythonic_parser_never_executes_anything():
    # A tool-call parser that eval()s model output is an RCE hole. The call
    # parses as a tree; its non-literal argument must simply be dropped.
    _, calls = nollama.parse_tool_calls(
        "<|tool_call_start|>[bash(command=open('/etc/passwd').read())]<|tool_call_end|>", TOOLS)
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {}


def test_unparsed_tool_markup_never_reaches_the_user():
    # A real session rendered bare <tool_call> markers into the assistant's
    # reply (2026-09-23, Qwen3-14B). The prose must survive, the markup not.
    assert nollama._strip_tool_markup("<tool_call>\n\n<tool_call>\n\nI will read it.") == "I will read it."
    assert nollama._strip_tool_markup("<tool_call>") == ""
    assert nollama._strip_tool_markup("a plain answer") == "a plain answer"


def test_tool_turn_vlm_path_uses_vlm_seam():
    (deltas, finish), slot = run_tool_stream(["hi ", CALL], vlm=("prompt", []))
    assert finish == "tool_calls" and joined(deltas, "content") == "hi "
    assert slot.last_raw_prompt is False


def test_tool_turn_vlm_path_threads_raw_prompt():
    (_, finish), slot = run_tool_stream(["hi ", CALL], vlm=("prompt", [], True))
    assert finish == "tool_calls"
    assert slot.last_raw_prompt is True


def test_native_prompt_reaches_the_seam():
    (_, finish), slot = run_tool_stream(["hi ", CALL], native_prompt="<|im_start|>x")
    assert finish == "tool_calls"
    assert slot.last_prompt == "<|im_start|>x"


def test_native_prompt_decides_preseeding_not_the_slot():
    # The slot's template preseeds <think>; this render does not, so the
    # answer must stay content rather than be swallowed as reasoning.
    (deltas, _), _ = run_tool_stream(["plain answer"], native_prompt="...assistant\n",
                                     preseeded=True)
    assert joined(deltas, "content") == "plain answer"
    (deltas, _), _ = run_tool_stream(["mulling</think>answer"],
                                     native_prompt="...assistant\n<think>\n")
    assert joined(deltas, "reasoning_content") == "mulling"
    assert joined(deltas, "content") == "answer"


# A cut-down LFM2.5 template: Pythonic call history, a native tool role, the
# HF-only {% generation %} tag, and an enable_thinking switch.
MINI_TEMPLATE = (
    "{{ bos_token }}{% if tools %}<|im_start|>system\nList of tools: {{ tools | tojson }}"
    "<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' %}{% generation %}{{ m.content }}"
    "{% for tc in m.tool_calls or [] %}<|tool_call_start|>[{{ tc.function.name }}("
    "{% for k, v in tc.function.arguments.items() %}{{ k }}={{ v | tojson }}{% endfor %}"
    ")]<|tool_call_end|>{% endfor %}{% endgeneration %}"
    "{% else %}{{ m.content }}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if not enable_thinking %}<think></think>{% endif %}{% endif %}")


def _template_dir(tmp, template=MINI_TEMPLATE):
    """A model dir holding only a chat template and a bos token."""
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(template)
    with open(os.path.join(tmp, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"bos_token": {"content": "<|startoftext|>"}}, f)
    nollama._NATIVE_TEMPLATES.pop(tmp, None)
    return tmp


def test_native_render_speaks_the_models_dialect():
    import tempfile
    d = _template_dir(os.path.join(tempfile.mkdtemp(), "m"))
    msgs = [{"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
    out = nollama.render_native_tool_prompt(d, msgs, TOOLS)
    assert out.startswith("<|startoftext|><|im_start|>system\nList of tools: [")
    assert "<|im_start|>user\nweather?<|im_end|>" in out            # parts flattened
    assert '[get_weather(city="Oslo")]' in out                      # args decoded once
    assert "<|im_start|>tool\nsunny<|im_end|>" in out               # native tool role
    assert out.endswith("<|im_start|>assistant\n")                  # thinking stays on


def test_native_render_honours_no_think_and_falls_back():
    import tempfile
    d = _template_dir(os.path.join(tempfile.mkdtemp(), "m"))
    msgs = [{"role": "system", "content": nollama.NO_THINK_MARKER},
            {"role": "user", "content": "hi"}]
    assert nollama.render_native_tool_prompt(d, msgs, TOOLS).endswith("<think></think>")
    empty = tempfile.mkdtemp()                                      # no template at all
    assert nollama.render_native_tool_prompt(empty, msgs, TOOLS) is None
    broken = _template_dir(os.path.join(tempfile.mkdtemp(), "b"),
                           "{{ raise_exception('no tools here') }}")
    assert nollama.render_native_tool_prompt(broken, msgs, TOOLS) is None


def test_debug_logs_the_raw_turn_including_a_call_hidden_in_think():
    # A call inside a CLOSED <think> is reasoning about a call, not a call:
    # it stays unexecuted (finish=stop). --debug must still show the text.
    import contextlib
    import io
    buf = io.StringIO()
    nollama.debug = True
    try:
        with contextlib.redirect_stdout(buf):
            (deltas, finish), _ = run_tool_stream(["<think>", "plan\n", CALL, "</think>"])
    finally:
        nollama.debug = False
    out = buf.getvalue()
    assert finish == "stop"                                  # no call reached the client
    assert "NO call parsed" in out
    assert "  | <function=get_weather>" in out               # ...but the log shows it
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_tool_stream(["hi ", CALL])
    assert "raw tool-turn output" not in buf.getvalue()      # silent without --debug


def test_debug_logging_survives_a_stdout_that_cannot_encode_the_model():
    # The B60 case: a scheduled task's stdout is cp1252, the model wrote '→',
    # and the debug print killed every turn at its last token.
    import contextlib
    import io
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    nollama.debug = True
    try:
        with contextlib.redirect_stdout(out):
            (deltas, finish), _ = run_tool_stream(["step → next ", CALL])
            out.flush()
    finally:
        nollama.debug = False
    assert finish == "tool_calls"                              # the turn survived
    assert "\\u2192" in raw.getvalue().decode("cp1252")         # ...and was logged, escaped


def test_calls_inside_an_unclosed_think_are_recovered():
    # Qwen3-14B on the B60: calls written into a <think> it never closed.
    # They streamed out as reasoning and the turn ended with nothing.
    (deltas, finish), _ = run_tool_stream(["<think>", "I'll edit calc.py.\n", CALL])
    assert finish == "tool_calls"
    calls = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    assert joined(deltas, "content") == ""          # the reasoning is not re-sent as prose


def test_answer_inside_an_unclosed_think_is_sent_as_content():
    (deltas, finish), _ = run_tool_stream(["<think>", "The tests pass now. Done."])
    assert finish == "stop"
    assert joined(deltas, "content") == "The tests pass now. Done."
    # ...but not when a real answer already went out after a closed block.
    (deltas, _), _ = run_tool_stream(["<think>plan</think>", "Answer.", "<think>", "more"])
    assert joined(deltas, "content") == "Answer."


def test_assistant_message_rescues_an_unclosed_answer():
    msg, finish = _assistant_message("<think>\nAll green, nothing left to do.", [])
    assert finish == "stop" and msg["content"] == "All green, nothing left to do."
    msg, _ = _assistant_message("<think>plan</think>Real answer.", [])
    assert msg["content"] == "Real answer."           # closed blocks behave as before


def test_opencode_title_request_skips_thinking():
    # The first sentence of opencode 1.18.32's title system prompt, verbatim
    # from a captured request; the rest of that prompt varies with the task.
    title = ("You are a title generator. You output ONLY a thread title. Nothing else.\n\n"
             "<task>\nGenerate a brief title that would help the user find this conversation later.")
    assert nollama._no_think_requested([{"role": "system", "content": title},
                                        {"role": "user", "content": "say hi"}])
    # A user quoting it does not switch thinking off, nor does the real agent turn.
    assert not nollama._no_think_requested([{"role": "user", "content": title}])
    assert not nollama._no_think_requested([{"role": "system", "content": "You are opencode, ..."}])
    assert not nollama._no_think_requested([{"role": "system",
                                             "content": [{"type": "text", "text": title}]}])


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
