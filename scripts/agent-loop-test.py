"""Two-turn agent loop against a running NoLlama — the round trip a real client makes.

Why this exists separately from tests/test_stream_tools.py: that suite drives
the SSE consumers with a fake token seam, so it proves the framing but never
touches a model or the history rewrite. An agent client does something the
single-turn tests cannot cover — it streams a turn that ends in `tool_calls`,
then sends the conversation BACK with a tool-call-only assistant message
(`content: null`) plus a `tool` role message, and every later turn re-renders
that history through prepare_messages_for_tools. Null content there is what
broke Zed (issue #24); reasoning arriving as `reasoning_content` is what
OpenCode needs (issue #36). This script exercises both against real weights.

Run it against a server already serving a GPU/CPU LLM (tools are refused on
the NPU by design):

    python scripts/agent-loop-test.py http://127.0.0.1:8000

Exit code 0 means the second turn used the tool result; 1 means it did not.
"""
import json
import sys
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]


def first_model():
    """Name of the first model the server advertises.

    Why: the tool loop has to address a specific model id, and it differs per
    machine (it is the model directory's name plus @DEVICE). In: nothing.
    Out: the id string; raises if the server is not up.
    """
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def stream_turn(model, messages, tools=None):
    """Stream one chat turn and collect the pieces an agent client reads.

    Why the split return: the point of the test is that reasoning arrives on
    `delta.reasoning_content` (not folded into content) and that a tool turn
    still ends with structured `tool_calls` — collecting them separately is
    what makes a regression visible.

    In: model id, OpenAI-shaped messages, optional tools. Out: tuple of
    (reasoning text, content text, list of tool_call deltas, finish_reason);
    finish_reason is None only if the stream ended without a final frame.
    """
    body = {"model": model, "messages": messages, "stream": True,
            "max_tokens": 400, "temperature": 0}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    reasoning, content, calls, finish = "", "", [], None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choice = json.loads(line[6:])["choices"][0]
            delta = choice["delta"]
            reasoning += delta.get("reasoning_content") or ""
            content += delta.get("content") or ""
            calls.extend(delta.get("tool_calls") or [])
            finish = choice["finish_reason"] or finish
    return reasoning, content, calls, finish


def main():
    """Run the loop and report; exit 1 if the model ignored the tool result."""
    model = first_model()
    print(f"model: {model}")

    messages = [{"role": "user",
                 "content": "What is the weather in Oslo? Use the tool."}]
    reasoning, content, calls, finish = stream_turn(model, messages, TOOLS)
    print(f"turn 1: reasoning={len(reasoning)}ch content={len(content)}ch "
          f"finish={finish} calls="
          f"{[(c['function']['name'], c['function']['arguments']) for c in calls]}")
    if finish != "tool_calls" or not calls:
        print("FAIL: turn 1 produced no tool call")
        return 1

    # Exactly what an agent client sends back: the assistant's tool-call-only
    # turn (content None, per the OpenAI spec) and the tool result.
    messages.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": calls[0]["id"], "type": "function",
                                     "function": calls[0]["function"]}]})
    messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                     "name": "get_weather",
                     "content": json.dumps({"city": "Oslo", "temp_c": 7,
                                            "sky": "rain"})})
    reasoning, content, calls2, finish = stream_turn(model, messages, TOOLS)
    print(f"turn 2: reasoning={len(reasoning)}ch content={len(content)}ch "
          f"finish={finish} calls={len(calls2)}")
    print(f"  answer: {content.strip()[:200]!r}")

    used_result = "7" in content or "rain" in content.lower()
    if finish == "stop" and used_result:
        print("PASS: the second turn used the tool result")
        return 0
    print("FAIL: the second turn did not use the tool result")
    return 1


if __name__ == "__main__":
    sys.exit(main())
