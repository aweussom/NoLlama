"""One tool-call round trip: the model asks for a function, we run it, it answers.

    pip install openai
    python tools.py [--base-url http://localhost:8000]

Tool calling works on GPU and CPU slots, never on the NPU. Small models
(under ~14B) call tools in a one-shot like this but fall apart in long agent
loops; see docs/AGENTS.md before pointing a coding agent at one.
"""
import argparse
import json
from datetime import datetime

from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://localhost:8000")
args = p.parse_args()

tools = [{"type": "function", "function": {
    "name": "get_time",
    "description": "Current local time as HH:MM.",
    "parameters": {"type": "object", "properties": {}},
}}]
messages = [{"role": "user", "content": "What time is it? Use the tool."}]
client = OpenAI(base_url=args.base_url + "/v1", api_key="unused")

msg = client.chat.completions.create(model="local", messages=messages, tools=tools).choices[0].message
if not msg.tool_calls:
    raise SystemExit(f"no tool call; the model said: {msg.content!r}")
call = msg.tool_calls[0]
print(f"model called {call.function.name}({call.function.arguments})")

messages += [msg.model_dump(exclude_none=True),
             {"role": "tool", "tool_call_id": call.id, "content": datetime.now().strftime("%H:%M")}]
final = client.chat.completions.create(model="local", messages=messages, tools=tools)
print(final.choices[0].message.content.strip())
