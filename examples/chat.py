"""One chat turn, then the same question streamed, through the openai package.

    pip install openai
    python chat.py [--base-url http://localhost:8000]

NoLlama serves whatever model it was started with; the `model` field is
required by the client library but the server answers with its own.
Thinking models (Qwen3) send their reasoning in `reasoning_content`, so
`content` is the answer alone.
"""
import argparse

from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://localhost:8000")
args = p.parse_args()

client = OpenAI(base_url=args.base_url + "/v1", api_key="unused")
question = [{"role": "user", "content": "What is the capital of Norway? One word."}]

reply = client.chat.completions.create(model="local", messages=question)
print("plain:   ", reply.choices[0].message.content.strip())

print("streamed: ", end="", flush=True)
for chunk in client.chat.completions.create(model="local", messages=question, stream=True):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
