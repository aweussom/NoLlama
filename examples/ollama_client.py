"""The same chat through the ollama package, on NoLlama's Ollama port.

    pip install ollama
    python ollama_client.py [--host http://localhost:11434]

Anything written for Ollama runs unchanged. The model name must be one that
/api/tags lists, so this asks first instead of hard-coding it.
"""
import argparse

import ollama

p = argparse.ArgumentParser()
p.add_argument("--host", default="http://localhost:11434")
args = p.parse_args()

client = ollama.Client(host=args.host)
name = client.list().models[0].model
print("model:", name)

for part in client.chat(model=name, stream=True, messages=[
        {"role": "user", "content": "What is the capital of Norway? One word."}]):
    print(part.message.content, end="", flush=True)
print()
