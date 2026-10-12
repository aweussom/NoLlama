"""The smallest RAG there is: embed five facts, answer from the nearest one.

    pip install openai
    python embed_rag.py ["When does the ferry leave?"]

Needs a server started with --embed-model-dir and a chat model. Vectors come
back L2-normalized, so the dot product is the cosine. Models whose card asks
for a `search_query: ` / `search_document: ` prefix expect the caller to add
it; the server adds nothing.
"""
import argparse

from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("question", nargs="?", default="When does the ferry leave?")
p.add_argument("--base-url", default="http://localhost:8000")
args = p.parse_args()

facts = [
    "The ferry to Lyngseidet leaves at 07:15 and 15:40 on weekdays.",
    "The shop in the village closes at 18:00, and at 15:00 on Saturdays.",
    "The road over the mountain is closed in winter from November to May.",
    "The doctor holds office hours on Tuesdays and Thursdays.",
    "Fishing licences are sold at the petrol station.",
]
client = OpenAI(base_url=args.base_url + "/v1", api_key="unused")
vectors = [d.embedding for d in client.embeddings.create(model="embed", input=facts + [args.question]).data]
*docs, query = vectors
scores = [sum(a * b for a, b in zip(query, d)) for d in docs]
best = facts[scores.index(max(scores))]
print(f"nearest fact ({max(scores):.2f}): {best}")

reply = client.chat.completions.create(model="local", messages=[
    {"role": "system", "content": f"Answer in one sentence, using only this fact: {best}"},
    {"role": "user", "content": args.question},
])
print(reply.choices[0].message.content.strip())
