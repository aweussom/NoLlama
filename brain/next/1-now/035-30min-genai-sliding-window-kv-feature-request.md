# N-035 — Ask genai for a sliding-window KV cache type?

**Cost:** 30 min · **Asks:** Tommy

## What

genai's continuous-batching pool keeps full-length KV for sliding-attention
layers. Glimmer has 39 of those against 13 full-attention layers, so a
64k-token context costs 3.25 GiB in our pool where llama.cpp spends 949 MiB
(`docs/dev/prefix-cache.md`, "Most of the gap to Ollama is sliding-window KV").
The CB cache manager already anticipates this: `cache_type.hpp` lists
`SLIDING_WINDOW` as a type that "can be added".

On a 24 GB B60 this is the difference between ~40k tokens of context at a
2 GB pool and more than 64k in half of it. It is also the thing that pushed
Glimmer over the edge (T-031).

## The decision

- **a)** File a feature request on openvino.genai with the llama.cpp numbers
  and the B60 measurement. The same offer as #37501: we have the hardware
  and can test a branch in minutes.
- **b)** Wait. T-031 already makes the B60 usable at 2 GB.

Session's lean: **a**. It is cheap, the numbers already exist, and it
affects every Gemma-family model, not only ours. Check first whether
genai#4146 (Gemma 4 E4B degrades past ~5–15k tokens) is related, because a
sliding-window mask problem would show up exactly like that. Do not claim it
is related without checking.

## Saying yes means

Draft the issue for review; nothing is posted without a look.
