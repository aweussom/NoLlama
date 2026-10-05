# N-038 — Kolibri-1 with Norwegian instead of German, on the Mac Studio?

**Cost:** 1 d for the first step, open-ended after · **Asks:** Tommy ·
**Blocked on:** the 128 GB Mac Studio arriving

## What

Aleph Alpha's Kolibri-1 (released 2026-10-03, Apache-2.0): 78.1B total,
3.46B active, 384 experts (6 routed + 1 shared), 50 layers, sliding window
of 513 tokens on 4 of every 5 layers, 256k context, Hermes-style tool calls,
reasoning mode. German and English only.
<https://huggingface.co/Aleph-Alpha/Kolibri-1>

No OpenVINO path: `kolibri1` exists only in Aleph Alpha's vLLM plugin, with no
transformers modeling code, so optimum-intel cannot export it. llama.cpp
support is an out-of-tree fork. MLX has it (`Sawfwair/Kolibri-1-MLX-8bit`,
mlx-vlm PR #2424). That makes a 128 GB Mac the one box we will have that runs
it without writing a model port first.

Idea: swap the German for Norwegian. Gets a Norwegian-capable agent model with
a tiny KV cache, and feeds T-011 (Norwegian translation) from a different
direction.

## What is realistic, in order

1. **Measure its Norwegian as-is.** [INFERRED] German is the nearest big
   language to Norwegian in its mix, so it may be better than "none". Run
   the T-011 test texts through the 8-bit MLX build. A day, and it decides
   whether the rest is worth it.
2. **LoRA on Norwegian text with MLX.** Plausible on 128 GB for the attention
   and shared expert. Training all 384 routed experts is not.
3. **A real tokenizer swap and continued pretraining.** [INFERRED] The 128k
   vocab is built for German morphology. Replacing it means re-initialising
   embeddings and billions of tokens of continued pretraining, which is a
   cluster job, not a Mac Studio job. Listed so nobody plans it by accident.

The architectural lesson stands whatever happens here: a 513-token window on
80 % of the layers is why 256k context is cheap. Worth knowing when we pick
agent models for small iGPU budgets.

## The decision

When the Mac arrives: run step 1, or drop this. Nothing before then.
