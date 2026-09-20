# T-032 — Bonsai 2 with its dspark drafter, to pair with Ollama's MTP row

**Effort:** 1 h · **Produces:** one more row in `docs/BENCHMARKS.md` ·
**Area:** benchmarks (not NoLlama code)

The Bonsai 2 vs Qwen3.8 comparison (2026-09-18) has a one-sided speculative
pairing: Ollama decodes Qwen3.8 with its multi-token-prediction drafter by
default (95 tok/s free text, 201 on a predictable count), while Bonsai ran
without its dspark drafter (131 tok/s). The like-for-like row today is
Ollama with `draft_num_predict=0` (77 tok/s) against Bonsai plain. The
"both speculating" pair is missing.

- [ ] on the 285K: convert the published bf16 dspark drafter once
      (`gguf-dspark-to-dflash`, see SPECULATIVE.md in the Bonsai-demo fork at
      `aweussom/Bonsai-demo`), then `BONSAI_SPECULATIVE=1` on the launcher
- [ ] `bench/bonsai-bench.py --url http://127.0.0.1:8080 --model bonsai
      --nothink template_kwargs --skip-probes --label bonsai2-cuda-dspark`
- [ ] add the row next to the two Ollama rows; note that dspark disables the
      prompt cache and forces one slot

## Done when

`docs/BENCHMARKS.md` shows Bonsai and Qwen3.8 each with and without their
drafter on the same card.
