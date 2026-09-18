# T-018 — Redo the Ollama head-to-head, now that temperature is pinned

**Effort:** 1 h · **Produces:** corrected figures in `docs/BENCHMARKS.md` ·
**Area:** benchmarks

The old comparison had **Ollama sampling at its default 0.8 against NoLlama
greedy at 0.0**, because `benchmark.py` sent no temperature at all. That is
fixed now, and the comparison has not been re-run.

The 1.6x decode figure probably survives — it is a throughput measurement and
both sides decode the same weights. **The task-time reading of it does not**,
because Ollama's build ignores `/no_think` and spends ~1755 tokens on a
291-character answer where NoLlama spends 293. Any "N times faster at the job"
claim built on that is measuring the thinking budget, not the runtime.

- [ ] Ollama on the 140V, same model, same prompts, temperature pinned on both
- [ ] separate the decode number from the task-time number in the write-up, and
      say which one the `/no_think` difference contaminates

Related and already correct: the llama.cpp OpenVINO-backend comparison added to
`docs/BENCHMARKS.md` (measured 2026-09-07, B60) pins greedy on both sides.

## Done when

`docs/BENCHMARKS.md` quotes an Ollama comparison where the only variable is the
runtime.
