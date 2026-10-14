# T-049 — `temperature > 0` is deterministic; the OpenAI `seed` is ignored

**Effort:** 1 h · **Produces:** a `seed` passthrough in `nollama.py` + a note in
`docs/API.md` · **Area:** `nollama.py` (`chat_completions`, the two Ollama
paths)

Sampling works — `do_sample=True` and `gen.temperature` are set whenever
temperature exceeds 0.01. But `openvino_genai.GenerationConfig` defaults
`rng_seed = 0` [OBSERVED 2026-10-14, genai 2026.4.1.0] and nothing ever
changes it, so the same prompt at the same temperature returns the same tokens
every single time.

Five calls at `temperature: 0.7`, same prompt, LFM2.5-24B-A2B on the B60:
byte-identical answers, `min == max == 5073` completion tokens. Five calls,
one sample.

`seed` is a documented field of the OpenAI Chat Completions request
[DOCUMENTED, OpenAI API reference]. NoLlama accepts it and ignores it, so a
client varying the seed to get different samples gets identical ones, with
nothing in the response saying why.

## Why it matters beyond tidiness

Determinism by default is a FEATURE for this project — it is why a benchmark
re-run reproduces, and `benchmark.py` quietly depends on it. The defect is
that it is undocumented and not overridable, which bites two ways:

- **Any sampling measurement is silently n=1.** A variance or
  best-of-N experiment looks like it ran and did not. It cost one here before
  `min == max` gave it away.
- **A client that asks for variety gets none.** Retry-with-a-different-seed
  is the standard escape when a model produces a bad answer, and it is a no-op
  against this server.

## Shape

- read `seed` from the OpenAI body and `options.seed` on the Ollama surface
- set `gen.rng_seed` when given; leave the default when not, so today's
  reproducibility is the unchanged default
- `docs/API.md`: say plainly that sampling is reproducible unless a seed is
  passed, because the surprise is worth one sentence

## Done when

Two calls at `temperature 0.7` with different `seed` values return different
text; two with the same seed return the same; omitting it keeps today's
behaviour. `docs/API.md` states the default.
