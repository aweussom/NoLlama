# Phi-3.5-vision repro — run this on the B60 box

`OpenVINO/Phi-3.5-vision-instruct-int4-ov` answers text prompts but fails on
**any** image with:

```
Check '(prompt_id >= 0) && (prompt_id < vocab_size)' failed at
.../sampling/logit_transformers.hpp:412: input_ids token out of bounds
```

**Cause found 2026-09-01: NoLlama's default `repetition_penalty` of 1.05.**
Same model, same image, greedy, differing only in that one value -- 1.0
answers correctly, 1.05 asserts. `presence_penalty` and `frequency_penalty`
are harmless, because only the repetition-penalty transformer walks the
*prompt* ids, and Phi-3 vision places image placeholders outside
`[0, vocab_size)`.

What was ruled out first, all of it true and all of it beside the point:

| Ruled out | How |
|---|---|
| Multi-image only | one image fails identically |
| The CB / prefix-caching path | `--no-prompt-cache` gives the same assertion |
| Runtime version | genai 2026.3.0.0-3277 **and** 2026.5.0.0-3402 nightly |
| Hardware / OS install | Arc 140T (community, #24) and Arc 140V (here) |
| A broken download | loads clean, 32 fused SDPA ops, text works |
| **The model or the images** | a bare `VLMPipeline` reads a real 807x552 screenshot correctly, and synthetic squares from 336 to 2048 px |

The lesson worth keeping: every one of those tests ran *through NoLlama*, so
the one variable never varied was NoLlama. This script exists to force the
comparison the other way round.

Still untested: a **discrete** GPU. Every observation above is an integrated
Xe part. Running this on the B60 box would close that gap.

## Run

```powershell
cd <repo>
.\venv\Scripts\python scripts\phi35v-repro\probe.py
```

It downloads the model if absent (2.2 GB), starts nothing — it drives
`VLMPipeline` directly, so NoLlama is not in the picture and cannot be
blamed. Add `-Nightly` handling by running it with `venv-nightly` instead.

Paste the whole output into the issue.
