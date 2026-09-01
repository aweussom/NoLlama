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
git pull
.\scripts\phi35v-reproun.ps1
```

That is the whole thing. It downloads the model if missing (2.2 GB), runs
`scriptsare-probe.py` under the release venv and the nightly venv if one
exists, and writes a timestamped `phi35v-report-*.txt` at the repo root to
paste into issue #24. It never starts NoLlama -- the probe drives
`openvino_genai` directly, so the server cannot be blamed for the result.

Options: `-Device CPU` to check a non-GPU path, `-ModelDir <path>` if the
model lives somewhere unusual.

**What to look for.** On an integrated Xe GPU every row is OK except
`repetition_penalty=1.05` + image. If the B60 matches that, the bug is
hardware-independent and the upstream report can say so. If it differs, that
is the first hardware-dependent result in this investigation and worth
saying loudly.
