# Prefix cache, KV pool, prewarm

Read this before touching caching, the KV pool, prewarm, `--idle-timeout`,
TTFT logging or the memory preflight.

## Prefix (KV) caching — default on

**Default on** for GPU/CPU **LLM and VLM** slots. They load via the
continuous-batching backend:

```python
LLMPipeline/VLMPipeline(..., scheduler_config=SchedulerConfig(
    enable_prefix_caching=True, cache_size=slot.kv_pool_gb))
```

A repeated prompt prefix (an agent's fixed system prompt + tool schemas,
identical every turn) is prefilled once, not every turn — measured **~47×
faster** on a cached turn (24.4s→0.5s for a ~2k-token prefix on the 285K
CPU). Auto-invalidated by any prefix change, so no staleness.
`--no-prompt-cache` disables it.

VLMPipeline honoring `scheduler_config` was verified 2026-08-18 on 2026.3
release (140V: ~9k-token prefix 21.7s→3.9s TTFT) and 2026.4 nightly (B60:
33k tokens 54.5s→1.3s) — the earlier **"CB backend is LLM-only" note was
stale**. Documented upstream gap: openvino.genai#4343.

NPU slots keep the plain pipeline (no CB path; NPU keeps MAX_PROMPT_LEN).
Any device or runtime that can't build the CB backend falls back to the
plain pipeline with a warning.

**It is not only devices and runtimes — an individual IR can refuse.** The
CB path is built by rewriting the graph (`SDPAToPagedAttention`), so a model
exported without an SDPA op cannot take it at all:

```
No ScaledDotProductAttention operation observed in the graph,
cannot perform the SDPAToPagedAttention transformation.
); using plain pipeline
```

Measured 2026-08-21 on the B60 / 2026.3 release across Intel's three Gemma 4
exports. The tell is static — count the ops in the language model's `.xml`:

| IR | SDPA ops | SoftMax ops | CB backend |
|---|---|---|---|
| `gemma-4-E2B-it-int4-ov` | 35 (= layers) | 0 | builds |
| `gemma-4-26b-a4b-it-int4-ov` | 30 (= layers) | 0 | builds |
| `gemma-4-E4B-it-int8-ov` | **0** | **42** | **refuses** |

**`--scan` reports this** (`Prefix caching : yes — N fused SDPA ops`, or a
`NO` with the reason), so the check no longer needs doing by hand. The
equivalent by hand:

```bash
grep -c 'ScaledDotProductAttention' openvino_language_model.xml   # want: > 0
```

A fused SDPA node means the rewrite has something to match; decomposed
matmul+softmax attention means it does not. E4B's *vision* tower has 32 SDPA
ops and is fine — only its language model is affected, which is why the
count must come from `openvino_language_model.xml` and not the tower.

**The predicate is `> 0`, not `== layers`.** An earlier version of this note
said "want: one per layer", which is true only for dense models and
condemns healthy hybrids [OBSERVED 2026-09-01, local IRs via `--scan`]:

| IR | layers | SDPA ops | why |
|---|---|---|---|
| `Qwen3-1.7B-int4-ov` | 28 | 28 | dense — one per layer |
| `Qwen2.5-VL-3B-int8-ov` | 36 | 36 | dense |
| `Qwen3.5-4B-int4-ov` | 32 | **8** | `full_attention_interval` 4; the other 24 layers are `linear_attention` and emit no SDPA node |
| `Muse-Glimmer-30B-int4-ov` | 52 | **52** | 39 `sliding_attention` + 13 `full_attention` — sliding attention is still attention and still fuses |
| `LFM2.5-1.2B-int4-cw-ov` | 16 | **6** | conv-heavy hybrid |

Those last three are all fine. Only zero is a defect. A warning keyed to
`layers` was written and then rejected for exactly this reason — it fired on
Glimmer, and a check that cries wolf on a good model is worse than no check.

**This is an export defect, not a model property.** Re-exporting the same
`google/gemma-4-E4B-it` weights to the same INT8 precision with a current
stack (optimum-intel 2.2.0.dev0, transformers 5.5.4, OpenVINO 2026.3) yields
**42 SDPA ops and a working CB backend** — same 7.8 GB, same 42-layer /
84 KB-per-token geometry, only the attention differs.

What we did *not* establish is which component caused Intel's build to
decompose. The transformers version recorded in the IRs (5.5.4 on the two
that work, 5.5.0 on the one that doesn't) is correlation, not cause:
`Gemma4ForConditionalGeneration._supports_sdpa` is `True` on both. The
likelier mechanism is the export **environment**, because optimum-intel only
pins the attention implementation for models listed in
`FORCE_ATTN_MODEL_CLASSES` (`phi3_v`, `gemma2`, `llama4`) and **`gemma4` is
not among them** — so whatever the environment resolves to is what gets
traced.

So a model can lose prefix caching for reasons invisible in its name, size
or precision. Check the load log rather than assuming; `/health`'s per-slot
`kv_pool_gb` is the other tell (null = fell back).

Cold-prefill trade-off, measured on the B60 with a 33k prompt: ~8.7s on the
plain pipeline vs 53.7s under CB (then 1.4s per repeat). **Agents win from
turn two; one-shot prompts pay more once.**

## KV pool sizing

**Auto-sized per slot** (`_resolve_kv_pool`), from what the weights leave
free in the device budget — `headroom = budget - weights*1.1`. Floored at
2 GB (`AUTO_KV_MIN_GB`), capped at ~64k tokens of the model's KV geometry
(`AUTO_KV_TOKENS`). Sized from the *total* budget, not free RAM, so it's
stable across restarts and reloads.

**The shape differs by device**, and that distinction is load-bearing:

| device | rule |
|---|---|
| GPU (discrete or iGPU) | larger of `headroom - 2 GB` (`AUTO_KV_RESERVE_GB`) and `headroom / 3` (`AUTO_KV_HEADROOM_SHARE`) |
| CPU | `headroom / 3` only |

A GPU budget is dedicated, or on an iGPU a carve-out the driver already
sized against the OS, so a fixed reserve is safe. **On CPU the budget is the
machine's entire RAM**, shared with the OS and everything else running —
there a 2 GB reserve is not a margin, it is a claim on the whole box.

**Why the reserve exists on GPU** (changed 2026-09-13): the flat third alone
under-spends a large budget badly. On the 140V — 25.3 GB budget, 15.2 GB of
weights, 8.6 GB free — it sized **2.87 GB**, truncated by `int()` to 2, which
looked like the floor had been hit when it had not. That is ~22k tokens
against the 60k context OpenCode declares, and a pool that cannot hold prompt
+ max_tokens lets a single request evict its own prefix mid-generation
(TODONT.md). The rule now rounds instead of truncating.

[OBSERVED 2026-09-13] Resolved values, all with the same runtime:

| box | device | model | pool |
|---|---|---|---|
| 140V laptop | iGPU, 25.3 GB budget | Qwen3-Coder-30B int4 | **6 GB** |
| B60 | dGPU, 24 GB | Qwen3-Coder-30B int4 | **5 GB** |
| B60 | CPU, 32 GB RAM | Phi-3.5-mini int4 | **10 GB** |

The laptop's 6 GB is independently the value the B60 rig had been pinned to by
hand for that model (`STATUS.md`, task `nollama-gpu-8000`); 6 GB is 65,536
tokens at the coder's 96 KB/token.

**Watch the cap on a model without GQA.** Phi-3.5-mini has `kv_heads ==
heads == 32`, so **384 KB/token** against the 30B coder's 96 KB — four times
the KV for an eighth of the weights. `AUTO_KV_TOKENS` alone then works out to
24 GB, and on the CPU path that was briefly what got asked for (the reserve
rule, before it was made GPU-only). A small model is not automatically a small
pool.

`--cache-size-gb N` pins it and skips auto.

## Prewarm

`--prewarm <file>` prefills a saved agent prompt at startup, so even the
first turn is a cache hit instead of a cold prefill that can trip a
client's idle watchdog. The file **auto-captures** the first big prompt
served (`_maybe_capture_prewarm`, on both the OpenAI and Ollama chat
paths) — so the workflow is: run once → restart with `--prewarm`.

- Auto-enabled as `prewarm-<port>.json` when `--idle-timeout 0` (opt out
  with `--no-prewarm`).
- `--prewarm` implies `--idle-timeout 0`. An explicit nonzero timeout
  alongside it is **REFUSED at startup** (2026-08-18; was a warning).
  Unload discards the warmed cache, and the reload path deliberately does
  **not** re-warm — a synchronous re-warm would stall the triggering
  request pre-SSE and trip exactly the client watchdogs the heartbeat
  exists to defeat.
- Covers **VLM slots** too (2026-08-18): capture on both API surfaces, and
  the startup prefill replays through `parse_messages`' flattening so the
  cached token prefix matches real requests. Measured, Glimmer/B60: first
  turn after restart 12.4s → 0.65s TTFT.
- **Only one slot is prewarmed** (2026-09-13), resolved from topology in
  `main()` as `PREWARM_SLOT`: the GPU slot when it holds an LLM, else the
  primary. In an agent setup the other slot serves **side requests** —
  OpenCode's `small_model`, which sends short titles and never the big system
  prompt the file holds. Warming it caches a prefix that can never hit *and*
  blocks that slot's queue while it runs. [OBSERVED 2026-09-13, B60 dual
  mode] Phi-3.5-mini on CPU spent **233.1s** prewarming the coder's 8k-token
  prompt, and a title request that arrived 5s after startup was answered
  **157s** later, having sat behind it. Resolved in `main()` rather than via
  `_route_request`, which consults `_slot_serviceable` — slots finish loading
  at very different times (B60: CPU 2s, GPU 44s), so asking mid-startup
  returns whichever is ready first and re-introduces the bug.
- Slots whose runtime fell back to the plain pipeline report a **null**
  `kv_pool_gb` at
  load, so prewarm skips them instead of burning a 30B-scale prefill for
  nothing (this also keeps `/health` honest about a dead cache).

## Observability

Per-request log lines include TTFT — streaming: wall-clock to first token;
non-streaming: `perf_metrics` via `extract_perf`. A prefix-cache hit is
sub-second against a cold multi-second/minute prefill, so hits and misses
are visible **without instrumentation**.

`/health` carries `prompt_cache_info` (pinned `pool_gb` or null, plus
`auto`, plus the prewarm file) and per-slot `kv_pool_gb` (resolved size),
`last_ttft_ms` and `prewarmed`. `prompt_cache` stays a bare bool —
launcher health checks truth-test it.

## Memory preflight

`_preflight_memory` at load **warns, never blocks**, when weights + KV pool
exceed the device budget (GPU: `GPU_DEVICE_TOTAL_MEM_SIZE`, which reflects
Windows' ~half-RAM iGPU policy and Intel's "Shared GPU Memory Override"
driver setting; CPU: total RAM). It logs the KV pool's token capacity from
`config.json` geometry — ~56 KB/token for a 7B coder, ~96 KB for 30B.

"Total RAM" on the CPU path means `min(MemTotal, cgroup limit)`, not
`/proc/meminfo` alone — inside a container `MemTotal` reports the **host**
total, so a `--memory=4g` container sized a 4 GB KV pool on top of 1.6 GB of
weights and warned about nothing [OBSERVED 2026-08-24, Docker 29.7.2 on WSL
2.7.12; MemTotal 23.5 GB vs `/sys/fs/cgroup/memory.max` 4 GB]. That is
issue #21 with extra steps. `_cgroup_mem_limit_bytes` reads cgroup v2 then
v1 and treats `max`/sentinel values as no limit; the same container now
sizes the 2 GB floor and prints the agent-prompt warning.

A too-small pool **hard-fails** generation with `Got unfinished
GenerationStatus` (issue #21); `explain_genai_error` annotates that error
with a `--cache-size-gb` hint wherever it surfaces.

The verdict has three words, not two (`_preflight_verdict`): **fits** is
silent, **tight** is an info line when the estimate overshoots the budget by
at most 5 % — the estimate's own 10 % margin is doing that, not the
hardware — and **over** is the warning. [OBSERVED 2026-09-12] The Arc Pro
B60 was told "needs ~23.3 GB but the device budget is 23.3 GB — this will
likely NOT work" for 15.2 GB of weights and a 6 GB pool, then served a 30B
coder through two OpenCode sessions; that case is now "tight". The hint on
"over" is typed by device: only an integrated GPU has a driver budget to
raise, so a discrete card is told to shrink the quant or the pool instead.

VLM configs nest geometry under `text_config` — `_text_config` handles
that, which is what fixed the KV half of this preflight silently
no-op'ing on every VLM.

## The scheduler path prefills cold three times slower on a no-XMX iGPU

[OBSERVED 2026-09-11, Core Ultra 9 285K Xe-LPG iGPU, Qwen3-8B-int4-cw, genai
2026.3.0] The same 11,198-token OpenCode request (25k-char system prompt +
ten tools rendered into it, captured from OpenCode 1.18.30) reaches its first
token in **72 s** on the plain `LLMPipeline` (`--no-prompt-cache`) and in
**216 s** through the continuous-batching scheduler that prefix caching
requires — identical in bare genai (217 s), so it is the runtime, not the
server. During those 216 s one CPU core sits at 100 % and the GPU compute
engine reads near 0 % in the samples. A 12.8k-token prose prompt takes 350 s
the same way. So on this GPU class the cache buys nothing on a cold turn and
costs 3× on it; it pays back only on the hit. Not measured on Xe2 (140V,
B60) or Xe3 (B390), where the published TTFT numbers for cached paths are
short — the CPU-bound phase may be an Xe-LPG-only artefact. Worth an
upstream question once someone has a second GPU class measured.

A process lesson from the same evening: three "stalls" chased for an hour
were measurement cutoffs of 140 s and 330 s against a path whose real TTFT
was 216–350 s. Set the budget from the plain-pipeline TTFT × 4 before
declaring a hang, and sample CPU as well as GPU: 100 % on one core with the
GPU idle is what this path looks like when it is *working*.
