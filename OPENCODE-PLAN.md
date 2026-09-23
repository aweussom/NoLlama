# OpenCode on weak Intel hardware: two servers, one plugin, and the NPU

Status 2026-09-11: recipe A (below) is verified; the compression plugin is a
plan, not code. This file is the thinking, the evaluation arms, and the gates.
Measurements land in `docs/BENCHMARKS.md` and `docs/AGENTS.md` as they come;
rejected paths go to `TODONT.md`.

## The premise

OpenCode against a local model on an iGPU is dominated by **cold prefill of
each turn's new suffix**, not by decode. Measured this week:

| Setup | Prompt | Cold prefill |
|---|---|---|
| 285K Xe-LPG iGPU, Qwen3-8B, OpenCode build request | ~30k chars | 100–155 s |
| B390 iGPU, once context outgrew the KV pool (#32) | 82k / 140k / 199k chars | 58 / 108 / 175 s |
| B60 dGPU, plain pipeline | 33k tokens | ~9 s |
| Any of them, prefix-cache hit | same | 0.2–1.4 s |

Every byte of new tool output is prefilled cold once, then cached — until the
session outgrows the pool, after which every turn misses and TTFT goes linear
in total context. So bytes of tool output cost three times: cold prefill on
arrival, pool consumption for the rest of the session, and (for an 8B–30B
model) attention spent on noise. Shorter tool blocks are the lever, and the
harness owns the tool blocks.

The compressor cannot be the GPU: it is serialised on the slot lock with the
turn itself. It can be the **NPU**, if the blocks fit its 4096-token prompt
cap. OpenCode's own truncation cap decides how big blocks get.

## What OpenCode already provides (verified 2026-09-11, v1.18.30)

| Knob | Where | Default | Verified in |
|---|---|---|---|
| `small_model` — separate model "for tasks like title generation" | `opencode.json` | provider's cheaper model, else main | config schema; live run |
| `tool_output.max_lines` / `max_bytes` — truncate, save full text to disk, return a preview | `opencode.json` | 2000 lines / 51 200 bytes | config schema |
| `compaction.auto` / `prune` / `reserved` | `opencode.json` | true / **false** / 10 000 | config schema + docs |
| `tool.execute.after` hook — `output: { title, output, metadata }`, mutable | plugin API | — | `packages/plugin/src/index.ts` |
| `experimental.chat.messages.transform` — rewrite messages per request | plugin API | — | same file; aikomp DESIGN.md §2 |
| `experimental.session.compacting` — add context / replace the compaction prompt | plugin API | — | same file |
| Two requests per turn: `agent=title` (~2k chars) + `agent=build` (30k+) | behaviour | — | OpenCode debug log |

## Recipe A — two NoLlama servers, NPU takes the side-tasks (DONE)

Coder on the GPU (port 8000), small chat model on the NPU (port 8002),
`model` → GPU provider, `small_model` → NPU provider. Verified: the title
request hit the NPU (12 tokens, 2.8 s) and the turn reached the GPU in the
same second instead of queueing behind it. Documented in `docs/AGENTS.md`.
Two processes rather than dual mode: separate locks, pools and crash domains.

## Recipe B — free knobs first (EVALUATING)

Before any code: `tool_output.max_bytes` down from 50 KB to **12 KB**
(~3k tokens, under the NPU cap with room for a distillation prompt),
`max_lines` to 300, and `compaction.prune: true`. Measure against defaults
on the same task. If this alone removes most of the cold-prefill pain, the
plugin's job shrinks to quality, not speed.

## Recipe C — compress-at-birth plugin, NPU as the compressor (PLANNED)

An OpenCode plugin on `tool.execute.after`: when a tool result exceeds a
threshold, send it to the `small_model` endpoint (the NPU server) with a
conservative distillation prompt, validate the result, and replace
`output.output` **once**, at birth. OpenCode stores the replaced text in the
session, so every later turn re-sends identical bytes: cache-neutral by
construction, no hash memo (aikomp needed one because a proxy sees the
original again every turn).

Inherited from aikomp, unchanged:

- **Let the LLM do the heavy lifting.** Code only in two thin roles: a
  mechanical pre-pass (ANSI strip, exact-line dedup) and a **contract
  validator** — every ERROR/FAIL/Traceback line, every `file:line`, every
  exit code from the original must survive in the distillate, or the original
  is kept. Code guarantees fallback, not quality.
- **Never block on the compressor.** Timeout or error → original.
- **Block level only.** Turn-level compaction stays OpenCode's
  (`compaction.auto`, `session.compacting`). Two lossy layers without
  coordination would fight (aikomp TODONT §2).

What is new here:

- The compressor runs on the **NPU**, at a few watts, while the GPU is busy
  with the turn. Blocks above the NPU cap are not distilled; OpenCode's
  truncation preview stands (Recipe B keeps them rare).
- The hook fires **once per tool result**, not once per request. The
  proxy's hardest problem (determinism across resends) does not exist.
- Where it lives: an OpenCode plugin repo (TypeScript, `.opencode/plugins/`),
  not NoLlama. NoLlama stays a server (pinned Discussion #41; the OpenClaw
  lesson in TODONT: trimming belongs in the client).

### The side lane is not free on an iGPU, 2026-09-23

[OBSERVED 2026-09-23, 258V laptop, 140V iGPU, OpenVINO 2026.4, OpenCode 1.18.30,
`opencode run` on a two-test fixture] Two servers as Recipe A prescribes, coder
on the GPU and Phi-3.5-mini on the **CPU** at port 8002.

| | Side request (128 tokens) |
|---|---|
| CPU server, box idle | **1.6-1.9 s** |
| CPU server, Qwen3-8B prefilling on the iGPU | 15.1 s (TTFT 8.5 s) |
| CPU server, Qwen3-Coder-30B prefilling on the iGPU | **40-42 s (TTFT 12.8-24.8 s)** |

The CPU is *faster than the NPU* (2.8 s baseline) when nothing else runs, and
roughly twenty times slower while an iGPU prefills beside it. An integrated GPU
is not a separate engine: the same package does the driver work, the tokenizer
and the memory traffic. On the B60 the same split worked well (2026-09-12),
because there the CPU really is idle.

**So the split is a dGPU recipe, not a laptop recipe.** An installer offering a
second CPU server by default on an iGPU box would be recommending a 40-second
side request in place of a 5-second one on the coder itself.

### A real task does complete on a 140V, 2026-09-23

Same session, `Qwen3-Coder-30B-A3B-Instruct-int4` on the 140V: read two files,
diagnose, edit, run the tests, confirm — **333 s wall clock**, correct one-line
fix, tests green. Roughly ten turns at 6-13 tok/s decode. TTFT is bimodal and
that is the whole story: **0.2-0.7 s on a cache hit, 16-28 s when the turn
carries a new suffix**. Nobody's patience is spent on decode.

Two harness lessons, recorded because both produced a false negative first:

- `opencode run` **auto-rejects edits outside the project directory**. A fixture
  under a scratch path failed with the model having diagnosed the bug correctly;
  the same fixture at `C:\develrena` passed. A rejected tool call reads like
  a model failure in the transcript.
- **Qwen3-8B cannot drive OpenCode.** It invented `src/calc.py` and never
  recovered, in both arms. Use it to measure plumbing, never to judge whether
  agentic coding works here.

### Which models can actually drive OpenCode, 2026-09-23

[OBSERVED 2026-09-23, 258V laptop / 140V iGPU, OpenVINO 2026.4, OpenCode 1.18.30,
`opencode run` on a two-test fixture, pinned KV pool] Criteria fixed before the
runs: PASS = finishes unaided, tests green, under ten minutes.

| Model | On disk | Verdict | How it behaves |
|---|---|---|---|
| `Qwen3-Coder-30B-A3B-int4` | 16 GB | **PASS** | 333 s, correct fix, tests green |
| `Qwen3-14B-int4` | 9.1 GB | **PARTIAL** | Correct fix in 5 turns with real tool calls, still going at 9 min |
| `Qwen2.5-Coder-14B-int4` | 7.9 GB | **FAIL** | Narrates tool use, fakes a call block in markdown, hands the task back |
| `Qwen2.5-Coder-7B-int4` | 4.2 GB | **FAIL** | Never calls a tool; writes instructions to the user |
| `Qwen3-8B-int4` | 4.6 GB | **FAIL** | Invents a path (`src/calc.py`), never recovers |

**Coding ability is not the binding constraint — tool-calling training is.** The
2.5-Coder family writes fine Python and cannot drive an agent loop; the 14B was
fast enough at 6-8 tok/s with TTFT under 1.2 s, and still failed. That is the
whole gap between "a coder model" and "an agent model", and it is why the
`agent` flag now comes off both 2.5-Coder entries in `models.json`.

**So there is no comfortable 16 GB answer.** Qwen3-14B is 9.1 GB before a KV
pool, and NoLlama's own warning on an 8 GB model with a 4 GB pool is explicit:
`~21k tokens — agent prompts (20k+) will exhaust it`. A 16 GB box can hold the
model or a usable cache, not both.

Two harness facts, recorded because each cost a run:

- **`opencode run` hangs at init after `opencode.json` changes.** Three times
  today: no request ever reaches the server, its own log stops right after
  `init`, and a plain retry with the same config works. Anyone editing a
  provider config will meet this.
- **The KV auto-sizer over-commits on a shared-memory iGPU.** It took 12 GB for
  an 8 GB model, driving free RAM to zero on a 32 GB box. Pin it with
  `--cache-size-gb` on any iGPU run.

## Economics to measure, not assert

Distilling a 3k-token block on the NPU costs its prefill plus ~200 tokens of
decode — order of 5–10 s on NPU 3, less on NPU 4. The same block costs the
GPU one cold prefill (30 s on Xe-LPG at ~100 tok/s; ~3 s on a B390) plus its
share of the pool for the rest of the session. So Recipe C should win
outright on Xe-LPG-class iGPUs and only pay for itself on Panther Lake
through pool longevity and answer quality. **That is the hypothesis, and it
is what the arms below test.** The distillation also runs concurrently with
the GPU turn, so its wall-clock cost is partly hidden.

## Evaluation arms

Same task, same servers, fresh session each. OpenCode 1.18.30 in Docker
(`ghcr.io/anomalyco/opencode`, `run --format json --print-logs`), NoLlama on
the host. Task: a read-heavy question over NoLlama's own source tree — it
forces `grep`/`read` on a 4 000-line file, which is the tool-output shape
that hurts.

| Arm | Config | Measures |
|---|---|---|
| 0 | Recipe A, OpenCode defaults | **Ran 2026-09-11, degenerate:** turn 1 = 46.6k chars, TTFT 217 s, then Qwen3-8B fired 27 tool calls (1 glob + 26 `read`s) and OpenCode sent 1.49 M chars back against a 40k-token window; killed after 4 min of prefill. Two lessons: declare the model's real `limit.context` (the config said 120k) so OpenCode compacts before sending, and a weak model's tool-call fan-out is the byte source, not any single result |
| 1 | + Recipe B (12 KB / 300 lines, prune on) | **B60 result below (2026-09-12).** 285K attempt 2026-09-11, no data: the run failed on something that did not reproduce afterwards (`brain/proposed/2026-09-11-opencode-arm-1-never-produced-a-token.md`); the exact request replayed later behaves normally at 216 s TTFT on the scheduler path vs 72 s plain (`docs/dev/prefix-cache.md`). Rerun with the honest 40k `limit.context`, `chunkTimeout` raised, and a 600 s patience |
| 2 | + Recipe C plugin, compressor on the NPU | **Ran 2026-09-12 with the CPU as compressor: failed the gate** (5× slower, 4 of 5 rejected, answer quality halved). Below |
| 3 | Arm 2 on the B390 class (community) | does it still pay on a fast iGPU? |

Instruments: NoLlama's request log (`<- [GPU] N chars`, `-> … TTFT`),
`/health` (`kv_pool_gb`, `last_ttft_ms`), OpenCode's `--format json` events
(tool output sizes per call), and aikomp's `metering.py` idea of per-block
accounting — how much of a turn's prompt is tool output, and how much is
repeat.

### First results: the B60 rig, 2026-09-12

Rig: `Qwen3-Coder-30B-A3B-Instruct-int4` on the Arc Pro B60 (port 8000,
6 GB pool, resident), `SmolLM3-3B-int4-cw` on the 5950X CPU as the NPU
stand-in (port 8002), both as scheduled tasks on the B60 box; OpenCode
1.18.30 in Docker on the 285K, reaching the B60 over Tailscale. Task: the
read-heavy cancel-mechanism question over NoLlama's own tree, fresh session
each. [OBSERVED 2026-09-12]

| Arm | Wall clock | Turns | Cold first token (46.6k chars, ~11k tok) | Later turns' TTFT | Tool results | Answer |
|---|---|---|---|---|---|---|
| 0, OpenCode defaults | **52 s** | 7 (grep, 4× read, write, summary) | 19.7 s | 1.1–4.4 s | 1.3–8.1 KB each | correct |
| 1, 12 KB / 300-line caps + prune | **56 s** | 8 (2× grep, 4× read, write, summary) | 13.2 s | 1.2–4.4 s | 0.9–7.9 KB each | correct, 70 lines |

Both answers are right: they name the per-request token, its binding inside
the lock, the consumer's finally, `/v1/cancel`, and the reason (issue #40).
The 30B coder issued targeted `read`s with offsets and limits — six of them
against a 4,700-line file — where the 8B on the 285K fired 26 whole-file
reads. Model quality, not the harness, decided the byte count.

What this does and does not show:

- **Recipe A holds on real hardware.** The title request went to the CPU
  server (15 tokens, 6–8 s) and the turn started on the B60 at the same
  second. Decode on the long generation: 41–44 tok/s.
- **Arms 0 and 1 are equal within noise on this task** because no tool
  result exceeded 8 KB — the 12 KB cap never bound. Recipe B's test needs a
  task that produces big blocks: a full-file read, a failing test run, a
  noisy build log. Same for Recipe C, which distils only above a threshold.
- **The scheduler path's cold prefill is fine on Xe2**: 13–20 s for ~11k
  tokens here against 216 s on the 285K's Xe-LPG for the same request. The
  3× penalty recorded in `docs/dev/prefix-cache.md` is Xe-LPG's, not the
  path's.
- NoLlama's memory preflight warned "needs ~23.3 GB, budget 23.3 GB — will
  likely NOT work" on a load that worked; the wording is the iGPU's
  (shared-memory override) and the equality case is a false alarm on a
  discrete card. Noted in `brain/todo/1-now/010`.
### Arm 2: the compress-at-birth plugin, first build, 2026-09-12

Built as `.opencode/plugins/nollama-distill.ts` (a `tool.execute.after`
hook: mechanical pre-pass, one chat call to the small-model server,
contract validator, 70 % size rule, never blocks; source and log in the
aikomp repo). Distiller: SmolLM3-3B on the 5950X CPU. Task 3: four bash
commands producing 6.2–9.1 KB each (`grep -n 'def '`, `grep -n 'print('`,
`grep -n -i error`, `sed -n 1,200p` over `nollama.py`), then an INDEX.md
from their content. Same rig, fresh sessions. [OBSERVED 2026-09-12]

| | Arm 1 (caps, no plugin) | Arm 2 (caps + plugin) |
|---|---|---|
| wall clock | **49 s** | **260 s** |
| distiller calls | — | 5, at 35–50 s each |
| verdicts | — | 1 replace (6.2 → 2.5 KB), 4 keep (not smaller / contract missing 16 of 16 / empty / contract missing 1 of 1) |
| functions counted (truth: 150 `def ` lines) | 114 | **57** — from the one distilled block |
| print sites (truth: 120) | 104 | 104 |
| extra turns | — | 1 (the coder re-ran a `^def` grep after seeing the distillate) |

**Gate failed on every axis.** Time: on the B60 a turn's first token is
0.5–1.4 s, so a 35–50 s distillation per block is a 5× slower session, and
the CPU stand-in is not the slow part of the story — NPU 3 would be slower
still. Quality: the one block the validator let through was a *listing*,
not noise, and the 3B distiller kept the first half of it and dropped the
rest; the coder's answer halved with it. The contract validator did its
job on error-shaped text (it rejected the `error` grep for dropping 16 of
16 contract lines) and has nothing to say about listings, which is exactly
the byte source in a coding session. Reject rate 4 of 5, against a gate of
~20 %.

What would change the verdict, in order of plausibility: (1) apply it only
to genuinely noisy output — test runs, build logs, stack-trace floods —
where aikomp's contract is the right test and listings never enter;
(2) a distiller that is both faster and stronger than a 3B on a CPU or
NPU, which today means the GPU itself, which is busy; (3) a GPU class
where prefill is the bottleneck — on the 285K's Xe-LPG a 9 KB block costs
~40 s of cold prefill, so 45 s of distillation is break-even there and a
loss everywhere faster. None of these is the B60, and the B60 is the
target. Recipe C is parked; see `TODONT.md`.
## Gates

- Arm 1 gates Recipe C: if caps + prune already cut the cold suffix by most
  of the way and the answers hold, the plugin is a quality tool, and the
  bar for building it rises.
- Recipe C gates on the contract validator's reject rate: above ~20 % the
  NPU model is too weak for the job and the plugin is a truncation tool
  with an expensive detour.
- Arm 3 decides whether the recipe is "for weak iGPUs" or "for iGPUs".

## Open questions

- Does OpenCode route **compaction** to `small_model`? The docs say "tasks
  like title generation"; compaction needs the whole context and cannot go
  to the NPU. If it does, the NPU provider's `limit.context` must keep it
  away, or compaction fails.
- NPU prefill throughput on a 3k-token block (NPU 3 and NPU 4) — the
  compressor's real cost. Not yet measured.
- Does the truncation preview (Recipe B) lose the lines the model needs?
  Arm 1 answers this per task.
- Where the plugin repo lives: aikomp (the ideas are its), or a fresh
  `opencode-nollama` repo. Decide after Arm 1.

## Manual test (WSL2)

Everything the Docker runs do can be done from a WSL2 shell with `npx
@deepseek-ai/dsh` replaced by `opencode` installed via the install script,
`opencode.json` in the working directory pointing at `http://<windows-host>:8000/v1`
and `:8002/v1` (the Windows host address from `ip route`), and `opencode run
--format json --print-logs "<task>"`. The NoLlama request log on the Windows
side is the primary instrument either way.
