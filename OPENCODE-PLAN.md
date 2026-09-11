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
| 1 | + Recipe B (12 KB / 300 lines, prune on) | same; did the answer survive the truncation? |
| 2 | + Recipe C plugin, compressor on the NPU | same + NPU time per block, validator reject rate |
| 3 | Arm 2 on the B390 class (community) | does it still pay on a fast iGPU? |

Instruments: NoLlama's request log (`<- [GPU] N chars`, `-> … TTFT`),
`/health` (`kv_pool_gb`, `last_ttft_ms`), OpenCode's `--format json` events
(tool output sizes per call), and aikomp's `metering.py` idea of per-block
accounting — how much of a turn's prompt is tool output, and how much is
repeat.

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
