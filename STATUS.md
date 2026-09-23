# STATUS.md — where things stand

**For whoever picks this up next, human or session.** What is true right now,
what is merely *built*, and what is in flight. Three minutes to read; updated at
the end of a working session. If `README.md` disagrees, this one is younger —
say so and fix the older.

**As of 2026-09-23. `v1.0.0` was tagged 2026-09-18; 21 commits since, all on
`main`.** The evening's theme: the installer and the registry now rest on
measurements rather than on plausible assumptions, and three of those
assumptions turned out to be wrong.

**Read this first if you are picking up the agent work.** Decode speed tracks
**active** parameters, not model size: a 30B-A3B MoE runs 18-32 tok/s on a 140V
where a dense 14B manages ~7 on the same box and task. That single fact
reorganised the model list, and it is why the search for a 16 GB-tier agent
model is a search for a *small MoE*, not a small dense model.

---

## The one-paragraph version

NoLlama serves chat, vision, transcription and embeddings from one process
across the full Intel stack, on both the OpenAI and Ollama wire protocols.
GPU and iGPU are the usable path: prefix caching on by default, tool calling on
LLM *and* VLM slots, agents (OpenCode, Goose, Copilot Chat) working. CPU works
and is deliberately not invested in — llama.cpp's CPU kernels beat every
OpenVINO row including ours, and `docs/BENCHMARKS.md` now measures that
directly. The NPU is real but narrow: short prompts, no tool calling, no prefix
cache, and a per-generation correctness split that the installer now gates on.
Working state is `brain/next/` (needs Tommy), `brain/todo/` (startable cold) and
`brain/proposed/` (noticed, not trusted), one file per item.

## Verified, with dates

| | |
|---|---|
| Embeddings closed by a real RAG client: LangChain over FAISS, 25 files / 483 chunks through `/api/embed`, 3 of 3 answers correct and grounded | 2026-09-17 |
| `embed()` slicing: 431 s and 18.52 GB peak unsliced vs 224 s and ~5 GB at `--embed-batch-size 16`, 375 chunks | 2026-09-17 |
| Idle alone does not poison an Intel GPU context: 210 min on a 140V, 180 min on a B60, generate within noise of the warm baseline | 2026-09-16 |
| dGPU idle penalty **plateaus**: 0.60 s warm, 1.82 s at 5 min, 3.68 / 3.13 / 3.02 s at 45 / 120 / 180 min — three hours costs no more than forty-five | 2026-09-16 |
| NoLlama vs llama.cpp's OpenVINO backend on a B60: cold prefill a tie, decode 65.7 vs 38.7 tok/s (1.7x), agent-shaped 8k prompt 3.98 s cold vs 0.205 s cached (19x) | 2026-09-07 |
| llama.cpp's own CPU backend beats **every** OpenVINO row including ours (91.5/9.74 vs 78.3/9.08) — the measurement behind "use Ollama if you are CPU-only" | 2026-09-07 |
| Prefix caching on VLM slots: 33k-token prefix 53.7 s → 1.4 s TTFT through the serving path (B60/Glimmer) | 2026-08-18 |
| Prewarm on VLM slots: first turn after restart 12.4 s → 0.65 s TTFT | 2026-08-18 |
| GPU in a container at native throughput: 74-79 vs 76-78 tok/s; prefix cache 1.9→0.3 s vs 2.1→0.2 s native | 2026-08-24 |
| Phi-3.5-vision fix (`_vlm_penalty_guard`) verified on 140V and B60, both API paths; hardware-independent across three GPUs and two GPU classes | 2026-09-01 |
| NPU generation gate: arch `4000` hides LFM2, simulated `3720` keeps it, GPU drops it as `npu_only`; 10/10 test modules; `check-docs -Render` ALL OK | 2026-09-18 |
| Runtime floor moved to OpenVINO **2026.4** (stable since 2026-09-16). Laptop and B60 both upgraded; 10/10 test modules after fixing a test double that had been failing since 3e9a918 on both boxes. Intel's re-exported gemma-4-E4B int8 segfaults on 2026.3.1 and runs on 2026.4, where its prefix cache works: ~13k-char prefix 10.3 s cold then 1.4 / 0.9 s | 2026-09-23 |
| OpenCode completes a real task on the 140V with `Qwen3-Coder-30B-A3B-int4`: 333 s, correct fix, tests green — TTFT 0.2–0.7 s cached against 16–28 s on a new suffix. The CPU side-model split is a **dGPU** recipe: 1.6 s idle, 40 s while an iGPU prefills | 2026-09-23 |
| Agent-capable models, measured: `Qwen3-Coder-30B-A3B` PASS, `Qwen3-14B` PARTIAL (correct but 9 min), `Qwen2.5-Coder-14B`/`-7B` and `Qwen3-8B` FAIL — they narrate or fake tool calls. Tool training, not coding ability, is the constraint | 2026-09-23 |

## Where the agent story stands (2026-09-23)

| model | on disk | agent verdict |
|---|---|---|
| `Qwen3-Coder-30B-A3B-int4` | 16.3 GB | **PASS** — 333 s, two boxes |
| `Qwen3-30B-A3B-int4` | 15.2 GB | passes the fix task by hand (2.5 min) |
| `Qwen3-14B-int4` | 9.1 GB | completes in **18 min**, writes a syntax error on the way — flag removed |
| `Qwen2.5-Coder-14B` / `-7B` | — | **retired from the registry** (TODONT): narrate tool use, never call |
| `Qwen3-8B-int4` | 4.6 GB | invents paths, never recovers |
| `LFM2.5-8B-A1B-int4` | 4.2 GB | **unknown** — its 0/2 was our parser (T-036) |

**A 32 GB machine has one agent model. A 16 GB machine has none**, and the
installer now says so rather than offering one that fails. LFM2.5-8B-A1B is the
open question: 4.2 GB, ~1B active, 48 KB/token KV, and it needs 30 minutes of
re-running to settle.

## Built, not yet proven

- **`brain/` itself.** Adopted from `ace-brain` on 2026-09-18 and this repo is
  its first instance outside that one. The `triage` and `premise-sweep` skills
  there have never run against a repo.
- **The v1.0.0 release ZIP.** Tagged 2026-09-18; nobody has yet unzipped it on
  a clean box and run `install-windows.bat` from it. That is the path
  release-ZIP users take and the one a checkout never exercises.
- **Docker Phase 3.** `Dockerfile`, `docker-compose.yml` and
  `docker-compose.wsl.yml` are in the repo and publish both ports, but
  `DOCKER-INSTALL.md` is the only thing that has exercised them. No clean-box
  `compose up`.
- **Native Linux.** Confirmed working by a user (issue #6); we have never run
  it. `/dev/dri` in a container is untested and is what #31 actually asks for —
  run book at `docs/dev/linux-native-gpu-test.md`, USB prepared. (`T-026`)
- **Embeddings on the GPU with a large slice** — the one case where dispatch
  overhead might amortise. Untested. (`T-022`)
- **The NPU's intended workload.** Norwegian document translation is why the
  NPU is in this project at all, and no translation model has been probed yet.
  (`T-011`)

## In flight

Nothing is running. No background jobs, no scheduled probes on this box.

The B60 rig is left up as scheduled tasks `nollama-gpu-8000`
(Qwen3-Coder-30B-A3B int4, 6 GB pool) and `nollama-cpu-8002` (SmolLM3-3B), logs
in `C:\Users\wossn\b60-eval\`, ports reachable from Tailscale. Stop with
`Stop-ScheduledTask`. It is the OpenCode evaluation arm-2 test bed.

## Known and accepted, so nobody re-investigates

- **The NPU cannot drive an agent.** 4096-token prompt cap, no tool calling, no
  prefix cache. This is structural, not a missing feature.
- **A model can be correct on one NPU generation and word salad on the next.**
  `FULL_DEVICE_NAME` is `Intel(R) AI Boost` on all of them, so
  `DEVICE_ARCHITECTURE` (`3720` = NPU 3, `4000` = NPU 4) is the only
  discriminator. The installer gates on it and the server warns; the evidence is
  openvino#38100.
- **The B60 box cannot run `venv-nightly` at all** — application-control policy
  blocks the unsigned `py_openvino_genai` DLL, and elevation does not lift it.
  So "release vs nightly on a discrete Intel GPU" has nowhere to run today.
- **We own no stock-cap iGPU with XMX.** The 285K is stock-cap but has no
  `GPU_HW_MATMUL`; the community 140T has both. The model-against-budget axis
  behind issues #24, #33 and #38 is not testable on our hardware.
- **NPU in a container: no.** No `/dev/accel*` in either WSL channel and no
  device flag on `wslc run`. Closed 2026-08-24, see `TODONT.md`.
- **CPU is a fallback, not a target**, and NVIDIA is never happening. Both in
  `TODONT.md` with the measurements.
- GitHub is used as backup; work goes straight to `main`, no branch ceremony.

## Blocked on someone else

Three community reporters and Intel. Details in `brain/next/waiting/` and
`brain/next/waiting-on-time/007` (the upstream ticket watch list — openvino
#38100, #37501, #38211, genai #4405, #4343, optimum-intel PR #1789).

## Where everything else lives

| | |
|---|---|
| `brain/next/` | Needs Tommy — a decision, or someone chased |
| `brain/todo/` | Startable cold |
| `brain/proposed/` | Noticed, not acted on, **not trusted** |
| `TODONT.md` | Rejected approaches and why. **Read before proposing anything structural** |
| `CLAUDE.md` | Conventions, and which deep note to read before touching an area |
| `docs/dev/machines.md` | Which box to run a test on, and which one is off-limits |

---

**Keeping this useful.** Update it when a session ends, keep it roughly this
length, and delete what stopped being true rather than appending its
correction. A count here is dated or it is absent.
