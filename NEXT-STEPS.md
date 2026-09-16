# Next steps

State after the 2026-08-18 merge. Anything settled lives in README, TODONT or
the docs — this file is only what's still open.

## Issue #38 idle crash — parked 2026-09-16, waiting on the reporter

**Do not start either run below until oligocene reports back.** He was asked
(comment 5687275131) to move from GPU driver `32.0.101.8508` to current, and
to put the NPU driver in at the same time. If `8991` fixes it, both runs cost
hours and tell us nothing.

**What is already settled.** The 140V does not reproduce it:
`scripts/idle-residency-probe.py` on `Qwen3-8B-int4-ov`, bare genai, rungs of
5/45/120/**210** minutes of untouched idle, all generating in 1.19–1.35 s
against a 1.42–1.55 s cold baseline, uptime 22.4 h → 32.0 h. Commit `c5d812c`;
numbers in `docs/dev/machines.md`. Bare genai is the right arm because at
`--idle-timeout 0` on an iGPU the server touches the pipeline *never* —
`_gpu_keepalive` is dGPU-gated and `_idle_watchdog` is not constructed — so a
bare pipeline left alone is his configuration minus Flask.

**What that does not settle**, and why a null result here is weak: his model
is ~15 GB inside a **stock 32 GB shared ceiling**; mine was 4.55 GB inside
this box's 27.2 GB override. The laptop cannot imitate a stock cap — same trap
as issue #24, and `machines.md` says so in as many words.

### Run 1 — the 285K, when he reports back

**Model against budget, the axis today's run could not touch.** The 285K is
the repro box for allocation-cap reports: stock-cap, no-XMX iGPU
(`GPU_DEVICE_MAX_ALLOC_MEM_SIZE` ~4.29 GB, no `GPU_HW_MATMUL`), and the same
Arrow Lake generation as his 285H. Nothing else we own is that shape.

- `scripts/idle-residency-probe.py`, unchanged, on
  `Qwen3-Coder-30B-A3B-Instruct-int4-ov` (15.2 GB) so the allocation sits near
  the ceiling rather than comfortably under it. Its 63 GB of RAM stages that
  without the paging that got the first 140V run killed.
- Its venv is OpenVINO **2026.3**; today's numbers are 2026.3.1. Close that
  gap or state it.
- It is a **working server** — Ollama serves from it and ComfyUI runs the
  graphic-novel work. Ask before touching either. SSH there is PowerShell:
  copy the script over rather than quoting it through `ssh '...'`.

### Run 2 — the B60, and it is a different question

The idle probe has **never run on a discrete GPU**, and the dGPU is where the
interesting machinery lives: WDDM evicts at ~80 s and `--gpu-keepalive` pings
to stop it. So the probe's premise — *nothing touches the allocation* — is not
even true there by default. Two things worth having, neither blocking on #38:

- The probe with `--gpu-keepalive 0`, to find out what a B60 allocation does
  across hours with the mitigation deliberately off. Today's iGPU answer
  (nothing decays) says nothing about a card that already evicts in 80 s.
- The same with the keepalive on, as a long-duration check that the ping keeps
  working for hours rather than minutes — the 2026-09-13 evidence is two
  windows of ~2h15m, which is good but is not "overnight".

Related and already written down: `TODO.md` on finding the residency control
the keepalive is standing in for.

## LFM2 on NPU 4 — closed 2026-09-01, and it is not the driver

The 258V laptop rebooted onto NPU driver **32.0.100.5540**
(`NPU_DRIVER_VERSION=1005540`). `LFM2.5-1.2B-Instruct-int4-cw-ov` still
emits the same word salad, **byte-identical** to every 4778 run, with the
plugin compiler and the driver compiler alike; SmolLM3-3B-int8-cw on the
same NPU and driver answers correctly at 15.4 tok/s. So the driver was the
last unvaried axis and it moved nothing — no downward bisect (4724/4512 are
older than a driver that already fails), and the user-facing fix is not
"update your NPU driver". Recorded in `TODONT.md`, `models.json` and
`docs/MODELS.md`; full log in
`C:\Users\tommyl\npu-driver-backup\FINDINGS.md`.

All three outward-facing items are done (2026-09-01):
- **Matrix reported** to openvinotoolkit/openvino#37322 —
  [comment 5498407344](https://github.com/openvinotoolkit/openvino/issues/37322#issuecomment-5498407344),
  addressed to @Zulkifli-Intel's 2026-08-13 "gibberish on NPU" observation,
  with the NPU 3 positive control and the two-driver result.

  **Answered 2026-09-12, and it has its own issue now:
  [openvinotoolkit/openvino#38100](https://github.com/openvinotoolkit/openvino/issues/38100)
  (filed 2026-09-13).** Zulkifli-Intel asked for a separate thread — *"keeping
  separate bugs and model versions in their own threads helps us track and
  triage them more effectively"* — which is fair: #37322 is titled for the
  2.6B `unordered_map` crash, that bug was fixed by a nightly in August, and
  our accuracy defect was living as a comment on someone else's resolved
  problem. #38100 restates the whole matrix standalone (3720 control, CPU/GPU
  controls, two drivers, three OpenVINO versions, both compilers, Intel's own
  350M export, and the `finish_reasons` evidence), scoped explicitly to
  **350M and 1.2B** with DavidDohmen's working 2.6B on a 256V cited as the
  boundary. #37322 cross-linked and otherwise left to its own discussion.

  Watch #38100 for a `Ref. <number>` — that is how Intel marks acceptance
  into engineering triage (see #4405 and #37501 below).
- **Both HF model cards** carry the caveat. The wording differs on purpose:
  LFM2.5-1.2B says a driver update does not fix it (measured on 4778 and
  5540); LFM2-1.2B says it is not expected to (that build was never
  re-probed on 5540 — the claim is inherited from its sibling, and the card
  says so).
- **Issue #24 answered** — the Phi-3.5 fix, the S-squared allocation
  explanation, and an explicit ask, since that reporter's 4.29 GB cap is the
  only place `_reset_vlm_state` can be verified.

Open, waiting on other people: triage of #38100, and the issue #24 reporter
confirming whether the slot reset actually works.

Note for whoever re-probes: the 5540 run covered LFM2.5-1.2B and Intel's
`OpenVINO/LFM2.5-350M-int8-ov` (re-downloaded; NPU garbage, CPU and GPU
correct same-venv). `LFM2-1.2B-int4-cw-ov` is no longer on the laptop.
A device-parameterised repro is at
`npu-driver-backup/probes/repro37322.py` — note `MAX_PROMPT_LEN` is
NPU-only and throws on CPU, which is why the script gates it. Also, `npu-probe.sh` prints only `content`, so
thinking models now log `text=''`; the answer is in `reasoning_content`.
Worth teaching the probe to print both before the next driver hunt.

## Open

- **Docker/#31: Phases 0-2 measured 2026-08-24, container path works.** Full
  results in `DOCKER-INSTALL.md`; the short version is that an Intel GPU is
  usable from a container at native throughput (74-79 vs 76-78 tok/s, prefix
  cache 1.9s → 0.3s vs 2.1s → 0.2s native), and two NoLlama bugs surfaced and
  were fixed on the way:

  1. **cgroup-blind memory sizing** (`_cgroup_mem_limit_bytes`): a
     `--memory=4g` container sized a 4 GB KV pool from the host's 23.5 GB
     `MemTotal`. Now `min(MemTotal, cgroup limit)`, v2 and v1.
  2. **WSL `/dev/dxg` 1 GiB allocation cap** (`_gpu_large_alloc_props`): the
     same B60 reports 25,055,051,776 bytes max-alloc natively and exactly
     1,073,741,824 through a container, so Gemma 4 E2B's 2.2 GB per-layer
     embedding table killed the load. A GPU whose max-alloc is below its own
     total budget now gets `GPU_ENABLE_LARGE_ALLOCATIONS`. Native installs
     take no hint and are unaffected.

  Still open, in rough priority order:

  - **`gemma-4-26b-a4b-it-int4-ov` produces deterministic garbage on the
    container GPU path.** Correct natively, correct on CPU inside the same
    container, correct on GPU for every other model tried including a 16 GB
    MoE. Byte-identical gibberish across pipelines and runs, so it is a
    compute defect, not corruption. Worth an upstream report against NEO
    26.31 on the WSL /dev/dxg path — not filed.
  - **Phase 3 packaging** — `Dockerfile` + `compose.yml` into the repo. The
    working image is still only in a session scratchpad. Compose must bind
    models under their real directory names, publish 8000 **and** 11434, and
    give `/app` a writable volume or prewarm silently never persists.
  - **NPU in a container: closed 2026-08-24, the answer is no.** 285K taken
    to WSL 2.9.8.0 (WSL Containers preview): no `/dev/accel*` in either
    channel, and `wslc run` exposes `--gpus` and no device flag at all, so
    there is no way to even ask for one. See TODONT.
  - **Native Linux `/dev/dri` is still untested**, and it is what #31 asks
    for. Neither limitation found here predicts the native answer. A live-USB
    run book is ready at `docs/dev/linux-native-gpu-test.md` — a USB is
    prepared for the B60 box, and the 285K can be made dual-boot if that
    turns out to be the better host.
  - Why models needing the large-allocation hint load ~3x slower (E2B 11.5s
    native vs 33-42.5s in-container; SmolLM3, which needs no hint, is at
    parity).
  - `_maybe_capture_prewarm` swallows `OSError`, so a read-only rootfs is a
    silent cold start forever. Wants a log line.

- **VLM slots are agent-grade (merged as PR #30 + the prewarm commit).**
  Three changes, all verified end-to-end 2026-08-18:

  0. **Prewarm on VLM slots** (followed the PR straight onto main): capture
     now happens on the VLM paths of both API surfaces, and the startup
     prefill replays through `parse_messages`' flattening so the cached
     token prefix matches real requests. Measured Glimmer/B60 through the
     network API: first turn after a restart 12.4s → **0.65s** TTFT
     (startup prewarm cost 12.1s, paid before the port answers requests).
     Slots whose runtime fell back to the plain pipeline zero `kv_pool_gb`
     at load, so prewarm skips them rather than burning a 30B prefill for
     nothing (this also makes `/health` honest about a dead cache).

  1. **Tool calling on VLM slots** (both API surfaces; buffered like LLM
     tool turns; images may ride along with tools). Qwen3.5-4B on the 140V
     and Glimmer on the B60 both return structured
     `get_weather({"city":"Oslo"})` with `finish_reason=tool_calls`;
     Glimmer's reasoning stays in `<think>` with no channel leak
     (`_AtemPlainFilter` also closes the think block at
     `<atem:function_calls>` so the tool XML reaches `parse_tool_calls`).
     This un-does the one regression the GenAI reroute had — Glimmer agent
     use no longer wants `--backend optimum`.
  2. **Prefix caching on VLM slots.** VLMPipeline honors `scheduler_config`
     — the long-standing "CB backend is LLM-only" belief was stale. Verified
     on 2026.3 *release* (140V, ~9k-token prefix 21.7s→3.9s TTFT) and the
     2026.4 nightly (B60/Glimmer, 33k-token prefix 53.7s→1.4s through
     NoLlama's serving path). Runtimes that reject the property fall back to
     the plain pipeline with a log line, like the LLM branch.

  Honest observations from the measurements:
  - **CB VLM prefill is slower cold**: the same 33k prompt prefilled in
    ~8.7s on the plain pipeline vs 53.7s under CB (then 1.4s per repeat).
    Agents win from turn two; one-shot prompts pay more once.
    `--no-prompt-cache` restores the plain pipeline if that bites.
  - The plain pipeline **OOM'd on the first 33k-token request** on the B60
    (16 GB USM allocation failed; the immediate retry succeeded). Under CB
    the same request completed first try. Unexplained — file upstream if it
    reproduces.
  - The "minutes of prefill" worry for agent prompts was wrong for the B60
    class: 33k tokens prefill in ~9s on the plain pipeline.

- **Intel docs gap — filed upstream as openvino.genai#4343 (2026-08-18).**
  The VLMPipeline API docs describe its kwargs only as "Device properties"
  and never mention `scheduler_config`/prefix caching; the GenAI guide shows
  SchedulerConfig on LLMPipeline only. The feature works (our measurements
  above, on 2026.3 release AND 2026.4 nightly) — undocumented, not
  unsupported. The issue also flags the slow cold CB prefill (~54s vs ~9s
  plain, same prompt/HW) as an observation; if Intel asks, offer the
  standalone repro. (Track: Intel has historically fixed our reports
  within a day.)

  **Status 2026-09-01.** The thread has moved onto the E4B export defect,
  which Intel accepted on 2026-08-31 and intends to fix by re-uploading the
  IR (see `TODONT.md`). We posted three things in reply: a second
  independent reproduction (Arc 140T, issue #24, dated the same day, HF repo
  untouched since 2026-04-23); the argument that the durable fix is adding
  `gemma4` to optimum-intel's `FORCE_ATTN_MODEL_CLASSES` rather than
  re-uploading one artifact; and the cold-prefill repro offer, now that a
  performance engineer is on the thread. **Item 1 — the actual documentation
  ask — is still unanswered**, and was restated so the issue does not close
  as "E4B fixed" with the docs untouched. Watch for that.

  **`Phi-3.5-vision`: the model works, we broke it.** Every image request
  died in the genai sampler (`input_ids token out of bounds`,
  `logit_transformers.hpp:412`) on a 140T (community) and a 140V (here),
  on the 2026.3 release and the 2026.5 nightly, with and without prefix
  caching. All of that was true and all of it was beside the point: driving
  `VLMPipeline` directly, the trigger is **NoLlama's default
  `repetition_penalty` of 1.05**. At 1.0 the same model reads the same
  images correctly; presence/frequency penalties are harmless. Phi-3 vision
  places image placeholders outside `[0, vocab_size)` and only the
  repetition-penalty transformer walks prompt ids.

  **Closed 2026-09-01.** Fix landed (`_vlm_penalty_guard`: retry once
  without the penalty, remember per slot, warn once; text turns keep it).
  Verified on the 140V through both the streaming and non-streaming server
  paths, and Qwen2.5-VL-3B confirmed unaffected. The B60 leg came back
  **identical on discrete Battlemage**, so the bug is hardware-independent
  across three GPUs, two GPU classes, two runtimes and both pipelines.

  **Filed upstream as openvino.genai#4405** (2026-09-01). The bar for
  filing was CPU reproducing it, not just our GPUs — an Intel-GPU-only
  repro is one the maintainers may not be able to run. CPU on the 2026.5
  nightly fails identically, so the report leads with an 18-line CPU-only
  script against their own published model.

  **Upstream status (2026-09-09):** first triage tested `LLMPipeline` on a
  text model and missed the path; answered 2026-09-06 with the divergence
  shown (penalty honoured on text, asserts on image). Then an Intel engineer
  posted `Ref. 194483` — their internal tracker id, which is how Intel marks
  an issue accepted into engineering triage. Nothing to do but wait; quote
  that number if anyone asks upstream about it.

  Also learned, and it constrains planning: the B60 box **cannot run
  `venv-nightly` at all** — its application-control policy blocks the
  unsigned `py_openvino_genai` DLL, and elevation does not lift it. Any
  future "release vs nightly on a discrete Intel GPU" question has nowhere
  to run today. → `docs/dev/machines.md`.

- **USM OOM: filed upstream as openvinotoolkit/openvino#37501 (2026-08-18).**
  (This section said `openvino.genai#4344` until 2026-09-13. No such issue
  exists — it is against the **openvino** repo, because the allocation is the
  GPU plugin's, not genai's. Corrected so a search for the number finds the
  thread.) Raw VLMPipeline (plain, no scheduler_config), Glimmer int4 on the
  B60: first ~33k-token generate fails with a USM Device allocation error;
  identical retry succeeds. 100% reproducible, with or without short
  generates first.

  **Diagnosed 2026-08-25, and it corrects what we filed.** The buffer is a
  **full-sequence logits allocation**: every failure size decodes exactly as
  `vocab_size × sequence_length × dtype_width`, no remainder, for all four
  numbers we have. Glimmer's `vocab_size` is 202,048 and the repro prompt is
  39,658 tokens:

  | run | requested | decodes as | vs prompt |
  |---|---|---|---|
  | 2026-08-25 control | 32,095,728,896 | 202,048 × 39,713 × 4 | +55 |
  | 2026-08-25 warmed | 32,056,935,680 | 202,048 × 39,665 × 4 | +7 |
  | 2026-08-18 #1 | 16,049,884,928 | 202,048 × 39,718 × 2 | +60 |
  | 2026-08-18 #2 | 16,031,296,512 | 202,048 × 39,672 × 2 | +14 |

  Generation needs the **last position only** — 202,048 × 4 = 808 KB. The
  allocation is ~39,700x that and scales with prompt length. The whole KV
  cache for the same prompt is 1.97 GB, so the logits buffer is 16x the KV.

  **The ×1.1 reading we filed was wrong — retracted upstream.**
  `202,048 = 11 × 18,368`, so *every* allocation of the form
  `vocab_size × n × width` for this model divides by 1.1 exactly, for any n.
  We pattern-matched a property of the vocabulary onto
  `buffers_preallocation_ratio` and sent Intel down that path. Worth
  remembering as a method failure, not just a wrong answer: two data points
  fitting a ratio is not evidence when the ratio's factors sit in the
  operands.

  Unexplained: the August sizes decode at width 2 and today's at width 4 on
  the same nominal build — the buffer appears to have gone fp16 → fp32 and
  doubled, which moved the failure from "16 GB alloc fails" to "32 GB exceeds
  the 25,055,051,776 device maximum outright". Only known change on the box
  is the Windows Intel graphics driver.

  **Upstream status (2026-09-01):** an Intel engineer posted `Ref. 193991` —
  the internal tracker id, same convention as `Ref. 194483` on #4405 above.
  Accepted into triage, nothing asked of us, nothing to do but wait. Quote
  the number if anyone upstream asks about this one.

  Intel's suggested workaround (a dummy short generate first) **does not
  work** — measured 2026-08-25, fails identically. It does tighten the
  predicted length from prompt+55 to prompt+7, i.e. 0.12% against a 7.0 GB
  gap. The CB path avoids the whole thing because chunked prefill never
  allocates the full-sequence buffer — consistent with scheduler_config being
  the workaround AND with CB's slower cold prefill. Bonus bug found while
  testing: setting `OV_GPU_SHAPE_PREDICTOR_SETTINGS` (a RELEASE_INTERNAL
  option) crashes pipeline construction — `ShapePredictor::Settings` has no
  string parser ("Bad as from std::string"), so the env knob is unusable
  and a bad value kills the load.

  **To file as its own ticket, deliberately deferred (2026-08-25.)** It is
  independent of the logits-allocation bug above, and now more orphaned than
  before: ShapePredictor is no longer implicated in that bug at all, so this
  will never get attention buried as a "bonus" in a ticket about something
  else. Holding it until #37501 is resolved rather than filing now — two open
  tickets from us on the same subsystem, one of which we already had to
  retract a theory in, is a good way to get both triaged slowly. When filing:
  minimal repro (set the env var, construct any pipeline, it dies), state
  plainly that the ask is either a string parser for
  `ShapePredictor::Settings` or for the option to reject bad input without
  killing construction. Re-run the repro first — it has not been retested
  since 2026-08-18. Weight staging through host/shared memory
  is by design (two-stage allocation, memory_allocation_gpu_plugin.md); no
  public knob for device-direct loading; `usm_policy`/`disable_usm` are
  debug-caps-only. Windows "shared GPU memory" is the WDDM half-of-RAM
  budget — discrete GPUs have it too, no iGPU required.

- **Local sparse checkouts of Intel sources** (for grepping docs + GPU
  plugin internals): `C:\devel\intel\openvino` (docs/articles_en +
  src/plugins/intel_gpu, shallow) and `C:\devel\intel\openvino.genai`
  (site + src). Machine has no git-lfs — clone with
  `GIT_LFS_SKIP_SMUDGE=1` and LFS filters disabled; partial-clone sparse
  blob fetch dies on this network, plain `--depth 1` works.

- **Loading a big model stages through host memory first.** Watched on the B60
  (17 GB Glimmer): shared GPU memory ramps to near its 16 GB ceiling and holds
  there while dedicated VRAM stays flat, then dedicated fills, then shared
  drains. So **peak host RAM during load is roughly model-sized even on a
  discrete card** — worth knowing before assuming 24 GB of VRAM makes system RAM
  irrelevant. [OBSERVED 2026-09-11, `scripts/load-mem-probe.ps1`, B60 box]:
  Qwen3-30B-A3B int4 (15.2 GB) — shared peaks 12.4 GB, free RAM 16.5 → 1.8 GB,
  then dedicated 22.0 GB steady and shared back to 0.25 GB, 43 s total.

- **`hf download` stalls on large files via Xet.** It sat at 0.00 CPU with a
  `.lock` on the 14.9 GB blob. `HF_HUB_DISABLE_XET=1` resumed it and ran at
  ~78 MB/s. Also leaves an abandoned partial in `.cache/huggingface/download`
  that has to be deleted by hand (17 GB of files, 28.7 GB on disk until then).

- **Glimmer and Qwen3.8 are back in the menu (2026-08-30)** — the gate was
  OpenVINO shipping them in a *release*, and 2026.3.1 (2026-08-26) did.
  Both verified on the Arc 140V with the release wheels; `requirements.txt`
  floors are `>=2026.3.1`; Qwen3.8 is pinned to its `2026.3.1` repo branch
  (see TODONT, "nightly stack in the default install"). Still open from that:
  - **B60 numbers on 2026.3.1** for both — the 140V figures (Qwen3.8 3.6–4.8
    tok/s, Glimmer ~2.5) are iGPU-bound and say nothing about the card users
    will actually buy for these models. `docs/MODELS.md` carries the 140V
    numbers until then.
  - **Existing installs are on 2026.3.0.** `install.ps1` re-run upgrades the
    venv via the new floors; a user who only `git pull`s and picks Qwen3.8
    from a stale venv gets a segfault at load, not a message. Worth a
    version check in `nollama.py` that names the fix (`pip install -U
    openvino openvino-genai openvino-tokenizers`) before loading a model
    whose registry entry declares a minimum.
  - **Retire `-Nightly`?** Nothing in the registry needs it; it stays as the
    test harness for "does the next runtime fix X" (used 2026-08-30 for the
    LFM2/NPU 4 question). Decide when the next release lands.
- **`benchmark.py` JSON records no provenance (noted 2026-08-30).** Not the
  OpenVINO/genai version, not the server flags (`--offload-ratio`,
  `--cache-size-gb`), not the driver, not the OS. Three community reporters
  in one week (#24, #32), and for none of them does the JSON say which runtime
  produced the number. Add `openvino.__version__`, the server's `/health`
  payload (model, device, `kv_pool_gb`) and `platform.platform()` to the
  JSON header; cheap, and it turns every future report into a citable one.
- **`benchmark.py --llm-only` skips text tests when the only slot is a VLM
  slot (#40, 2026-09-07).** It assigns text tests from `/health` `type ==
  "llm"` only, so a text-capable model that landed on the VLM slot (Qwen3.6-
  35B-A3B) prints "No LLM model found". Workaround today: `--model
  Qwen3.6-35B-A3B@GPU`. Fix: fall back to the VLM slot for text when no LLM
  slot is ready.
- **#33 `matmul primitive` on the 140T is down to the MoE axis.** Asymmetric
  int8 dense at 60k chars passes on a non-XMX Xe-LPG (285K iGPU, 2026-09-11,
  `docs/dev/machines.md`), and the reporter's symmetric `int8-cw` re-convert
  fails identically — so neither zero-points nor XMX alone. Untested lever:
  `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` on the GPU (his verbose log shows an
  s8 x s8 gemm with per-token src scales, i.e. quantized activations). Needs
  a small MoE int8 export to vary the last axis here.
  **Update 2026-09-11 (evening):** the export exists now — `LFM2-8B-A1B`
  converted to INT8 asymmetric (7.8 GB, 32 experts) with `venv-2026.3`, on
  both boxes (`~\models\LFM2-8B-A1B-int8-asym`). Results [OBSERVED
  2026-09-11, 60k/100k chars of docs prose, 8-token answers]:
  - **B60 (XMX), 2026.3.0:** plain and scheduler paths pass at 60k and 100k
    (3–7 s each). Control clean.
  - **285K iGPU (no XMX), 2026.3.0:** scheduler path passes at 60k (657 s)
    and 100k (475 s); plain passes at 60k (863 s) and hits the per-allocation
    cap at 100k (11.8 GB buffer). **No `matmul primitive` error.** Prefill ran
    at ~21 tok/s, i.e. the *unfused* expert path — TODONT's XMX gate holding.
  - The reporter's verbose log shows the **fused grouped gemm being
    attempted** on his 140T — which, it turns out, **has XMX** (Xe-LPG+; the
    docs were wrong about it until 2026-09-11), so on his box the fusion
    engages and here it never does. The scratch `venv-2026.3.1` run on the
    285K passed too (60k chars, scheduler path, 699 s): runtime version is
    not the axis either. What is left is a device the project does not own:
    an Xe-LPG+ part with XMX, where oneDNN's grouped-gemm generator has no
    kernel for the fused u8-zero-point expert matmul ("insufficient
    registers", then every reference fallback rejects the datatype). The B60
    and 140V are Xe2 and have the kernels. **Cannot be reproduced on our
    hardware**; the reporter's verbose log is the upstream report.
  - Also his own bare probe passed and NoLlama failed on the same weights
    (#33, 2026-09-11) — so the cached (scheduler) path is implicated on his
    box; the discriminating runs for him are `--no-prompt-cache` and a
    120k-char probe.

- **A genai GPU load of a MoE aborts while a NoLlama NPU server runs on the
  same box** [OBSERVED 2026-09-11, 285K, 2026.3.0, three for three]:
  `LLMPipeline(<LFM2-8B-A1B int4 or int8>, "GPU.0")` dies with "Fatal Python
  error: Aborted" (exit 3), no exception, no event-log entry, while
  `nollama.py --device NPU` (SmolLM3-3B) is up; with it stopped, the same
  load compiles (212 s core, 60–69 s genai) and runs. Dense Qwen3-8B loaded
  fine beside the NPU server all afternoon. Hypothesis, unverified: the
  MoE compile's ~25 GB of shared-memory staging plus the NPU process
  exceeds the 32.9 GB shared budget and the driver aborts rather than
  throws. Matters for the two-server recipe in `docs/AGENTS.md`: start the
  GPU server first, then the NPU one — untested; and it is another reason
  to raise this box's shared-memory override.

- **OpenCode evaluation arm 1 (2026-09-11, 16:37–18:02) failed for a reason
  that did not reproduce.** Six identical 46.6k-char build requests, each
  retried by OpenCode ~10–15 min apart, never produced a token — no text
  event on the client, no completion line on the server (which, before
  `ac4fc66`, logged nothing for a disconnected client). The same body
  replayed after a full restart gave a first token at 216 s, as did bare
  genai, and OpenCode's 5-min chunk timer is reset by our keep-alives
  (verified from a container). The window followed two MoE GPU-load aborts
  and an hour of 25 GB MoE compiles on the same iGPU while an NPU server was
  up. Unexplained; recorded so the next occurrence is measured with the
  client-gone log line and a 600 s budget instead of guessed at.

- **Memory preflight false alarm on a discrete card** [OBSERVED 2026-09-12,
  B60]: "model (~15.2 GB) + KV pool (6 GB) needs ~23.3 GB but the device
  budget is 23.3 GB — this will likely NOT work (raise the iGPU budget …)".
  It loaded and ran. Two fixes: treat equality as fits, and word the hint
  by device type — "Shared GPU Memory Override" means nothing on a dGPU.

- **The B60 rig is left running** as scheduled tasks `nollama-gpu-8000`
  (Qwen3-Coder-30B-A3B int4, 6 GB pool) and `nollama-cpu-8002` (SmolLM3-3B),
  logs in `C:\Users\wossn\b60-eval\`, ports open from Tailscale. Stop with
  `Stop-ScheduledTask`. It is the arm 2 test bed; the next task must produce
  tool results over 12 KB or the caps and the distiller have nothing to do.

- **The VLM slot reset is now known NOT to cover `CL_OUT_OF_RESOURCES`**
  [OBSERVED 2026-09-12, oligocene, Arc 140T, #38]: the slot stays dead until
  NoLlama restarts, once until a reboot — a poisoned driver context, as
  OpenVINO's own error text warns. NoLlama now takes the slot out of service
  (`_note_poisoned`: status "error", reason in `/health`) and the explainer
  names the remedy. Still open: the *allocation-cap* throw from #24 (a
  different class) — whether `_reset_vlm_state` rescues that one remains
  unverified. Also noted by the reporter and seen here the same day: GPU
  misbehaviour that clears with a reboot after long uptime under heavy load;
  keep asking for driver versions.
- **`transformers` main breaks the optimum backend's text-only path.**
  `5.16.0.dev0` calls `get_experts_implementation()` from
  `_optimize_model_for_decode()`; `OVModelForCausalLM` doesn't implement it, so
  `generate()` dies. `OVModelForVisualCausalLM` has its own `generate()` and is
  unaffected — the only reason Glimmer works. This will bite `nemotron_h`, which
  is text-only. `scripts\New-OptimumVenv.ps1 -TransformersRef main` is the exposure:
  decide between pinning a known-good ref and waiting for optimum-intel.
- **Offload non-determinism on the B60, unexplained.** At
  `--offload-ratio 30`, greedy decoding returned 87-2040 tokens for the same
  prompt across five runs (resident: 478 every time). Varying length proves
  something varies; nobody has looked at whether the content is wrong or merely
  different. Detail in TODONT.
- **The offload split didn't track the ratio.** Ratio 30 left 3.2 GB of 15.2 GB
  resident (~24%) where the 140V measured 10.8 GB (~71%), with 21 GB of VRAM
  free. Either the ratio is a ceiling a demand-driven expert LRU never fills, or
  it behaves differently on discrete hardware. Runs at 50 and 90 would tell.
- **Ollama head-to-head needs redoing with the temperature pin.** The old
  comparison had Ollama sampling (its default 0.8) against NoLlama greedy (0.0),
  because `benchmark.py` sent no temperature. Fixed now. The 1.6× decode figure
  probably survives; the *task-time* reading of it does not, because Ollama's
  build ignores `/no_think` and spends ~1755 tokens on a 291-character answer
  where NoLlama spends 293. Needs Ollama on the 140V.
- **Nemotron Lightning: still blocked upstream.** PR #1789 merged descoped — no
  `nemotron_h` exporter. Decide whether to file the optimum-intel feature request
  offering to test (the pattern that worked for Glimmer, issue #1927).
- Re-run the TODONT comprehension test on each new OpenVINO release.
- Qwen3.5-4B vision verdict for the registry note (`models.json`).
- SmolLM3 registry notes could mention thinking-mode + `/no_think`.

## Benchmarking notes for whoever runs the next one

- **Use the 285K or the B60 box, not the laptop.** A busy 140V reads ~30% low
  (Qwen3-8B int4-cw: 14.8 tok/s with a browser and chat apps running, 19.4
  quiet). Decode figures across the table were verified sound on the 285K
  (SmolLM3 iGPU 29.4 vs 29.7 published, Qwen3-8B 14.6 vs 15.4).
- **Kill servers by port owner, not by pid.** A venv built from the Microsoft
  Store Python has a redirector at `venv\Scripts\python.exe`, so
  `Start-Process -PassThru` returns the launcher's pid and the real server
  survives being stopped. The next server then fails to bind and the benchmark
  quietly keeps talking to the previous model. `scripts/bench-b60.ps1` kills by
  port and asserts `/health` reports the expected model; copy both.
- **Detached `pwsh` launched over SSH dies when the session ends.** Long
  orchestration runs need to be started locally, or driven one step per SSH
  call.
