# Next steps

State after the 2026-08-18 merge. Anything settled lives in README, TODONT or
the docs — this file is only what's still open.

## Pending a reboot of the 258V laptop — LFM2 garbage on NPU 4

NPU driver **32.0.100.5540** is installed (pnputil, `oem43.inf`, exit 3010)
but the kernel driver + firmware swap waits for a reboot; every process still
reports `NPU_DRIVER_VERSION=1004778`. Do not reboot for it — the WSL2 project
comes first. When the laptop does reboot anyway, run this **first** (Git Bash,
no admin):

```bash
cd /c/devel/aweussom/python/NoLlama
./venv/Scripts/python.exe -c "import openvino as ov; print(ov.Core().get_property('NPU','NPU_DRIVER_VERSION'))"   # expect 1005540
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe /c/Users/tommyl/npu-driver-backup/compiler-probe.py C:/Users/tommyl/models/LFM2.5-1.2B-Instruct-int4-cw-ov
bash /c/Users/tommyl/npu-driver-backup/npu-probe.sh SmolLM3-3B-int8-cw-ov control   # NPU still healthy?
```

Read it as:
- `Hello!`-type answers → the garbage was a **4778 runtime bug on NPU 4**; the
  fix for users is "update the NPU driver". Then bisect downward to find the
  first good driver: elevated pwsh 7,
  `C:\Users\tommyl\npu-driver-backup\rollback-npu-driver.ps1 -Target 32.0.100.4724`
  (and `.4512`), probe again after each.
- still `cohclclcl…` → **NPU 4 is wrong for LFM2 regardless of driver**;
  report the full matrix to openvinotoolkit/openvino#37322 (Intel already
  reproduced "LFM2.5-1.2B gibberish on NPU" there on 2026-08-13).

Either way, afterwards: TODONT entry, `models.json` + `docs/MODELS.md` caveat
on the two LFM builds ("verified NPU 3 only"), the two HF model cards, and a
reply on issue #24. To go back to the shipped driver:
`rollback-npu-driver.ps1` (no args) — 4778 is still staged and backed up.

What is already established (2026-08-30, full log in
`C:\Users\tommyl\npu-driver-backup\FINDINGS.md`): with OpenVINO 2026.3.0,
driver 4778 and the same files held constant, **NPU 3 (285K, arch 3720) is
correct and NPU 4 (258V, arch 4000) emits byte-identical garbage** — through
2026.3.0, 2026.3.1, the 2026.5.0 nightly, plugin *and* driver compiler, every
NPUW knob tried, and Intel's own `LFM2.5-350M-int8-ov`. Same file on CPU/GPU
in the same venv: correct. SmolLM3/Qwen3 on the same NPU: correct.

## PR #34 (NPU_PLATFORM pin) — author has until 2026-09-01 ~07:00 UTC

Third revision `606b779` was sent back 2026-08-30 18:46 UTC: the commit that
says it drops the subprocess isolation still contains it (`subprocess.run`,
`_detect_devices_inproc`, `[RETEST]`, "≈0.3 s" all present), and its merge
commit's second parent is PR #35's branch, so the PR carries #35's three
files. The *pin itself* is verified here (Lunar Lake: `NPU_PLATFORM="4000"`
and `"NPU4000"` both load; `--npu-platform` works; banner shows it).

Decision (owner, 2026-08-30): wait 36 h for a clean branch. If nothing by
the deadline, write the pin ourselves — `npu_platform` on `DeviceSlot` and
`OptimumSlot`, the `NPU_PLATFORM` kwarg in `load()` for NPU slots,
`--npu-platform`, `platform` in `detect_devices()`'s NPU dict, the banner
suffix, and the `explain_genai_error` hint for `Unsupported platform:
'AUTO_DETECT'` — in-process detection as on `main`, `nollama.py` only,
credit PearTr0191 in the commit, close #34 as superseded. Verify with the
`npu-probe.sh` SmolLM3 control before pushing.

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

  Also learned, and it constrains planning: the B60 box **cannot run
  `venv-nightly` at all** — its application-control policy blocks the
  unsigned `py_openvino_genai` DLL, and elevation does not lift it. Any
  future "release vs nightly on a discrete Intel GPU" question has nowhere
  to run today. → `docs/dev/machines.md`.

- **USM OOM: filed upstream as openvino.genai#4344 (2026-08-18).**
  Raw VLMPipeline (plain, no scheduler_config), Glimmer int4 on the B60:
  first ~33k-token generate fails with a USM Device allocation error;
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
  irrelevant.

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
- **`transformers` main breaks the optimum backend's text-only path.**
  `5.16.0.dev0` calls `get_experts_implementation()` from
  `_optimize_model_for_decode()`; `OVModelForCausalLM` doesn't implement it, so
  `generate()` dies. `OVModelForVisualCausalLM` has its own `generate()` and is
  unaffected — the only reason Glimmer works. This will bite `nemotron_h`, which
  is text-only. `install-optimum.ps1 -TransformersRef main` is the exposure:
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
