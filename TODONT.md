# TODONT

Things we tried that didn't work, or that work but aren't worth doing. Each
entry explains *why not* so we don't re-litigate it in six months.

## Waiting for WSL to expose the NPU to containers (2026-08-24)

Idea: NoLlama is NPU-first, so a container path that reaches the NPU would
be worth having even though it is Windows-only. Build 2026 coverage promised
exactly that — "WSL 3" with GPU **and** NPU passthrough, several outlets
naming Meteor Lake and Lunar Lake specifically.

**Verdict:** the reporting is wrong. There is no NPU in WSL, in either the
stable or the preview channel. Stop waiting for it.

**Why not**, in the order the evidence arrived:

- **There is no "WSL 3".** Microsoft denied the name; what Build 2026
  announced is **WSL Containers**, built on WSL 2. Much of the press
  mislabelled it, and the NPU-passthrough claim rides along with the wrong
  name.
- **The primary source does not say it.** Microsoft's own WSL Containers
  announcement mentions GPU exactly once — a CUDA example — and never
  mentions NPU, AI accelerators or `/dev/accel`. Its only performance claim
  is a virtiofs filesystem speedup.
- **Stock WSL 2 has no NPU device node.** Measured on a Core Ultra 9 285K
  with a healthy `Intel(R) AI Boost`: `/dev/accel*` does not exist, only
  `/dev/dxg`. Confirmed on hardware rather than inferred (microsoft/WSL#40842).
- **Neither does the preview.** Updated that box to WSL **2.9.8.0** (kernel
  6.18.40.1) and re-checked: still no `/dev/accel*`. A `wslc` container gets
  a minimal `/dev` with no `dxg` at all; `wslc run --gpus all` adds `dxg`
  **and nothing else**.
- **The CLI cannot express the request.** `wslc run` has exactly one
  hardware-passthrough flag, `--gpus`. No `--device`, no NPU flag, no
  privileged mode. That is not a driver gap that userspace work could close
  — there is no way to ask.

So the container story for NoLlama is GPU and CPU, on every platform, and
anything we publish says "no NPU" in plain words.

Re-evaluate only on a concrete trigger: a WSL release whose `wslc run --help`
grows a device-passthrough flag, or a `/dev/accel*` node appearing in a
distro. Both are ten-second checks. Do not re-open this on press coverage —
that is what sent us round the loop the first time.

## The stock `openvino/ubuntu24_runtime` image as the container base (2026-08-24)

Idea: Intel publishes an OpenVINO runtime image; use it as the base for a
NoLlama container and get the GPU stack for free. It is what the Docker test
protocol's own Phase 0 command uses.

**Verdict:** unusable on Battlemage. Build the driver stack yourself.

**Why not:** `openvino/ubuntu24_runtime:latest` (OpenVINO 2026.3) ships
`intel-opencl-icd 24.48.31907.7` and `intel-level-zero-gpu 1.6.31907.7` —
NEO 24.48, December 2024, which predates Arc Pro B60 (BMG-G31, `0xe211`).
On the B60 box it enumerates **CPU only**. The failure is silent and reads
exactly like "the container cannot see the GPU": `zeInit` returns
`ZE_RESULT_ERROR_UNINITIALIZED`, `clGetPlatformIDs` returns `-1001`
(`CL_PLATFORM_NOT_FOUND_KHR`). The driver *does* reach the card — NEO debug
output prints `Created Wddm context. Status: :0, engine: 4` — it simply does
not recognise the device id.

Installing the current upstream release over it (compute-runtime
26.31.39395.13 + IGC 2.40.13, both from GitHub releases) makes the GPU
appear and compute correctly. Two dpkg wrinkles when doing that: `intel-ocloc`
collides with the image's `intel-opencl-icd` over `libocloc.so`, and
`libze-intel-gpu1` supersedes the older `intel-level-zero-gpu` package name —
so remove that one first and install the rest with `--force-overwrite`.

Since every Intel-supplied layer has to be replaced anyway, the eventual
image is better built `FROM ubuntu:24.04` with the driver stack and the pip
wheels installed directly: 1.28 GB model-free, versus fighting a base image
whose only remaining contribution is a Python it also has to be told not to
use. Measurements and the full result set are in `DOCKER-INSTALL.md`.

Re-evaluate when Intel's runtime image ships a NEO recent enough for the
hardware in question — the check is one `available_devices` call, and the
answer is unambiguous.

## A Gemma-family guard to honour upstream's `requires_sdpa()` (2026-08-21)

Idea: openvino_genai's `requires_sdpa()` forces the plain SDPA backend by
default for GEMMA3 + GEMMA4_UNIFIED on `releases/2026/3` (Intel tickets
171180 / 189844) because PagedAttention could not represent Gemma's
non-diagonal attention masks. Passing `scheduler_config` takes the
`explicitly_requires_paged_attention` branch and walks straight past that
guard -- which is exactly what NoLlama's default-on prefix caching does. The
proposed fix was a version-gated special case: skip the CB backend for those
architectures when the runtime is older than 2026.4.

**Verdict:** not needed. Measured, don't add it.

**Why not:** A/B'd CB (PagedAttention) against `--no-prompt-cache` (plain
SDPA) on the Arc Pro B60, 2026.3 release, greedy, identical inputs:

- `gemma-4-26b-a4b-it-int4-ov` -- **7/7 answers byte-identical**, including
  an exact three-line OCR transcription and the same wrong dot count. CB
  genuinely engaged (`prefix caching on`, `kv_pool_gb: 2`). This is the
  load-bearing result: biggest model, real PagedAttention.
- `gemma-4-E2B-it-int4-ov` -- 5/7 identical. The two divergences were
  second-pass punctuation (`Red green blue` / `Red, green, blue`) and one
  grid-reading case where SDPA answered correctly and CB denied the image.
  Across the repeat runs that is **1 success in 24 attempts**, on the one
  task sitting exactly at that model's capability ceiling. Noise, but it is
  the single datapoint a skeptic would cite, so it is recorded rather than
  rounded away.
- `gemma-4-E4B-it-int8-ov` -- inapplicable: its IR cannot build the CB
  backend at all (below), so both runs were the same plain pipeline and the
  match is tautological.

Corroborating: the same prompts through Ollama/llama.cpp on an RTX 5090
produced byte-identical OCR and the same per-size counting bias, i.e. the
stacks feed the model equivalent visual information. Nothing in the
measurements looks like corrupted attention.

Also note the guard is gone on master (`requires_sdpa()` is a stub returning
false), so upstream considers this fixed for 2026.4 -- a special case would
have been dead on arrival there anyway.

Re-evaluate if: a Gemma vision model shows *systematic* divergence between
the CB and plain paths on the same inputs -- multi-image, long-context, or
the audio path, none of which were exercised here.

## Gemma 4 E4B for agent serving on OpenVINO 2026.3 (2026-08-21)

Idea: `OpenVINO/gemma-4-E4B-it-int8-ov` is the sweet spot on paper -- reads
detail E2B cannot, a third of the 26B's footprint, and Intel publishes it
ready to run.

**Verdict:** not for agent workloads on this runtime. It gets **no prefix
caching**, so every turn re-prefills the whole system prompt.

**Why not:** the CB backend is built by rewriting the graph, and this IR has
nothing to rewrite:

```
No ScaledDotProductAttention operation observed in the graph,
cannot perform the SDPAToPagedAttention transformation.
); using plain pipeline
```

NoLlama degrades correctly (warning at load, `kv_pool_gb` null in
`/health`, prewarm skips the slot) -- it simply cannot cache.

**It is an export defect, and we proved it by re-exporting.** The same
`google/gemma-4-E4B-it` weights, same INT8 precision, current stack
(optimum-intel 2.2.0.dev0, transformers 5.5.4, OpenVINO 2026.3) produce 42
fused SDPA ops -- one per layer -- and prefix caching engages. Identical
7.8 GB, identical 42-layer / 84 KB-per-token geometry; only the attention
differs. Nothing in the published model's name, size or precision hints at
which one you have.

Correction to an earlier reading of this: the transformers version in the
IRs (5.5.4 on the two that work, 5.5.0 on E4B) is **correlation, not
cause** -- `_supports_sdpa` is True on both. optimum-intel pins the
attention implementation only for models in `FORCE_ATTN_MODEL_CLASSES`
(`phi3_v`, `gemma2`, `llama4`), and `gemma4` is absent, so the export
environment decides. That is how two builds of one model diverge.

Second trap for anyone re-exporting it themselves: the chat template is
baked into `openvino_tokenizer.xml` rt_info **at export time**, so replacing
`chat_template.jinja` afterwards does nothing. Google's raw template uses
Python-style implicit string concatenation inside `raise_exception(...)`,
which openvino_genai's C++ Jinja parser rejects ("Expected closing
parenthesis in call args"). Intel post-processes that away -- their template
is 16,317 bytes against Google's 18,569, with zero `raise_exception`. A DIY
export needs the patched template in the source directory *before* running
the exporter, or it loads and caches but dies at warmup.

**What the defect actually costs**, measured on the B60 with a ~7.9k-token
repeated prefix (the shape of an agent's fixed system prompt), 2026-08-21:

| | turn 1 | turn 2 | turn 3 |
|---|---|---|---|
| Intel's IR (no cache possible) | 2.63s | 3.06s | 3.03s |
| our re-export (prefix cache) | 5.92s | **1.16s** | **1.16s** |

Note the honest shape: Intel's IR is *faster cold*, because the CB path
prefills more slowly (the same trade recorded for Glimmer). Ours pays a
+3.3s premium once and then runs **2.6x faster per turn**, breaking even
cumulatively by turn 3 -- and with prewarm the cold turn is paid before the
port answers, so every user-visible turn is the fast one. So this is "wrong
model for agent loops", not "catastrophically broken".

Quality is unaffected either way: our re-export answered **7/7 probe cases
byte-identically** to Intel's, including both of its wrong answers (16 for
17 dots, 8 for 7). The re-export adds a capability; it changes nothing else.

Re-evaluate if: Intel re-exports E4B, or `gemma4` lands in
`FORCE_ATTN_MODEL_CLASSES`. Verify with `--scan`, not by assuming -- see
`docs/dev/prefix-cache.md`.

**Update 2026-08-31 -- Intel accepted it; the trigger is armed, not fired.**
An OpenVINO engineer reproduced the defect on openvino.genai#4343: the
stored IR has no fused SDPA node, a local export from the same weights does,
and *"we need to update the IRs stored in `OpenVINO/gemma-4-E4B-it-int8-ov`
with the one exported using the latest tool set"*. A third party
independently hit it the same day on unrelated hardware (Arc 140T, issue
#24), so it is still shipping. **Nothing has been re-uploaded** -- the HF
repo is untouched since 2026-04-23 -- so `models.json` still points at our
re-export. `REVISION_WATCH` in `scripts/model_watch.py` now watches that
repo's commit sha and files an issue when it moves; do not flip the entry
on the strength of a changelog, re-download and check `--scan`.

We also argued upstream that re-uploading one artifact is the narrow fix:
`gemma4` is still absent from `FORCE_ATTN_MODEL_CLASSES` in optimum-intel
2.1.0 **and** 2.2.0.dev0+dd4ed1a, so the export environment keeps deciding
and the next gemma4 export can land decomposed again.

## Phi-3.5-vision as a GPU VLM entry (2026-09-01)

Idea: `OpenVINO/Phi-3.5-vision-instruct-int4-ov` is small (2.2 GB), Intel
publishes it ready to run, and its text path is quick. It sat in issue #24's
"untested" table for weeks.

**Verdict (2026-09-01, corrected the same day): the model works. NoLlama
broke it.** An earlier version of this entry said the model "does no vision
on this runtime" and recommended rejecting it. That was wrong, and the
mistake is worth keeping visible because of how it happened -- see below.

**What actually fails:** every image request returns

```
Check '(prompt_id >= 0) && (prompt_id < vocab_size)' failed at
.../sampling/logit_transformers.hpp:412: input_ids token out of bounds
```

**The trigger is our default `repetition_penalty` of 1.05** [OBSERVED
2026-09-01, Arc 140V, driving `VLMPipeline` directly with no server in the
loop]:

| generation config | text | image |
|---|---|---|
| bare `GenerationConfig` | OK | **OK** |
| `repetition_penalty = 1.0` | OK | **OK** |
| `repetition_penalty = 1.05` | OK | **FAILS** |
| `presence_penalty = 0.5` | OK | OK |
| `frequency_penalty = 0.5` | OK | OK |

Only the repetition-penalty transformer walks the *prompt* ids; presence and
frequency penalties score generated tokens. Phi-3 vision puts image
placeholders outside `[0, vocab_size)`, and that bound check rejects them.
Text prompts contain no placeholders, which is why text always worked.

The model itself is fine: a bare `VLMPipeline` reads a real screenshot
correctly at every size from 336x336 to 2048x2048, and describes synthetic
images correctly.

**How the wrong verdict happened, because the next person will repeat it.**
Four negative results were collected -- not multi-image, not the caching
path, not the runtime version (2026.3 release *and* 2026.5 nightly), not the
hardware (140T and 140V) -- and each one genuinely narrowed the problem. But
every single test ran *through NoLlama*, so the one variable never varied
was NoLlama itself. Ruling out four things you thought of is not the same as
ruling out the thing you did not. The bug was found in the first ten minutes
of driving `VLMPipeline` directly, which is what the standalone repro in
`scripts/phi35v-repro/` exists to force.

**Still to do:** decide the fix. Options considered, none implemented yet --
skip the repetition penalty when a request carries images (penalises every
VLM for one model's convention); catch this assertion and retry once with
the penalty off, remembering per slot (general, self-healing, costs a
re-prefill once); or key it to the `phi3_v` architecture (narrow and
hardcoded, with precedent in `NEEDS_OPTIMUM`). Do not simply lower the
global default -- 1.05 is a deliberate compromise, documented at
`REPETITION_PENALTY`.

Worth an upstream report too: a repetition penalty over a VLM prompt should
skip placeholder ids rather than assert.

## Pointing every Gemma download at our own re-exports (2026-09-01)

Idea: we already ship `aweussom/gemma-4-E4B-it-int8-ov` because Intel's is
defective. Do the same for the rest of the family and stop depending on
Intel's export quality at all.

**Verdict:** no. Fix the artifact that is broken, not the family.

**Why not:**

- **Only one of three is defective.** E2B carries 35 fused SDPA ops and
  26b-a4b carries 30, one per layer, and both cache correctly. There is
  nothing to fix on either.
- **26b-a4b is INT4-AWQ.** Re-exporting it is not re-running the exporter,
  it is reproducing a calibrated quantization. Getting that subtly wrong
  trades a real quality regression for a caching fix that was not needed.
- **Intel's E4B is ~2.2x faster on a cold turn**, which is why the entry
  above deliberately keeps both and picks by workload. A blanket "ours"
  policy deletes a measured choice.
- **Every DIY Gemma export re-arms the chat-template trap** (the
  `raise_exception` implicit-concatenation failure above). That footgun is
  Gemma-specific and recurs on every re-export, forever.
- Upstream has accepted the E4B defect, so the one reason we self-host is
  scheduled to disappear.

The standing policy is unchanged and is the right one: **host our own only
when the published artifact is defective or does not exist.** The real gap
was never hosting, it was detection -- closed by the `Prefix caching` line
in `--scan` (2026-09-01).

## Warning when SDPA op count != layer count (2026-09-01)

Idea: `--scan` knows the fused SDPA count and the layer count, so flag any
IR where they disagree -- a cheap early warning for a half-broken export.

**Verdict:** rejected, built and removed the same day.

**Why not:** it fires on healthy models. Attention type per layer is not
uniform, and only *linear* attention omits the node [OBSERVED 2026-09-01,
local IRs]: Qwen3.5-4B has 8 ops for 32 layers (`full_attention_interval`
4), LFM2.5-1.2B has 6 for 16, and Muse Glimmer has all 52 because its 39
`sliding_attention` layers still fuse. The first version of this warning
condemned Glimmer, a model we ship and know works.

Refining it to "expected = layers minus linear_attention layers" fits all
four observed families, but on two architectures' worth of evidence it is a
guess dressed as a check, and a checker that cries wolf on a good model
teaches people to ignore the line that matters. **The predicate is `> 0`.**
`--scan` reports the count so a human can judge; only zero is a verdict.

Re-evaluate if: a real export defect turns up that has a *non-zero* op count
-- that is the only case this would have caught.

## Untracking the model-watch state file (2026-08-18 -> reverted 2026-08-21)

Idea: `scripts/seen_models.json` is `model_watch.py`'s state. It churned on
every branch and showed up in four merge diffs for no reason, so cc387fc
gitignored and untracked it, leaving the file on disk.

**Verdict:** reverted. The GitHub Action cannot work without it tracked. Do
not untrack it again.

**Why not:**
- `.github/workflows/model-watch.yml` finishes with `git add
  scripts/seen_models.json` + `git push`. `git add` on an ignored, untracked
  path is a **fatal error** ("The following paths are ignored by one of your
  .gitignore files"), so the last step fails and every run ends red.
- Worse, the run is already useless before it goes red. A CI checkout has no
  state file, so `model_watch.py` reads nothing, falls into `seen = set()`,
  and takes the `baseline` branch: it prints "Baseline established", returns
  0, and **never sets `new=true`**. No issue is ever opened again. The
  notifier silently stops notifying, and the only visible symptom is a red X
  on a `git` step that reads like a permissions hiccup.
- The state has to survive a week between scheduled runs. `actions/cache` is
  the obvious alternative, but a cache miss reproduces exactly the
  silent-baseline failure above — trading a visible cosmetic problem for an
  invisible correctness one.

The churn complaint was real but cosmetic, and has a proper fix:
`.gitattributes` marks the file `linguist-generated=true`, which collapses it
in GitHub diffs while keeping it tracked.

Caught before it bit. The untrack landed Tuesday 2026-08-18, after that
Monday's run (the one that opened #29), so the next scheduled run — Monday
2026-08-24 — would have been the first broken one. Restored snapshot is
byte-identical to 500a15c, the last state CI committed, so nothing gets
re-reported.

## GigaChat-20B-A3B conversion (2026-08-13, issue #27)

Idea: Dmitriy Teteruk tried converting `ai-sage/GigaChat-20B-A3B-base`
(Sber's DeepSeek-MoE-based 20B-A3B). On paper a good NoLlama shape — MoE,
~3B active, would suit the offload path and the agent workload.

**Verdict:** unconvertible today. He abandoned it and closed the issue.
Don't recommend it or spend a download until upstream moves (conditions
below).

**Why not:**
- The exporter is NOT the wall: optimum-intel registers `deepseek`
  (`DeepseekOpenVINOConfig`, model_configs.py:4239 in the venv's 1.27
  stack), alongside `deepseek_v2`/`deepseek_v3`.
- The wall is the model repo's remote code. GigaChat ships its own
  `modelling_deepseek.py` (trust-remote-code), which imports `LossKwargs`
  from `transformers.utils` — removed in modern transformers. 4.55 still
  failed for him; pinning 4.53.3 let the conversion start.
- At 4.53.3 the stack then failed anyway — his final report: the model
  "used old transformers version that is not compatible with openvino".
  The exact second error was never captured, so we don't know whether that
  wall is optimum/nncf versioning or something deeper.
- Net: remote code needs transformers ≤4.53, the export stack needs newer,
  and no overlap window was found. (Reading note: his closing comment
  "Finally, it does now work" is missing a *not* — the rest of the
  sentence and the issue closure make the meaning unambiguous.)

Re-evaluate if: (a) Sber updates the repo's remote code for current
transformers — the `LossKwargs` import is the visible blocker, and a hand
shim (`class LossKwargs(TypedDict, total=False)` in a patched local copy)
might bridge it for a determined attempt, unverified; (b) transformers
gains a native implementation of the V1 `deepseek` MoE architecture so
remote code isn't needed at all; (c) anyone captures the actual
4.53.3-era failure, which would tell us what the second wall really is.

## Glimmer (optimum backend) on Intel iGPUs (2026-08-13)

Idea: serve Muse-Glimmer-30B int4 on the iGPU instead of CPU — the 140V
warmed up in 2.9s and streamed at ~2.8 tok/s, looked like a win.

**RESOLVED 2026-08-15 — fixed in OpenVINO 2026.4.** On
`2026.4.0.dev20260814` the issue's own repro quotes the prompt verbatim on
the B60 GPU and decodes at 8-9 tok/s (vs ~2 while corrupt, and 1.0 on the CPU
control). The verdict below stands for **2026.3 and earlier**, which is still
what a `pip install openvino` gives you today — `nollama.py` therefore keys
its GPU warning on the runtime version, not the device. Everything after this
paragraph is the pre-fix evidence; keep it, because it is what that version
check is protecting users from, and because the *method* (same-venv CPU
control) is what made the verdict trustworthy in both directions.

**Verdict (OpenVINO <= 2026.3):** don't, on **any Intel GPU — integrated or
discrete** — until a new OpenVINO GPU plugin passes the comprehension test
below. Xe-LPG fails
loudly at warmup (`Count is called for dynamic shape`). Xe2 (Arc 140V,
OpenVINO 2026.3, Windows) is the trap: it runs and *looks* healthy, but
inference is numerically wrong — and a community report (issue #24,
2026-08-13) reproduced the identical fingerprint on **Xe3** (Arc B390 iGPU,
Panther Lake, **Fedora**), so this is the GPU plugin's handling of the
graph, not one generation's numerics or one OS's driver. The model
half-perceives the prompt (asked `Respond only with the text "HELLO!"`, its
think channel quoted the user as saying "Respond only text") and greedy
decoding degenerates into a two-word loop inside the think channel that
never ends. The identical IR with identical sampling on CPU quotes the
instruction verbatim and complies exactly. Diagnosed 2026-08-13 after three
red herrings (think-block history round-trip, stale failed sends in web-UI
history — both real bugs, both fixed, neither the cause).

**Discrete Battlemage settled it (2026-08-15).** Arc Pro B60 24 GB, Windows,
OpenVINO 2026.3: same corruption. The shared-memory hypothesis the iGPU-only
evidence had suggested is dead — dedicated VRAM behaves identically, so four
device classes across two OSes now share one fingerprint. Two details worth
keeping:

- **It is a runaway generation, not a hang.** GPU sat near 100% throughout;
  the loop would have run to `max_tokens` (16384 from the web UI) and
  returned garbage after an hour. Cancel works, because the streamer is
  yielding — this is *not* the uninterruptible-native-code case. Cap
  `max_tokens` when testing so a corrupt run ends in seconds.
- **Restating the system prompt in the think channel is normal**, on CPU
  too — so not every mention of it is a symptom. On the B60 the think
  channel read `Respond directly.` where the real system prompt is 15
  words; the CPU control produced the full text. That is the *dropped
  words* signature, not hallucination. Keep it distinct from the Xe2
  observation of a wholly **fake** system prompt ("You are an expert in
  competitive programming…", never sent) — that one is real hallucination
  and still stands. Two different symptoms; don't merge them.

**Always run the CPU control on the same box, same venv, same session.**
`install-optimum.ps1` tracks transformers `main`, so the stack moves between
test runs. Without the control, "the GPU plugin is broken" and "transformers
regressed since the last test" fit the evidence equally well, and you would
file the wrong bug upstream.

**Comprehension test** (cheap, definitive): multi-turn chat, ask
`Respond only with the text "HELLO!"`, expand the thinking. If the model
can't quote the instruction back, the plugin is corrupting inference —
no error is raised anywhere.

Scope note: this is the **optimum backend** only. The GenAI path is fine on
the same hardware — Qwen3.8-27B runs correctly on that same B60 (2026-08-15).
Don't cite a GenAI-path success as evidence about this bug, or vice versa.

## Port-availability check via bind() probe (2026-05 -> 2026-08-11)

What we had: check_port() tried bind(("0.0.0.0", port)) and treated success
as "free". Shipped that way from the start.

**Verdict:** replaced with a connect-test (does anything ACCEPT on loopback
or any hostname-resolved local IPv4 address?). Never use a bind probe for
"is somebody serving here" on Windows.

**Why not:** Windows treats a specific-address binding and a wildcard bind
as distinct — bind("0.0.0.0", 11434) SUCCEEDS while real Ollama holds
127.0.0.1:11434, and Flask then double-binds the same way. Two servers on
one port: localhost clients reach Ollama (most-specific binding wins), LAN
clients reach NoLlama. Found live on the fresh-Ryzen test box (2026-08-11);
reproduced in isolation the same day — a loopback-only listener + a
successful wildcard bind on the same port, same machine. On Linux the same
bind fails EADDRINUSE, which is why it looked correct for months.
_identify_ollama() now also names the incumbent in the warning.

This does not replace exclusivity on the real server socket: NoLlama's
Werkzeug listeners use `SO_EXCLUSIVEADDRUSE` on Windows so a process started
later cannot claim a more-specific address on the same port. The connect-test
exists only to identify an incumbent cleanly before startup.

---

## Meta Muse Glimmer 30B for NoLlama (2026-08-10)

Idea: Meta released Muse Glimmer on HF the day it landed here — Apache 2.0,
agentic-first (tool calling, multi-step reasoning), multimodal (interleaved
text+image), explicitly pitched as "runs locally on consumer hardware". On
paper that is exactly NoLlama's story: GPU/CPU tool calling + VLM routing.

**Verdict:** don't. Not a measurement — arithmetic, from the model card. The
disqualifier is one architectural fact: it is **dense**, and dense removes
the only lever that makes a 30B-class model usable on this hardware.

**Shape:** ~29.6B total = **dense** 28B text decoder + ~1.8B ViT perception
encoder. 52 layers, hidden 6656, GQA 32 q / 2 kv (head_dim 208), interleaved
local(2048 window)/global attention, RoPE θ=500k, 128k context, vocab 202k.
Meta's own floor is a **24 GB VRAM envelope** at 4-bit.

**Why not:**
- **Dense ⇒ `--offload-ratio` is inapplicable.** OFFLOAD_RATIO streams *MoE
  expert weights*; there are no experts. The thing that made Qwen3-30B-A3B
  interactive on the 140V (ratio 30 → 10.8 GB resident @ 25.3 tok/s) simply
  does not exist here. Everything must be resident, with no knob to claw back.
- **Doesn't fit the 140V.** int4 decoder ≈ 15 GB + vision tower (OV VLM
  exports keep the encoder at higher precision, ~3.5 GB) + KV (~86 KB/token
  at full attention; less if OV ever implements the 3-in-4 sliding window,
  which a brand-new arch won't get on day one) against ~16.5 GB usable.
  Over budget before the first token.
- **Dense 28B is ~6× the compute of a peer we already call slow.**
  `gemma-4-26b-a4b-it-int4` (4B active) does 21.0 tok/s steady-state on the
  285K CPU. Same total size, dense, touches all 28B per token → expect ~3.
  The desktop iGPU fits it by memory (33 GB shared) but has no XMX and is
  bandwidth-bound; prefill on a real agent prompt would be worse than the
  ~6 min TTFT already logged there.
- **NPU: out entirely.** 30B dense, and the NPU path caps at MAX_PROMPT_LEN
  4096 regardless.
- **No exporter.** New `AutoModelForMultimodalLM` architecture; optimum-intel
  2.1.0 in `venv-2026.3` predates it. Blocked upstream even if the numbers
  were good — and per #19 that conversion would be RAM-bound anyway.

**Where it does belong today:** the RTX 5090 (32 GB), outside NoLlama's
Intel/OpenVINO remit. Ollama shipped an `-mlx` build at launch and says the
CUDA build follows "in the following days" — that is the path for this model
on this desk, and it costs us nothing.

**But the verdict inverts on an Arc Pro B70 (32 GB) — the exporter is then
the only blocker.** Xe2-HPG, 256 XMX engines, **32 GB dedicated GDDR6 at
608 GB/s**. int4 decoder ~15 GB + vision tower ~3.5 GB ≈ 18-19 GB resident,
~13 GB left for KV. Bandwidth ceiling 608/15 ≈ 40 tok/s; at the 40-60% of
peak these reach in practice, **~15-25 tok/s — interactive**. On that card
the missing offload lever is irrelevant: nothing needs to stream.

**On the B60 (24 GB, 456 GB/s, arriving 2026-08-10 week) it fits — but only
for agent/chat, not whole-book.** Weight budget swings on how the vision
tower is exported: decoder int4 ≈ 14.5 GB (≈16 if embeddings/lm_head stay
int8 — vocab is 202k × 6656, so those two tensors are ~2.7B params on their
own), vision 1.8B at int8 ≈ 1.8 GB / at fp16 ≈ 3.6 GB. So **16.3 GB best
case, 19.6 GB worst**, leaving 4-7 GB of KV. At ~84.5 KB/token (52 layers ×
2 kv heads × 208 head_dim × 2 × fp16) that is **~40-75k tokens** — plenty for
a 21k agent prompt, and the reason to export the vision tower at int8.
Meta's own figure is 55 GB bf16 → **18-20 GB at 4-bit**, i.e. the worst case
above — so budget ~4 GB of KV, ~45k tokens. That is no longer a constraint
worth worrying about: secondreader retired whole-book prompting entirely on
2026-08-09 (depth collapse is architectural — every ≤35B model goes thin and
loses anchors at 120k, while the same models scoped per-chapter produce 20×
the richness), so the workload this card serves is **scoped chapter calls of
5-15k tokens**, not 113k. 45k tokens is ample.

**And even with a perfect exporter it is the wrong model class for the
surviving workload.** Scoped means *many serial calls* (one per chapter), so
throughput is the figure of merit — and the two local models that actually
deliver there are **MoE**: gemma4-26b-**a4b** (11,553 words, 901 anchors, 0
bad) and qwen3.6-35b-**a3b** (the only ≤35B arm to hold [ChN:M] perfectly at
122k). Both buy 26-35B quality at 3-4B active cost. A dense 28B pays full
28B compute on every token of every chapter call. Glimmer would be a *peer*
of gemma4-26b-a4b in capability and several times its cost per artifact.

**The exporter is the whole blocker for now, and it is proven, not suspected
(checked 2026-08-10, release day):**
- `config.json` declares `model_type: "muse_glimmer"`,
  `MuseGlimmerForConditionalGeneration`, nested `muse_glimmer_text` /
  `muse_glimmer_vision`. **No `muse_glimmer` registration exists in
  optimum-intel `main`** (`optimum/exporters/openvino/model_configs.py`;
  newest multimodal entries are gemma4_unified, gemma3n, qwen3_omnimoe).
  `optimum-cli export openvino` fails at config lookup — no flag routes around
  a missing exporter config.
- It requires **transformers 5.15.0.dev0**; `venv-2026.3` is pinned to **5.4**
  by the LFM2 exporter's cap. That is a third venv, not an upgrade.
- No `OpenVINO/Muse-Glimmer-*-int4-ov` pre-convert exists yet. Intel shipped
  the Qwen3-VL pre-export within weeks, so **waiting most likely obtains the
  export for free** and skips a RAM-bound 55 GB conversion. Do not spend the
  download until either the pre-convert appears or `muse_glimmer` lands in
  optimum-intel.

**Capacity is the wrong axis for a dense model — don't reach for the
big-RAM machine.** Dmitriy's Arc 140T (285H, 128 GB RAM, 110 GB shared-memory
override) has 2× this desk's RAM and still loses: his ratio-0 Qwen3-Coder-Next
int8 does 9.1 tok/s on an 80B-**A3B** (~3B active), implying ~45 GB/s
effective for weight reads. Dense 28B int4 reads ~15 GB *per token* on that
path → **~2-3 tok/s**. The rule: **MoE is capacity-bound** (huge shared
memory is the fix — it's why he can run a 74 GB model at all); **dense is
bandwidth-bound** (only dedicated VRAM is the fix). Adding host RAM to an
iGPU does nothing for Glimmer.

**Decision 2026-08-11: let Intel do the heavy lifting.** The DIY path is a
third venv on transformers >=5.15 (5.4 has no `muse_glimmer` module at all —
and Meta shipped no `modeling_*.py`, so `trust_remote_code` is not a door
either), plus a hand-written `MuseGlimmerOpenVINOConfig`, plus probable
`VLMPipeline` per-architecture work on the C++ side. Not worth it for a model
that would at best tie `gemma4-26b-a4b`. Weights not downloaded — nothing is
staged locally, and there is no reason to stage it until an export exists.

**The gate is a quality measurement, not the hardware and not the export.**
Ollama's CUDA build lands within days; run Glimmer through `facts-scoped` on
the 5090 against oldgods with the terra ruler — a *model* question, fully
separable from OpenVINO plumbing, answerable in an afternoon with the harness
that already exists. The row to beat is `gemma4:26b` scoped: **11,553 words,
901 anchors (0 bad), 54 quotes / 2 flagged, both nine-minute facts + late
fifties**. If Glimmer does not clear that, the OpenVINO question never needs
asking. Only if it clears it decisively does the export matter — and by then
Intel has probably shipped `OpenVINO/Muse-Glimmer-*-int4-ov` anyway.

Re-evaluate if: (a) Glimmer beats gemma4-26b-a4b scoped on the 5090 **and** an
OpenVINO export exists; (b) a smaller or **MoE** Glimmer variant ships, which
would restore the offload lever, suit the many-serial-calls shape, and make
even 16 GB XMX viable. Do **not** re-evaluate on the grounds that some machine
has more system RAM, more pagefile, or faster internet — none of the three
walls (transformers implementation, optimum-intel exporter, genai VLM arch) is
resource-bound.

**Update 2026-08-11:** On hold *until Intel ships a pre-exported OpenVINO
variant* -- no self-export attempts before that, even though the incoming
B60 (24 GB) matches Meta's stated 4-bit envelope. The gate is support, not
hardware. The model-watch bot tracks the OpenVINO org, so the gate opening
files its own issue; nothing to poll.

**Update 2026-08-13 — two of the three walls fell within 48 h; decision
superseded by events.** optimum-intel merged `muse_glimmer` support the
evening the entry above was written (PR #1924, 2026-08-11); we exported
int4 on the 128 GB workstation the next day and published it
(`aweussom/Muse-Glimmer-30B-int4-ov`, ~17 GB) — the export cost turned out
to be one lounge afternoon, not a project. transformers wall: solved by the
model-lab venv (git-main stack, `scripts/glimmer-export/`). The third wall
(genai VLM arch) was *bypassed*, not climbed: the `optimum-backend` branch
serves `muse_glimmer`/`nemotron_h` through optimum-intel's python runtime
(`OptimumSlot`), text-only, tools working. What this does NOT supersede:
the model-class analysis above. Dense 28B is still the wrong shape for the
desktop's scoped-chapter workload, and the quality gate (beat
`gemma4-26b-a4b` scoped on the 5090) still stands for *that* use. The slot
exists because (a) it's one implementation serving two models — Nemotron
3.5 Lightning is 30B-**A3B** MoE, which fits the throughput argument
perfectly (optimum-intel `nemotron_h` export merged 2026-08-12, PR #1789) —
and (b) OpenClaw/agent use on owned hardware is a different workload than
scoped book runs. Ollama-side quality signal so far: Glimmer subjectively
best-in-class on secondreader (5090, 2026-08-12), formal facts-scoped run
still pending.

## Whole-book (100k+) prompts on CPU serving (2026-08-09)

Idea: serve secondreader's whole-novel prompts (~113k tokens) from the 285K
CPU slot — decode is fine there (25 tok/s short-context), 64 GB RAM holds
weights + a 12 GB KV pool, and the prefix cache makes repeat artifacts cheap.

**Verdict:** don't. CPU prefill is the wall, and the client-retry dynamics
around it are actively destructive. Whole-book serving waits for XMX
(140V/B60 — protocol in `docs/LAPTOP-140V-BOOKRUN.md`).

**Why not (Qwen3-30B-A3B-Instruct-2507 int4, 285K CPU, genai 2026.1):**
- Cold prefill measured ~45 tok/s at 10.6k tokens (TTFT 234s) and
  superlinear beyond: 112,753 tokens produced NO first token in 90 minutes.
  Decode with 10k context: ~6 tok/s (not the 25 of the short-context bench).
- The client's timeout+retry then created a death spiral: identical requests
  at exactly timeout-interval (5400s ×3 observed), each entering the CB
  engine while the previous still ran — OpenVINO cannot cancel, a dead
  socket does not stop a sequence. Three ~113k sequences in a ~131k-token
  pool = permanent preemption, zero completions in 3h20m. Any client of an
  uncancellable backend must set timeout > worst-case total and attempts=1.
- Sizing rule that was missed: the KV pool must hold prompt + max_tokens
  (113k + 32k = 145k > the 12 GB pool's 131k), so even a single request can
  evict its own prefix during generation.

Flow itself is fine — the same stack completed a 2-chapter book end-to-end
(819s total, artifact + clean citation check). The failure is CPU prefill
compute at book scale, not the pipeline.

Re-evaluate if: OpenVINO's CPU plugin gains a dramatically faster prefill
path (AMX-heavy), or a future NoLlama gains chunked-prefill progress
reporting + duplicate-request rejection, which would at least defang the
retry spiral.

**Update 2026-08-10 — don't re-evaluate: the workload is gone.** The entry
above says whole-book serving "waits for XMX (140V/B60)". It no longer waits;
secondreader retired whole-book prompting outright on branch `scoped-facts`,
on **quality** grounds that no hardware fixes. The runtime confound was closed
in both directions: the identical 113k-token payload ran on the 5090 at full
speed (prefill 2,100 tok/s, whole artifact in 4 min) and still collapsed —
2,707 words, 74 anchors all in the wrong format, zero precision probes — while
the 140V/OpenVINO run of the same model landed statistically identical (2,753
words, zero anchors, 0/3 probes) after 4h26m. Four local families ≤35B all
collapse at 120k depth. **The depth collapse is the model, not the runtime and
not the device.**

What replaced it: **per-chapter scoped calls in a ~10-30k envelope**, merged by
code. Same gemma4:26b that produced 1,267 words whole-book produces 11,553
words with **901 anchors, 0 bad** scoped, and recovers precision facts every
whole-book local arm missed. So for NoLlama the serving profile inverts:
- KV pool: `--cache-size-gb 12` was a whole-book requirement. Scoped needs
  ~1-3 GB. The 15+ GB advice in secondreader's `models.ini` block applies only
  to the retired path.
- The **prefix cache matters far more now, not less**: scoped runs issue one
  call per chapter sharing a fixed instruction preamble, so `--idle-timeout 0`
  + prewarm turns every chapter after the first into a warm-prefix hit. Whole-
  book had one giant prefix reused across few calls; scoped has a small prefix
  reused across dozens.
- Uncancellable-backend hygiene still stands (attempts=1, generous timeout) —
  but at 10-30k the death spiral is far less reachable.

## Gemma 4 on the NPU (2026-08-07)

Idea: Gemma 4 launched this week; the E-series (E2B/E4B) are edge-sized
multimodal models with Intel pre-exports — natural NPU candidates, and the
blog post promised we'd test them.

**Verdict:** no Gemma 4 on the NPU for now, on any precision. CPU (and
presumably XMX GPU) is the way to run them.

**Why not (285K NPU, driver 32.0.100.4778, genai 2026.3):**
- `gemma-4-E4B-it-int8-ov` (Intel's own export) **compiles** for NPU
  (103 s) but generates **garbage at 0.5 tok/s** — multilingual token
  salad, three identical runs. The same file on CPU: 13.0 tok/s, perfectly
  coherent. Export is sound; the NPU path is numerically broken for
  `Gemma4ForConditionalGeneration`.
- int4 variants are already documented (zenn.dev, 2026-08) to crash the
  vpux compiler with the duplicated-names bug; we did not re-prove that.
- Both failure modes differ from the LFM int8 traps (fast-garbage /
  slow-correct) — this is slow-AND-garbage, a distinct NPU-path defect.

The models themselves are good: `gemma-4-26b-a4b-it-int4-ov` (VLM MoE,
128 experts) does **21.0 tok/s steady-state on the 285K CPU**, coherent,
16 s load. E4B int8 does 13.0 on CPU. Gemma 4 belongs in the CPU/GPU
columns, not the NPU column.

**Update 2026-08-07 (VLM + offload premiere, same 140V):**
`gemma-4-26b-a4b` served through NoLlama with `--offload-ratio 30` works —
7.9 tok/s on the cold first request, 12.5 on the second (LRU warming),
vs 26.6 resident. So VLMPipeline forwards OFFLOAD_RATIO and it engages;
the desktop 35B failures were XMX-only after all. Side-finding: VLM slots
use the PLAIN pipeline, and two sequential generates with offload active
did NOT hang — the second-generate hang is therefore LLMPipeline-specific,
not plain-pipeline-general. Vision also verified under offload: an image
request (XKCD strip) answered correctly at 8.9 tok/s — the vision encoder
is not expert weights, stays resident, works. (Entry below is Gemma-on-NPU.)

**Update 2026-08-07 (laptop NPU4, Arc 140V machine):** same E4B int8 on
the newer NPU generation produces **coherent** output — but at
**0.1 tok/s** (8 minutes per answer, three consistent runs). So two
separate defects: the numerical garbage is specific to the older NPU
arch/driver (3720 wrong, NPU4 right), while the speed is broken on BOTH
generations (0.1-0.5 tok/s smells like most of the graph falling back off
the NPU via NPUW partitioning). Verdict unchanged — no Gemma 4 on any NPU
we own — but the upstream report can now be precise: wrong-on-3720,
~100x-too-slow-everywhere. For comparison the same file does 16.4 tok/s
on the same laptop's GPU and 13.0 on the desktop CPU.

Re-evaluate if: an NPU driver or openvino release notes gemma4 fixes —
retest is `scripts/vlm-bench.py`, three minutes; or Intel ships a
`-int4-cw-ov` build of a gemma-4 (none exist today, unlike gemma-3).

## OFFLOAD_RATIO (2026.3 MoE disk offload) on the desktop 285K iGPU (2026-08-06)

Idea: OpenVINO 2026.3's MoE disk offload ("30B on 16 GB of memory") should
let big MoE models (Qwen3.6-35B-A3B, Qwen3-30B-A3B, and Dmitriy's 74 GB
Qwen3-Coder-Next from #19) run on this 33 GB-shared-memory iGPU.

**Verdict:** could not be made to work on this machine, on ANY model, at ANY
ratio, after a full day of controlled experiments. Do not recommend it to
users (incl. #19) as more than "exists upstream, unverified by us."

**What was measured (genai 2026.3.0, iGPU shared mem 33 GB, 64 GB RAM,
141 GB pagefile):**
- Qwen3.6-35B-A3B int4 VLM (2026.2 export): USM **Device** OOM (512 MB
  alloc) at ratio absent/40/90 — identical failure, ~9.5 min in. Compiling
  its language model directly with the property (no VLM wrapper) fails the
  same, so it is not a property-forwarding problem.
- Qwen3-30B-A3B-int4-ov, Intel pre-convert (2026.0 export): ratio 0 → USM
  **Host** OOM (384 MB); ratio 90 → USM Device OOM, one minute later and
  after staging ~120 GB of host commit. The offload machinery clearly
  *engages* — and still fails.
- LFM2-24B-A2B-int4-ov, Intel pre-convert (2026.2 export, **11.6 GB** —
  fits the 33 GB pool three times over): ratio 0 AND 90 → USM Host OOM
  (384 MB). An 11.6 GB model failing a 33 GB device on load is the smoking
  gun: the failure is in the GPU plugin's **weight-staging phase**, before
  any device-residency savings from offload can apply.
- Control that the pool itself works: Qwen3-8B int4 (~5 GB) and
  Qwen2.5-Coder-14B (~8 GB) load and generate fine on this iGPU. The
  practical ceiling on this box sits between ~8 and ~11.6 GB for MoE IRs.

**ROOT CAUSE (definitive, from source + device query):** the entire MoE
fusion path is gated in `transformations_pipeline.cpp`:

```cpp
// Gated on supports_immad (systolic-only) and oneDNN (required for expert GEMM dispatch).
if (device_info.supports_immad && config.get_use_onednn() && !config.get_moe_disable_fusion())
```

`supports_immad` = XMX/DPAS systolic hardware. The desktop 285K's Xe-LPG
iGPU has none (`OPTIMIZATION_CAPABILITIES` lists no `GPU_HW_MATMUL`;
verified 2026-08-06). No XMX → no TiledMoeBlock→MOECompressed fusion →
`OFFLOAD_RATIO` is a **silent no-op**, and experts stay as giant plain
constants — which is also why big-MoE loads OOM in staging on this device.
Proven end-to-end on a fusable IR: LFM2-8B-A1B exported fresh with the
2026.3 stack (tiled `u4 [32,1792,16,128]` expert constants confirmed in
the XML) loads fine and shows byte-identical device memory (14.91 GB) and
identical tok/s at ratio 0 and 90.

Intel's demos run on XMX-capable GPUs (Lunar Lake Arc 140V, Panther Lake,
Arc dGPUs). The release notes never mention the hardware gate.

**Consequence:** MoE disk offload is a hardware capability, not a software
setting, on this box. Raising the pagefile to re-export Qwen3-30B-A3B is
pointless *for offload on this machine* (the export itself would still be
useful only on an XMX-capable device). The Arc 140V laptop (original
NoLlama dev machine) HAS XMX — that is the machine to validate offload on.

Re-evaluate if: (a) testing on an XMX GPU (Arc 140V laptop / any Arc dGPU)
— use a fresh-stack export, ratio 0 vs 90, `GPU_MEMORY_STATISTICS`;
(b) Intel lifts the immad gate for non-systolic GPUs in a future release
(watch `transformations_pipeline.cpp`); (c) recommending it to anyone —
ask for their GPU model first, `OPTIMIZATION_CAPABILITIES` containing
`GPU_HW_MATMUL` is the tell.

**Update 2026-08-06 (same evening):** condition (a) tested on the Arc 140V
laptop (Core Ultra 7 258V, XMX confirmed) — **offload works exactly as
advertised there**. LFM2-8B-A1B int4: 4.10 GB resident at ratio 0 →
0.70 GB at ratio 90 (−83%), SSD streaming visible. Qwen3-30B-A3B int4
(15.2 GB weights, Intel's 2026.0 pre-convert — so old IRs DO fuse on XMX;
the tiled layout was never the blocker): loads and generates at ratio 90
with **2.35 GB resident**, 2.5 tok/s (ratio was oversized; tuning the knee
is follow-up). The verdict above is thus purely about non-XMX hardware —
the feature itself is real, first reproduction outside Intel we know of.
NoLlama grew `--offload-ratio` the same evening, with a startup warning on
non-XMX GPUs. install.ps1 surfaces XMX at device detection.

**Update 2026-08-09 (whole-book prefill inverts the offload economics):**
140V, Qwen3-30B-A3B-Instruct-2507 int4, 112.7k-token prompt, ratio 90,
13 GB KV pool (15.78/16.5 GB used — pool + weights co-load fine): completed
coherently but TTFT was **2h25m** and post-TTFT decode **0.38 tok/s** —
total 4h26m for one artifact. Mechanism (from the laptop run's analysis):
**prefill activates EVERY expert**, so at high ratios each prefill chunk
re-uploads essentially the whole offloaded expert set — ~7 TB of logical
reads over the run (served from the Windows file cache; the SSD idled while
the GPU compute engine ran 95%). The RATIO, not the token count, is the
multiplier: ratio-30 steady-state decode numbers (25.3 tok/s) say nothing
about prefill-heavy workloads at ratio 90. Sizing rule: pick the ratio for
the WORKLOAD — chat/agent (decode-heavy) tolerates high ratios; long-prompt
work wants the lowest ratio that fits, because prefill pays the streaming
tax per chunk. A 16 GB XMX AI-PC is thus an *overnight* whole-book appliance,
not an interactive one; the B60's 24 GB fits the same job at ratio ≤50
(possibly 0), which removes the multiplier — measure next week.

**Update 2026-08-09 (field data, issue #19: ratio doesn't change decode speed
when the model dwarfs the device):** Dmitriy Teteruk's Arc 140T (285H laptop,
128 GB RAM), Qwen3-Coder-Next int8 (74 GB, 80B-A3B): ratio 30 AND 60 both
decode at **3.7 tok/s** steady-state. His memory lines explain it: with any
nonzero ratio, `usm_device` stays constant at 2.21 GB (attention/non-expert
weights only) and the *retained* experts land in `usm_host` (50.5 GB at 30,
28.8 GB at 60, 7.2 GB at his earlier 90) — and with 128 GB RAM the offloaded
experts' file pages all live in the OS page cache anyway. So "retained" and
"offloaded" experts are read over the same host-memory path either way; the
ratio only changes how much host RAM is *pinned*, not the bandwidth
bottleneck. Consequence: the "smallest ratio that fits" knee we measured on
the 140V (where retained experts occupy device memory: 10.8/8.1/2.35 GB at
30/50/90) applies when retained experts are device-resident. When the runtime
puts them in host USM — observed when weights vastly exceed the device
budget — a HIGHER ratio strictly dominates (same speed, less RAM pinned).
Also his ratio 0 run (74.4 GB `usm_device`, via the 110 GB shared-memory
override) does 21.3 tok/s — ~6× the offload path — so on huge-shared-memory
machines, offload is a RAM-saving knob, not a speed-neutral one. Don't hand
out the "smallest ratio" rule without asking where the retained experts land.

**Update 2026-08-10 (the "21.3 tok/s" ratio-0 baseline above is an artefact —
retract it, and with it the "6×" gap):** a re-run of the identical
configuration on the same machine (Arc 140T, Qwen3-Coder-Next int8, ratio 0,
74.42 GB `usm_device`, 140.1 s load) measured **9.1 tok/s** steady-state.
This *is* the real-token-counting bug, contrary to what was first said here
and in the issue thread: the 21.3 run was posted 2026-08-06 21:56, the fix
(df0340e) landed 2026-08-07 08:06. Two tells in the pasted log itself —
its memory lines lack the `(post-load)` suffix added by cb4215f the same
morning, and it reads `OK: 64 tokens in 3.0s -> 21.3 tok/s` / `OUTPUT:
Hello!`: the old code divided the *token budget* by wall-clock regardless
of how many tokens were actually produced, so a generation that hit EOS
after a couple of tokens reported ~3 s of mostly-prefill as 64 tokens of
decode. **9.1 tok/s is the representative figure**, and the
resident-vs-offload gap on that machine is ~2.5×, not 6×. Lesson for
reading any pasted benchmark: date the log against the script's git
history before theorising about power profiles — the log's own format is
the version stamp. Second reading hazard, same thread: `offload-test.py`
used to take the device positionally and default to `GPU`, so a request
for a CPU baseline came back looking like a perfectly valid GPU run. It now
accepts the device anywhere in the arguments and prints "(default …)" in
the banner when it wasn't given — but always check the banner line.

**Update 2026-08-07 (steady-state correction — the evening numbers above
were 2-5× too pessimistic):** the offload LRU needs ~60 tokens to warm,
and single-generate measurements reported cold-cache speed as the verdict.
Proper steady-state on the 140V, Qwen3-30B int4: ratio 30 → **25.3 tok/s**
(interactive — matches the 24-core desktop CPU running the same model
resident), 50 → 22.1, 90 → 5.1. Two benchmark bugs fixed the same morning:
warm-up contamination, and rate computed from ASSUMED token counts (a
4-token "Hello!" + EOS once reported 645 tok/s — real LFM2-8B GPU number
is 86.8). Also found: **a second generate() on an offload-active plain
pipeline hangs in native code, uninterruptible** (140V, 30B ratio 50) —
upstream-repro-worthy, and it means NoLlama's own `--offload-ratio` serving
path (which reuses one pipeline across requests, though via the CB backend,
not the plain pipeline) MUST be verified with two sequential chat requests
before recommending the flag in production.

## int8 exports of LFM2 / LFM2.5 for the NPU (2026-08-06)

Idea: channel-wise int4 is the lossiest int4 variant and the NPU forces it,
so ship int8 builds of the LFM models for quality-sensitive use — it worked
for SmolLM3-3B (int8-cw-sym: coherent, 12.3 tok/s vs int4-cw's 23.3 on the
285K NPU, a fair trade).

**Verdict:** no publishable int8 variant exists for LFM2-family on NPU.
int4-cw is the only good configuration. The SmolLM3 result does NOT
generalize.

**Why not (both variants measured on 285K NPU, genai 2026.3):**
- `--weight-format int8 --sym --group-size -1` (mirroring the int4-cw
  recipe): compiles and runs FAST (32-33 tok/s) but generates garbage —
  LFM2-1.2B emits whitespace, LFM2.5-1.2B-Instruct emits "BY-AL-AN-AN-…"
  loops. Silent numerical breakage, not a crash: the worst failure mode.
- `--weight-format int8` (asymmetric, Intel's own recipe — their
  LFM2.5-350M-int8-ov uses it): output is coherent but decode is
  **1.4 tok/s** (119 tokens in 89 s). Intel's own 350M reference runs
  4.5 tok/s the same way — asymmetric zero-points evidently fall off the
  NPU fast path. Correct but unusable.
- SmolLM3-3B int8-cw-sym is fine (fast AND coherent), so this is
  LFM-architecture-specific (its short-conv/linear-attention blocks),
  not a general int8-on-NPU rule.

Re-evaluate if: a newer NPU driver or openvino release changes either half
(retest is two 5-minute benches with scratchpad `npu_bench.py`-style
timing), or Intel publishes a fast LFM int8 NPU build — read its rt_info
for the recipe before assuming ours was wrong.

## LFM2 / LFM2.5 on the Lunar Lake NPU 4 (2026-08-30)

Idea: the two LFM builds are the fastest NPU models we publish (38.8 /
36.5 tok/s on the 285K), and Lunar Lake laptops are NoLlama's flagship
target — so offer them there too. The installer already did.

**Verdict:** don't recommend either LFM build on NPU 4 until the numerics
are fixed. They run at 46–48 tok/s and emit word salad. Registry notes and
`docs/MODELS.md` say "NPU 3 only".

**Why not (258V laptop, `DEVICE_ARCHITECTURE=4000`, driver 32.0.100.4778):**
- `aweussom/LFM2.5-1.2B-Instruct-int4-cw-ov`: `Say hello.` →
  `cohclclclcl…`, `What is 2+2?` → `an anthankankank…`, `capital of Norway`
  → `ablelelele…`. Deterministic — byte-identical across OpenVINO
  **2026.3.0, 2026.3.1 and the 2026.5.0 nightly**, with the **plugin
  compiler and the driver compiler** (`NPU_COMPILER_TYPE=DRIVER`) alike, and
  under every NPUW knob tried (`MAX_PROMPT_LEN` 1024/4096,
  `GENERATE_HINT=BEST_PERF`, `PREFILL_HINT=DYNAMIC`). Never emits EOS, so
  every request runs to `max_tokens`.
- It is the architecture, not our export: Intel's own
  `OpenVINO/LFM2.5-350M-int8-ov` is garbage on the same NPU
  (`Tong、/神可乱 … shove shove shove`), `aweussom/LFM2-1.2B-int4-cw-ov` is
  degenerate, and a fresh export with the 2026.5 toolchain (new IR) is
  garbage of a different flavour. The 350M file on **CPU and GPU in the same
  venv** answers `Hello! How can I help you today?` / `2 + 2 = 4` / `Oslo`.
- **Positive control:** the 285K (NPU 3, arch 3720), same driver 4778, same
  OpenVINO 2026.3.0, same two files — correct with both compilers. With
  software held constant, the NPU generation is the only variable left.
- Not the device: SmolLM3-3B-int8-cw and Qwen3-8B-int4-cw are coherent on
  the same NPU in the same session.
- Precedent, mirror image: Gemma 4 E4B int8 was garbage on NPU 3 and coherent
  on NPU 4 (entry above). Per-generation numerical bugs in the NPU stack for
  non-standard blocks (LFM2's short-conv layers here) are a pattern.
- Upstream already sees it: in openvinotoolkit/openvino#37322 an Intel
  engineer reproduced "LFM2.5-1.2B on NPU gives gibberish, CPU/GPU fine" on
  2026.3 / Lunar Lake (2026-08-13). The nightly that fixed the reporters'
  *2.6B new-IR* models does not fix ours; the 2026.3.1 notes' "fixed accuracy
  issue for lfm2-1.2b … NPU" did not hold on this NPU 4 either.
- The 2026.3.1 runtime **segfaults** (exit 139, reproducible) when handed an
  IR exported by the 2026.5 toolchain — separate hazard, same day: a newer
  IR is not a drop-in on an older runtime.

**Still open — the one axis not varied:** every failing run shared the
**4778 kernel driver/firmware**. Driver 32.0.100.5540 is installed
(pnputil, `oem43.inf`) but a reboot is needed before a process sees it
(`NPU_DRIVER_VERSION` still reads 1004778). Correct after the reboot →
4778 runtime bug on NPU 4, fix = "update your NPU driver", then bisect
down through the staged 4724/4512. Still garbage → hardware/firmware path,
report the whole matrix to #37322. Commands are at the top of
`NEXT-STEPS.md`; the full log is `C:\Users\tommyl\npu-driver-backup\FINDINGS.md`.

Re-evaluate if: the post-reboot probe is correct; an NPU driver or
OpenVINO release names LFM2 on NPU4000; or Intel publishes an LFM2 build
validated on Lunar Lake. Retest is `npu-probe.sh LFM2.5-1.2B-Instruct-int4-cw-ov`
(two minutes) plus the SmolLM3 control.

## Qwen3.6-35B-A3B (Qwen3.5-MoE arch) on the NPU (2026-08-06)

Idea: with OpenVINO 2026.3 passing regression, put the new Qwen3.6-35B-A3B
INT4 export on the NPU — NPU coverage is the stated priority of the 2026.3
move, and an A3B MoE (3B active) looks NPU-sized on paper.

**Verdict:** doesn't load. Not a memory problem — an architecture-vs-plugin
incompatibility. Serve this model on GPU/CPU only until the NPU plugin
catches up.

**Why not:**
- Both `VLMPipeline` and `LLMPipeline` on NPU fail in ~3 s at shape
  inference, before compile, with
  `Check '!dim::is_empty(minus_one_dim)' failed ...
  reshape_shape_inference.hpp:357` on node
  `__module.model.model.language_model/aten::index/Reshape`
  ("Non-'-1' output dimensions do not evenly divide the input dimensions").
  The NPU's static-shape import can't reshape a boolean-mask `aten::index`
  in the Qwen3_5Moe language model. genai 2026.3.0.0-3277, 285K NPU
  ("AI Boost"), driver as of 2026-08-06.
- It is *not* the earlier commit failure: that was fixed (141 GB pagefile,
  33 GB iGPU shared-memory override) and this failure reproduces identically
  with memory to spare. Don't respond to this error by adding RAM/pagefile.
- Nothing NoLlama can patch: the export is Intel-toolchain-fresh
  (OpenVINO 2026.2 export, optimum-intel 1.27.0.dev0) and the failure is in
  OpenVINO's NPU plugin shape inference, upstream of anything we configure
  (`MAX_PROMPT_LEN` etc. never comes into play).

Re-evaluate if: a later OpenVINO release notes NPU support for Qwen3.5-MoE /
`Qwen3_5MoeForConditionalGeneration` (retest is one `--scan`-verified dir +
a 3-second load attempt), or Intel publishes an NPU-targeted export of this
family.

## `--model-name` / `--model-description` override flags (2026-08-06)

Idea: let the user set the name shown in the web UI and reported as the
model ID, since renaming the model folder appeared to do nothing. Raised by
Dmitriy Teteruk (issue #19) after converting Qwen3-Coder-Next himself and
wanting it to show up as something sensible.

**Verdict:** don't add the flags. Fix the rename and add `--scan` instead.

**Why not:**
- **The bug was ours, not the interface's.** `model_display_name()` called
  `os.path.realpath()` unconditionally, so on a junction (which is what
  `install.ps1` creates) the name came from the *link target* and the user's
  rename was silently discarded. Renaming a directory is already the naming
  interface — it needed no documentation, no flag, and no knowledge. It just
  had to work. Now it does: the given name wins, and the link is only
  followed when the directory name is generic (`model/`, `gpu-model/`).
- **A flag puts the cost in the wrong place.** It has to be discovered in
  `--help`, then threaded through the generated `start.ps1`, then kept in
  sync per slot (primary, GPU, whisper). The person most likely to need it
  is the person least likely to be editing launch scripts — the exact user
  who reported it.
- **Most of what a description would say is already on disk, and more
  reliably.** The IR's model-level `<rt_info>` records the real nncf
  weight-compression mode, group size, ratio and AWQ flag; `config.json`
  gives architecture, layer count, context and MoE expert counts. `--scan`
  reports those as facts. A hand-typed description would just be an
  opportunity to be wrong — a folder named `-int4-ov` holding int8 weights
  is exactly the confusion the feature would have entrenched.

**The one thing detection genuinely cannot do:** recover the *variant*.
`config.json` in an OpenVINO export has no `_name_or_path`, and
Qwen3-Coder-Next vs Qwen3-Next-Instruct are identical in architecture and
geometry — indistinguishable from the files. That's precisely why the
directory name must stay authoritative for naming instead of being
second-guessed by a heuristic.

Re-evaluate if: someone needs two directories with the same basename served
under different IDs (two quantizations of one model in one process). That's
a real case a rename can't express — but nobody has asked for it, and the
dual-slot routing (`_route_request`) would need work first anyway.

## `--cpu-model-dir` — a third generative slot in one process (2026-08-03)

Idea: add a third `DeviceSlot` so one NoLlama process could serve chat +
vision + coding at once (e.g. NPU chat, iGPU vision, CPU coder). Prompted by
a user question (Manuel Destouesse, email 2026-08-02) asking whether three
models could run simultaneously.

**Verdict:** works in principle, don't build it.

**Why not:**
- It adds a slot to serve the exact case we should be recommending *away*
  from NoLlama. CPU is the one device where Ollama is unambiguously the
  better tool (see the next entry) — so the feature's whole purpose is to
  do badly what a `ollama serve` next door does well.
- `_route_request` (`nollama.py:1142`) is built around exactly two
  generative slots: `for slot in (primary, secondary)` for explicit
  `model@DEVICE` selection, then a two-way heuristic (images → whichever
  slot is a VLM, text → the GPU if it holds an LLM, else primary). A third
  slot turns that heuristic into a policy question — with a coder model
  loaded, which slot gets an unlabelled text request? There's no good
  default, so it becomes config, which is the complexity we're avoiding.
- Memory-bound anyway on the target hardware. The NPU and iGPU both draw
  on system RAM, so three resident 4-bit models plus KV caches are
  competing for one pool on a 32 GB laptop. The device count was never the
  scarce resource.
- **Zero-code alternative already works:** two NoLlama instances on
  different `--port`/`--ollama-port` values, or the recommended split
  (NoLlama for NPU+iGPU, Ollama for the CPU model). Both are documented in
  README "When to use NoLlama, and when to use Ollama".

Re-evaluate if: a single-device machine ever needs three models on that one
device (the two-slot cap, not the device count, would then be the real
blocker) — or if OpenVINO's CPU path decisively beats llama.cpp, which
would reverse the entry below and with it this one's first argument.

## Recommending / building out NoLlama's CPU path (2026-08-03)

Recurring temptation: NoLlama already runs on CPU, the `--device CPU` path
is tested and works, and the desktop 285K benchmarks are respectable
(17.8 tok/s on Qwen3-8B INT4, faster than that box's iGPU *and* NPU). So
it's tempting to present CPU as a first-class NoLlama target and invest in
it — tool-calling on CPU is already enabled (`_tools_supported`).

**Verdict:** keep the CPU path as a working fallback, but recommend Ollama
for CPU-only users, and don't invest further in it.

**Why not:**
- Ollama's llama.cpp CPU backend is far more mature than our OpenVINO CPU
  path, `ollama pull` avoids the conversion/export problem entirely, and
  its tool calling uses per-model chat templates rather than our
  `render_tools_prompt` + `parse_tool_calls` regex approach. That parser
  already needs to recognize six native formats (Qwen3-Coder XML, Hermes,
  bare `<function=>`, Mistral, Llama, DeepSeek) precisely because models
  ignore our prompt — that's a maintenance treadmill Ollama doesn't have.
- ~~It contradicts the project's stated scope. NoLlama exists for the Intel
  **NPU**; GPU/CPU are explicitly provisional (README "Roadmap note"),
  kept only while OpenVINO is meaningfully faster. Advertising CPU dilutes
  the one claim nothing else makes.~~ *(Update 2026-08-04: the provisional
  stance is reversed — GPU/CPU are committed long-term, since no
  OpenVINO-class Ollama Intel backend is coming and most users run agents
  (OpenClaw) on GPU/CPU. This argument no longer applies; the entry's
  verdict still stands on the ecosystem-maturity argument above.)*
- Coexistence is free: Ollama keeps 11434, and NoLlama's port check
  (`nollama.py:2193`) already detects that and disables its own Ollama
  shim rather than failing. There is no integration cost to pay.

**Caveat — this verdict is not measured.** We have benchmarked NoLlama vs
Ollama on the Arc 140V iGPU (NoLlama ~1.6× faster on decode, 2026-06-16)
but **never on CPU**. `bench-results/` has `cpu-qwen3-at-CPU-*.json` for
NoLlama only. The recommendation above rests on ecosystem maturity and
scope, not on a throughput comparison.

Re-evaluate if: someone runs `benchmark.py --backend ollama` on CPU against
the same model/quantization and OpenVINO wins by a wide margin — that would
make CPU worth defending on the same measured grounds as the iGPU. Until
then, don't claim a speed verdict on CPU in docs or in replies to users.

## Putting the OpenVINO nightly stack in the default install (2026-08-15)

Qwen3.8-27B's Intel-published IR needs OpenVINO 2026.4.0-nightly plus an
openvino-genai nightly from 2026-08-14+. The obvious move is to bump
`requirements.txt` — the floors are already `>=`, so a one-line change to
`openvino>=2026.4.0.dev0` plus the nightly `--extra-index-url` would make
the model Just Work for everyone.

**Verdict:** don't. Nightlies live behind an opt-in `-Nightly` switch that
builds a separate `venv-nightly/`, and models that need them are hidden
from the menus unless it's passed.

**Why not:**
- `requirements.txt` is the reproducibility promise for every existing
  user. A nightly index resolves to a different build every day, so two
  people running the same `install.ps1` on the same commit get different
  runtimes — and one of them gets whatever broke last night. This is the
  same objection that kept Glimmer out of the installer (NEXT-STEPS
  "stack gate"): an installer that builds from a moving target promises
  reproducibility it can't keep.
- Intel marks the export itself EXPERIMENTAL / "not fully validated with
  OpenVINO". Shipping an unvalidated runtime to serve an unvalidated model
  compounds two unknowns for users who asked for neither.
- The nightly venv needs `transformers==5.2`, which is incompatible with
  the `<5` cap `requirements.txt` carries for the qwen3_next exporter. One
  venv genuinely cannot hold both stacks; forcing it would silently break
  Qwen3-Next conversions to enable one untested model.
- The cost of the split is small and already paid elsewhere:
  `install-optimum.ps1` established the second-venv pattern, and
  `start-template.ps1` now takes `-VenvName` so both runtimes coexist.

Re-evaluate when: OpenVINO 2026.4.0 ships as a **release** and genai's
qwen3_5 VLM support lands with it. At that point the nightly switch stops
being about Qwen3.8 (a plain `requirements.txt` bump serves it) and is only
worth keeping if a *new* pre-runtime model has taken its place. If none
has, delete `-Nightly` and `requirements-nightly.txt` rather than
maintaining an unused path.

**Update 2026-08-30 — the release arrived early, as 2026.3.1 (2026-08-26),
and the bump happened.** Its notes list Qwen3.8-27B and Muse-Glimmer-30B as
"functionally enabled"; both verified here on the Arc 140V with the
2026.3.1 wheels (Qwen3.8 3.6–4.8 tok/s, Glimmer ~2.5 tok/s, both correct),
so `requirements.txt` floors are now `>=2026.3.1` and both are back in
`models.json` — Qwen3.8 pinned to the repo's **`2026.3.1` branch**, because
its `main` branch is a 2026.4-toolchain export that segfaults the 2026.3.x
runtime at load (`revision` field in the registry, `--revision` in the
downloaders). `-Nightly` stays for now as the *testing* path — it is how the
LFM2-on-NPU-4 bug was shown to persist into 2026.5 the same day — not as a
way to ship models; nothing in the registry needs it.
## `--offload-ratio` on a discrete GPU that fits the model (2026-08-18)

Idea: the Arc Pro B60 has XMX, and TODONT/README already record offload as a
win on XMX hardware (140V: ratio 30 -> 10.8 GB resident @ 25.3 tok/s). A 24 GB
card should do better still.

**Verdict:** don't. On a card where the model fits, offload is a 5x loss.
Measured, Qwen3-30B-A3B int4 (15.2 GB) on a B60 (24 GB):

| | resident | `--offload-ratio 30` |
|---|---|---|
| decode | **50.8 tok/s** | ~10.5 tok/s |
| dedicated VRAM | ~15 GB | **3.2 GB** |
| host (shared) memory | - | **10.2 GB** |
| GPU copy engine | idle | **97%** |
| disk activity | - | **0%** |

**Why not:**
- **It is not streaming from disk.** Both disks sat at 0%. The offloaded experts
  live in the OS page cache and every token DMAs them across PCIe. The copy
  engine at 97% is the bottleneck, not storage.
- **A dGPU pays a bus hop that an iGPU does not.** On the 140V the "offloaded"
  weights sit in system RAM the GPU addresses directly, so the cost is memory
  bandwidth (~136 GB/s). On a discrete card it is PCIe against 450 GB/s of VRAM.
  **Memory topology is the axis, not XMX** - XMX is merely required. The earlier
  framing ("requires XMX", implying XMX is the qualifier) reads as an
  endorsement on any XMX card; it is not.
- **The resident/offload split did not track the ratio.** Ratio 30 left 3.2 GB
  of 15.2 GB resident (~24%), where the 140V measured 10.8 GB (~71%) at the same
  setting, with 21 GB of VRAM free and nothing forcing eviction. Either the ratio
  is a ceiling that a demand-driven expert LRU never fills, or it behaves
  differently on discrete hardware. Unresolved; would need runs at 50 and 90 to
  tell which.
- **Generation stopped being reproducible.** Under greedy decoding (temperature
  0) the same prompt returned 87, 1944, 1951, 1962 and 2040 tokens across five
  runs, four of them hitting the token cap, where the resident run returned 478
  every time. Varying length proves *something* varies; it does not prove the
  numerics are wrong. Flagged, not diagnosed.

Re-evaluate if: a later OpenVINO makes the split honour the ratio on discrete
hardware, or someone measures offload against `--device CPU` on a dGPU that
genuinely cannot fit the model. That second case is the only one where offload
on a dGPU might still be the right answer, and nobody has measured it.
