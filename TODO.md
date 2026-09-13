# TODO

## A better dGPU keepalive than pinging the model (2026-09-13)

`_gpu_keepalive` works and should stay until something better is proven, but
it is a workaround wearing the shape of a fix: we defeat an idle timer by
manufacturing non-idleness, once a minute, forever. It burns a little power,
it takes the slot lock, it costs a 1-token prefill, and — as
`TODONT.md` now records — it needed its own `CL_OUT_OF_RESOURCES` handling
because a ping that poisons the context, swallowed, feeds a dead slot to the
idle watchdog. **There is almost certainly a residency control to ask for
instead.** Find it.

### What we already know

- The eviction is **not memory pressure**: 20.34 GB → 0 at 80s, 80s and 79s
  after the last request on an Arc Pro B60, with 21.5 GB of host RAM free
  [OBSERVED 2026-09-13, `docs/slot-lifecycle.mmd`]. Three for three, at a
  consistent threshold. That is an **idle timer**, and an idle timer is the
  kind of thing that has a setting.
- **iGPUs are immune** — their VRAM is host RAM, nothing to evict to. So
  whatever we find only ever applies to the discrete path, same as today's
  gate.
- NoLlama is not involved: status stays `ready`, the allocation is never
  released from our side, committed memory does not change.

### Ruled out already: the OpenVINO GPU plugin has no such property

[OBSERVED 2026-09-13, enumerated `SUPPORTED_PROPERTIES` on the 285K box,
OpenVINO 2026.3.0.] The complete RW set is `PERF_COUNT`, `MODEL_PRIORITY`,
`GPU_HOST_TASK_PRIORITY`, `GPU_QUEUE_PRIORITY`, `GPU_QUEUE_THROTTLE`,
`GPU_ENABLE_SDPA_OPTIMIZATION`, `GPU_ENABLE_LORA_OPERATION`,
`GPU_ENABLE_LARGE_ALLOCATIONS`, `GPU_ENABLE_LOOP_UNROLLING`,
`GPU_DISABLE_WINOGRAD_CONVOLUTION`, `CACHE_DIR`, `CACHE_MODE`,
`PERFORMANCE_HINT`, `EXECUTION_MODE_HINT`, `COMPILATION_NUM_THREADS`,
`NUM_STREAMS`, `PERFORMANCE_HINT_NUM_REQUESTS`, `INFERENCE_PRECISION_HINT`,
`ENABLE_CPU_PINNING`, `ENABLE_CPU_RESERVATION`, `DEVICE_ID`,
`DYNAMIC_QUANTIZATION_GROUP_SIZE`, `ACTIVATIONS_SCALE_FACTOR`,
`WEIGHTS_PATH`, `KV_CACHE_PRECISION`, `OFFLOAD_RATIO`, `CONFIG_FILE`.

Nothing resembling residency, eviction, reservation or keep-alive. So the
lever, if it exists, is **below** OpenVINO — which also means asking Intel
for one at the plugin layer is a legitimate second outcome of this hunt.

Caveat on that enumeration: taken on a Xe-LPG iGPU + an RTX 5090, not on the
B60. The list is plugin-level rather than per-device so it should hold, but
re-run it on the B60 before treating the negative as final.

### Leads, in the order worth trying

1. **`MODEL_PRIORITY` (`ov::hint::model_priority`).** One property, ten
   minutes, already exposed. `HIGH` may map onto something the driver
   consults when deciding what to page out. Weak prior — the trigger is an
   idle timer, not contention — but it is the cheapest test we have and it
   is a real RW property, not a guess.
2. **DXGI residency, which is the actual WDDM-level lever.**
   `IDXGIAdapter3::SetVideoMemoryReservation` tells Windows how much video
   memory this process wants kept resident, and
   `QueryVideoMemoryInfo` reports `CurrentReservation` /
   `CurrentUsage` / `Budget` so the effect is **directly measurable** rather
   than inferred from TTFT. Reachable from Python by `ctypes` against
   `dxgi.dll` with no OpenVINO involvement, which makes it testable as a
   standalone probe beside a running server. Best candidate.
3. **Level Zero residency.** `zeContextMakeMemoryResident()` /
   `zeContextEvictMemory()` are the documented L0 API for exactly this. The
   obstacle: OpenVINO's GPU plugin on Windows is on **OpenCL**, not L0 — the
   issue #38 traceback names `ocl_common.hpp` — so we would be making a
   residency call against allocations we do not own. Worth understanding
   before dismissing; may be the thing to ask Intel to wire up.
4. **Intel USM extensions.** `GPU_USM_MEMORY` is in the iGPU's
   `OPTIMIZATION_CAPABILITIES`, so `cl_intel_unified_shared_memory` is
   present. It carries `clEnqueueMigrateMemINTEL` and
   `clEnqueueMemAdviseINTEL` — check whether any advice value expresses
   "keep resident". Same ownership problem as (3).
5. **Driver-side and Windows-side settings.** An 80-second idle threshold
   that fires with memory to spare smells like driver power management, not
   the Windows memory manager. Check Intel Graphics Software's power
   settings on the B60, the Windows power mode, and whether any documented
   Intel registry value governs idle VRAM release. **Do not change a driver
   setting on the B60 without asking** — `docs/dev/machines.md` rules apply,
   and a setting silently changed is a measurement quietly invalidated.

### How it would land

If a lever exists: set it once at load on eligible slots, keep
`_keepalive_eligible` as the gate, and retire `_gpu_keepalive` to a fallback
behind a flag rather than deleting it — the eviction threshold is a driver
behaviour and can change under us. Keep the poisoned-context handling
whichever way this goes.

If no lever exists: that is a finding, and it is Intel-shaped. File it with
the B60 numbers (three evictions, consistent ~80s, ample free host RAM,
~10s TTFT against ~0.5s warm) and ask for a residency hint on the GPU
plugin. Either way the measurement is already done — that is the expensive
part and it is behind us.

### Smaller thing, same function

`_keepalive_eligible` gates on `"(dGPU)" in FULL_DEVICE_NAME`, which is
"discrete", not "Intel discrete". On the 285K box `GPU.1` is an RTX 5090 and
reports `(dGPU)` [OBSERVED 2026-09-13]. Unreachable today because an NVIDIA
slot never reaches `ready` (NVIDIA is a settled non-goal, below), so this is
tidiness rather than a bug — but the day something changes there, the gate
says yes to a card whose driver has none of this behaviour.

---

## Test the available Norwegian translation models — the NPU's real workload (2026-09-04)

The NPU is not primarily meant for coding LLMs. The intended workload is
**translating internal documents on-device**, so company material never leaves
the laptop. Privacy is the point; throughput is secondary.

This workload is a better fit for the NPU than anything we currently route
there, because every documented NPU limitation stops mattering:

- `MAX_PROMPT_LEN=4096` is a hard ceiling for agent prompts. Translation is
  chunked per paragraph, so it never approaches it.
- The NPU keeps the plain pipeline with **no prefix cache**. Translation would
  miss that cache on every chunk anyway — each chunk is different text. The one
  feature the NPU lacks is the one this workload does not want.
- Tool calling never works on the NPU. Translation needs no tools.
- Sustained low-power throughput over a long document is the NPU's design
  point, and it leaves the GPU free.

**Hard requirement: Bokmål and Nynorsk.** This is what disqualifies the
otherwise ideal candidate. Tencent's `HY-MT1.5-1.8B` is exactly the right
shape — 1.8B, translation-specific, and claimed to beat Tower-Plus-72B and
Qwen3-32B — but its 33 languages contain **no Nordic language at all**, not
Norwegian, Danish, Swedish or Finnish (checked 2026-09-04 against the HF tag
list on `tencent/HY-MT1.5-1.8B`). Worth recording that the *export* path is
fine: `model_type` is `hunyuan_v1_dense`, and optimum-intel already registers
`HunyuanV1DenseOpenVINOConfig` as a `LlamaOpenVINOConfig` subclass for
text-generation-with-past (`optimum/exporters/openvino/model_configs.py:6219`,
verified in our venv). A future Nordic-capable HY-MT converts with no work.

### Candidates to probe

| Model | Size | Licence | Note |
|---|---|---|---|
| `NbAiLab/borealis-1b` | 1B | NB-licence (Apache-derived) | Gemma 3 based; smallest sane NPU target |
| `NbAiLab/borealis-4b` | 4B | NB-licence | the one to beat |
| `norallm/normistral-7b-warm` | 7B | Apache-2.0 | NB's own writeup calls NorMistral stronger *on translation* |

Borealis is Nasjonalbiblioteket's AI-lab family, released 2026-05-26, covering
Bokmål + Nynorsk + English, commercial use permitted, sizes 270m/1b/4b/12b/27b.
`gemma3_text` is already registered for OpenVINO text-generation, so the export
path exists. Skip `normistral-11b-thinking` — reasoning tokens are pure waste
on a translation turn, and we already know thinking models can burn the whole
budget before answering.

Caveat on Borealis: the instruct variants are still `-instruct-preview` and
carry the **Gemma licence** rather than NB's own. Check the terms before an
installer entry, not after.

### How to run it — the standing orders apply in full

1. **Bare `openvino_genai` first**, no NoLlama:
   `.\venv\Scripts\python scripts\bare-probe.py <model-dir>`
2. Then under the server.
3. Then **every device** — CPU, iGPU, B60, and the NPU — each bare *and*
   served. A device that refuses the model is a result, not a skip; record the
   error in the verified list. → `docs/dev/machines.md`, `docs/dev/models.md`.

NPU export must be channel-wise (`-Weight int4-cw` or `int8-cw`); default
group-quantised int4 IRs crash the NPU driver compiler.

**What "tested" means for this one.** Translation quality is not a tok/s
number, and our existing benchmarks will not catch a model that is fluent and
wrong. Fix a small held-out set of real internal-doc paragraphs, both
directions, and compare candidates on the same set. Record the driver with
every measurement as usual.

Background on why this direction and not another: `docs/dev/machines.md` for
the boxes, and the trend read is that narrow small models now beat broad large
ones on their one job — a 1.8B translator outscoring a 32B generalist is the
datapoint, not an outlier.

---
## SmolLM3-3B returns EMPTY content on the NPU after thinking — CPU is fine (2026-09-13)

**This started as "strip markdown fences from small-model titles" and the
device axis turned it into something else.** Recorded that way deliberately:
the fence was the symptom that got noticed, and it was the wrong thing to fix.

OpenCode's `small_model` writes the session title. On the 140V laptop the tab
read ``OC | ```python `` where an earlier turn had correctly produced *"Hello
World Example in Python"*. The obvious reading — a 3B being sloppy — is wrong.

[OBSERVED 2026-09-13] Identical request to both slots: same model files
(`SmolLM3-3B-int4-cw-ov`), same runtime (OpenVINO 2026.3.1-22476, genai
2026.3.1.0-3290), `max_tokens=300`, prompt *"Generate a short title for this
conversation: … Reply with the title only."*

| slot | `content` | `reasoning_content` | finish |
|---|---|---|---|
| B60, **CPU** (5950X) | `"Python Main Calls Hello World"` | 1296 chars | stop |
| laptop, **NPU 4** | **`''`** | 1430 chars | stop |

The NPU result is **deterministic: 3/3 runs byte-identical**, same 1430-char
reasoning, same empty content, `finish_reason: stop` every time. So it is not
sampling variance — the NPU path emits its `<think>` span and then stops
without ever producing an answer, while the CPU emits the same kind of
reasoning and then the title.

Note `usage` came back `{completion_tokens: -1, prompt_tokens: -1,
total_tokens: -1}` on that path too — probably unrelated, but check it while
you are in there.

**Why it matters beyond a wrong tab name:** an empty `small_model` response is
a silent failure. The client gets a 200 with no content and renders whatever
it falls back to. Anything else routed to the NPU that is a thinking model has
the same exposure.

**Do NOT write the fence stripper.** It would paper over an empty-content bug
with a cosmetic scrub and make the real fault harder to see.

### RESOLVED the same day — and it was not a bug at all

**Bare openvino_genai on the NPU reproduced it with NoLlama absent**, so the
server was never implicated:

```
total chars: 1323      has </think>: False
tail: '...Alternatively, "Call hello_world in Python" is also possible'
```

It hit `max_new_tokens=300` mid-thought, never closed the block, never
answered. Then, with `extra_context={"enable_thinking": False}`:

```
closed_think=True   answer='```python\ndef hello_world():\n    print("Hello, World!")...'
```

So the fence was never a formatting quirk. **Both symptoms are one cause:
SmolLM3-3B cannot follow this instruction.** Thinking on, it reasons past its
budget and returns nothing; thinking off, it writes the program it sees
described instead of the title it was asked for. The B60's CPU producing a
correct title was it scraping through, not a device advantage — which is why
the CPU-vs-NPU table above reads like a device divergence and is not one.

**Fixed two ways, both landed:**

1. **`Phi-3.5-mini-instruct-int4-cw` is now the recommended small model.** No
   thinking channel at all (its chat template has zero think markers), so the
   failure is structurally impossible. Verified on NPU 4, bare genai: a correct
   title 3/3, deterministic, 3.2-3.6s — and it loads in **41s** against
   SmolLM3-3B's 84-101s. `models.json` carries a `small_model: true` flag and
   `install.ps1` sorts flagged entries to the top of the side-task menu, since
   menu order is the recommendation.
2. **`enable_thinking` is wired** (`_apply_thinking_switch`), closing the
   "No-think toggle is prose" item below for the LLM path.

**Still open from this:** the VLM path still passes a flattened string with no
`extra_context` hook, so no-think there is still prose only — see the entry
below. And `usage` came back `{completion_tokens: -1, prompt_tokens: -1,
total_tokens: -1}` on the NPU path; unrelated, but nobody has looked.

**The lesson worth keeping:** the fence was the symptom that got noticed and
the wrong thing to fix. Four steps got to the real answer — device axis (CPU
worked), bare genai (never closed `</think>`), the thinking switch (answered
the wrong question), and only then "this model is not good enough". A fence
stripper would have hidden every one of them.

Related: CLAUDE.md's "never emits EOS" entry, also an NPU-only claim that CPU
reframed — except there the device *was* the variable, and here it was not.
Which is the point of running both.

## Shim fixes from the fresh-Ryzen first-contact test (2026-08-11, still open)

Live test of install-windows.bat on a fresh Win11 box (Ryzen 5950X/RX580):
pwsh prompt, winget install, Store-stub detection and both re-run paths all
worked -- but two things to fix:

1. **pwsh probe missed after winget install.** The direct-continue probe only
   checks `%ProgramFiles%\PowerShell\7\pwsh.exe`; winget did a per-user/MSIX
   install, so the .bat fell back to "close and re-run". Also probe
   `%LocalAppData%\Microsoft\WindowsApps\pwsh.exe`.
2. **Don't make Python the user's problem.** NoLlama needs Python, the user
   shouldn't need to know. Cheap version: install via winget automatically
   (notice, not Y/N prompt), then skip the second re-run by probing
   `%LocalAppData%\Programs\Python\Python3xx\python.exe` and passing it to
   install.ps1 via a new `-PythonExe` param (install.ps1 currently only does
   Get-Command python/python3).

## Later: uv instead of system Python

The thorough answer to "don't require Python": `uv` is a single static
binary, no admin, that fetches its own private CPython and builds the venv
(and installs deps much faster than pip). Would remove the Python
prerequisite entirely -- but means reworking install.ps1's venv creation,
so it's a minor-version project, not a patch.

---

## No-think toggle is prose, not the native switch (2026-08-15)

> **LLM path DONE 2026-09-13** — `_apply_thinking_switch` sets
> `enable_thinking=false` via `ChatHistory.set_extra_context` in both
> `generate_llm` and `stream_llm` when the UI's prose marker is present. The
> prose is left in the history deliberately: for Muse Glimmer it IS the native
> control. **The VLM path is still prose-only** (it passes a flattened string,
> which has no extra_context hook) — that is what remains of this entry.

The web UI's "No-think" checkbox sends a **system prompt in English**:

```js
const NO_THINK_PROMPT = 'Respond directly and concisely, with no internal
                         reasoning preamble. Reasoning strength: minimal.';
```

The second sentence is Muse Glimmer's native control and works there. For
the Qwen3 family it is just prose. Qwen3.8's `chat_template.jinja` shows the
real switch:

```jinja
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}     {# pre-closed: no channel to fill #}
{%- else %}
    {{- '<think>\n' }}                   {# generation starts inside the block #}
{%- endif %}
```

`enable_thinking=false` **structurally forecloses** the channel before the
model writes a token. Prose cannot: the template has already opened the
block, so the best the model can do is reason briefly and then stop.
Observed on the B60 (2026-08-15, Qwen3.8-27B): it acknowledged the request
*inside* its reasoning — "Constraint 3: Reasoning strength: minimal" — and
reasoned anyway. Not a bug in the model; we asked in the wrong language.

Same story for SmolLM3, whose family switch is `/no_think`. Two of the three
model families in the registry don't speak the dialect we send.

To do:
1. ~~Find out whether openvino-genai exposes chat-template kwargs~~
   **RESOLVED 2026-08-24 — yes, on the 2026.3 we already ship.** The hook is
   `extra_context`: `Tokenizer.apply_chat_template(..., extra_context={...})`
   and `ChatHistory.set_extra_context({...})`, and the pipelines forward the
   history's extra_context at generate time. Verified end-to-end on
   Qwen3.5-4B-int4-ov via `VLMPipeline` on CPU: default spends the whole
   budget on "Thinking Process:…", `{"enable_thinking": False}` answers
   directly. Our LLM serving path already builds `ovg.ChatHistory`
   (`generate_llm`/`stream_llm`) — wiring it there is one
   `set_extra_context` call. The VLM path still passes a flattened string
   (no extra_context hook on the `prompt=str` overload); it needs to move to
   the `generate(ChatHistory, images=...)` overload — verify the
   `<ov_genai_image_N>` anchor tags still work inside ChatHistory message
   content before switching. No need to wait for 2026.4.
   Caveats: upstream #3937 — Qwen3.6-35B-A3B int4 IR renders the pre-closed
   block but reasons in plain prose anyway (honoring the switch is
   per-model, so keep the prose fallback); findings queued in
   openvino-map `PROPOSED_UPDATES.md`.
2. Route the toggle per model family rather than sending one string to all:
   `enable_thinking=false` (Qwen3.x), `/no_think` (SmolLM3), the
   `Reasoning strength:` line (Glimmer). Keep the prose as the default for
   families we don't recognise — it degrades gracefully, as seen above.
3. Only then consider exposing it on the API surface
   (`chat_template_kwargs`, which is what OpenAI-compatible clients send).

Do NOT put the literal string `<think>` in a system prompt as a shortcut —
models mimic the tags straight into their answer text (observed on Glimmer,
2026-08-13; the existing comment in app.js records this).

---

## Load GGUF directly — openvino-genai already can (2026-08-06)

openvino-genai 2026.1 ships a **GGUF reader** we don't expose at all
(`gguf_modeling.cpp`, `gguf_quants.cpp`, `gguf_tokenizer.cpp`, `load_gguf`,
`GGUFAdapterImpl` in the shipped `openvino_genai.dll`). If NoLlama passed a
`.gguf` path to `LLMPipeline`, users could skip `optimum-cli export` — and
skip the RAM wall that makes conversion the worst part of the project.

Scope limit, read out of the binary's dispatch table (the three names sit
immediately before the `Unsupported model architecture '` string):

    llama    qwen2    qwen3

Probed and **absent**: `qwen3next`, `qwen3moe`, `qwen2moe`, `deepseek2`,
`glm4`, `granite`, `bitnet`. So dense Qwen3/Qwen2.5-Coder and Llama GGUFs
are in scope; **MoE is not** — Qwen3-Coder-Next GGUF will not load, which
kills the obvious hope that GGUF routes around the 400 GB conversion.

Next step is a 10-minute experiment, not a design: point `LLMPipeline` at a
small Qwen3 GGUF on GPU and see whether it loads and generates. Everything
above is inferred from strings in a stripped DLL — verify before writing it
down as fact anywhere user-facing. Expect NPU not to work (it needs its own
compiled blob path).

Worth knowing even if we don't ship it: it's the answer to "why do I have to
convert when Ollama just pulls?" for a chunk of the model space.

---

## Make more memory available to the iGPU (2026-08-04)

By default Windows budgets the iGPU ~half of system RAM (the Arc 140V on a
32 GB laptop reports a 16.5 GB budget via `GPU_DEVICE_TOTAL_MEM_SIZE`).
Intel's driver 32.0.101.6987+ added **"Shared GPU Memory Override"** in the
Intel Graphics Software app (Core Ultra series 1/2): default ~57%, up to
~87% of RAM — explicitly marketed for local AI. That's the difference
between "30B model won't fit" and "fits with a fat KV pool".

Status / open questions:

- The memory preflight (`_preflight_memory`) already reads the driver's
  reported budget and names the override in its "will NOT work" hint.
- **Verify on this laptop** that flipping the override actually moves
  `GPU_DEVICE_TOTAL_MEM_SIZE` (expect 16.5 → ~27 GiB at 87% of 32 GB).
  Blocked so far: Intel Graphics Software crashes when toggling the
  override (2026-08-04); retry after a reboot.
- Is there a programmatic/registry way to set it (so install.ps1 could
  offer it, or at least link the setting)? Driver-version detection +
  a pointer in the install summary may be the realistic scope.
- Document in README once verified (currently one line in the KV-pool
  section).

---

## Spinoff idea: claude-code CLI as Ollama backend (2026-05-26)

**Not part of NoLlama.** Separate repo if pursued. NoLlama is local-Intel
inference; this is the opposite (cloud Anthropic via local CLI). Captured
here so it doesn't get lost.

### The idea

`claude-code -p "<prompt>"` runs in non-interactive print-mode: prompt
in via argv/stdin, response out via stdout, no REPL. So in principle a
small bridge could:

1. Listen on `localhost:11434` speaking Ollama API
2. Translate each `/api/generate` or `/api/chat` request into a
   `claude-code -p` invocation
3. Stream stdout back as NDJSON

Result: any Ollama-aware tool (Open WebUI, Continue, mobile Ollama
clients, etc.) gets Claude as the model, using whatever auth the
local `claude-code` install already has.

### IPC choice

- **Anonymous pipes** (`subprocess.Popen(stdin=PIPE, stdout=PIPE)`) —
  simplest, cross-platform, one-shot per request. Probably the right
  default.
- **Named pipes** — Windows has them too (`\\.\pipe\<name>`), and
  `System.IO.Pipes.NamedPipeServerStream` is cross-platform via .NET.
  Worth it only if the bridge wants a long-lived `claude-code` process
  serving many requests (would need claude-code to support a
  daemon/streaming-stdin mode, which `-p` doesn't currently).

### Practical concerns before pursuing

- **claude-code is interactive-oriented.** `-p` mode works but isn't
  the supported "stable backend" surface. Behavior might change between
  releases.
- **Session state.** Ollama clients expect stateless or
  client-managed history. claude-code might carry conversation context
  across invocations in ways that surprise an Ollama client.
- **Per-request startup cost.** Spawning claude-code per request adds
  latency. For chat that's fine; for batch agents it's not.
- **Auth model mismatch.** claude-code uses the user's logged-in
  account; that's a single tenant. Fine for personal use, doesn't
  scale to a shared server.
- **Already exists upstream?** Worth checking if Anthropic ships a
  similar bridge or if anyone in the community has one — don't reinvent.

### If pursued

One Python file (`claude_ollama_bridge.py`), uses subprocess.PIPE,
maps `POST /api/chat` and `POST /api/generate` to `claude-code -p`,
ignores `/api/pull`/`/api/delete` (stubs returning success), forwards
stdout as NDJSON streaming response. Reusable shape from NoLlama's
existing `ollama_app` Flask blueprint.

---

## Suppress OpenVINO native chatter on model load — low priority (2026-05-26)

openvino_genai 2026.1 prints model-property dumps (`Model: OV
Tokenizer / NETWORK_NAME / NUM_STREAMS / INFERENCE_NUM_THREADS / …`)
plus `[INFO] pruning_ratio` and `[XAttention] DISABLED` lines from
native C++ during pipeline construction and warmup. Roughly 25 lines
per loaded model, only at startup; per-request inference is silent.

### What didn't work

- `OPENVINO_LOG_LEVEL=0` env var (the dump shows `LOG_LEVEL: LOG_NONE`
  already — it's a deliberate print, not a log statement).
- A `contextlib.contextmanager` that did `dup2(devnull, 1/2)` around
  pipeline construction (`657f4bb`, reverted in `8824eca`). Works
  single-threaded but races when both NPU and GPU loaders run
  concurrently — both touch process-global fd 1/2 at the same time
  and the "save original stdout" step can land *after* the other
  thread already redirected. Models silently went to `status=error`
  because their own error prints were lost.

### Path forward

Thread-safe version: global `threading.Lock` around the dup2
sequence. Cost: NPU and GPU model loads serialize through the
suppress block (~30s + 30s instead of ~30s parallel). Acceptable —
loading already mostly serial inside OpenVINO's plugin machinery.

Or: find an upstream-supported flag. Search openvino_genai 2026.1
source for the property-dump call site; there may be a builder
option, runtime property, or env var we missed.

### Why low priority

- Only visible at startup, not during use.
- Doesn't affect correctness.
- The interim "broken suppress" cost was much higher than the
  chatter itself — better to accept the chatter than ship a fix
  that breaks model loading.

---

## CPU as primary on NPU/GPU systems — settled non-goal (2026-05-26)

`install.ps1` already offers CPU as the primary slot when no NPU
and no GPU are detected (line ~450). On NPU- or GPU-equipped
systems, NPU > GPU > CPU is the install-time default and stays so.

### Context that briefly suggested otherwise

`0bbb948` benchmarked Qwen3-8B (text LLM) on Arrow Lake desktop and
found **CPU > iGPU > NPU** — decode is memory-bandwidth-bound, and
DDR5 + many CPU cores beat the 4-core Xe-LPG iGPU and the NPU's
power-sipping memory path. That made it tempting to expose CPU as
a deliberate choice for desktop users.

### Why we're not adding the install prompt

The "CPU wins" rule turned out narrower than first thought:

1. **VLM flips the result.** 2026-05-26 QA: same desktop, same
   Qwen3-VL-8B-INT4 model, image-bearing prompts ran ~2.2x **slower**
   on CPU than on Xe-LPG iGPU (15.29s vs 6.93s avg). VLM prefill is
   compute-bound on the vision encoder; iGPU wins. Text-only on the
   same model: CPU only ~10-30% slower than GPU.

2. **NPU has its own memory path.** Intel AI Boost uses dedicated
   DMA, separate from CPU/GPU memory controllers. Benchmark numbers
   are best-case for CPU/GPU (idle system) and unchanged for NPU.
   Under real load (browser open, build running, game rendering),
   CPU and iGPU lose bandwidth they share; NPU keeps its own.
   For "always-on assistant" / "while-I-work" workloads — which is
   what most users actually have — NPU is undervalued by idle
   benchmarks.

3. **Adding a prompt adds friction for the 95%.** The "Keep it
   simple" preference in CLAUDE.md argues against interactive
   choices for niche power-user scenarios.

4. **The runtime override already exists.** Anyone who's measured
   and wants CPU can do `python nollama.py --device CPU --model-dir
   .\model` — discoverable from `--help`.

### Settled position

NPU > GPU > CPU stays as the install default on Ultra hardware.
CPU is only offered when no NPU and no GPU are present. The
benchmark data and NPU memory-path nuance live here as context for
any future "why not let users pick CPU?" question.

Intel now ships pre-exported, pre-quantized Whisper models on
[huggingface.co/OpenVINO](https://huggingface.co/OpenVINO). Our
`models.json` whisper entries still convert from `openai/whisper-*`
to FP16 (slower install, larger files). Worth benchmarking the
pre-exported variants and replacing the entries if they're competitive.

Candidates to test:

- **`OpenVINO/distil-whisper-large-v3-int8-ov`** — most-downloaded
  whisper variant in the OpenVINO org (9k+ downloads). Distilled
  large-v3 is reportedly ~6× faster than the original at similar
  accuracy. INT8 quantization on top should be a real win.
- **`OpenVINO/whisper-large-v3-int4-ov`** — best accuracy if size
  fits. INT4 makes large-v3 viable on 16 GB GPUs.
- **`OpenVINO/whisper-medium-int8-ov`** and **`-int4-ov`** — direct
  upgrade path for our current "Whisper Medium" FP16 entry.
- **`OpenVINO/whisper-small-int4-ov`** — smallest viable multi-language.

What to measure (extend `benchmark.py --backend whisper`?):
- WER on Norwegian + English samples (you have local audio)
- Wallclock per second-of-audio
- Cold-load time
- Memory footprint

Replace `models.json` whisper entries with whatever benchmarks best.
Tag survivors as proven; drop the rest. Don't add untested ones to
the install menu.

---

## NVIDIA support — deliberate non-goal (settled 2026-05-21)

Settled: NoLlama will **never** support NVIDIA GPUs, even though there
is now a working path.

**The path exists.** OpenVINO 2026 ships an experimental NVIDIA plugin
via `openvino-extensibility`. It's possible to run inference on an
RTX through OpenVINO — but it drags CUDA/cuDNN into the stack, lives
in contrib/plugin land, and is a developer backend rather than a
drop-in user feature. Docs:
https://docs.openvino.ai/2026/documentation/openvino-extensibility/openvino-plugin-library/plugin.html

**Why we won't.** Ollama already does NVIDIA inference excellently.
Anyone with an NVIDIA card should use Ollama; NoLlama's whole reason
to exist is the Intel NPU + ARC story that Ollama doesn't cover.
Supporting both would dilute the project's identity, multiply the
test matrix, and compete with a much better tool on its home turf.

**What changed.** `0bbb948` filtered non-Intel GPUs out of
`detect_devices()` in `nollama.py` so the RTX 5090 wouldn't be offered
as a footgun (compile errors + `CL_INVALID_VALUE` at warmup). On
2026-05-21, the same filter was added to `install.ps1`, plus
multi-GPU enumeration handling (`GPU.0`/`GPU.1` → canonical `GPU` with
the actual OpenVINO id tracked separately). Multi-GPU desktops (iGPU +
non-Intel dGPU) now detect the Intel GPU correctly.

---

## NoLlama on NVIDIA GPUs — verified does NOT work (2026-05-03, historical)

Kept for context. On a desktop with both Intel iGPU and an RTX 5090,
`python nollama.py --device GPU.1` (when GPU.1 was the RTX) failed:

- Model compile: 144 errors generated by the `intel_gpu` plugin's kernels.
- Warmup crashes with `CL_INVALID_VALUE` from `clEnqueueMapBuffer`.

Root cause: OpenVINO's stock `intel_gpu` plugin enumerates any
OpenCL-capable device. NVIDIA's driver provides OpenCL, so the 5090
shows up — but the plugin's kernels use Intel-specific GPU intrinsics
that NVIDIA's OpenCL runtime doesn't support. Enumeration ≠ executable.
The new NVIDIA-specific plugin (above) is a separate story, but we've
chosen not to pursue it.

---

## Text-to-Speech (TTS) — `/v1/audio/speech`

`openvino_genai.Text2SpeechPipeline` exists. Only SpeechT5 supported so far.

**Export:**
```bash
optimum-cli export openvino \
  --model microsoft/speecht5_tts \
  --weight-format int4 \
  --model-kwargs '{"vocoder":"microsoft/speecht5_hifigan"}' \
  speecht5_tts
```

**What's needed:**
- `--tts-dir` flag, similar to `--whisper-dir`
- `POST /v1/audio/speech` endpoint (OpenAI-compatible)
- Speaker embedding files (512×float32 `.bin`), map OpenAI voice names (`alloy`, `echo`, etc.) to them
- CPU or GPU only — no NPU support for encoder-decoder models

**Caveats:**
- SpeechT5 is serviceable but clearly first-gen neural TTS, not ElevenLabs quality
- English-centric — Norwegian output would be rough
- Voice selection via embedding files, not named presets — UX is awkward
- Small model (~few hundred MB), fast on CPU

**Verdict:** Clean API surface, completes the OpenAI compatibility story. Worth adding
once STT (Whisper) is proven. Low priority until then.

---

## Spinoff project idea: Ollama API wrapper for any OpenAI-compatible server

**Not part of NoLlama.** Separate repo if pursued.

### Honest assessment (verified 2026-04-13)

Initial motivation was "tools that speak Ollama but not OpenAI." This
turned out to be weaker than hoped:

- **Major tools support both.** Continue.dev, Zed, Cursor, Open WebUI,
  VS Code extensions — all take custom OpenAI-compatible base URLs.
- **Walled-garden tools don't help either way.** Android Studio's AI
  (Gemini-only) and JetBrains AI Assistant (their own backend) won't
  accept any local endpoint, Ollama or OpenAI.
- **Genuinely Ollama-only tools are niche**: Llama Coder (VS Code),
  Enchanted (macOS), Maid, various mobile clients. Real but small audience.

### The narrower valid case

- Protocol quirks: `/api/tags` vs `/v1/models` have different shapes.
  Some tools nominally "OpenAI" still call Ollama-specific endpoints
  (`/api/show` for metadata).
- Ollama NDJSON vs OpenAI SSE framing trips tools tested against only one.
- Dev ecosystems built around `ollama` CLI expect a real Ollama server.

### If pursued

| Ollama endpoint | Upstream call | Translation |
|---|---|---|
| `GET /api/tags` | `GET /v1/models` | reshape model list |
| `POST /api/show` | `GET /v1/models` | pick one, reshape |
| `POST /api/chat` | `POST /v1/chat/completions` | SSE → NDJSON |
| `POST /api/generate` | `POST /v1/chat/completions` | wrap prompt as user msg |
| `POST /api/pull`/`delete`/`copy` | stub — return success | |

One Python file, single config (`--upstream http://ovms:8080 --port 11434`).
Reusable chunks already exist in nollama.py's `ollama_app` and
`_ollama_stream_*` functions.

**Verdict**: Interesting afternoon project, but the audience is smaller
than the initial Reddit comment suggested. Not urgent.
