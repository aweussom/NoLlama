# NoLlama

OpenAI-compatible LLM/VLM server for Intel hardware. NPU-first.

## Architecture at a glance

- `nollama.py` — Flask server, `DeviceSlot` class per device, auto-detects
  VLM/LLM from `config.json`. `threaded=True`, concurrency via per-device locks.
- NPU: `LLMPipeline` with MAX_PROMPT_LEN=4096 (driver default is 1024),
  streaming via SSE.
- GPU: `VLMPipeline` (images) or `LLMPipeline` (text). Both stream as of
  openvino-genai 2026.1 — verified on Arc 140V iGPU.
- Routing: images go to GPU, text goes to NPU (or GPU if no NPU).
- Whisper: `WhisperSlot` + `WhisperPipeline` for STT,
  `POST /v1/audio/transcriptions`, CPU or GPU.
- Prefix (KV) caching is **default on** for GPU/CPU **LLM and VLM** slots;
  NPU keeps the plain pipeline. → `docs/dev/prefix-cache.md`
- Tool calling works on **GPU/iGPU + CPU only, never the NPU**, on both LLM
  and VLM slots. Tool turns stream on the OpenAI endpoint, gated at the
  tool-call opener (`_ToolCallGate`); `<think>` spans travel as
  `reasoning_content` (`_ThinkSplitter`; `--think-in-content` restores the
  old shape). → `docs/dev/tool-calling.md`
- Most models run on openvino_genai; a few need optimum-intel's python
  runtime (`--backend`). → `docs/dev/runtime-stacks.md`
- `models.json` — curated model registry (npu, gpu_vlm, gpu_llm, whisper).
  `install.ps1` detects devices, shows the model menu, generates `start.ps1`;
  agent setups get `--idle-timeout 0` (keeps the prefix cache alive,
  auto-enables prewarm).
- Web UI: `templates/index.html` + `static/css/style.css` +
  `static/js/app.js`. Collapsible `<think>` blocks, "Just answer me,
  dammit!" button, temperature slider. Markdown rendering
  (`mdEscapeAndRender`, PR #23) is hand-rolled, no library — **attribute
  values must go through `escapeAttr` + `safeUrl` (scheme whitelist), never
  `escapeHtml` alone** (XSS fixed in review). Streaming scroll is
  pinned-but-releasable (`streamState`/`updateStreamBubble`): follows the
  stream until the user scrolls up.
- OpenVINO GenAI may unify VLM/LLMPipeline — when that happens, simplify the
  dual-pipeline routing.

## Deeper notes — read the relevant one before you touch that area

These are this project's memory files. Keep them current; keep this file thin.
The docs-toolkit block at the end is managed — edit above or below it,
never inside.

| File | Read it before touching |
|---|---|
| `docs/dev/prefix-cache.md` | caching, KV pool sizing, prewarm, `--idle-timeout`, TTFT logging, `/health`, memory preflight |
| `docs/dev/tool-calling.md` | `tools` handling, `render_tools_prompt`, `parse_tool_calls`, SSE heartbeat, agent-client quirks |
| `docs/dev/models.md` | model discovery/naming, `--scan`, weight integrity, `download-model.ps1`, NPU export rule, the verified-model list |
| `docs/dev/runtime-stacks.md` | installs, dependency pins, which venv runs what, genai vs optimum backend |
| `docs/dev/moe-offload.md` | `--offload-ratio` and anything XMX-dependent |
| `docs/dev/machines.md` | which box to run a test on, and which one is off-limits |
| `TODONT.md` | **anything structural** — it records approaches already rejected, with the reason |
| `NEXT-STEPS.md` | what is currently open/unresolved |
| `docs/DIAGRAMS.md` + `docs/*.mmd` | any function a diagram `covers:` — the diagram moves in the same commit, and `.\check-docs.ps1` says which |
| `docs/` (`MODELS.md`, `API.md`, `DEVICES.md`, `BENCHMARKS.md`, `AGENTS.md`, `INTERNALS.md`) | user-facing behaviour and measured numbers |

## Development preferences

- Read `TODONT.md` before proposing anything structural. Add an entry
  whenever an approach is abandoned.
- Keep it simple. One file (`nollama.py`) is fine. Don't split into modules
  unless it gets unwieldy.
- PowerShell for install/launch scripts (Windows-native users).
- Runtime flags over hardcoded config (e.g. `--port`, `--device`).
- When testing, use small payloads / short prompts. Don't run full model
  loads unless needed.
- VLM prompts must be dead simple for small models (3B): one question, one
  answer, minimal JSON. All logic in Python, not in the prompt.
- Leading edge, not bleeding edge: nightly-only models don't get installer
  entries.

## A new model gets tested outside the server first — standing order

Before a model is judged, run it through **bare openvino_genai**, no NoLlama:

```powershell
.\venv\Scripts\python scripts\bare-probe.py <model-dir>
```

Only then bring it up under the server. If bare works and NoLlama doesn't,
the bug is ours — and that is the common case, not the rare one.

**Why this is a rule and not a suggestion.** Phi-3.5-vision was declared
broken and written off after a full day of testing: four separate negative
results (not multi-image, not the prefix cache, not the runtime version
across a release *and* a nightly, not the hardware across two GPUs). Every
one was true. Every one ran through NoLlama, so the variable never varied
was NoLlama. The actual cause — our own default `repetition_penalty` of
1.05 colliding with Phi-3's out-of-vocab image placeholders — fell out in
ten minutes of calling `VLMPipeline` directly. Ruling out four things you
thought of is not the same as ruling out the thing you didn't.
→ `TODONT.md`, "Phi-3.5-vision as a GPU VLM entry".

The same order applies to a model that *works* but looks wrong: wrong
answers, odd token counts, suspicious speed. Establish what the runtime
does with it before blaming or crediting the server.

## Known issues

- Qwen3 thinking models can exhaust the token budget on `<think>` before
  producing an answer.
- Cancel (`/v1/cancel`) relies on OpenVINO invoking the streamer callback.
  If the native code blocks without yielding, cancel won't take effect and
  generation completes naturally. Same root cause as the uncancellable
  prefill in `docs/dev/tool-calling.md`.
- Chat history is unbounded in the web UI — the user clears with Ctrl+N when
  a long session approaches MAX_PROMPT_LEN.
- Big agent prompts prefill slowly on weak iGPUs; the Ollama surface still
  buffers tool turns (the OpenAI endpoint streams them since 2026-08-30)
  → `docs/dev/tool-calling.md`.

<!-- docs-toolkit:begin — managed block, edit above or below but not inside -->

## Every function gets documented — no exceptions

Add a function, write its docstring in the same edit. Change a function so its
contract, its failure mode, or the reason it exists moved? The words above it
move too, same edit. Not "before commit" — same edit, while you still remember
why.

Format is identical in every language: one line saying what it does, a `Why:`
giving the reason it exists (the failure it prevents, the upstream quirk it
absorbs, the behaviour it serves), then `In:`/`Out:` in prose naming the edge
case that will bite.

**No `@param`/`@returns`/`Args:` tags** unless something actually consumes them
— a docs site, `mypy --strict`, IDE contract checks. Otherwise they crowd out
the *why*, which is the part you cannot recover from reading the code.

```python
def _msg_signature(msg):
    """Collapse counter drift in a message so it dedups.

    Why: the upstream re-fires with a ticking counter ('for 901 seconds' →
    'for 954 seconds'); grouping on raw text inflates the count 30+x per
    real event.

    In: raw string, or None/''. Out: same string with every digit run
    replaced by 'N'; '' for falsy input (never None).
    """
```

Brace languages use the same prose in a `/** */` block — the block form, not
`//`, so a contract is visually distinct from an inline aside.

**Exempt**: test functions (the test name is the documentation), package
markers, and single-expression one-liners whose name says everything
(`const toRad = d => d * Math.PI / 180`). The checker knows about that last
exemption; it does not need arguing with.

**Cross-file twins get the same docstring**, plus an explicit note on the one
genuine difference. Where two near-identical functions must NOT be unified, say
so — otherwise someone will helpfully unify them.

Bring a file to 100% before adding it to the checker's `$ENFORCED_*` arrays,
never the other way round, or the output fills with known misses and stops
being read.

## Epistemic tagging in docstrings

Docstrings carry two kinds of statement. One is derivable from the code by
reading it. The other is not — it came from a spec, a production run, or domain
knowledge, and **it cannot be recovered if it is lost**. Tag the second kind so
a future rewrite can see the boundary.

- `[DOCUMENTED]` — stated in a vendor spec, RFC, or stdlib docs. Name the
  source.
- `[OBSERVED <date>]` — measured, or seen in a real run. Needs a date and
  enough provenance to re-run it.
- `[INFERRED]` — reasoned from code or evidence, not verified against reality.
  Say what would confirm it.
- `[GUESS]` — no basis. A placeholder asking to be checked or deleted.

### Do not tag what the code already says

**This rule matters more than the definitions above.** `TTL-cached 300s`,
`returns a list of ints`, `raises on HTTP failure` are all visible in the
function. Tagging them dilutes the signal until nobody reads the tags.

The test: **could this line be reconstructed from the code alone?** If yes, no
tag. If it needs a spec, a run, or domain knowledge, tag it.

A design *rationale* is not evidence — it belongs in untagged `Why:` prose.

The failure mode of this convention is **over**-tagging, not under-tagging.
Marking `returns a list of ints` as `[DOCUMENTED]` is syntactically fine and
destroys the entire point, because the tags become decoration and a diff
touching them stops meaning anything. When reviewing a regeneration, check for
over-tagging **first** — that is the direction the error comes from.

### Tags survive rewrites

- Untagged prose may be rewritten freely.
- **A tagged line is evidence, not phrasing.** Do not reword, replace, or drop
  it as part of an unrelated edit.
- If a tagged line must change because the code changed, change the tag with it
  and **say so explicitly in the summary** — never fold it into a diff that
  reads as a wording improvement.
- **Updates preserve or DOWNGRADE a tag; upgrades require cited evidence.**
  `[INFERRED]` becomes `[OBSERVED]` only when a run produced the number, and the
  date comes from that run. Downgrading is always allowed, and is the honest move
  when a claim turns out to rest on less than you thought.

  The drift vector this closes is specific: an eager session "improves" a file
  and rewrites `[INFERRED]` material in `[DOCUMENTED]` voice. Nothing in the diff
  looks wrong — the prose got better — and a guess is now load-bearing.
- Pair a checkable `[OBSERVED]` claim with a real test, reading real data rather
  than a fixture, so it fails when the convention changes. A claim written only
  as prose is a test that never runs.

## Diagrams follow the code

Logic diagrams live in `docs/*.mmd`, indexed by `docs/DIAGRAMS.md`. Node IDs
are the real function names, so every box is greppable. Solid edges are the
normal path; **dashed edges are failure, degraded, dead, or
scheduled-for-removal**. ALL-CAPS annotation nodes carry prose explaining *why*.

Each diagram declares its own coverage:

```
%% covers: src/alarms.py:refresh_state,reconcile_snapshot
```

Touch a function named in a `covers:` line → update that diagram in the same
commit. Add a function to a flow that is already diagrammed → it gets a node
and a `covers:` entry. Add a whole new flow → a new `.mmd`, an index row, and a
link. Not a paragraph of prose pretending to be a diagram.

Before committing:

```powershell
.\check-docs.ps1              # docs + covers integrity + what your diff invalidated
.\check-docs.ps1 -Deep        # narrow the staleness check to changed functions
.\check-docs.ps1 -Render      # confirm every diagram still parses
.\check-docs.ps1 -Audit       # coverage across the repo; gates nothing
```

Fix what it names, **or say in the commit message why you didn't** — a
docstring-only change to a covered function genuinely does not invalidate a
diagram, and saying so is the honest close-out.

Two Mermaid traps, both caught by `-Render`:

- A **bare `%%` line** survives Mermaid's comment stripper (it needs at least
  one character after the `%%`) and then collides with the graph header. Write
  `%% ---` for a separator.
- **`call` is a reserved node ID** (the `click <id> call fn()` directive). So
  are `end`, `graph`, `subgraph`, `class`, `style`. And `A -. .-> B` is not an
  unlabelled dotted edge — `A -.-> B` is.

## Rejected approaches go in TODONT.md

When an approach is abandoned — including ones that sound obviously sensible —
log it in `TODONT.md`: what was tried, the **verdict**, and **why not**, with
the measurement or the concrete failure rather than an opinion. Read it before
proposing anything structural. Update an existing entry rather than duplicating
it when a verdict later narrows or reverses.

A docstring is the wrong home for a decision that spans the codebase; nobody
finds it there.

<!-- docs-toolkit:end -->
