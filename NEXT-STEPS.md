# Next steps

State after the 2026-08-18 merge. Anything settled lives in README, TODONT or
the docs — this file is only what's still open.

## Open

- **Waiting on mikestahili's Xe3/B390 iGPU Glimmer run.** Asked 2026-08-18
  (issue #29, with the exact recipe: `git pull`, `install.ps1 -Nightly`,
  Intel's export, `venv-nightly` NOT `venv-optimum`, `--device GPU`). The
  Glimmer→GenAI reroute shipped AND was verified end-to-end through
  NoLlama's serving path on the B60 (2026-08-18, nightly runtime, Intel's
  export): loads as a plain VLM on GPU, greedy answers correct, `<think>`
  framed on both the non-streaming and streaming OpenAI surfaces, no
  channel-routing leak, 18.5 tok/s decode (vs 8–11 on optimum). Every
  previous Glimmer GPU result came from the optimum path, where all four
  Intel GPU classes corrupted — Glimmer-on-GenAI on an *integrated* GPU is
  the datapoint worth having.

  How the reroute works — by not existing: `muse_glimmer` is simply **out of
  `NEEDS_OPTIMUM`** (only `nemotron_h` remains). Every Glimmer export in
  existence (Intel's and ours) is VLM-shaped, so Glimmer is served as a
  plain VLM, no special routing at all. The one Glimmer-specific piece is
  `_AtemPlainFilter`: it translates the channel routing that survives
  detokenisation (`to=self` glued onto reasoning, `assistant to=user` glued
  onto the answer) into `<think>` blocks on both `generate_vlm` and
  `stream_vlm`. That part is irreducible app-side work — the model emits
  channels as text and the pipeline strips the markers; vLLM/TGI would have
  to do the same. Unit-tested against every chunk split, plus the live run
  above.

  Corrections to what this file used to claim: our own export
  (`aweussom/Muse-Glimmer-30B-int4-ov`) is **VLM-shaped too**
  (`openvino_vision_embeddings_model.xml` is right there), so the earlier
  "is_vlm overrides the blocklist, keeping our LLM-shaped export on optimum"
  plan protected nothing — dropping the set entry is equivalent and simpler.
  On a release runtime, where VLMPipeline lacks the arch, Glimmer now
  **fails at load** instead of limping along on optimum — docs say so, and
  `--backend optimum` (from `venv-optimum/`) is the escape hatch. MODELS.md's
  example command gained that flag.

  Still needs the nightly runtime — Intel exported it with a
  `2026.4.0-...-muse_onyx` build and the card wants 2026.3.1+ with a genai
  pre-release. So Glimmer's stack gate becomes **"2026.4 ships stable"**,
  the same gate Qwen3.8 is already waiting on. VLM slots get no prefix
  cache (the CB backend is LLM-only), so that win does not arrive with this.

- **Branch `vlm-tool-calling`: VLM slots are agent-grade — review & merge.**
  Two changes, both verified end-to-end 2026-08-18:

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
  - **Prewarm is still LLM-only** (`_maybe_capture_prewarm` and the prewarm
    prefill gate on `model_type == "llm"`). With VLM caching working, wiring
    prewarm up for VLM slots is the natural follow-up — it converts the one
    remaining cold 53.7s prefill into a startup cost.
  - The "minutes of prefill" worry for agent prompts was wrong for the B60
    class: 33k tokens prefill in ~9s on the plain pipeline.

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

- **Glimmer into `install.ps1`/`models.json`: waits for OpenVINO 2026.4 as a
  *release*.** Standing rule: leading edge, not bleeding edge. The GenAI
  reroute works on the 2026.4 nightly, but a menu item that needs a nightly
  wheel is bleeding. When 2026.4 releases, the entry is Intel's
  `OpenVINO/Muse-Glimmer-30B-int4-ov` with `"requires_nightly"` dropped —
  the manual path until then is `install.ps1 -Nightly` plus a hand
  download (`install-optimum.ps1` is no longer the recommended Glimmer
  path, only the `--backend optimum` fallback). Docs may say we know it
  will work; the installer may not act on it.
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
