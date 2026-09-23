# T-036 — Qwen3-14B as the 16 GB agent model, in its own dialect

**Effort:** 1 h, mostly waiting · **Produces:** the 16 GB answer for
agentic coding, and whether `Qwen3-14B`'s PARTIAL was ours · **Area:**
`--tool-template native`, `models.json`, `docs/MODELS.md`, installer

It is **the** 16 GB candidate now. Every smaller model has failed in both
dialects, and MoE offload is out (`TODONT.md`, 2026-09-23). A stock 16 GB
laptop gives the iGPU ~8 GB, too little for 9.1 GB of weights. But Shared
GPU Memory Override raises that, as it does on this laptop: weights plus a
3 GB pool (~19k tokens at 160 KB/token, enough for OpenCode's ~10k-token
first prompt) is ~12 GB on the iGPU. That leaves ~4 GB for Windows and
OpenCode. Tight, but a real configuration, and `--cache-size-gb 3` below *is*
that configuration. Speed measured here transfers: same iGPU, same bandwidth.

The A/B so far [OBSERVED 2026-09-23, 140V, OpenVINO 2026.4; table in
`docs/dev/models.md`, tier 2]: LFM2.5-8B-A1B and Qwen3-8B fail 0/2 in
**both** dialects, the same way each time — invented paths. The dialect
changed how far LFM2.5 got, not the outcome. So the native template is not
the rescue it looked like; this run checks whether it moves the one model
that was already close. Under XML, Qwen3-14B finished the fix task in 18 min
and wrote a syntax error on the way.

The first attempt was killed on turn 1 by Claude Code's low-memory reaper
(9.1 GB weights + auto-sized 5 GB pool + OpenCode, with the session idle).
Run it by hand, with a smaller pool:

```powershell
.\venv\Scripts\python nollama.py --model-dir $HOME\models\Qwen3-14B-int4-ov --device GPU `
    --port 8000 --ollama-port 0 --tool-template native --cache-size-gb 3 `
    --log-file bench\qwen3-14b-native.log
# second window, once it prints "NoLlama ready":
.\scripts\agent-probe.ps1 -Url http://127.0.0.1:8000/v1 -TimeoutSec 1500
```

- [ ] read `bench/agent-probe/*Qwen3-14B*.log.err` before believing the verdict
- [ ] note peak committed memory during the run: that, plus the override
      setting it needs, is what a 16 GB owner is told
- [ ] a clean 2/2 → `"agent": true` stays off until it also passes on the B60;
      write the row into the tier-2 table either way
- [ ] if native beats XML here: open the question of making native the
      default for non-Coder models — that is structural, `TODONT.md` first

## Done when

The tier-2 table has a Qwen3-14B row with both dialects, and `docs/MODELS.md`
tells a 16 GB owner either "Qwen3-14B, override to N%, expect M minutes a
task" or that nothing qualifies.
