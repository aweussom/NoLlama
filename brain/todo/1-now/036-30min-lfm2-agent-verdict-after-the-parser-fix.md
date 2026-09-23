# T-036 — Re-run LFM2.5-8B-A1B's agent probe now that we parse its calls

**Effort:** 30 min · **Produces:** an `agent` verdict for the only model that
fits a 16 GB box · **Area:** `models.json`, `OPENCODE-PLAN.md`

`OpenVINO/LFM2.5-8B-A1B-int4-ov` scored **0/2** on the agent probe with the
classifier reporting "never called a tool". That verdict is void: it *was*
calling tools, in LFM2's Pythonic syntax, which `parse_tool_calls` did not
understand until `b950e75`. The model is already downloaded.

Why it matters more than another model verdict: it is the **only candidate that
fits a 16 GB machine with a usable cache**, and right now that tier has no
agent-capable model at all.

What the scan and a clean laptop run already established [OBSERVED 2026-09-23,
140V, OpenVINO 2026.4]:

| | |
|---|---|
| on disk | 4.2 GB (int4), 32 experts top-4, ~1B active |
| KV | **48 KB/token** — a 3 GB pool holds ~65k tokens |
| load | 5 s |
| bare-probe | 6/6 on the 2026.4 release (the card's nightly hint is stale) |
| tier 1 (pelican) | valid SVG in 18.1 s |
| free RAM with it loaded | 12.5 GB of 32 |

- [ ] `nollama.py --model-dir ~/models/LFM2.5-8B-A1B-int4-ov --device GPU --port 8000 --ollama-port 0 --log-file bench/lfm2-8b-a1b.log`
- [ ] `scripts\agent-probe.ps1 -Url http://127.0.0.1:8000/v1 -TimeoutSec 600`
- [ ] read `bench/agent-probe/*.log` before believing either outcome — it is a
      **thinking** model, so `<think>` blocks eat turn budget, and a FAIL may
      again be ours rather than its
- [ ] on a pass: `"agent": true` in `models.json` with the numbers, and
      `docs/MODELS.md` gains a 16 GB recommendation it does not currently have

## Done when

The entry carries a measured verdict either way, and MODELS.md says what a
16 GB laptop should install for agentic coding — or states plainly that nothing
qualifies.
