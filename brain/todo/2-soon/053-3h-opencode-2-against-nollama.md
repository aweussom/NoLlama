# T-053 — OpenCode 2.x against NoLlama

**Effort:** 3 h · **Produces:** an agent-probe verdict on 2.x, fixed probe
scripts, a refreshed prewarm prompt · **Area:** `scripts/agent-probe.ps1`,
`prewarm-8000.json`, `docs/AGENTS.md`, `OPENCODE-PLAN.md`

OpenCode 2.0 shipped 2026-10-12/13 (Anomaly, formerly SST); v2.0.32 by
2026-10-16 [DOCUMENTED, thenewstack.io/opencode-2-shared-service,
releasebot.io/updates/sst/opencode, read 2026-10-26]. Every OpenCode number
in this repo is 1.18.30/1.18.32. 1.x stays a separate line; 2.x is opt-in
(`opencode upgrade --major`).

What changed that touches us, per the coverage, not yet checked here:

- One background service per user, shared by TUI, desktop and web
  (`opencode service set disabled true` turns it off). `agent-probe.ps1`
  drives `opencode run` and already needed stdin/`PWD` fixes for 1.18.32.
- Rebuilt server API and SDK; V1 plugins do not run. The compress-at-birth
  plugin plan in `OPENCODE-PLAN.md` was written against V1.
- Terminal settings move to a global `cli.json`. Our provider config
  instructions in `docs/AGENTS.md` may be stale.
- "Code Mode": the agent writes JavaScript that chains tools. Changes the
  shape of tool calls a small model has to produce.
- A ~5x token-usage increase was reported in one interview [GUESS until
  measured]. On weak hardware that is the whole story: prefill dominates.

- [ ] Install 2.x beside 1.18.x (separate dir or box) so both arms run
- [ ] `agent-probe.ps1` on 2.x, Qwen3-Coder-30B, B60 and 140V; fix the probe
- [ ] Capture 2.x's system prompt + tools; compare size to 1.18.30's; refresh
      `prewarm-8000.json` if it changed (a stale prewarm warms the wrong prefix)
- [ ] Measure prompt tokens per turn, 1.18 vs 2.x, same task
- [ ] Update `docs/AGENTS.md` setup; mark the plugin plan V1-only if so

## Done when

`docs/AGENTS.md` names the OpenCode version it was verified against, and
that version is 2.x with a pass/fail from `agent-probe.ps1`.
