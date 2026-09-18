# T-017 — `transformers` main breaks the optimum backend's text-only path

**Effort:** 1 h · **Produces:** a pin or a documented exposure ·
**Area:** `scripts/New-OptimumVenv.ps1`, `docs/dev/runtime-stacks.md`

`transformers` `5.16.0.dev0` calls `get_experts_implementation()` from
`_optimize_model_for_decode()`. `OVModelForCausalLM` does not implement it, so
`generate()` dies. `OVModelForVisualCausalLM` has its own `generate()` and is
unaffected — **which is the only reason Glimmer still works**.

This will bite `nemotron_h`, which is text-only. The exposure is
`scripts\New-OptimumVenv.ps1 -TransformersRef main`.

- [ ] decide between pinning a known-good ref and waiting for optimum-intel
- [ ] whichever way, `docs/dev/runtime-stacks.md` should say which refs are
      known good, since that file is what a future session reads before
      touching a venv

## Done when

`-TransformersRef main` either works for a text-only model or refuses with an
explanation, instead of dying inside `generate()`.
