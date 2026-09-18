# N-003 — File the optimum-intel feature request for `nemotron_h`?

**Cost:** 30 min · **Asks:** Tommy

## What

Nemotron Lightning (30B-A3B) is the agent model we want and cannot convert:
optimum-intel PR #1789 merged **descoped**, with no `nemotron_h` exporter.
`NEEDS_OPTIMUM = {"nemotron_h"}` in `nollama.py` is already waiting for it.

The pattern that worked for Glimmer was issue #1927 — a feature request that
offered to test on real Intel hardware, which got it moving.

Second-order problem, same area: `transformers` `5.16.0.dev0` breaks the
optimum backend's **text-only** path (`_optimize_model_for_decode()` calls
`get_experts_implementation()`, which `OVModelForCausalLM` does not implement).
`OVModelForVisualCausalLM` has its own `generate()`, which is the only reason
Glimmer still works. Nemotron is text-only, so it walks straight into this.

## The decision

- **a)** File it, offering the same testing deal that worked for Glimmer.
- **b)** Leave it. We have no user asking for Nemotron by name.

Session's lean: **a** — it costs half an hour and the precedent is good. Note
in the request that the transformers-main breakage above will hit any text-only
model, so the exporter alone is not enough.

## Saying yes means

Pick a letter. On (a): the filing is the work; T-017 (the transformers pin) is
the part we can do without them.
