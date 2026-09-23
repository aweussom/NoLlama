# N-033 — The optimum backend serves no model we ship. Keep it or cut it?

**Cost:** 2 h · **Asks:** Tommy · **Area:** `nollama.py` (`OptimumSlot`,
`NEEDS_OPTIMUM`), `scripts/New-OptimumVenv.ps1`, `docs/dev/runtime-stacks.md`

`OptimumSlot` is ~300 lines — the second serving backend, with its own venv, its
own install script and its own section in three docs. What it currently serves:

- `NEEDS_OPTIMUM = {"nemotron_h"}`, and **no nemotron_h model is in
  `models.json`**, because none could be exported.
- Muse Glimmer used to live here; Intel's VLM-shaped export moved it to the
  GenAI path on 2026-08-18. The code keeps it as a fallback "on release runtimes
  whose VLMPipeline lacks the arch" — a runtime we no longer support, now that
  the floor is 2026.4.
- `--backend optimum` as a manual escape hatch, which nothing in the docs asks a
  user to reach for.

So today it is a maintained path with zero shipping models on it.

**What changed and is worth checking first:** optimum-intel
[#1789](https://github.com/huggingface/optimum-intel/pull/1789) (Mamba 2
selective-SSM representation, merged 2026-08-12) is the class of fix the
nemotron_h export was waiting on. If a Nemotron model now exports and serves,
this backend stops being dead and becomes the thing that gets us
Nemotron 3.5 Lightning — a wanted agent model. If it still refuses, we are
maintaining 300 lines against a hope.

- [ ] try an export in `venv-optimum` and serve it with `--backend optimum`
- [ ] if it works: add the model, and this item closes as "alive, keep"
- [ ] if it does not: decide between (a) keep for the escape hatch, (b) cut
      `OptimumSlot`, `NEEDS_OPTIMUM`, `New-OptimumVenv.ps1` and the doc
      sections, with a `TODONT.md` entry recording why it was cut and what
      would bring it back

## The decision

Only Tommy's if the export fails. **Do not cut anything before that test** — the
whole point of the backend is that it runs what GenAI cannot, and the question
is whether that set is empty *today*.
