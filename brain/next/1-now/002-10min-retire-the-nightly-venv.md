# N-002 — Keep `venv-nightly` and `-Nightly`?

**Cost:** 10 min · **Asks:** Tommy

## What

Nothing in `models.json` needs the nightly stack any more — the gate was
OpenVINO shipping Glimmer and Qwen3.8 in a *release*, and 2026.3.1 did
(2026-08-26). `requirements.txt` floors are `>=2026.3.1`.

It still earns its keep as the harness for "does the next runtime fix X" — used
2026-08-30 for the LFM2/NPU 4 question, which is how we learned the answer was
no across three OpenVINO versions.

Constraint worth knowing before deciding: **the B60 box cannot run the nightly
stack at all** — its application-control policy blocks the unsigned
`py_openvino_genai` DLL and elevation does not lift it. So "release vs nightly
on a discrete Intel GPU" already has nowhere to run.

## The decision

- **a)** Keep both. Cost is disk and an occasional stale-wheel surprise.
- **b)** Retire `-Nightly` from `install.ps1`, keep `scripts/New-OptimumVenv.ps1`
  and let anyone who needs a nightly build one by hand.

Session's lean: **a** until the next OpenVINO release lands, then re-ask. It is
the only tool we have for dating an upstream fix, and `TODONT.md` already
records why the nightly must never be the default install.

## Saying yes means

Pick a letter. On (b): a small `install.ps1` change and a `TODONT.md` entry.
