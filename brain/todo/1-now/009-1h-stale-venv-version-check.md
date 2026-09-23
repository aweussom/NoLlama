# T-009 — A stale venv segfaults instead of saying what is wrong

**Effort:** 1 h · **Produces:** a pre-load check in `nollama.py` ·
**Area:** `nollama.py`, `models.json`

Existing installs are on an older OpenVINO. Re-running `install.ps1` upgrades
the venv via the `>=2026.4.0` floors, but a user who only `git pull`s and then
loads a model exported with a newer runtime **gets a segfault at load, not a
message**.

This stopped being hypothetical on 2026-09-23: Intel re-exported gemma-4 with
2026.4, and that IR takes the process down under 2026.3.1 on **GPU and CPU
alike** — bare openvino_genai, NoLlama absent. It is not a model-specific
guard any more; it is every Hub model Intel rebuilds from now on.

- [ ] a `min_openvino` field on the registry entries that need one — and a
      floor for the *installed runtime itself*, since the failing case is now
      "any IR newer than your venv", not a named model
- [ ] check it before the pipeline is constructed, not after
- [ ] the error names the fix verbatim:
      `pip install -U openvino openvino-genai openvino-tokenizers`

Keep it a *check*, not an auto-upgrade: `TODONT.md` records why the nightly
stack must never install itself, and the same argument applies to reaching into
someone's venv.

## Done when

A 2026.3.0 venv loading Qwen3.8 prints the pip line and exits, instead of
dying in native code.
