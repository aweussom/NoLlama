# T-009 — A stale venv segfaults instead of saying what is wrong

**Effort:** 1 h · **Produces:** a pre-load check in `nollama.py` ·
**Area:** `nollama.py`, `models.json`

Existing installs are on OpenVINO 2026.3.0. Re-running `install.ps1` upgrades
the venv via the new `>=2026.3.1` floors, but a user who only `git pull`s and
then picks Qwen3.8 from a stale venv **gets a segfault at load, not a message**.

- [ ] a `min_openvino` field on the registry entries that need one (Glimmer,
      Qwen3.8)
- [ ] check it before the pipeline is constructed, not after
- [ ] the error names the fix verbatim:
      `pip install -U openvino openvino-genai openvino-tokenizers`

Keep it a *check*, not an auto-upgrade: `TODONT.md` records why the nightly
stack must never install itself, and the same argument applies to reaching into
someone's venv.

## Done when

A 2026.3.0 venv loading Qwen3.8 prints the pip line and exits, instead of
dying in native code.
