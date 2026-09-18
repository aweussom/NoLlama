# T-025 — Suppress OpenVINO's native chatter on model load

**Effort:** 1 h · **Produces:** a quieter startup · **Area:** `nollama.py`

Low priority and recorded as such since 2026-05-26 — it is cosmetic, and the
startup banner already carries the information that matters.

What did not work: the Python-level logging knobs. The dump comes from native
code before any Python handler is in play; `OPENVINO_LOG_LEVEL` and friends do
not cover it.

Path forward if anyone cares enough: redirect the file descriptors around the
pipeline construction, which is ugly and risks swallowing a real error message
— which is the actual reason this has stayed at the bottom of the list.

## Done when

Load is quiet, or this file is `git rm`'d with a `TODONT.md` line saying the
fd-redirect cure is worse than the disease.
