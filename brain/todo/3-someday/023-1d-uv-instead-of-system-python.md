# T-023 — `uv` instead of system Python

**Effort:** 1 d · **Produces:** reworked venv creation in `install.ps1` ·
**Area:** installers

The thorough answer to "don't require Python". `uv` is a single static binary,
no admin, that fetches its own private CPython and builds the venv — and
installs dependencies much faster than pip. It would remove the Python
prerequisite entirely.

It means reworking `install.ps1`'s venv creation, which makes it a
minor-version project rather than a patch. T-014 is the cheap version of the
same goal; do that first and see whether this is still worth it.

## Done when

A box with no Python at all installs and runs NoLlama.
