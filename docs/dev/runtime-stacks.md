# Environment, venvs, backends

Read this before installing, changing pins, or deciding which python runs
a given model.

## Platform

- Primary: Windows 11, Python 3.10+.
- Hardware here: Intel Core Ultra (NPU) + Intel ARC 140V 16 GB (GPU);
  desktop Core Ultra 9 285K; Arc Pro B60 24 GB.
- OpenVINO 2026.4+ with `openvino_genai` (the floor moved from 2026.3.1 on
  2026-09-23; Intel now publishes IRs built with 2026.4, and an older runtime
  does not refuse them cleanly — it segfaults on load. See `TODONT.md`).
- Cross-platform: scripts use `#requires -Version 7.0` and branch on
  `$IsWindows`. Linux + PowerShell 7 is confirmed working (user-reported on
  Core Ultra 7 258V with NPU + GPU, issue #6). **There is no install.sh** —
  Linux runs the same `install.ps1` via pwsh. On Linux, NPU/GPU need the
  Intel userspace drivers or only CPU is detected, and the Linux NPU stack
  (`intel-npu-driver`) is less mature than Windows.

## The venvs

Four, none interchangeable. Activate before running.

| venv | Built by | Stack | Use for |
|---|---|---|---|
| `venv/` | `install.ps1` | `requirements.txt` — OpenVINO 2026.4 release + genai 2026.4, transformers `<5` (4.57.6), optimum-intel `>=1.27` | Everything normal. **Qwen3-Next conversions must use this one** — the exporter refuses to run on transformers 5.x |
| `venv-nightly/` | `install.ps1 -Nightly` | `requirements-nightly.txt` + nightly OpenVINO/genai/tokenizers wheels (2026.5.x), transformers `==5.2`, optimum-intel from git | IRs Intel published ahead of the runtime: Qwen3.8-27B, Muse-Glimmer-30B |
| `venv-optimum/` | `scripts\New-OptimumVenv.ps1` | transformers **and** optimum-intel from git `main`, release runtime | `NEEDS_OPTIMUM` architectures via `--backend optimum` |
| `venv-optimum-nightly/` | `scripts\New-OptimumVenv.ps1 -Nightly` | same, nightly runtime | Testing an OpenVINO GPU fix while `venv-optimum/` stays as the control |

Two things that keep biting:

- The transformers pin in `requirements.txt` (`<5`) is the **opposite** of
  the one in `requirements-nightly.txt` (`==5.2`), on purpose. Read the
  comment blocks in both files before "harmonising" them.
- `scripts\New-OptimumVenv.ps1` tracks git `main` for two packages, so without a
  same-venv, same-session CPU control run, "the GPU plugin changed" and
  "transformers main moved" are indistinguishable. Keep the control.
- Nightly wheels move daily and carry no reproducibility promise, which is
  why `venv/` must keep pointing at `requirements.txt` (see `TODONT.md`,
  "Making the nightly OpenVINO stack the default install").

## Two serving backends

Most models run on **openvino_genai** (`LLMPipeline`/`VLMPipeline`) — the
fast path, and everything else in these notes assumes it.

Some architectures land in optimum-intel (export) before openvino_genai
(serving) learns to run them. Those are served through **optimum-intel's
python runtime** instead: `NEEDS_OPTIMUM` in `nollama.py`, currently just
`{"nemotron_h"}`. Detection is automatic from `config.json`'s `model_type`;
`--backend auto|genai|optimum` overrides, and `--scan` prints a `Backend`
line.

What an optimum slot does **not** get: images (clean 400), prefix cache,
prewarm, `--offload-ratio`, NPU. Tool calling works, and both API surfaces
behave identically.

Muse Glimmer lived on this backend until Intel published a VLM-shaped
export — it is now a plain GenAI VLM slot, and `muse_glimmer` is out of
`NEEDS_OPTIMUM`. `--backend optimum` remains the escape hatch on a release
runtime. See `docs/MODELS.md`.

## Policy: leading edge, not bleeding edge

A model that needs a nightly wheel does not get a menu entry in
`install.ps1`/`models.json` — the manual path is the honest offering until
the runtime ships as a *release*. Docs may say we know it will work; the
installer may not act on it.
