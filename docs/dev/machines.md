# The test machines

Which box to use for what, and what not to touch. Every measured number in
`docs/` came from one of these three, so knowing which one matters when
comparing results.

## 1. B60 box — the primary workstation

Where this repo lives and where most measurements are taken.

| | |
|---|---|
| CPU | AMD Ryzen 9 5950X, 16C/32T (DDR4) |
| GPU | **Intel Arc Pro B60, 24 GB, discrete** (Battlemage, XMX) |
| NPU | **none** |
| RAM | 32 GB |
| OS | Windows 11 Pro 26200 |
| WSL | 2.7.12.0, Ubuntu 24.04.4, kernel 6.18.33.2-2 |
| Docker | Docker Desktop 4.87.0 / engine 29.7.2 |

**The clean Intel GPU test.** No other GPU vendor in play, so an enumeration
result here is unambiguous. Also where the Docker work happens (#31) — the
container GPU results in `DOCKER-INSTALL.md` are all from this box.

**An application-control policy blocks venv console-script shims here**
[OBSERVED 2026-09-01]. `venv\Scripts\hf.exe` — a generated launcher, not a
signed binary — is refused with *"En programkontrollpolicy har blokkert
denne filen"*, while `python.exe` from the same venv runs fine.
`download-model.ps1` no longer depends on it (it calls
`scripts/hf_download.py` through python instead, and `optimum-cli` now goes
through `python -m optimum.commands.optimum_cli`). Prefer `python -m
<module>` or a small script over a console-script name when writing anything
that has to run here.

**It is not only `.exe` shims — the policy blocks unsigned native modules
too, so `venv-nightly` cannot import openvino_genai on this box at all:**

```
ImportError: DLL load failed while importing py_openvino_genai:
En programkontrollpolicy har blokkert denne filen.
```

The release venv is fine; the nightly wheels are not. **Elevation does not
help** — this is code-integrity policy, not a permission check. So *the B60
box cannot run the nightly stack*, which matters when planning: anything
needing "release vs nightly" on a discrete Intel GPU has nowhere to run
today. Check what is enforcing it with
`Get-CimInstance Win32_DeviceGuard | Select CodeIntegrityPolicyEnforcementStatus`.

Watch the RAM: 32 GB with ~19-26 GB typically free. Loading a big model
stages through host RAM at roughly model size, so **one model server at a
time** — two concurrent 14 GB loads thrashed the pagefile for 40 minutes on
2026-08-21 and produced nothing. `.wslconfig` sets `memory=24GB` so WSL does
not take its default ~50% share and squeeze this further.

## 2. 285K box — the NPU + Ollama + ComfyUI machine

Reachable over SSH at `wossn@100.98.33.88` (Tailscale-range address). **The
SSH shell is PowerShell, not bash** — `uname`, `/dev/null` and `2>/dev/null`
all mangle, and PowerShell quoting inside a bash-side `ssh '...'` is a
reliable source of `SyntaxError`. Copy a script over and run that instead of
fighting the quoting.

**The banner problem is fixed (2026-08-24).** The login profile used to print
a shortcut banner and PSReadLine errors into every non-interactive session,
which contaminated piped output and made `scp` fail outright with `Received
message too long`. The profile now returns early when stdio is redirected
(`[Console]::IsOutputRedirected -or ::IsInputRedirected`), so `scp`, `sftp`
and `ssh host <command>` are all clean. The coreutils block below that guard
still loads, so aliases non-interactive scripts might use keep working.
Backup of the original: `Microsoft.PowerShell_profile.ps1.bak-20260824`
beside the profile.

Its NoLlama venv was on **OpenVINO 2026.1** and was upgraded to **2026.3** on
2026-08-24 — byte-identical build strings to the B60 box, so results from the
two are now comparable.

Two failures seen there on 2026.1, and re-tested after the upgrade, because
"old runtime" is a tempting explanation that is only sometimes the right one:

| Failure on 2026.1 | On 2026.3 | Verdict |
|---|---|---|
| gemma-4 VLM dies at `vlm_config.cpp:34` | loads and answers | **version gap** — gemma-4 VLM needs 2026.3 |
| `LFM2-1.2B-int4-cw` dies on a broadcast shape (`infer_request.cpp:224`, dim 19 vs 10) | **still dies**, now at warmup | **not** the version. That export, or LFM2 support generally, is broken |

Also carries the local LFM2 quant experiments (several `LFM2-*` directories)
— those are the owner's, not project models, and at least one of them does
not load.

| | |
|---|---|
| CPU | **Intel Core Ultra 9 285K** (Arrow Lake) |
| GPU | NVIDIA RTX 5090 32 GB **+** Intel Xe-LPG iGPU |
| NPU | **Intel AI Boost, present, status OK** |
| RAM | 63 GB |
| OS | Windows 11 Pro |
| WSL | **2.9.8.0** (pre-release channel), Ubuntu, kernel 6.18.40.1 — includes the WSL Containers preview (`wslc.exe`) |
| Docker | 29.4.1 — already installed |
| Ollama | 0.32.14, well stocked (gemma4, muse-glimmer, qwen3-coder-next, …) |

**The only machine available for NPU-in-container work**, since the laptop is
off-limits (below). Also the cross-stack reference: Ollama/llama.cpp results
come from here.

Poor choice for an *Intel GPU* test — the RTX 5090 has populated
`/usr/lib/wsl/lib` with 24 NVIDIA libraries and zero Intel ones, so two
vendors muddy any result.

It is a **working server**. Ollama serves from it and ComfyUI runs the
graphic-novel work — ask before mutating those.

**WSL there is fair game**, granted 2026-08-24: the owner does not use WSL on
this box (that happens on the laptop), so WSL experiments do not need asking
each time. That is how it reached the pre-release channel for the B2 test.
Ollama, ComfyUI and the drivers are still ask-first.

## 3. The 258V laptop — DO NOT TOUCH WSL

| | |
|---|---|
| CPU | Intel Core Ultra 7 258V (Lunar Lake) |
| GPU | Intel Arc 140V, 16 GB iGPU |
| NPU | yes — the stronger of the two NPUs we have |

**Hands off.** The owner needs this machine to work and is unwilling to have
WSL messed with on it. That is a hard constraint, not a preference: do not
install, update, or reconfigure WSL, Docker, or drivers here.

It can `git pull` and run scripts. That is the extent of it.

Also, for benchmarking: a busy 140V reads about **30% low**, so numbers from
this machine are not comparable to the desktops unless it is otherwise idle.

## Choosing a machine

| Question | Machine | Why |
|---|---|---|
| Intel GPU / OpenVINO GPU plugin | B60 box | only Intel GPU present, no vendor confusion |
| NPU behaviour | 285K | the laptop is off-limits |
| NPU in a container | 285K only | ditto, and it already has WSL + Docker |
| Ollama / llama.cpp comparison | 285K | that is where Ollama lives |
| Big-model memory pressure | 285K (63 GB) | the B60 box has half the RAM |
| Anything needing a spare reboot | **not** the laptop | see above |
