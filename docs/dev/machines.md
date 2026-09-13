# The test machines

Which box to use for what, and what not to touch. Every measured number in
`docs/` came from one of these three, so knowing which one matters when
comparing results.

**A model gets run on CPU, iGPU, B60 and (where it fits) NPU** — the standing
order in `CLAUDE.md`. No single box has all of those, so the matrix below is
what makes that rule executable. The two NPUs are **different generations**,
and that has already been the whole answer once (LFM2), so "the NPU" is never
one device.

## 1. B60 box — the primary workstation

Where this repo lives and where most measurements are taken.

| | |
|---|---|
| CPU | AMD Ryzen 9 5950X, 16C/32T (DDR4) |
| GPU | **Intel Arc Pro B60, 24 GB, discrete** (Battlemage, XMX), driver `32.0.101.8805` (checked 2026-09-11) — **plus an AMD Radeon RX 580**, so it is not a single-vendor box after all |
| NPU | **none** |
| RAM | 32 GB |
| OS | Windows 11 Pro 26200 |
| WSL | 2.7.12.0, Ubuntu 24.04.4, kernel 6.18.33.2-2 |
| Docker | Docker Desktop 4.87.0 / engine 29.7.2 |

**The clean Intel GPU test.** No other GPU vendor in play, so an enumeration
result here is unambiguous. Also where the Docker work happens (#31) — the
container GPU results in `DOCKER-INSTALL.md` are all from this box.

Reachable over SSH at **`wossn@100.81.4.88`** (Tailscale range; hostname
`r9-5950x`). Not to be confused with `wossn@100.98.33.88`, which is the 285K
— same username, different box, and the addresses differ by two digits.

**Working on it over SSH — four rules learned the hard way** [OBSERVED
2026-09-11]:

- **A process started with `Start-Process` from an SSH session dies when the
  session ends**, even `-WindowStyle Hidden`. Two conversions and a download
  died silently that way before the pattern was seen. Long work runs either
  inside an SSH session kept alive for the duration, or as a **scheduled
  task**: `Register-ScheduledTask` with `-UserId $env:USERNAME` (the
  `WORKGROUP\` prefix does not resolve to a SID on this box), `-LogonType
  Interactive` (the console session is normally logged in), and the **full
  path to the executable** — the task environment has no `pwsh` on its PATH
  because PowerShell 7 is a Store app here (result `0x80070002`). The NoLlama
  CPU server runs that way as task `nollama-cpu-8002`.
- **Docker is unusable over SSH**: every pull, even anonymous, dies on
  "error getting credentials" because Docker Desktop's credential helper
  needs the interactive logon, and a config without a store does not stop
  the CLI from calling it. Run the harness container on the 285K instead
  and point it at this box.
- **Inbound firewall**: rule "NoLlama 8000-8002 from Tailscale" allows TCP
  8000–8002 from `100.64.0.0/10`, created 2026-09-11. Health and chat on
  8002 verified from the 285K and from a container on the 285K.
- The Tailscale interface is on the Private profile; the RX 580 shares the
  box with the B60 (see the table); RAM is the constraint, not disk (630 GB
  free).
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

**Its iGPU is the stock-cap, no-XMX class the community reports come from**
[OBSERVED 2026-09-11]: `OPTIMIZATION_CAPABILITIES` has no `GPU_HW_MATMUL`, and
a 60k-char (12,825-token) prompt through bare genai on `SmolLM3-3B-int8-cw`
died with *Exceeded max size of memory object allocation: requested
5,466,652,672 bytes, max 4,294,959,104* — the same ~4.29 GB cap as the 140T
in #24, which the 140V with its 27.2 GB override cannot hit. **This is the
repro box for allocation-cap reports.** `GPU_ENABLE_LARGE_ALLOCATIONS=True`
got the same prompt through (80 s prefill). Same session, same device:
`Qwen3-0.6B-int8-ov` (INT8 asymmetric, dense) prefilled the same 60k chars
cleanly with and without `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`, so the #33
`matmul primitive` failure is not "int8 zero-points on a non-XMX GPU" — with
the reporter's symmetric re-convert also failing, MoE is the axis left.

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
| venvs | `venv` (2026.3.0, serving), `venv-2026.3` (transformers 5.4 — the one that knows `lfm2_moe`, use it for `-Convert`), `venv-2026.3.1` (scratch, runtime-version tests only, created 2026-09-11) |
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
| GPU | Intel Arc 140V iGPU — **~25.5 GB budget**, not the stock 16 GB (Intel Shared GPU Memory Override is on) |
| NPU | yes — **NPU 4, `DEVICE_ARCHITECTURE=4000`.** The newer generation, and not the better one for every model |

**WSL and Docker: hands off.** The owner needs this machine to work and is
unwilling to have WSL messed with on it. That is a hard constraint, not a
preference — do not install, update or reconfigure either, here, ever.

**Drivers: ask first, and the owner does the reboot** — see *Drivers* below.
The NPU work in Aug–Sep 2026 happened here because it is the only NPU 4; it
was asked for each time. Consent does not carry to the next swap.

Running scripts, probes and model loads needs no permission — that is what
the box is for.

Also, for benchmarking: a busy 140V reads about **30% low**, so numbers from
this machine are not comparable to the desktops unless it is otherwise idle.

**It is the only NPU 4 we have**, so every NPU-generation comparison needs it
and the 285K together.

### The override is lost easily, and restoring it needs elevation

[OBSERVED 2026-09-13] The budget collapsed from 25.34 GB to **16.5 GB** — the
stock ~50%-of-RAM default — and Intel Graphics Software crashed when used
normally. **The fix was running Intel Graphics Software as Administrator**;
the split could then be changed, and `GPU_DEVICE_TOTAL_MEM_SIZE` came back to
**27,208,896,512**, byte-identical to the 2026-09-01 reading.

Three hypotheses died on the way, and each is worth not re-running:

- **Not HVCI.** Memory Integrity was on before and is still on after
  (`Enabled = 1`, VBS status 2, services {1,2,3,4}). It never mattered.
- **Not the BIOS.** A flash from 1.14 (N4IET28W) to 1.16 (N4IET30W) happened
  in the same session and looked causal for a while. It was not — the
  elevation is what changed the outcome. Do not spend a reflash on this.
- **Not the registry.** `SharedSystemMemorySize`, `DedicatedSegmentSize` and
  `AllowSharedMemoryOverride` under the adapter key are **absent in both
  states**, at 16.5 GB and at 25.3 GB. Reading them tells you nothing; they
  are not the mechanism.

**The only trustworthy check** is the runtime's own view:

```powershell
.\venv\Scripts\python.exe -c "import openvino as ov; c=ov.Core(); print('%.1f GB' % (c.get_property('GPU','GPU_DEVICE_TOTAL_MEM_SIZE')/2**30))"
```

Note the adapter's *name* is literally `Intel(R) Arc(TM) 140V GPU (16GB)` — a
driver INF label that appears in Task Manager, the registry `DriverDesc` and
OpenVINO's `FULL_DEVICE_NAME`. It is not a measurement and does not change
with the override. `Win32_VideoController.AdapterRAM` reports 4.00 GB, the
int32 overflow. Both are traps.

Shared GPU memory is a **ceiling, not a carve-out**: with the override active
and nothing loaded, Windows still reported 25.26 GB of 31.52 GB free. A drop
in available RAM means a model is resident, not that the iGPU reserved it.

**The memory override also moved the per-allocation cap, and that costs repro
ability** [OBSERVED 2026-09-01]: `GPU_DEVICE_MAX_ALLOC_MEM_SIZE` reads
**27,208,896,512** here against the ~4.29 GB (`4,294,959,104`) a stock iGPU
reports — which is what users actually have. So this box **cannot reproduce an
allocation-cap bug** at any sane prompt length: the dense `[1,1,S,S]` mask that
dies at 67k tokens elsewhere would need ~165k tokens here. When a report names
*"Exceeded max size of memory object allocation"*, reach for another box, and
never read a clean run here as a refutation (issue #24).

## Drivers — never forcibly, but never stale either

Two halves, and the second is the one that gets forgotten.

**Never update a driver unprompted.** Ask, and let the owner reboot. Leading
edge, not bleeding edge: a driver swap costs a reboot on a machine someone is
working on, and it can move a result you are mid-way through explaining.

**But do not let a box drift.** Users run the latest driver, and when
something breaks, *updating the driver is the first thing they will try* —
and the first thing Intel will tell them to try. Two consequences we have
already paid for:

- **A bug reported from a stale driver gets closed as "update your driver."**
  The LFM2 report only survived contact with Intel because we were on
  32.0.100.5540 and could say the failure is byte-identical on two driver
  generations. On 4778 alone it would have been dismissed in one reply, and
  correctly so.
- **A bug we cannot reproduce may simply be newer than us.** If a user is on
  a driver we have never run, "works here" means nothing.

So: **record the driver with every measurement**, and check the gap before
believing a negative result. `explain_genai_error` and the model cards name
driver versions for this reason.

Read 2026-09-01:

| Box | GPU driver | NPU driver |
|---|---|---|
| 258V laptop | `32.0.101.8991` (driver date 2026-08-24, **checked 2026-09-13**) — current | `32.0.100.5540` (2026-08-20) — current |
| 285K | `32.0.101.8860` (2026-06-25) — ~2 months behind (Xe-LPG; the RTX 5090 is on `32.0.16.1088`) | `32.0.100.4778` (2026-04-28) — **old** |
| B60 box | `32.0.101.8805` (re-checked 2026-09-13, unchanged) — **now the stale box** | none (no NPU) |

**The laptop is no longer behind (2026-09-13)** — it is on `.8991`, the
release named as latest below. That inverts the warning in the paragraph
that follows: cross-box GPU comparisons now carry a **driver delta between
the laptop and the B60** (`.8991` vs `.8805`), and the B60 is the side to
bring forward.

Latest Intel Arc driver at that date was **`32.0.101.8991`** (2026-08-25,
WHQL, re-certified 08-29), with `.8974` before it on 08-15 — per the driver
trackers; intel.com 403s an automated fetch, so confirm by hand before
acting. **Both boxes are two releases behind on the GPU**, so a GPU negative
from either is worth re-checking against a current driver before it is
believed — which matters right now, because the gemma-4-E4B work (issue #24)
is a GPU story.

**The two NPUs are on different drivers, and that is a live gap.** The laptop
(NPU 4) is on 5540; the 285K (NPU 3) is still on 4778. Every NPU 3 positive
control we have — including the one in the #37322 report — was taken on 4778.
The controlled comparison still holds, because the NPU 4 side was measured on
*both* drivers and did not move, so 4778-vs-4778 is a like-for-like pair. But
we have never run NPU 3 on 5540 ourselves; the only evidence that NPU 3 stays
correct there is a third party's token counts (issue #24). Worth closing if
the 285K is ever updated.

Read them back with:

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate
Get-CimInstance Win32_PnPSignedDriver |
  Where-Object { $_.DeviceName -match 'AI Boost' } |
  Select-Object DeviceName, DriverVersion, DriverDate
```

From inside OpenVINO, the NPU reports its own:
`core.get_property("NPU", "NPU_DRIVER_VERSION")` — note that this reads the
*running* driver, so it still shows the old one after an install until the
machine reboots. That gap cost a day once: pnputil said 5540, every process
said 4778, and the difference was a pending reboot.

## Choosing a machine

| Question | Machine | Why |
|---|---|---|
| The full device matrix for a new model | all three | CPU+B60 here, iGPU+NPU4 on the laptop, NPU3+Xe-LPG on the 285K |
| Intel GPU / OpenVINO GPU plugin | B60 box | the only Intel GPU; an RX 580 sits beside it, but OpenVINO's GPU plugin is Intel-only so enumeration stays unambiguous |
| iGPU behaviour | laptop (140V) or 285K (Xe-LPG) | two iGPU generations; the B60 is discrete and not a substitute |
| NPU behaviour | **both** — 285K is NPU 3, laptop is NPU 4 | one NPU proves nothing; the generation has been the whole variable before |
| NPU in a container | 285K only | the laptop's WSL/Docker is off-limits |
| An allocation-cap or per-buffer-limit report | **not** the laptop | its cap is 27.2 GB, ~6x a stock iGPU |
| NPU in a container | 285K only | ditto, and it already has WSL + Docker |
| Ollama / llama.cpp comparison | 285K | that is where Ollama lives |
| Big-model memory pressure, **CPU loads** | 285K (63 GB) | the B60 box has half the RAM |
| Big MoE on an iGPU | **nowhere here** | the 285K's no-XMX iGPU stages ~3x the weights in shared memory and pages the box to a standstill (TODONT, 2026-09-11); the laptop's 140V has XMX and a 25.5 GB budget — ask the owner first |
| Anything needing a spare reboot | **not** the laptop | see above |
