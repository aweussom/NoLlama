#requires -Version 7.0
# install.ps1 — NoLlama setup: venv, dependencies, model selection
#
# Usage:
#     .\install.ps1                       # interactive setup
#     .\install.ps1 -SkipModel            # venv + deps only
#     .\install.ps1 -HfToken hf_xxx       # auth for gated/private models
#     .\install.ps1 -Nightly              # OpenVINO nightly stack in venv-nightly/
#
# Detects available devices (NPU, GPU, CPU), then asks what you want to DO
# (chat / coding agent / vision / combos) and places each model on the best
# device. Coding-agent models (OpenCode / Copilot, tool-calling) and CPU are
# first-class choices, not buried.
#
# -HfToken: a HuggingFace access token (https://huggingface.co/settings/tokens).
# Only needed for gated or private models — the curated OpenVINO models are
# public and download anonymously. We can't rely on 'hf auth login' here
# because this script is what installs 'hf' in the first place, so the token
# is passed through the HF_TOKEN env var that huggingface_hub reads at
# download time.
#
# -Nightly: build a SECOND venv (venv-nightly/) on the OpenVINO nightly
# wheels instead of touching venv/. Some Intel-published IRs land months
# before the runtime that reads them ships — Qwen3.8-27B needs OpenVINO
# 2026.4.0-nightly and an openvino-genai nightly from 2026-08-14+. Nightly
# wheels change daily and Intel marks those exports EXPERIMENTAL, so the
# stable venv keeps its reproducible requirements.txt and the generated
# start.ps1 records which venv it belongs to. Models that need this stack
# carry "requires_nightly": true in models.json and are hidden from the
# menus unless -Nightly is passed.

param(
    [switch]$SkipModel,
    [string]$HfToken,
    [switch]$Nightly,
    # Override if the nightly index moves, or point at a local wheel dir.
    [string]$NightlyIndex = "https://storage.openvinotoolkit.org/simple/wheels/nightly",
    # Catch-all. 'pwsh -File install.ps1 -Unknown' does NOT error: it binds what
    # it recognises, drops the rest, and runs with exit 0. Passing -Nightly to a
    # checkout that predates it therefore looked like success while building the
    # STABLE venv (hit twice, 2026-08-18).
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

if ($ExtraArgs) {
    Write-Host "ERROR: unrecognized argument(s): $($ExtraArgs -join ', ')" -ForegroundColor Red
    Write-Host "  Flags: -Nightly -SkipModel -HfToken <token> -NightlyIndex <url>" -ForegroundColor Yellow
    Write-Host "  If you expected one of these, check you are on a branch that has it:" -ForegroundColor Yellow
    Write-Host "    git branch --show-current" -ForegroundColor Yellow
    exit 1
}

# Drop PATH entries this session cannot traverse. Windows refuses to cross a
# user-created junction under a network logon (an SSH session), and pip walks
# PATH during install: it dies with ERROR_UNTRUSTED_MOUNT_POINT / WinError 448.
# The OpenAI Codex CLI ships exactly such a junction at
# %LOCALAPPDATA%\Programs\OpenAI\Codexin. Probed rather than name-matched,
# so any future offender is handled too. Interactive installs never see this.
$badPath = @()
$env:Path = (($env:Path -split ';') | Where-Object {
    if (-not $_) { return $false }
    try { Get-ChildItem -LiteralPath $_ -ErrorAction Stop | Out-Null; $true }
    catch { $script:badPath += $_; $false }
}) -join ';'
if ($badPath) {
    Write-Host "[i] Ignoring $($badPath.Count) untraversable PATH entry/entries for this install:" -ForegroundColor DarkGray
    $badPath | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
}

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelsRoot = Join-Path $HOME "models"
Push-Location $ScriptDir

# Make a passed HF token available to every 'hf download' below. huggingface_hub
# reads HF_TOKEN from the environment automatically, so the download calls need
# no change. Only set when -HfToken was given; otherwise any token already
# stored via 'hf auth login' is used as before.
if ($HfToken) {
    $env:HF_TOKEN = $HfToken
    Write-Host "[+] HF token set for this session (gated/private model auth)" -ForegroundColor DarkGray
}

# Cross-platform venv layout: Windows uses Scripts/<tool>.exe, POSIX uses bin/<tool>.
$VenvBinDir = if ($IsWindows) { "Scripts" } else { "bin" }
$ExeExt     = if ($IsWindows) { ".exe" }   else { "" }

Write-Host ""
Write-Host "=== NoLlama Install ===" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Create venv
# ---------------------------------------------------------------------------

# -Nightly gets its own venv so a nightly experiment can never leave the
# stable install in a half-upgraded state. Both are complete NoLlama runtimes;
# start.ps1 is generated pointing at whichever one built it.
$VenvName = if ($Nightly) { "venv-nightly" } else { "venv" }
$VenvDir  = Join-Path $ScriptDir $VenvName

if ($Nightly) {
    Write-Host "[!] Nightly mode: building $VenvName/ on OpenVINO nightly wheels." -ForegroundColor Yellow
    Write-Host "    Your stable venv/ is left untouched. Nightly builds are not" -ForegroundColor DarkGray
    Write-Host "    reproducible and Intel marks the models needing them EXPERIMENTAL." -ForegroundColor DarkGray
    Write-Host "    Index: $NightlyIndex" -ForegroundColor DarkGray
    Write-Host ""
    # optimum-intel is installed from git in this stack (qwen3_5 VLM support
    # is unreleased), and pip shells out to git to fetch it.
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: -Nightly needs 'git' on PATH (optimum-intel installs from GitHub)." -ForegroundColor Red
        Pop-Location; exit 1
    }
}

# Validate existing venv. A venv that was moved or renamed is genuinely dead:
# every script launcher bakes the absolute path to python.exe into itself at
# install time, so they all fail with "Unable to create process". Catch that
# here and recreate, rather than failing mid-install. The probe itself must
# not use one of those launchers — see the comment on the check below.
function Remove-VenvOrExplain {
    <#
    Delete a venv directory, naming the process that prevents it.

    Why: Windows refuses to delete a file a running process has open, so a
    NoLlama server still serving from this venv makes Remove-Item throw
    mid-install. With $ErrorActionPreference = "Stop" that aborts the whole
    installer on an error that says "access denied" and not "your server is
    still running" — which is the one thing the user needs to be told.
    Observed 2026-09-13 on the B60 (a server left running over SSH).

    In: the venv directory. Out: nothing on success; on failure it prints the
    holding process ids and exits 1, because continuing would build on top of
    a venv that is half-deleted.
    #>
    param([string] $Dir)

    try {
        Remove-Item -Recurse -Force $Dir -ErrorAction Stop
    } catch {
        Write-Host ""
        Write-Host "ERROR: could not delete $Dir" -ForegroundColor Red
        $holders = @(Get-Process python, pythonw -ErrorAction SilentlyContinue |
                     Where-Object { $_.Path -and $_.Path.StartsWith($Dir, [StringComparison]::OrdinalIgnoreCase) })
        if ($holders) {
            Write-Host "  A process is still using it - almost certainly a running NoLlama:" -ForegroundColor Yellow
            foreach ($h in $holders) { Write-Host "    PID $($h.Id)  $($h.Path)" -ForegroundColor Yellow }
            Write-Host "  Stop it and re-run:" -ForegroundColor Yellow
            Write-Host "    Stop-Process -Id $($holders[0].Id) -Force" -ForegroundColor Yellow
        } else {
            Write-Host "  $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "  Nothing obvious is holding it; close anything using the venv and re-run." -ForegroundColor Yellow
        }
        exit 1
    }
}

if (Test-Path $VenvDir) {
    # Probe with python.exe -m pip, NEVER pip.exe. pip.exe is a generated
    # console-script shim, and an application-control policy can refuse to run
    # one while python.exe from the same venv is fine — that is the B60 box
    # (docs/dev/machines.md: venv\Scripts\hf.exe is blocked the same way).
    # Probing the shim made a HEALTHY venv look broken there, and the broken
    # branch below deletes without asking, so this mis-probe was one stray
    # process away from destroying a working install on a machine where pip
    # cannot easily rebuild one (its PyPI downloads die on an SSL record-MAC
    # error). Verified 2026-09-13 on the B60: pip.exe --version produces
    # nothing, python.exe -m pip --version prints pip 26.2.1.
    $venvPython = Join-Path $VenvDir $VenvBinDir "python$ExeExt"
    $venvOk = $false
    if (Test-Path $venvPython) {
        & $venvPython -m pip --version 2>&1 | Out-Null
        $venvOk = ($LASTEXITCODE -eq 0)
    }
    if ($venvOk) {
        # Show what's actually installed: "venv exists" hid that the runtime
        # could be releases behind (requirements.txt floors are >=, so a
        # fresh venv always gets the newest OpenVINO).
        $genaiVer = & $venvPython -c "import openvino_genai as og; print(og.__version__)" 2>$null
        if (-not $genaiVer) { $genaiVer = "openvino-genai not installed?" }
        Write-Host "[OK] $VenvName already exists (openvino-genai $genaiVer)"
        Write-Host "     Recreating it pulls the newest OpenVINO runtime." -ForegroundColor DarkGray
        $reply = Read-Host "     Delete and recreate venv for a fresh install? [y/N]"
        if ($reply -in @("y", "Y", "yes")) {
            Remove-VenvOrExplain -Dir $VenvDir
        }
    } else {
        Write-Host "[!] $VenvName at $VenvDir is broken (likely moved from another path). Recreating..." -ForegroundColor Yellow
        Remove-VenvOrExplain -Dir $VenvDir
    }
}

if (-not (Test-Path $VenvDir)) {
    # Windows ships 'python'; most Linux distros only ship 'python3'. Find
    # whichever is on PATH for the bootstrap. After the venv exists, plain
    # 'python' resolves to the venv's binary on both platforms.
    $sysPython = @(
        (Get-Command python  -ErrorAction SilentlyContinue),
        (Get-Command python3 -ErrorAction SilentlyContinue)
    ) | Where-Object { $_ } | Select-Object -First 1
    if (-not $sysPython) {
        Write-Host "ERROR: Neither 'python' nor 'python3' found in PATH." -ForegroundColor Red
        Write-Host "  Install Python 3.10+ (python.org on Windows, your package manager on Linux)." -ForegroundColor Yellow
        Pop-Location; exit 1
    }
    Write-Host "Creating Python $VenvName (using $($sysPython.Source))..."
    & $sysPython.Source -m venv $VenvDir
    if (-not $?) { Write-Host "ERROR: Failed to create venv." -ForegroundColor Red; Pop-Location; exit 1 }
    Write-Host "[OK] $VenvName created"
}

$ActivateScript = Join-Path $VenvDir $VenvBinDir "Activate.ps1"
& $ActivateScript

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip wheel setuptools 2>&1 | Out-Null
if ($Nightly) {
    # Two passes, deliberately. Everything that lives on PyPI installs
    # normally; only the three OpenVINO packages get --pre against the
    # nightly index. A single --pre pass over the whole set would also let
    # pip pick release candidates of numpy/pillow/etc, which is a different
    # (and unwanted) experiment.
    python -m pip install -r (Join-Path $ScriptDir "requirements-nightly.txt")
    if (-not $?) { Write-Host "ERROR: pip install failed (base deps)" -ForegroundColor Red; Pop-Location; exit 1 }

    Write-Host "Upgrading OpenVINO to nightly..." -ForegroundColor Yellow
    python -m pip install --pre -U openvino openvino-tokenizers openvino-genai --extra-index-url $NightlyIndex
    if (-not $?) { Write-Host "ERROR: pip install failed (nightly OpenVINO)" -ForegroundColor Red; Pop-Location; exit 1 }

    # Print what actually landed. The versions are the thing to quote when
    # reporting a nightly result upstream — "openvino nightly" is not a
    # reproducible statement, "2026.4.0.dev20260814" is.
    $ovVers = python -c "import openvino, openvino_genai; print(openvino.__version__); print(openvino_genai.__version__)" 2>$null
    if ($ovVers) {
        $lines = @($ovVers -split "`r?`n" | Where-Object { $_ })
        Write-Host "[OK] Nightly runtime: openvino $($lines[0])" -ForegroundColor Green
        Write-Host "                      openvino-genai $($lines[1])" -ForegroundColor Green
        Write-Host "     Quote these two lines in any bug report." -ForegroundColor DarkGray
    } else {
        Write-Host "[!] Installed, but 'import openvino_genai' failed - the nightly may be broken today." -ForegroundColor Yellow
        Write-Host "    Retry tomorrow, or pin an older nightly with -NightlyIndex." -ForegroundColor DarkGray
    }
} else {
    python -m pip install -r (Join-Path $ScriptDir "requirements.txt")
    if (-not $?) { Write-Host "ERROR: pip install failed" -ForegroundColor Red; Pop-Location; exit 1 }
}
Write-Host "[OK] Dependencies installed"
Write-Host ""

# ---------------------------------------------------------------------------
# 2. Detect devices
# ---------------------------------------------------------------------------

Write-Host "Detecting devices..." -ForegroundColor Cyan
# Mirror nollama.py's detect_devices(): canonical-keyed {kind: {id, name}}.
# Filter non-Intel GPUs (NVIDIA/AMD enumerated via OpenCL are unusable —
# crash with CL_INVALID_VALUE at warmup). Normalize multi-GPU enumeration
# (GPU.0/GPU.1) to a single canonical "GPU" entry pointing at the first
# Intel GPU; "id" preserves the real OpenVINO device id for --device.
$DeviceInfo = python -c @"
import openvino as ov, json
core = ov.Core()
out = {}
for dev in core.get_available_devices():
    try: full = core.get_property(dev, 'FULL_DEVICE_NAME')
    except: full = dev
    if dev.startswith('GPU'):
        if 'intel' not in full.lower(): continue
        if 'GPU' not in out:
            # XMX (systolic) gates OpenVINO's MoE disk offload: without it,
            # big MoE models must fit entirely in GPU memory (see TODONT.md).
            try: caps = core.get_property(dev, 'OPTIMIZATION_CAPABILITIES')
            except: caps = []
            # DEVICE_TYPE decides whether the CPU is a free lane. On a
            # discrete card it is: the side-model split measured well on the
            # B60. On an integrated one the CPU shares the package with the
            # GPU, and the same 128-token side request went from 1.6 s idle
            # to 40 s while the iGPU prefilled (2026-09-23, 140V).
            try: kind = 'discrete' if 'DISCRETE' in str(core.get_property(dev, 'DEVICE_TYPE')).upper() else 'integrated'
            except: kind = 'unknown'
            try: mem_gb = round(core.get_property(dev, 'GPU_DEVICE_TOTAL_MEM_SIZE') / 2**30, 1)
            except: mem_gb = 0
            out['GPU'] = {'id': dev, 'name': full, 'xmx': 'GPU_HW_MATMUL' in caps,
                          'type': kind, 'mem_gb': mem_gb}
    elif dev in ('NPU', 'CPU'):
        out[dev] = {'id': dev, 'name': full}
        if dev == 'NPU':
            # DEVICE_ARCHITECTURE is the ONLY thing that separates NPU
            # generations: FULL_DEVICE_NAME is 'Intel(R) AI Boost' on all of
            # them. '3720' = NPU 3 (Meteor/Arrow Lake), '4000' = NPU 4 (Lunar
            # Lake). Some models are correct on one and silently wrong on the
            # other, so the model menu gates on this (models.json
            # npu_arch_deny).
            try: out[dev]['arch'] = core.get_property(dev, 'DEVICE_ARCHITECTURE')
            except: out[dev]['arch'] = ''
print(json.dumps(out))
"@ | ConvertFrom-Json

# Turn a DEVICE_ARCHITECTURE string into the name people actually use. The
# generation matters because model defects track it: an IR can be correct on
# NPU 3 and word salad on NPU 4 with everything else held constant.
function Get-NpuGenLabel {
    param([string]$Arch)
    switch ($Arch) {
        "3720" { "NPU 3, Meteor/Arrow Lake" }
        "4000" { "NPU 4, Lunar Lake" }
        ""     { "generation unknown" }
        default { "arch $Arch" }
    }
}

$HasNPU = $null -ne $DeviceInfo.NPU
$HasGPU = $null -ne $DeviceInfo.GPU
$IsDiscreteGPU = $HasGPU -and $DeviceInfo.GPU.type -eq "discrete"
$SystemRamGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
# On an integrated GPU the "GPU memory" is system RAM, so a model competes
# with the OS, the user's applications and the loader's own staging copy.
# 8 GB goes to the OS and the applications the person is actually using; a
# 16 GB model with a 5 GB pool on a 32 GB laptop left 1-2 GB free with
# ordinary apps open (2026-09-23, 140V), which is where paging starts.
$UsableModelGB = if ($IsDiscreteGPU) { $DeviceInfo.GPU.mem_gb } else { [math]::Max(0, $SystemRamGB - 8) }

Write-Host ""
if ($HasNPU) {
    Write-Host "  [+] NPU: $($DeviceInfo.NPU.name) ($(Get-NpuGenLabel $DeviceInfo.NPU.arch))" -ForegroundColor Green
} else {
    Write-Host "  [-] NPU: not found" -ForegroundColor DarkGray
}
if ($HasGPU) {
    $gpuSuffix = if ($DeviceInfo.GPU.id -ne "GPU") { " [$($DeviceInfo.GPU.id)]" } else { "" }
    Write-Host "  [+] GPU$($gpuSuffix): $($DeviceInfo.GPU.name)" -ForegroundColor Green
    if ($IsDiscreteGPU) {
        Write-Host "      Discrete, $($DeviceInfo.GPU.mem_gb) GB of its own memory" -ForegroundColor DarkGray
    } else {
        Write-Host "      Integrated: its memory IS system RAM ($SystemRamGB GB total), so a model" -ForegroundColor DarkGray
        Write-Host "      competes with everything else you run. Budgeting $UsableModelGB GB for models." -ForegroundColor DarkGray
    }
    if ($DeviceInfo.GPU.xmx) {
        Write-Host "      XMX: yes — large MoE models can stream experts from disk (OpenVINO 2026.3+)" -ForegroundColor DarkGray
    } else {
        Write-Host "      XMX: no — MoE disk offload will NOT work on this GPU; models must fit" -ForegroundColor Yellow
        Write-Host "      entirely in GPU memory. Size your model choice accordingly." -ForegroundColor Yellow
    }
} else {
    Write-Host "  [-] GPU: not found (non-Intel GPUs are filtered)" -ForegroundColor DarkGray
}
Write-Host "  [+] CPU: $($DeviceInfo.CPU.name)" -ForegroundColor DarkGray
Write-Host ""

# ---------------------------------------------------------------------------
# 3. Scan existing local models in ~/models/
# ---------------------------------------------------------------------------

$LocalModels = @()
if (Test-Path $ModelsRoot) {
    # Require the .bin + .xml pair: a folder with only the big weights file
    # (interrupted download) would be offered as "Already on disk" and then
    # fail at load with "Could not find a model in the directory" (#17).
    $LocalModels = @(Get-ChildItem -Path $ModelsRoot -Directory | Where-Object {
        ((Test-Path (Join-Path $_.FullName "openvino_language_model.bin")) -and
         (Test-Path (Join-Path $_.FullName "openvino_language_model.xml"))) -or
        ((Test-Path (Join-Path $_.FullName "openvino_model.bin")) -and
         (Test-Path (Join-Path $_.FullName "openvino_model.xml")))
    } | ForEach-Object {
        $vlmBin = Join-Path $_.FullName "openvino_language_model.bin"
        $llmBin = Join-Path $_.FullName "openvino_model.bin"
        $binPath = if (Test-Path $vlmBin) { $vlmBin } else { $llmBin }
        $binSize = (Get-Item $binPath).Length
        $sizeGB = [math]::Round($binSize / 1GB, 1)
        # Mirror nollama.py is_vlm(): the definitive VLM signal is the
        # presence of a separate vision encoder; fall back to arch/model_type
        # keys for older exports. Catches new generations (Qwen3.5 reports
        # Qwen3_5ForConditionalGeneration / qwen3_5, matching no key).
        $mtype = "llm"
        # An embedding model ships the same openvino_model.bin/.xml pair as a
        # chat model, so without this it lands in the chat and coding-agent
        # menus -- which is exactly what happened (all-MiniLM offered as a
        # coding agent, 2026-09-23). Encoder architectures and the
        # sentence-transformers layout are the tells.
        # The IR itself is the reliable tell, and it needs no config.json --
        # which matters, because Intel's nomic-embed export ships only the
        # OpenVINO files. A generative model produces "logits"; an encoder
        # produces last_hidden_state and no logits.
        #
        # NOT past_key_values, which was the obvious choice and is wrong:
        # LFM2.5, Qwen3.5-4B and Qwen3.8-27B have no past_key_values anywhere
        # in their IR because of linear/hybrid attention, and would have been
        # classified as embedding models and dropped from every chat menu
        # (caught before shipping, 2026-09-23). The whole file is read because
        # results are declared at the end.
        $irXml = if (Test-Path (Join-Path $_.FullName "openvino_model.xml")) {
            Join-Path $_.FullName "openvino_model.xml"
        } else { Join-Path $_.FullName "openvino_language_model.xml" }
        $isEmbed = $false
        try { $isEmbed = -not ([System.IO.File]::ReadAllText($irXml) -match '"logits"') } catch {}
        if ($isEmbed) {
            $mtype = "embed"
        } elseif (Test-Path (Join-Path $_.FullName "openvino_vision_embeddings_model.xml")) {
            $mtype = "vlm"
        } else {
            $cfgPath = Join-Path $_.FullName "config.json"
            if (Test-Path $cfgPath) {
                try {
                    $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
                    $arch = ""
                    if ($cfg.architectures -and $cfg.architectures.Count -gt 0) { $arch = $cfg.architectures[0].ToLower() }
                    $mt = if ($cfg.model_type) { $cfg.model_type.ToLower() } else { "" }
                    if ($arch -match "vl|vision|llava|qwen2vl|internvl|minicpm" -or $mt -match "vl|vision") {
                        $mtype = "vlm"
                    }
                } catch {}
            }
        }
        # Detect NPU compatibility: needs int4 quantization and NPU-class size.
        # Matches the older "-int4-cw" / "-cw-ov" naming and the newer plain
        # "-int4-ov" exports (e.g. Qwen3.5). Soft filter — user still confirms.
        # Size cap 6 GB ~ 8B params at int4: proven NPU models top out at 8B
        # (~4.4 GB); a 14B int4 (~8 GB) slipped under the old 10 GB cap and
        # died in the NPU compiler (#20).
        $npuOk = ($_.Name -match "int4") -and $sizeGB -lt 6
        [PSCustomObject]@{ Name = $_.Name; Path = $_.FullName; SizeGB = $sizeGB; Type = $mtype; NpuOk = $npuOk }
    })
}

if ($LocalModels.Count -gt 0) {
    Write-Host "  Local models ($ModelsRoot):" -ForegroundColor DarkGray
    foreach ($lm in $LocalModels) {
        Write-Host "    $($lm.Name)  ($($lm.SizeGB) GB, $($lm.Type.ToUpper()))" -ForegroundColor DarkGray
    }
    Write-Host ""
}

if ($SkipModel) {
    Write-Host "Skipping model selection (-SkipModel)"
    Write-Host ""
    Write-Host "=== Install complete (no model) ===" -ForegroundColor Yellow
    Pop-Location; exit 0
}

# Say whether a model fits the budget worked out at detection, and why not.
#
# Why annotate instead of filtering: a hidden option is a fact withheld. The
# user is entitled to know that the 17 GB model exists and what it would take.
# $UsableModelGB is VRAM on a discrete card and (system RAM - 8 GB) on an
# integrated one, where the model, the KV pool and the OS share the same
# memory.
function Get-FitTag {
    param([double]$SizeGB)
    if ($script:UsableModelGB -le 0 -or $SizeGB -le 0) { return "" }
    # Weights are not the whole cost: the prompt cache is what makes a second
    # turn fast, and it scales with the model (2-4 GB in practice). A tag that
    # counted weights alone would wave through a model that then has no cache.
    $poolGB  = [math]::Min(4, [math]::Max(2, [math]::Round($SizeGB / 3)))
    $needGB  = $SizeGB + $poolGB
    if ($needGB -gt $script:UsableModelGB) {
        return "  [WON'T FIT: ~$needGB GB with a cache, you have ~$($script:UsableModelGB) GB]"
    }
    if ($needGB -gt ($script:UsableModelGB - 2)) {
        return "  [TIGHT: ~$needGB GB with a cache — close the browser]"
    }
    return ""
}

# ---------------------------------------------------------------------------
# Helper: show a model menu and return the selection
# ---------------------------------------------------------------------------

$Registry = Get-Content (Join-Path $ScriptDir "models.json") -Raw | ConvertFrom-Json

function Show-ModelMenu {
    param(
        [string]$Title,
        [array]$RegistryModels,
        [array]$LocalModels,
        [string]$LocalLabel = "Already on disk (instant)",
        [bool]$AllowSkip = $false
    )

    if ($null -eq $LocalModels) { $LocalModels = @() }

    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host ""

    $items = @()

    # Partition into on-disk vs downloadable.
    #
    # On-disk has two sources:
    #   1. The generic ~/models scan passed in as $LocalModels.
    #   2. Any registry model whose cache already exists. This catches
    #      multimodal models that the type-based scan filtered out of THIS
    #      menu — e.g. Qwen3.5 reports architecture Qwen3_5ForConditional-
    #      Generation, so the scan tags it "llm" and it never reaches the
    #      vlm-filtered vision menu. Checking the cache directly (the same
    #      path Install-Model would use) is independent of that fragile
    #      classification, so it shows as instant instead of a bogus download.
    $onDisk = @()
    foreach ($lm in $LocalModels) {
        # AgentTag is set by Sort-AgentLocal and is the only thing that tells
        # a user which of their downloaded models has actually been measured.
        $note = if ($lm.PSObject.Properties['AgentTag'] -and $lm.AgentTag) { $lm.AgentTag } else { "Already on disk" }
        $onDisk += [PSCustomObject]@{
            Action = "local"; Name = $lm.Name; Path = $lm.Path
            HfId = $null; Source = $null; Weight = $null; Trust = $false
            SizeGB = $lm.SizeGB; Notes = $note
        }
    }

    $localNames = @($LocalModels | ForEach-Object { $_.Name.ToLower() })
    $downloads = @()
    $hiddenNightly = 0
    foreach ($dm in $RegistryModels) {
        # Models whose IR only loads on the OpenVINO nightly runtime stay out
        # of the stable menu entirely — offering a 16 GB download that then
        # fails to load is worse than not offering it. Same filter for the
        # on-disk branch below: having the weights doesn't help if this venv
        # can't read them.
        if ($dm.requires_nightly -and -not $Nightly) { $hiddenNightly++; continue }

        $repoName = ($dm.hf_id -split '/')[-1]
        # Already surfaced by the generic scan (matched on folder name)?
        if ($repoName.ToLower() -in $localNames) { continue }

        # Compute the cache path Install-Model would use (convert appends the
        # weight format so int4/int8 of the same model don't collide).
        $cacheName = if ($dm.source -eq "convert") { "$repoName-$($dm.weight_format)" } else { $repoName }
        $cachePath = Join-Path $ModelsRoot $cacheName

        if (Test-ModelCacheValid -Path $cachePath) {
            $onDisk += [PSCustomObject]@{
                Action = "local"; Name = $dm.name; Path = $cachePath
                HfId = $dm.hf_id; Source = $dm.source
                Weight = $dm.weight_format; Trust = $dm.trust_remote_code
                SizeGB = $dm.est_size_gb; Notes = "Already on disk"
            }
        } else {
            $downloads += [PSCustomObject]@{
                Action = $dm.source; Name = $dm.name; Path = $null
                HfId = $dm.hf_id; Source = $dm.source
                Weight = $dm.weight_format; Trust = $dm.trust_remote_code
                SizeGB = $dm.est_size_gb; Notes = $dm.notes
            }
        }
    }

    if ($onDisk.Count -gt 0) {
        Write-Host "  $LocalLabel" -ForegroundColor Yellow
        foreach ($od in $onDisk) {
            $items += $od
            $i = $items.Count
            $noteColor = if ($od.Notes -like "NOT an agent*") { "DarkYellow" } else { "DarkGray" }
            Write-Host "    $i. $($od.Name)" -NoNewline
            Write-Host "  ($($od.SizeGB) GB)" -ForegroundColor DarkGray -NoNewline
            Write-Host "  $($od.Notes)" -ForegroundColor $noteColor -NoNewline
            Write-Host (Get-FitTag $od.SizeGB) -ForegroundColor Yellow
        }
        Write-Host ""
    }

    if ($downloads.Count -gt 0) {
        Write-Host "  Download from HuggingFace:" -ForegroundColor Yellow
        foreach ($dm in $downloads) {
            $items += $dm
            $dlTag = if ($dm.Source -eq "pre-exported") { "download" } else { "convert" }
            $i = $items.Count
            $fit = Get-FitTag $dm.SizeGB
            Write-Host "    $i. $($dm.Name)" -NoNewline
            Write-Host "  (~$($dm.SizeGB) GB, $dlTag)" -ForegroundColor DarkGray -NoNewline
            if ($fit) { Write-Host $fit -ForegroundColor Yellow -NoNewline }
            Write-Host "  $($dm.Notes)" -ForegroundColor DarkGray
        }
    }

    if ($script:AgentLocalOmitted -gt 0) {
        Write-Host ""
        Write-Host "  ($($script:AgentLocalOmitted) smaller local model(s) not listed: too small to drive a tool" -ForegroundColor DarkGray
        Write-Host "   loop. Any model still runs with 'python nollama.py --model-dir <path>'.)" -ForegroundColor DarkGray
        $script:AgentLocalOmitted = 0
    }

    if ($hiddenNightly -gt 0) {
        Write-Host ""
        Write-Host "  ($hiddenNightly model(s) hidden: they need the OpenVINO nightly runtime." -ForegroundColor DarkGray
        Write-Host "   Re-run as '.\install.ps1 -Nightly' to see them.)" -ForegroundColor DarkGray
    }

    Write-Host ""

    if ($AllowSkip) {
        $prompt = "Pick a model [1-$($items.Count)] or press Enter to skip"
    } else {
        $prompt = "Pick a model [1-$($items.Count)]"
    }

    while ($true) {
        $choice = Read-Host $prompt
        if ($AllowSkip -and [string]::IsNullOrWhiteSpace($choice)) {
            return $null
        }
        $num = 0
        if ([int]::TryParse($choice, [ref]$num) -and $num -ge 1 -and $num -le $items.Count) {
            return $items[$num - 1]
        }
        Write-Host "Enter a number between 1 and $($items.Count)" -ForegroundColor Red
    }
}

# ---------------------------------------------------------------------------
# Helper: download or link a model into a target directory
# ---------------------------------------------------------------------------

function Test-ModelCacheValid {
    # A cache is valid only if the main weights .bin file exists AND is
    # substantial (>100 MB) AND its matching .xml descriptor exists. Partial
    # downloads fail both ways: the XML + small tokenizer files complete
    # quickly while the multi-GB weights file may be 0 bytes or missing —
    # and the reverse (big .bin, no .xml) also happens and makes OpenVINO
    # fail at load with "Could not find a model in the directory" (#17).
    # Smallest real model in our registry (DeepSeek-1.5B INT4) is ~700 MB;
    # tokenizer .bin files top out around 10 MB. 100 MB separates the two.
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $false }
    foreach ($base in @("openvino_language_model", "openvino_model")) {
        $bin = Join-Path $Path "$base.bin"
        $xml = Join-Path $Path "$base.xml"
        if (-not ((Test-Path $bin) -and ((Get-Item $bin).Length -gt 100MB) -and (Test-Path $xml))) {
            continue
        }
        # Truncation check: the IR .xml records each weight blob's
        # offset+size into the .bin, so max(offset+size) is the exact
        # minimum byte count the .bin must have. Catches a download that
        # lost even the last few bytes; a truncated cache is re-fetched.
        $need = [int64]0
        foreach ($m in [regex]::Matches((Get-Content $xml -Raw), 'offset="(\d+)" size="(\d+)"')) {
            $end = [int64]$m.Groups[1].Value + [int64]$m.Groups[2].Value
            if ($end -gt $need) { $need = $end }
        }
        if ($need -gt 0 -and (Get-Item $bin).Length -lt $need) {
            Write-Host "  $base.bin in $Path is truncated ($((Get-Item $bin).Length) of $need bytes) - re-fetching." -ForegroundColor Yellow
            continue
        }
        return $true
    }
    return $false
}

function New-ModelJunction {
    # Windows: junction (works without admin/dev-mode).
    # POSIX:   symlink.
    param([string]$TargetDir, [string]$CachePath)
    if (Test-Path $TargetDir) {
        $item = Get-Item $TargetDir -Force
        if ($item.LinkType) {
            # Remove the link without following it.
            if ($IsWindows) { cmd /c rmdir "`"$TargetDir`"" | Out-Null }
            else            { Remove-Item -Force $TargetDir }
        } else {
            Remove-Item -Recurse -Force $TargetDir
        }
    }
    if ($IsWindows) {
        cmd /c mklink /J "`"$TargetDir`"" "`"$CachePath`"" | Out-Null
    } else {
        New-Item -ItemType SymbolicLink -Path $TargetDir -Target $CachePath | Out-Null
    }
}

function Install-Model {
    param(
        [PSCustomObject]$Selected,
        [string]$TargetDir
    )

    if ($Selected.Action -eq "local") {
        Write-Host "Linking to: $($Selected.Path)" -ForegroundColor Green
        New-ModelJunction -TargetDir $TargetDir -CachePath $Selected.Path
        Write-Host "[OK] $($Selected.Name)" -ForegroundColor Green
        return $true
    }

    # pre-exported and convert both cache into ~/models/<name>/ first, then
    # junction $TargetDir → cache. Lets re-installs detect the existing
    # model (scan looks at ~/models/) and skip the download.
    if ($Selected.Action -eq "pre-exported") {
        $cacheName = ($Selected.HfId -split '/')[-1]
        $cachePath = Join-Path $ModelsRoot $cacheName

        if (Test-ModelCacheValid -Path $cachePath) {
            Write-Host "Using cached $($Selected.Name) at $cachePath" -ForegroundColor Green
        } else {
            if (Test-Path $cachePath) {
                Write-Host "  Found incomplete cache at $cachePath, removing." -ForegroundColor DarkGray
                Remove-Item -Recurse -Force $cachePath
            }
            New-Item -ItemType Directory -Path $ModelsRoot -Force | Out-Null
            Write-Host "Downloading $($Selected.Name)..." -ForegroundColor Cyan
            Write-Host "  From: $($Selected.HfId)"
            Write-Host "  To:   $cachePath"
            Write-Host ""
            $env:PYTHONIOENCODING = "utf-8"
            # An entry may pin a repo branch/tag ("revision"): Intel keeps the
            # IR for the current release on a branch while main tracks the next
            # runtime (Qwen3.8's main branch segfaults 2026.3.x at load).
            $revArgs = @()
            if ($Selected.PSObject.Properties['revision'] -and $Selected.revision) {
                $revArgs = @("--revision", $Selected.revision)
                Write-Host "  Revision: $($Selected.revision)"
            }
            hf download $Selected.HfId --local-dir $cachePath @revArgs
            if (-not $?) {
                Write-Host "ERROR: Download failed." -ForegroundColor Red
                Write-Host "  If 401/403 (gated/private model): re-run with a token --" -ForegroundColor Yellow
                Write-Host "    .\install.ps1 -HfToken hf_xxx   (get one at https://huggingface.co/settings/tokens)" -ForegroundColor Yellow
                return $false
            }
        }

        New-ModelJunction -TargetDir $TargetDir -CachePath $cachePath
        Write-Host "[OK] $($Selected.Name)" -ForegroundColor Green
        return $true
    }

    if ($Selected.Action -eq "convert") {
        # Include weight format in cache name so int4 and int8 conversions
        # of the same model don't collide.
        $cacheName = "$(($Selected.HfId -split '/')[-1])-$($Selected.Weight)"
        $cachePath = Join-Path $ModelsRoot $cacheName

        if (Test-ModelCacheValid -Path $cachePath) {
            Write-Host "Using cached $($Selected.Name) at $cachePath" -ForegroundColor Green
        } else {
            if (Test-Path $cachePath) {
                Write-Host "  Found incomplete cache at $cachePath, removing." -ForegroundColor DarkGray
                Remove-Item -Recurse -Force $cachePath
            }
            New-Item -ItemType Directory -Path $ModelsRoot -Force | Out-Null
            Write-Host "Converting $($Selected.Name)..." -ForegroundColor Cyan
            Write-Host "  From: $($Selected.HfId)"
            Write-Host "  To:   $cachePath"
            Write-Host "  This may take 5-20 minutes."
            Write-Host ""
            $args = @("export", "openvino", "--model", $Selected.HfId, "--weight-format", $Selected.Weight)
            if ($Selected.Trust) { $args += "--trust-remote-code" }
            $args += $cachePath
            Write-Host "Running: optimum-cli $($args -join ' ')" -ForegroundColor DarkGray
            & optimum-cli @args
            if (-not $?) {
                Write-Host "ERROR: Conversion failed." -ForegroundColor Red
                Write-Host "  If unsupported architecture: needs newer optimum-intel" -ForegroundColor Yellow
                return $false
            }
        }

        New-ModelJunction -TargetDir $TargetDir -CachePath $cachePath
        Write-Host "[OK] $($Selected.Name)" -ForegroundColor Green
        return $true
    }

    Write-Host "ERROR: Unknown action '$($Selected.Action)'" -ForegroundColor Red
    return $false
}

# ---------------------------------------------------------------------------
# 4. Model selection — use-case first
# ---------------------------------------------------------------------------
# Ask what the user wants to DO, then place each model on the best device.
# NoLlama runs ONE primary model + (optionally) ONE GPU secondary, so the combos
# are: chat / agent / vision alone, or NPU(or CPU) chat + a GPU coder/vision.

$ModelDir = Join-Path $ScriptDir "model"
$GpuModelDir = Join-Path $ScriptDir "gpu-model"
$StartArgs = @()  # collect args for start.ps1

function Select-Device {
    param([string]$Purpose, [string[]]$Choices, [string]$Note = "")
    if ($Choices.Count -eq 1) { return $Choices[0] }
    Write-Host ""
    Write-Host "  Run $Purpose on which device?" -ForegroundColor Cyan
    if ($Note) { Write-Host "    $Note" -ForegroundColor DarkGray }
    for ($i = 0; $i -lt $Choices.Count; $i++) { Write-Host "    $($i + 1). $($Choices[$i])" }
    while ($true) {
        $c = Read-Host "  [1-$($Choices.Count)]"
        $n = 0
        if ([int]::TryParse($c, [ref]$n) -and $n -ge 1 -and $n -le $Choices.Count) { return $Choices[$n - 1] }
        Write-Host "  Enter 1-$($Choices.Count)" -ForegroundColor Red
    }
}

# Drop registry entries that cannot work on the device they would be placed on.
# Two gates, both measured rather than inferred:
#   npu_arch_deny - correct on one NPU generation and silently wrong on another.
#                   LFM2.5-1.2B decodes at full speed on NPU 4 and returns word
#                   salad; models.json records the drivers, runtimes and
#                   compilers that were held constant while proving it.
#   npu_only      - an NPU-shaped export (static shapes, int4 channel-wise) that
#                   does not load on CPU/GPU at all.
# Listing a model IS recommending it, so one that cannot work here does not
# belong in the list. Say what was dropped and why, though: a silent absence
# reads as a missing download and sends people hunting for it.
function Where-NpuUsable {
    param([object[]]$Models, [string]$Device)
    $arch = if ($HasNPU) { [string]$DeviceInfo.NPU.arch } else { "" }
    $blocked = @($Models | Where-Object {
        ($Device -eq "NPU" -and $arch -and ($_.npu_arch_deny -contains $arch)) -or
        ($Device -ne "NPU" -and $_.npu_only)
    })
    if ($blocked.Count -gt 0) {
        $why = if ($Device -eq "NPU") { "not usable on this NPU ($(Get-NpuGenLabel $arch))" }
               else { "NPU-only builds, they do not load on $Device" }
        Write-Host ""
        Write-Host "  Not offered - ${why}:" -ForegroundColor Yellow
        foreach ($b in $blocked) { Write-Host "    - $($b.name)" -ForegroundColor DarkGray }
        Write-Host "  docs/MODELS.md explains." -ForegroundColor DarkGray
    }
    @($Models | Where-Object { $blocked -notcontains $_ })
}

# What the registry knows about a local folder, matched on the repo name the
# downloader used. Without this a local model is an anonymous directory and the
# menu cannot say whether we have tested it.
function Find-RegistryEntry {
    param([string]$Name)
    foreach ($bucket in @("npu", "gpu_llm", "gpu_vlm", "embed", "whisper")) {
        foreach ($e in $Registry.$bucket) {
            if ((($e.hf_id -split '/')[-1]) -ieq $Name) { return $e }
        }
    }
    # Fall back to the family, ignoring the quantization suffix: whether a
    # model can drive a tool loop is a property of its training, not of
    # int4 vs int8. Without this, Qwen3-8B-int4-ov learns nothing from what
    # we measured on Qwen3-8B-int4-cw-ov.
    $stem = ($Name -replace '-(int4|int8|fp16|bf16)(-cw)?(-ov)?$', '') -replace '-ov$', ''
    foreach ($bucket in @("npu", "gpu_llm", "gpu_vlm")) {
        foreach ($e in $Registry.$bucket) {
            $rstem = ((($e.hf_id -split '/')[-1]) -replace '-(int4|int8|fp16|bf16)(-cw)?(-ov)?$', '') -replace '-ov$', ''
            if ($rstem -ieq $stem) { return $e }
        }
    }
    return $null
}

# Rank and label local models for the coding-agent menu.
#
# Why: the agent menu was listing every folder in ~/models alphabetically --
# embedders included -- so the models measured to FAIL sat above the one that
# works (2026-09-23). Verified-capable first, untested next, measured-failures
# last and labelled. Nothing is hidden: a model you downloaded stays visible,
# it just stops being presented as a candidate.
function Sort-AgentLocal {
    param([object[]]$Models, [int]$KeepUntested = 4)
    $ranked = foreach ($m in $Models) {
        $reg = Find-RegistryEntry $m.Name
        $rank = 1; $tag = "untested for agents"
        if ($null -ne $reg -and $reg.PSObject.Properties['agent']) {
            if ($reg.agent) { $rank = 0; $tag = "verified agent model" }
            else            { $rank = 2; $tag = "NOT an agent model (tested)" }
        }
        [PSCustomObject]@{ M = $m; Rank = $rank; Tag = $tag }
    }
    $sorted = @($ranked | Sort-Object Rank, @{Expression = {-$_.M.SizeGB}})
    # Keep the list short enough to read. Everything verified stays; the
    # untested tail is cut to the largest few, because a 0.3 GB model is not a
    # coding agent candidate on any hardware. The count of what was left out
    # is printed by the caller -- summarised, not silently dropped.
    $keep = @($sorted | Where-Object { $_.Rank -eq 0 }) +
            @($sorted | Where-Object { $_.Rank -eq 1 } | Select-Object -First $KeepUntested) +
            @($sorted | Where-Object { $_.Rank -eq 2 })
    $script:AgentLocalOmitted = $sorted.Count - $keep.Count
    @($keep | ForEach-Object {
        $_.M | Add-Member -NotePropertyName AgentTag -NotePropertyValue $_.Tag -Force -PassThru
    })
}

# Chat can run anywhere; small NPU-class models + bigger GPU LLMs both work on GPU/CPU.
function Get-ChatRegistry { param([string]$Device)
    if ($Device -eq "NPU") { return Where-NpuUsable $Registry.npu "NPU" }
    return Where-NpuUsable (@($Registry.npu) + @($Registry.gpu_llm)) $Device
}
function Get-ChatLocal { param([string]$Device, [string]$Exclude = "")
    @($LocalModels | Where-Object { $_.Type -eq "llm" -and (($Device -ne "NPU") -or $_.NpuOk) -and $_.Name -ne $Exclude })
}
# The coding-agent menu: LLMs only (an embedder is not a coding agent), ranked
# by what has actually been measured.
function Get-AgentLocal { param([string]$Exclude = "")
    Sort-AgentLocal @($LocalModels | Where-Object { $_.Type -eq "llm" -and $_.Name -ne $Exclude })
}

# --- Use-case menu (filtered by available hardware) ---
Write-Host ""
Write-Host "=== What will you use NoLlama for? ===" -ForegroundColor Cyan
Write-Host ""
$cases = @()
# Labels lead with the device map — "[NPU] Chat + [GPU] Vision" tells you
# what runs where, which is the actual decision being made (user feedback).
# Combos pin chat to NPU (CPU when absent); single-purpose options let you
# pick the device on the next screen.
$chatDevices  = @(); if ($HasNPU) { $chatDevices += "NPU" }; if ($HasGPU) { $chatDevices += "GPU" }; $chatDevices += "CPU"
$agentDevices = @(); if ($HasGPU) { $agentDevices += "GPU" }; $agentDevices += "CPU"
$comboChatDev = if ($HasNPU) { "NPU" } else { "CPU" }
$cases += [PSCustomObject]@{ Key = "chat";   Label = "Chat";         Desc = "text assistant (you pick the device next: $($chatDevices -join '/'))" }
$cases += [PSCustomObject]@{ Key = "agent";  Label = "Coding agent"; Desc = "OpenCode / Goose / VS Code Copilot, tool-calling (pick GPU or CPU next)" }
if ($HasGPU) {
    $cases += [PSCustomObject]@{ Key = "vision";      Label = "[GPU] Vision + chat"; Desc = "image understanding; a vision model answers plain chat too" }
    $cases += [PSCustomObject]@{ Key = "chat+agent";  Label = "[$comboChatDev] Chat + [GPU] Coding agent"; Desc = "two models, one server" }
    $cases += [PSCustomObject]@{ Key = "chat+vision"; Label = "[$comboChatDev] Chat + [GPU] Vision";       Desc = "two models, one server (the classic NoLlama setup)" }
}
for ($i = 0; $i -lt $cases.Count; $i++) {
    Write-Host ("  {0}. {1}" -f ($i + 1), $cases[$i].Label) -NoNewline
    Write-Host "  $($cases[$i].Desc)" -ForegroundColor DarkGray
}
Write-Host ""
$useKey = $null
while ($null -eq $useKey) {
    $c = Read-Host "Pick [1-$($cases.Count)]"
    $n = 0
    if ([int]::TryParse($c, [ref]$n) -and $n -ge 1 -and $n -le $cases.Count) { $useKey = $cases[$n - 1].Key }
    else { Write-Host "Enter 1-$($cases.Count)" -ForegroundColor Red }
}

$coders = @($Registry.gpu_llm | Where-Object { $_.agent })   # OpenCode/Copilot-ready
$isAgent = $false
$SmallModelDir = Join-Path $ScriptDir "small-model"
$OpenCodeArgs = $null   # set by the agent cases; consumed after start.ps1 is written

function Install-Primary { param($Sel, [string]$Device)
    if (-not (Install-Model -Selected $Sel -TargetDir $ModelDir)) {
        Write-Host "Model installation failed. Re-run install.ps1 to retry." -ForegroundColor Yellow; Pop-Location; exit 1
    }
    $script:StartArgs += @("--device", $Device)
}

function Get-InstalledDirName { param($Sel)
    # The directory name Install-Model ends up linking to — what nollama.py
    # derives the advertised model id from (resolve_display_name follows the
    # generic model/ link to this). Mirrors the three branches of Install-Model:
    # local pick = its own folder; pre-exported = the HF repo name; convert =
    # repo name plus weight format. New-OpenCodeConfig.ps1 strips the export
    # suffixes the same way the server does.
    if ($Sel.Action -eq "local") { return (Split-Path $Sel.Path -Leaf) }
    $repo = ($Sel.HfId -split '/')[-1]
    if ($Sel.Source -eq "convert") { return "$repo-$($Sel.Weight)" }
    return $repo
}

switch ($useKey) {
    "chat" {
        $dev = Select-Device -Purpose "chat" -Choices $chatDevices
        $sel = Show-ModelMenu -Title "Chat model ($dev)" -RegistryModels (Get-ChatRegistry $dev) -LocalModels (Get-ChatLocal $dev)
        if ($sel) { Install-Primary $sel $dev }
    }
    "agent" {
        $dev = Select-Device -Purpose "the coding agent" -Choices $agentDevices `
            -Note "GPU is usually faster; CPU often wins on strong desktops / weak iGPUs."
        $loc = Get-AgentLocal
        $sel = Show-ModelMenu -Title "Coding agent model ($dev) - OpenCode / Copilot ready" -RegistryModels $coders -LocalModels $loc
        if ($sel) {
            Install-Primary $sel $dev; $StartArgs += @("--prewarm", "prewarm.json", "--vscode-compat", "--idle-timeout", "0"); $isAgent = $true
            # The two-server recipe (docs/AGENTS.md): OpenCode sends a small
            # "title" request beside every turn; on one server it queues in
            # front of the turn. A second NoLlama takes it, and opencode.json
            # points OpenCode's small_model there.
            #
            # Only offered on a DISCRETE GPU. Measured 2026-09-23 on a 140V:
            # the same 128-token side request runs in 1.6 s with the box idle
            # and 40 s while the iGPU prefills, because an integrated GPU
            # shares its package with the CPU. The NPU is not the alternative
            # either — it cannot hold a coding session at all (4k prompt cap,
            # no tool calling). So on an iGPU the split trades a 5-second
            # request on the coder for a 40-second one, and we do not ask.
            $smallDev = "CPU"
            if (-not $IsDiscreteGPU -or $dev -eq "CPU") {
                Write-Host ""
                Write-Host "  Single server: on an integrated GPU a second model on the CPU is starved while" -ForegroundColor DarkGray
                Write-Host "  the GPU prefills (measured: a side request goes 1.6s -> 40s). OpenCode's side" -ForegroundColor DarkGray
                Write-Host "  requests will use the coder itself. See docs/AGENTS.md." -ForegroundColor DarkGray
                $OpenCodeArgs = @{ CoderDir = (Get-InstalledDirName $sel); CoderDevice = $dev; Mode = "two-servers" }
                break
            }
            Write-Host ""
            Write-Host "  OpenCode sends a small side request with every turn. A second, small model on the CPU" -ForegroundColor Cyan
            Write-Host "  keeps it off the coder's queue — worth it here because your GPU is discrete, so the" -ForegroundColor DarkGray
            Write-Host "  CPU is genuinely free (Enter to skip; docs/AGENTS.md explains)." -ForegroundColor DarkGray
            # Models flagged small_model come first. A side-request slot wants a
            # model that ANSWERS, not the best reasoner: a thinking model spends
            # the whole side-request budget in <think> and can return empty
            # content (SmolLM3-3B on NPU 4 did exactly that, 3/3 — models.json
            # carries the measurement). Menu order is the recommendation, so the
            # non-thinking ones have to be at the top.
            $smallUsable = Where-NpuUsable $Registry.npu $smallDev
            $smallModels = @($smallUsable | Where-Object { $_.small_model }) +
                           @($smallUsable | Where-Object { -not $_.small_model })
            $smallSel = Show-ModelMenu -Title "Small model for OpenCode side-tasks ($smallDev)" -RegistryModels $smallModels -LocalModels (Get-ChatLocal $smallDev -Exclude $sel.Name) -AllowSkip $true
            if ($smallSel -and (Install-Model -Selected $smallSel -TargetDir $SmallModelDir)) {
                $OpenCodeArgs = @{ CoderDir = (Get-InstalledDirName $sel); CoderDevice = $dev; SmallDir = (Get-InstalledDirName $smallSel); SmallDevice = $smallDev; Mode = "two-servers" }
            } else {
                $OpenCodeArgs = @{ CoderDir = (Get-InstalledDirName $sel); CoderDevice = $dev; Mode = "two-servers" }
            }
        }
    }
    "vision" {
        $loc = @($LocalModels | Where-Object { $_.Type -eq "vlm" })
        $sel = Show-ModelMenu -Title "Vision model (GPU)" -RegistryModels $Registry.gpu_vlm -LocalModels $loc
        if ($sel) { Install-Primary $sel "GPU" }
    }
    "chat+agent" {
        $chatDev = if ($HasNPU) { "NPU" } else { "CPU" }
        $chatSel = Show-ModelMenu -Title "Chat model ($chatDev)" -RegistryModels (Get-ChatRegistry $chatDev) -LocalModels (Get-ChatLocal $chatDev)
        if ($chatSel) {
            Install-Primary $chatSel $chatDev
            $cloc = Get-AgentLocal -Exclude $chatSel.Name
            $coderSel = Show-ModelMenu -Title "Coding agent model (GPU) - OpenCode / Copilot ready" -RegistryModels $coders -LocalModels $cloc -AllowSkip $true
            if ($coderSel -and (Install-Model -Selected $coderSel -TargetDir $GpuModelDir)) {
                $StartArgs += @("--gpu-model-dir", "gpu-model", "--prewarm", "prewarm.json", "--vscode-compat", "--idle-timeout", "0"); $isAgent = $true
                # Dual mode: one process, two slots — OpenCode addresses them
                # as <name>@GPU (the coder) and <name>@NPU/CPU (small_model).
                $OpenCodeArgs = @{ CoderDir = (Get-InstalledDirName $coderSel); CoderDevice = "GPU"; SmallDir = (Get-InstalledDirName $chatSel); SmallDevice = $chatDev; Mode = "dual" }
            }
        }
    }
    "chat+vision" {
        $chatDev = if ($HasNPU) { "NPU" } else { "CPU" }
        $chatSel = Show-ModelMenu -Title "Chat model ($chatDev)" -RegistryModels (Get-ChatRegistry $chatDev) -LocalModels (Get-ChatLocal $chatDev)
        if ($chatSel) {
            Install-Primary $chatSel $chatDev
            $vloc = @($LocalModels | Where-Object { $_.Type -eq "vlm" })
            $visSel = Show-ModelMenu -Title "Vision model (GPU)" -RegistryModels $Registry.gpu_vlm -LocalModels $vloc -AllowSkip $true
            if ($visSel -and (Install-Model -Selected $visSel -TargetDir $GpuModelDir)) {
                $StartArgs += @("--gpu-model-dir", "gpu-model")
            }
        }
    }
}

if ($isAgent) {
    Write-Host ""
    Write-Host "Coding agent ready. Point OpenCode, Goose or Copilot Chat at it - see docs/AGENTS.md" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 5. Generate start.ps1
# ---------------------------------------------------------------------------

$StartScript = Join-Path $ScriptDir "start.ps1"
$TemplateScript = Join-Path $ScriptDir "start-template.ps1"
$ArgsStr = $StartArgs -join " "

# Generate start.ps1 — a one-liner that calls the template with the right args.
# -VenvName pins the launcher to the venv this install built, so a machine can
# hold both a stable venv/ and an experimental venv-nightly/ without start.ps1
# silently launching the wrong runtime.
$Content = "# Auto-generated by install.ps1`n"
$Content += "& '$(Join-Path $ScriptDir "start-template.ps1")' -ServerArgs '$ArgsStr' -VenvName '$VenvName' @args"
Set-Content -Path $StartScript -Value $Content -Encoding UTF8
Write-Host "[OK] Generated start.ps1" -ForegroundColor Green

# Coding-agent installs also get opencode.json — and, for the two-server
# recipe, start-small.ps1 for the second server (same template, other args).
if ($OpenCodeArgs) {
    if ($OpenCodeArgs.Mode -eq "two-servers" -and $OpenCodeArgs.SmallDir) {
        $SmallArgs = "--port 8002 --ollama-port 0 --device $($OpenCodeArgs.SmallDevice) --model-dir small-model --idle-timeout 0"
        $SmallContent = "# Auto-generated by install.ps1 — the second NoLlama server for OpenCode's small_model`n"
        $SmallContent += "& '$(Join-Path $ScriptDir "start-template.ps1")' -ServerArgs '$SmallArgs' -VenvName '$VenvName' @args"
        Set-Content -Path (Join-Path $ScriptDir "start-small.ps1") -Value $SmallContent -Encoding UTF8
        Write-Host "[OK] Generated start-small.ps1 (port 8002, $($OpenCodeArgs.SmallDevice))" -ForegroundColor Green
    }
    & (Join-Path $ScriptDir "scripts" "New-OpenCodeConfig.ps1") @OpenCodeArgs -Out (Join-Path $ScriptDir "opencode.json")
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "=== NoLlama install complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start the server:"
Write-Host "  .\start.ps1"
if ($OpenCodeArgs) {
    if ($OpenCodeArgs.SmallDir -and $OpenCodeArgs.Mode -eq "two-servers") { Write-Host "  .\start-small.ps1        # second terminal: the small model for OpenCode's side-tasks" }
    Write-Host ""
    Write-Host "For OpenCode, one command does the rest:" -ForegroundColor Cyan
    Write-Host "  .\launch-agent.ps1 -Path <your project>"
    Write-Host "  Starts the server if it is down, waits for the model to finish loading," -ForegroundColor DarkGray
    Write-Host "  puts opencode.json where OpenCode reads it, and opens OpenCode there." -ForegroundColor DarkGray
    Write-Host ""

    # Check the client here rather than only in launch-agent.ps1: this is the
    # moment the user learns what the setup needs, and "npm install -g" is a
    # download they would rather start now than after the next command fails.
    $ocCmd = Get-Command opencode -ErrorAction SilentlyContinue
    if ($ocCmd) {
        $ocVer = try { (& opencode --version 2>$null | Select-Object -First 1) } catch { $null }
        Write-Host "  [OK] opencode found$(if ($ocVer) { " ($ocVer)" }) at $($ocCmd.Source)" -ForegroundColor Green
    } else {
        Write-Host "  [!] 'opencode' is not on PATH - launch-agent.ps1 will stop until it is." -ForegroundColor Yellow
        Write-Host "      Install it with:  npm install -g opencode-ai" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "  By hand instead - point OpenCode at the config without copying it:" -ForegroundColor DarkGray
    Write-Host "    `$env:OPENCODE_CONFIG = '$(Join-Path $PSScriptRoot 'opencode.json')'" -ForegroundColor DarkGray
    Write-Host "    opencode" -ForegroundColor DarkGray
    Write-Host "  That layers on top of your existing settings rather than replacing them." -ForegroundColor DarkGray
    Write-Host "  The config carries the model ids NoLlama advertises, an honest context limit," -ForegroundColor DarkGray
    Write-Host "  and the timeouts a slow iGPU needs. Details: docs/AGENTS.md" -ForegroundColor DarkGray
}
Write-Host ""

Pop-Location
