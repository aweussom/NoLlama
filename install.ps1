#requires -Version 7.0
# install.ps1 — NoLlama setup: venv, dependencies, model selection
#
# Usage:
#     .\install.ps1                       # interactive setup
#     .\install.ps1 -SkipModel            # venv + deps only
#     .\install.ps1 -HfToken hf_xxx       # auth for gated/private models
#
# Detects available devices (NPU, GPU, CPU), then asks what you want to DO
# (chat / coding agent / vision / combos) and places each model on the best
# device. Coding-agent models (OpenClaw / Copilot, tool-calling) and CPU are
# first-class choices, not buried.
#
# -HfToken: a HuggingFace access token (https://huggingface.co/settings/tokens).
# Only needed for gated or private models — the curated OpenVINO models are
# public and download anonymously. We can't rely on 'hf auth login' here
# because this script is what installs 'hf' in the first place, so the token
# is passed through the HF_TOKEN env var that huggingface_hub reads at
# download time.

param(
    [switch]$SkipModel,
    [string]$HfToken
)

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

$VenvDir = Join-Path $ScriptDir "venv"

# Validate existing venv. Script launchers (pip.exe, hf.exe, ...) bake the
# absolute path to python.exe into themselves at install time. If the venv
# folder is moved or renamed, every launcher fails with "Unable to create
# process". Catch that here and recreate, rather than failing mid-install.
if (Test-Path $VenvDir) {
    $venvPip = Join-Path $VenvDir $VenvBinDir "pip$ExeExt"
    $venvOk = $false
    if (Test-Path $venvPip) {
        & $venvPip --version 2>&1 | Out-Null
        $venvOk = ($LASTEXITCODE -eq 0)
    }
    if ($venvOk) {
        Write-Host "[OK] venv already exists"
    } else {
        Write-Host "[!] venv at $VenvDir is broken (likely moved from another path). Recreating..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VenvDir
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
    Write-Host "Creating Python venv (using $($sysPython.Source))..."
    & $sysPython.Source -m venv $VenvDir
    if (-not $?) { Write-Host "ERROR: Failed to create venv." -ForegroundColor Red; Pop-Location; exit 1 }
    Write-Host "[OK] venv created"
}

$ActivateScript = Join-Path $VenvDir $VenvBinDir "Activate.ps1"
& $ActivateScript

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip wheel setuptools 2>&1 | Out-Null
python -m pip install -r (Join-Path $ScriptDir "requirements.txt")
if (-not $?) { Write-Host "ERROR: pip install failed" -ForegroundColor Red; Pop-Location; exit 1 }
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
        if 'GPU' not in out: out['GPU'] = {'id': dev, 'name': full}
    elif dev in ('NPU', 'CPU'):
        out[dev] = {'id': dev, 'name': full}
print(json.dumps(out))
"@ | ConvertFrom-Json

$HasNPU = $null -ne $DeviceInfo.NPU
$HasGPU = $null -ne $DeviceInfo.GPU

Write-Host ""
if ($HasNPU) { Write-Host "  [+] NPU: $($DeviceInfo.NPU.name)" -ForegroundColor Green }
else         { Write-Host "  [-] NPU: not found" -ForegroundColor DarkGray }
if ($HasGPU) {
    $gpuSuffix = if ($DeviceInfo.GPU.id -ne "GPU") { " [$($DeviceInfo.GPU.id)]" } else { "" }
    Write-Host "  [+] GPU$($gpuSuffix): $($DeviceInfo.GPU.name)" -ForegroundColor Green
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
    $LocalModels = @(Get-ChildItem -Path $ModelsRoot -Directory | Where-Object {
        (Test-Path (Join-Path $_.FullName "openvino_language_model.bin")) -or
        (Test-Path (Join-Path $_.FullName "openvino_model.bin"))
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
        if (Test-Path (Join-Path $_.FullName "openvino_vision_embeddings_model.xml")) {
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
        # Detect NPU compatibility: needs int4 quantization and reasonable size.
        # Matches the older "-int4-cw" / "-cw-ov" naming and the newer plain
        # "-int4-ov" exports (e.g. Qwen3.5). Soft filter — user still confirms.
        $npuOk = ($_.Name -match "int4") -and $sizeGB -lt 10
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
        $onDisk += [PSCustomObject]@{
            Action = "local"; Name = $lm.Name; Path = $lm.Path
            HfId = $null; Source = $null; Weight = $null; Trust = $false
            SizeGB = $lm.SizeGB; Notes = "Already on disk"
        }
    }

    $localNames = @($LocalModels | ForEach-Object { $_.Name.ToLower() })
    $downloads = @()
    foreach ($dm in $RegistryModels) {
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
            Write-Host "    $i. $($od.Name)" -NoNewline
            Write-Host "  ($($od.SizeGB) GB)" -ForegroundColor DarkGray -NoNewline
            Write-Host "  Already on disk" -ForegroundColor DarkGray
        }
        Write-Host ""
    }

    if ($downloads.Count -gt 0) {
        Write-Host "  Download from HuggingFace:" -ForegroundColor Yellow
        foreach ($dm in $downloads) {
            $items += $dm
            $dlTag = if ($dm.Source -eq "pre-exported") { "download" } else { "convert" }
            $i = $items.Count
            Write-Host "    $i. $($dm.Name)" -NoNewline
            Write-Host "  (~$($dm.SizeGB) GB, $dlTag)" -ForegroundColor DarkGray -NoNewline
            Write-Host "  $($dm.Notes)" -ForegroundColor DarkGray
        }
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
    # substantial (>100 MB). The previous "file exists" check let partial
    # downloads sneak through: the XML descriptor + small tokenizer files
    # complete quickly, but the multi-GB weights file may be 0 bytes or
    # missing if the download was interrupted. Smallest real model in our
    # registry (DeepSeek-1.5B INT4) is ~700 MB; tokenizer .bin files top
    # out around 10 MB. 100 MB cleanly separates the two.
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $false }
    foreach ($bin in @("openvino_language_model.bin", "openvino_model.bin")) {
        $file = Join-Path $Path $bin
        if ((Test-Path $file) -and ((Get-Item $file).Length -gt 100MB)) {
            return $true
        }
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
            hf download $Selected.HfId --local-dir $cachePath
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

# Chat can run anywhere; small NPU-class models + bigger GPU LLMs both work on GPU/CPU.
function Get-ChatRegistry { param([string]$Device)
    if ($Device -eq "NPU") { return $Registry.npu }
    return @($Registry.npu) + @($Registry.gpu_llm)
}
function Get-ChatLocal { param([string]$Device, [string]$Exclude = "")
    @($LocalModels | Where-Object { $_.Type -eq "llm" -and (($Device -ne "NPU") -or $_.NpuOk) -and $_.Name -ne $Exclude })
}

# --- Use-case menu (filtered by available hardware) ---
Write-Host ""
Write-Host "=== What will you use NoLlama for? ===" -ForegroundColor Cyan
Write-Host ""
$cases = @()
$cases += [PSCustomObject]@{ Key = "chat";   Label = "Chat";         Desc = "text assistant" }
$cases += [PSCustomObject]@{ Key = "agent";  Label = "Coding agent"; Desc = "OpenClaw / VS Code Copilot (tool-calling)" }
if ($HasGPU) {
    $cases += [PSCustomObject]@{ Key = "vision";      Label = "Vision";             Desc = "image understanding (GPU)" }
    $cases += [PSCustomObject]@{ Key = "chat+agent";  Label = "Chat + Coding agent"; Desc = "chat model + a GPU coder, together" }
    $cases += [PSCustomObject]@{ Key = "chat+vision"; Label = "Chat + Vision";       Desc = "chat model + GPU vision (classic)" }
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

$chatDevices  = @(); if ($HasNPU) { $chatDevices += "NPU" }; if ($HasGPU) { $chatDevices += "GPU" }; $chatDevices += "CPU"
$agentDevices = @(); if ($HasGPU) { $agentDevices += "GPU" }; $agentDevices += "CPU"
$coders = @($Registry.gpu_llm | Where-Object { $_.agent })   # OpenClaw/Copilot-ready
$isAgent = $false

function Install-Primary { param($Sel, [string]$Device)
    if (-not (Install-Model -Selected $Sel -TargetDir $ModelDir)) {
        Write-Host "Model installation failed. Re-run install.ps1 to retry." -ForegroundColor Yellow; Pop-Location; exit 1
    }
    $script:StartArgs += @("--device", $Device)
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
        $loc = @($LocalModels | Where-Object { $_.Type -eq "llm" })
        $sel = Show-ModelMenu -Title "Coding agent model ($dev) - OpenClaw / Copilot ready" -RegistryModels $coders -LocalModels $loc
        if ($sel) { Install-Primary $sel $dev; $StartArgs += @("--prewarm", "prewarm.json", "--vscode-compat"); $isAgent = $true }
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
            $cloc = @($LocalModels | Where-Object { $_.Type -eq "llm" -and $_.Name -ne $chatSel.Name })
            $coderSel = Show-ModelMenu -Title "Coding agent model (GPU) - OpenClaw / Copilot ready" -RegistryModels $coders -LocalModels $cloc -AllowSkip $true
            if ($coderSel -and (Install-Model -Selected $coderSel -TargetDir $GpuModelDir)) {
                $StartArgs += @("--gpu-model-dir", "gpu-model", "--prewarm", "prewarm.json", "--vscode-compat"); $isAgent = $true
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
    Write-Host "Coding agent ready. To drive it with OpenClaw:" -ForegroundColor Green
    Write-Host "  npm install -g openclaw@latest      # once" -ForegroundColor DarkGray
    Write-Host "  openclaw onboard --install-daemon   # once" -ForegroundColor DarkGray
    Write-Host "  ./start-openclaw.ps1 -Setup -Warmup # configures + launches the agent" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 5. Generate start.ps1
# ---------------------------------------------------------------------------

$StartScript = Join-Path $ScriptDir "start.ps1"
$TemplateScript = Join-Path $ScriptDir "start-template.ps1"
$ArgsStr = $StartArgs -join " "

# Generate start.ps1 — a one-liner that calls the template with the right args
$Content = "# Auto-generated by install.ps1`n"
$Content += "& '$(Join-Path $ScriptDir "start-template.ps1")' -ServerArgs '$ArgsStr'"
Set-Content -Path $StartScript -Value $Content -Encoding UTF8
Write-Host "[OK] Generated start.ps1" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "=== NoLlama install complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start the server:"
Write-Host "  .\start.ps1"
Write-Host ""

Pop-Location
