# install.ps1 — NoLlama setup: venv, dependencies, model selection
#
# Usage:
#     .\install.ps1              # interactive setup
#     .\install.ps1 -SkipModel   # venv + deps only
#
# Detects available devices (NPU, GPU, CPU), then walks the user
# through model selection. NPU-first: if you have an NPU, that's
# your primary chat device.

param(
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelsRoot = Join-Path $HOME "models"
Push-Location $ScriptDir

Write-Host ""
Write-Host "=== NoLlama Install ===" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Version check and warnings
# ---------------------------------------------------------------------------

$PSVersion = $PSVersionTable.PSVersion
if ($PSVersion.Major -lt 7) {
    Write-Host "WARNING: PowerShell $PSVersion detected." -ForegroundColor Yellow
    Write-Host "  This script requires PowerShell 7+ for full compatibility." -ForegroundColor Yellow
    Write-Host "  Current version may have syntax errors or unexpected behavior." -ForegroundColor Yellow
    Write-Host "  Please upgrade to PowerShell 7: https://github.com/PowerShell/PowerShell" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Host "Install cancelled. Please upgrade PowerShell and try again." -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Write-Host ""
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "  Please install Python 3.10+ from https://www.python.org/" -ForegroundColor Yellow
    Pop-Location
    exit 1
}

Write-Host ""

# ---------------------------------------------------------------------------
# 1. Create venv
# ---------------------------------------------------------------------------

$VenvDir = Join-Path $ScriptDir "venv"
if (Test-Path $VenvDir) {
    Write-Host "[OK] venv already exists"
} else {
    Write-Host "Creating Python venv..."
    python -m venv $VenvDir
    if (-not $?) { Write-Host "ERROR: Failed to create venv. Is Python installed?" -ForegroundColor Red; Pop-Location; exit 1 }
    Write-Host "[OK] venv created"
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
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
$DeviceInfo = python -c @"
import openvino as ov, json
core = ov.Core()
d = {}
for dev in core.get_available_devices():
    try: d[dev] = core.get_property(dev, 'FULL_DEVICE_NAME')
    except: d[dev] = dev
print(json.dumps(d))
"@ | ConvertFrom-Json

$HasNPU = $null -ne $DeviceInfo.NPU
$HasGPU = $null -ne $DeviceInfo.GPU

Write-Host ""
if ($HasNPU) { Write-Host "  [+] NPU: $($DeviceInfo.NPU)" -ForegroundColor Green }
else         { Write-Host "  [-] NPU: not found" -ForegroundColor DarkGray }
if ($HasGPU) { Write-Host "  [+] GPU: $($DeviceInfo.GPU)" -ForegroundColor Green }
else         { Write-Host "  [-] GPU: not found" -ForegroundColor DarkGray }
Write-Host "  [+] CPU: $($DeviceInfo.CPU)" -ForegroundColor DarkGray
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
        $mtype = "llm"
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
        # Detect NPU compatibility: needs int4-cw quantization and reasonable size
        $npuOk = ($_.Name -match "int4-cw" -or $_.Name -match "cw-ov") -and $sizeGB -lt 10
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

    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host ""

    $items = @()

    # Local models first
    if ($LocalModels.Count -gt 0) {
        Write-Host "  $LocalLabel" -ForegroundColor Yellow
        foreach ($lm in $LocalModels) {
            $items += [PSCustomObject]@{
                Action = "local"; Name = $lm.Name; Path = $lm.Path
                HfId = $null; Source = $null; Weight = $null; Trust = $false
                SizeGB = $lm.SizeGB; Notes = "Already on disk"
            }
            $i = $items.Count
            Write-Host "    $i. $($lm.Name)" -NoNewline
            Write-Host "  ($($lm.SizeGB) GB)" -ForegroundColor DarkGray -NoNewline
            Write-Host "  Already on disk" -ForegroundColor DarkGray
        }
        Write-Host ""
    }

    # Registry models — skip any already on disk
    $localNames = @($LocalModels | ForEach-Object { $_.Name.ToLower() })
    $filteredRegistry = @($RegistryModels | Where-Object {
        $repoName = ($_.hf_id -split '/')[-1].ToLower()
        $repoName -notin $localNames
    })
    if ($filteredRegistry.Count -gt 0) {
        Write-Host "  Download from HuggingFace:" -ForegroundColor Yellow
        foreach ($dm in $filteredRegistry) {
            $dlTag = if ($dm.source -eq "pre-exported") { "download" } else { "convert" }
            $items += [PSCustomObject]@{
                Action = $dm.source; Name = $dm.name; Path = $null
                HfId = $dm.hf_id; Source = $dm.source
                Weight = $dm.weight_format; Trust = $dm.trust_remote_code
                SizeGB = $dm.est_size_gb; Notes = $dm.notes
            }
            $i = $items.Count
            Write-Host "    $i. $($dm.name)" -NoNewline
            Write-Host "  (~$($dm.est_size_gb) GB, $dlTag)" -ForegroundColor DarkGray -NoNewline
            Write-Host "  $($dm.notes)" -ForegroundColor DarkGray
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

function Install-Model {
    param(
        [PSCustomObject]$Selected,
        [string]$TargetDir
    )

    if ($Selected.Action -eq "local") {
        Write-Host "Linking to: $($Selected.Path)" -ForegroundColor Green
        if (Test-Path $TargetDir) {
            $item = Get-Item $TargetDir
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                cmd /c rmdir "`"$TargetDir`""
            } else {
                Remove-Item -Recurse -Force $TargetDir
            }
        }
        cmd /c mklink /J "`"$TargetDir`"" "`"$($Selected.Path)`""
        Write-Host "[OK] $($Selected.Name)" -ForegroundColor Green
        return $true
    }

    if ($Selected.Action -eq "pre-exported") {
        Write-Host "Downloading $($Selected.Name)..." -ForegroundColor Cyan
        Write-Host "  From: $($Selected.HfId)"
        Write-Host ""
        if (Test-Path $TargetDir) {
            $item = Get-Item $TargetDir
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                cmd /c rmdir "`"$TargetDir`""
            } else {
                Remove-Item -Recurse -Force $TargetDir
            }
        }
        $env:PYTHONIOENCODING = "utf-8"
        hf download $Selected.HfId --local-dir $TargetDir
        if (-not $?) {
            Write-Host "ERROR: Download failed." -ForegroundColor Red
            Write-Host "  If 401/403: run 'huggingface-cli login' first" -ForegroundColor Yellow
            return $false
        }
        Write-Host "[OK] $($Selected.Name)" -ForegroundColor Green
        return $true
    }

    if ($Selected.Action -eq "convert") {
        Write-Host "Converting $($Selected.Name)..." -ForegroundColor Cyan
        Write-Host "  From: $($Selected.HfId)"
        Write-Host "  This may take 5-20 minutes."
        Write-Host ""
        if (Test-Path $TargetDir) {
            $item = Get-Item $TargetDir
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                cmd /c rmdir "`"$TargetDir`""
            } else {
                Remove-Item -Recurse -Force $TargetDir
            }
        }
        $args = @("export", "openvino", "--model", $Selected.HfId, "--weight-format", $Selected.Weight)
        if ($Selected.Trust) { $args += "--trust-remote-code" }
        $args += $TargetDir
        Write-Host "Running: optimum-cli $($args -join ' ')" -ForegroundColor DarkGray
        & optimum-cli @args
        if (-not $?) {
            Write-Host "ERROR: Conversion failed." -ForegroundColor Red
            Write-Host "  If unsupported architecture: needs newer optimum-intel" -ForegroundColor Yellow
            return $false
        }
        Write-Host "[OK] $($Selected.Name)" -ForegroundColor Green
        return $true
    }

    Write-Host "ERROR: Unknown action '$($Selected.Action)'" -ForegroundColor Red
    return $false
}

# ---------------------------------------------------------------------------
# 4. Model selection — NPU-first
# ---------------------------------------------------------------------------

$ModelDir = Join-Path $ScriptDir "model"
$GpuModelDir = Join-Path $ScriptDir "gpu-model"
$StartArgs = @()  # collect args for start.ps1

if ($HasNPU) {
    # --- Step 1: NPU chat model ---
    # Only show local models that are NPU-compatible (int4-cw, reasonable size)
    $npuLocal = @($LocalModels | Where-Object { $_.Type -eq "llm" -and $_.NpuOk })
    $sel = Show-ModelMenu -Title "Step 1: Chat Model (NPU)" `
        -RegistryModels $Registry.npu `
        -LocalModels $npuLocal `
        -LocalLabel "Already converted (instant)"

    $NpuSelectedName = $null
    if ($sel) {
        $NpuSelectedName = $sel.Name
        $ok = Install-Model -Selected $sel -TargetDir $ModelDir
        if (-not $ok) { Write-Host ""; Write-Host "Model installation failed. You can re-run install.ps1 to try again." -ForegroundColor Yellow; Pop-Location; exit 1 }
        $StartArgs += @("--device", "NPU")
        Write-Host ""
    }

    # --- Step 2: GPU model (optional) ---
    if ($HasGPU) {
        Write-Host ""
        Write-Host "=== Step 2: GPU Model (optional) ===" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  You also have an Intel ARC GPU. What do you want to use it for?"
        Write-Host ""
        Write-Host "    A. Vision model  — image understanding alongside NPU chat"
        Write-Host "    B. Bigger LLM    — much smarter chat than the NPU model"
        Write-Host "    C. Skip          — NPU chat only"
        Write-Host ""
        while ($true) {
            $gpuChoice = (Read-Host "  [A/B/C]").ToUpper()
            if ($gpuChoice -in @("A", "B", "C", "")) { break }
            Write-Host "  Enter A, B, or C" -ForegroundColor Red
        }

        if ($gpuChoice -eq "A") {
            $vlmLocal = @($LocalModels | Where-Object { $_.Type -eq "vlm" })
            $sel = Show-ModelMenu -Title "GPU Vision Model" `
                -RegistryModels $Registry.gpu_vlm `
                -LocalModels $vlmLocal
            if ($sel) {
                $ok = Install-Model -Selected $sel -TargetDir $GpuModelDir
                if ($ok) { $StartArgs += @("--gpu-model-dir", "gpu-model") }
                Write-Host ""
            }
        } elseif ($gpuChoice -eq "B") {
            $llmLocal = @($LocalModels | Where-Object { $_.Type -eq "llm" -and $_.Name -ne $NpuSelectedName })
            $sel = Show-ModelMenu -Title "GPU LLM (bigger chat model)" `
                -RegistryModels $Registry.gpu_llm `
                -LocalModels $llmLocal
            if ($sel) {
                $ok = Install-Model -Selected $sel -TargetDir $GpuModelDir
                if ($ok) { $StartArgs += @("--gpu-model-dir", "gpu-model") }
                Write-Host ""
            }
        }
    }
}
elseif ($HasGPU) {
    # --- No NPU, GPU only ---
    Write-Host "No NPU detected. Selecting a GPU model." -ForegroundColor Yellow
    Write-Host ""
    $allGpu = @($Registry.gpu_vlm) + @($Registry.gpu_llm)
    $sel = Show-ModelMenu -Title "GPU Model" `
        -RegistryModels $allGpu `
        -LocalModels $LocalModels
    if ($sel) {
        $ok = Install-Model -Selected $sel -TargetDir $ModelDir
        if (-not $ok) { Pop-Location; exit 1 }
        $StartArgs += @("--device", "GPU")
        Write-Host ""
    }
} else {
    # --- No NPU, no GPU — CPU fallback ---
    Write-Host "No NPU or GPU detected. Models will run on CPU (slower)." -ForegroundColor Yellow
    Write-Host ""
    $sel = Show-ModelMenu -Title "CPU Model" `
        -RegistryModels $Registry.npu `
        -LocalModels @($LocalModels | Where-Object { $_.Type -eq "llm" })
    if ($sel) {
        $ok = Install-Model -Selected $sel -TargetDir $ModelDir
        if (-not $ok) { Pop-Location; exit 1 }
        $StartArgs += @("--device", "CPU")
        Write-Host ""
    }
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
