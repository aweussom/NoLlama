#requires -Version 7.0
<#
.SYNOPSIS
  Run the Phi-3.5-vision repetition-penalty repro and save a paste-ready report.

.DESCRIPTION
  Every observation of this failure so far is on an *integrated* Xe GPU
  (Arc 140T community, Arc 140V here). This closes the discrete-GPU gap.

  Downloads the model if it is missing (2.2 GB), then runs scripts\bare-probe.py
  under the release venv and, if present, the nightly venv — the runtime axis
  matters because the failure survives genai 2026.3 and 2026.5 alike on
  integrated parts.

  NoLlama itself is never started: the probe drives openvino_genai directly,
  so nothing here can be blamed on the server.

.EXAMPLE
  .\scripts\phi35v-repro\run.ps1
  .\scripts\phi35v-repro\run.ps1 -Device CPU
#>
param(
    [string]$Device = "GPU",
    [string]$ModelDir = (Join-Path $HOME "models\Phi-3.5-vision-instruct-int4-ov")
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$report = Join-Path $repo "phi35v-report-$stamp.txt"

function Write-Both {
    param([string]$Text)
    Write-Host $Text
    Add-Content -Path $report -Value $Text -Encoding utf8
}

Write-Both "=== Phi-3.5-vision repetition-penalty repro ==="
Write-Both "host   : $env:COMPUTERNAME"
Write-Both "os     : $((Get-CimInstance Win32_OperatingSystem).Caption) $([Environment]::OSVersion.Version)"
Write-Both "commit : $(git rev-parse --short HEAD 2>$null)"
Write-Both "device : $Device"
Write-Both ""

if (-not (Test-Path $ModelDir)) {
    Write-Host "Model not found, downloading (2.2 GB)..." -ForegroundColor Yellow
    & (Join-Path $repo "download-model.ps1") "OpenVINO/Phi-3.5-vision-instruct-int4-ov"
}
if (-not (Test-Path $ModelDir)) {
    Write-Both "FAILED: model still missing at $ModelDir"
    exit 1
}

# Both venvs, because "fixed in a newer runtime" is the first thing anyone
# will ask and it costs one extra run to answer.
$venvs = @(
    @{ Name = "release"; Python = Join-Path $repo "venv\Scripts\python.exe" },
    @{ Name = "nightly"; Python = Join-Path $repo "venv-nightly\Scripts\python.exe" }
)

$ran = 0
foreach ($v in $venvs) {
    if (-not (Test-Path $v.Python)) {
        Write-Both "--- $($v.Name): venv not present, skipped"
        Write-Both ""
        continue
    }
    Write-Both "--- $($v.Name) venv ---"
    $env:PYTHONIOENCODING = "utf-8"
    $out = & $v.Python (Join-Path $repo "scripts\bare-probe.py") $ModelDir --device $Device 2>&1
    $out | ForEach-Object { Write-Both "  $_" }
    Write-Both ""
    $ran++
}

if ($ran -eq 0) {
    Write-Both "No venv found. Run install.ps1 first."
    exit 1
}

Write-Both "Expected on an integrated Xe GPU: every row OK except"
Write-Both "'repetition_penalty=1.05' + image, which asserts with"
Write-Both "  Check '(prompt_id >= 0) && (prompt_id < vocab_size)' failed"
Write-Both "If the discrete GPU differs from that, say so — it would be the"
Write-Both "first hardware-dependent result in this investigation."
Write-Host ""
Write-Host "Report written to: $report" -ForegroundColor Green
Write-Host "Paste it into https://github.com/aweussom/NoLlama/issues/24" -ForegroundColor Green
