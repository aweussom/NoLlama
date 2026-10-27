# The installer's fit check: what it tags, when it asks, and the limit advice.
#
#     pwsh -NoProfile -File tests\test_install_fit.ps1
#
# Why: on an integrated GPU the installer budgeted RAM - 8 GB and never
# mentioned the iGPU's own limit, so a 32 GB laptop at the default 16.5 GB
# limit got Qwen3-Coder-30B (15.2 GB) with no word, and the server refused it
# (2026-10-27). That limit is a setting, so such a model is installed, tagged
# RAISE GPU LIMIT, and the end of the install says how to raise it. Only a
# model past RAM - 8 is WON'T FIT, and that one asks first. Get-FitNeedGB uses
# the server preflight's x1.1. Loads install.ps1's functions from its AST.

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$script:failed = 0
function Assert-Equal($expected, $actual, $what) {
    if ("$expected" -ne "$actual") {
        Write-Host "FAIL $what`n  expected: $expected`n  actual:   $actual" -ForegroundColor Red
        $script:failed++
    } else { Write-Host "ok   $what" -ForegroundColor DarkGray }
}
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $repo "install.ps1"), [ref]$null, [ref]$null)
foreach ($f in $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    . ([scriptblock]::Create($f.Extent.Text))
}
$script:asked = 0
function Read-Host { param($Prompt) $script:asked++; $script:answer }

# The laptop: 32 GB RAM (23.5 GB for models), iGPU limit at the default 16.5 GB.
$SystemRamGB = 32; $VenvName = "venv"; $HasGPU = $true
$script:UsableModelGB = 24; $script:IsDiscreteGPU = $false; $script:IgpuLimitGB = 16.5
$script:RaiseLimitToGB = 0
Assert-Equal $true  ((Get-FitTag 15.2).Contains("RAISE GPU LIMIT")) "16.5 GB limit: Coder-30B needs the limit raised"
Assert-Equal ""     (Get-FitTag 9)                                  "16.5 GB limit: Qwen3-14B fits as is"
Assert-Equal $true  ((Get-FitTag 24).Contains("WON'T FIT"))        "past RAM - 8: won't fit"

$coder = [PSCustomObject]@{ Name = "Qwen3 Coder 30B-A3B MoE (INT4)"; SizeGB = 15.2 }
$script:asked = 0
Assert-Equal $true (Confirm-Fit $coder 6>$null) "RAISE GPU LIMIT: installs"
Assert-Equal 0     $script:asked                "RAISE GPU LIMIT: no question asked"
Assert-Equal 21.1  $script:RaiseLimitToGB       "RAISE GPU LIMIT: need recorded ((15.2 + 4) x 1.1)"

$text = (Write-RaiseLimit 6>&1 | Out-String)
Assert-Equal $true ($text -match 'RUN AS ADMINISTRATOR')                       "advice: elevated Intel Graphics Software"
Assert-Equal $true ($text -match 'SystemPartitionCommitLimitPercentage .*/d 75 ') "advice: reg add with 75% (21.1 / (0.92 x 32) = 72, +3)"
Assert-Equal $true ($text -match '\.\\venv\\Scripts\\python')                  "advice: check command uses this install's venv"

$script:RaiseLimitToGB = 0
Assert-Equal "" (Write-RaiseLimit 6>&1 | Out-String) "nothing needing the limit: no advice"

$big = [PSCustomObject]@{ Name = "too big"; SizeGB = 24 }
$script:answer = ""; $script:asked = 0
Assert-Equal $false (Confirm-Fit $big 6>$null) "WON'T FIT + Enter: back to the menu"
Assert-Equal 1      $script:asked               "WON'T FIT: asked once"
$script:answer = "y"
Assert-Equal $true  (Confirm-Fit $big 6>$null) "WON'T FIT + y: install anyway"

# The B60: no iGPU limit; 30B-class models are tight, not refused.
$script:UsableModelGB = 23.3; $script:IsDiscreteGPU = $true; $script:IgpuLimitGB = 0
Assert-Equal $true  ((Get-FitTag 17).Contains("TIGHT"))   "B60: Glimmer-size model is tight"
Assert-Equal $false ((Get-FitTag 17).Contains("browser")) "B60: tight hint is not about the browser"
Assert-Equal $false ((Get-FitTag 17).Contains("LIMIT"))   "B60: no limit advice on a discrete card"

if ($script:failed) { Write-Host "$($script:failed) failed" -ForegroundColor Red; exit 1 }
Write-Host "all passed"
