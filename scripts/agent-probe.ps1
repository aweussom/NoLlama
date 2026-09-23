#requires -Version 7.0
<#
Tier 2 model check — can this model actually drive OpenCode?

Builds a throwaway project with two failing tests, runs `opencode run` against
a served model, and checks whether the code was fixed. Pass/fail is decided by
running the tests, not by reading the transcript, because the failure this
catches looks like success in a transcript: a model that narrates tool use and
hands the work back sounds exactly like one that did it.

Why this exists as a gate. `pelican-probe.py` measures the model; this measures
the model AND the loop. Qwen2.5-Coder-14B draws a fine picture, writes valid
Python, and cannot call a tool [OBSERVED 2026-09-23] -- only this probe
separated it from Qwen3-Coder-30B-A3B. A model is flagged `agent` in
models.json on the strength of this and nothing else.

    .\scripts\agent-probe.ps1 -Url http://100.81.4.88:8000/v1
    .\scripts\agent-probe.ps1 -Url http://127.0.0.1:8000/v1 -Model Qwen3-14B@GPU
    .\scripts\agent-probe.ps1 -Url ... -TimeoutSec 600 -KeepWorkdir

Needs `opencode` on PATH and a server already running -- it deliberately does
not start one, so the same script works against a laptop, the B60 over
Tailscale, or anything else that speaks the OpenAI API.

Out: one line per task, and an exit code (0 = every task passed). Criteria are
printed before the run so the verdict cannot be written afterwards.
#>
[CmdletBinding()]
param(
    [string]$Url = "http://127.0.0.1:8000/v1",
    [string]$Model,
    [int]$TimeoutSec = 900,
    [string]$WorkDir = (Join-Path ([System.IO.Path]::GetTempPath()) "nollama-agent-probe"),
    [switch]$KeepWorkdir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-Fixture {
    # Write the project the model has to repair.
    #
    # Why a percentage bug rather than a syntax error: it is two characters to
    # fix and impossible to fix by accident, so a pass means the model read the
    # test, understood the intent and edited the right line. The test file is
    # runnable with plain python -- no pytest, because the probe must work on a
    # machine that only has the venv.
    #
    # In: a directory, created fresh. Out: nothing; the directory is left with
    # failing tests and a git repo so each task starts from the same state.
    param([string]$Dir)
    Remove-Item -Recurse -Force $Dir -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path (Join-Path $Dir "tests") -Force | Out-Null
    @'
"""A tiny expression helper used by the billing report."""


def apply_discount(amount, percent):
    """Return amount with percent taken off, rounded to two decimals."""
    return round(amount - (amount * percent), 2)


def total(items):
    """Sum a list of (amount, discount_percent) pairs."""
    return round(sum(apply_discount(a, p) for a, p in items), 2)
'@ | Set-Content -Path (Join-Path $Dir "calc.py") -Encoding utf8
    @'
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calc import apply_discount, total


def test_apply_discount_takes_a_percentage_not_a_fraction():
    assert apply_discount(200.0, 10) == 180.0


def test_total_sums_discounted_items():
    assert total([(200.0, 10), (100.0, 50)]) == 230.0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("ok  ", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    raise SystemExit(1 if fails else 0)
'@ | Set-Content -Path (Join-Path $Dir "tests\test_calc.py") -Encoding utf8
    Push-Location $Dir
    git init -q 2>$null; git add -A 2>$null
    git -c user.email=probe@local -c user.name=probe commit -qm fixture 2>$null
    Pop-Location
}

function Write-OpenCodeConfig {
    # Point OpenCode at one NoLlama model and nothing else.
    #
    # Why no small_model: the split is a discrete-GPU recipe (measured
    # 2026-09-23 -- a CPU side model is starved while an iGPU prefills), and a
    # probe that quietly added one would be measuring a different setup than
    # the installer builds.
    param([string]$Dir, [string]$Url, [string]$ModelId)
    $cfg = @{
        '$schema' = "https://opencode.ai/config.json"
        provider  = @{ probe = @{
            npm     = "@ai-sdk/openai-compatible"
            name    = "NoLlama probe"
            options = @{ baseURL = $Url; chunkTimeout = 1800000; headerTimeout = 1800000 }
            models  = @{ $ModelId = @{ limit = @{ context = 32000; output = 4096 } } }
        } }
        model     = "probe/$ModelId"
    }
    $cfg | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $Dir "opencode.json") -Encoding utf8
}

function Invoke-Task {
    # Run one task and decide it by running code, never by reading prose.
    #
    # In: the project dir, a prompt, and a verification scriptblock that exits
    # non-zero on failure. Out: a result object with the wall clock and the
    # verdict. A timeout is a FAIL with its own label, because "still going
    # after fifteen minutes" is a different fact from "got it wrong".
    param([string]$Dir, [string]$Label, [string]$Prompt, [scriptblock]$Verify, [int]$TimeoutSec)
    Push-Location $Dir
    git checkout -q . 2>$null
    Remove-Item -Recurse -Force (Join-Path $Dir "__pycache__"), (Join-Path $Dir "tests\__pycache__") -ErrorAction SilentlyContinue
    $log = Join-Path $Dir "$Label.log"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $job = Start-Job -ScriptBlock {
        param($d, $p, $l)
        Set-Location $d
        & opencode run $p *> $l
    } -ArgumentList $Dir, $Prompt, $log
    $done = Wait-Job $job -Timeout $TimeoutSec
    $sw.Stop()
    if (-not $done) { Stop-Job $job -ErrorAction SilentlyContinue; $verdict = "TIMEOUT" }
    else {
        Receive-Job $job -ErrorAction SilentlyContinue | Out-Null
        $verdict = if ((& $Verify) -eq $true) { "PASS" } else { "FAIL" }
    }
    Remove-Job $job -Force -ErrorAction SilentlyContinue
    Pop-Location
    [PSCustomObject]@{ Task = $Label; Verdict = $verdict; Seconds = [int]$sw.Elapsed.TotalSeconds; Log = $log }
}

# --- run -------------------------------------------------------------------
$base = $Url.TrimEnd('/')
if (-not $Model) {
    $Model = (Invoke-RestMethod -Uri "$base/models" -TimeoutSec 30).data[0].id
}
if (-not (Get-Command opencode -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: opencode is not on PATH" -ForegroundColor Red; exit 2
}

Write-Host "=== agent probe ===" -ForegroundColor Cyan
Write-Host "  server : $base"
Write-Host "  model  : $Model"
Write-Host "  PASS   : the tests pass afterwards, unaided, within $TimeoutSec s" -ForegroundColor DarkGray
Write-Host "  (criteria printed before the run on purpose)" -ForegroundColor DarkGray
Write-Host ""

New-Fixture -Dir $WorkDir
Write-OpenCodeConfig -Dir $WorkDir -Url $base -ModelId $Model

$py = if (Test-Path ".\venv\Scripts\python.exe") { (Resolve-Path ".\venv\Scripts\python.exe").Path } else { "python" }
$results = @()
$results += Invoke-Task -Dir $WorkDir -Label "fix" -TimeoutSec $TimeoutSec `
    -Prompt "Run 'python tests/test_calc.py'. Both tests fail. Fix calc.py so they pass, then run the tests again to confirm." `
    -Verify { & $py (Join-Path $WorkDir "tests\test_calc.py") *> $null; $LASTEXITCODE -eq 0 }

$results += Invoke-Task -Dir $WorkDir -Label "feature" -TimeoutSec $TimeoutSec `
    -Prompt "Add a function apply_tax(amount, percent) to calc.py that ADDS that percentage to the amount and rounds to two decimals. Add a test for it in tests/test_calc.py asserting apply_tax(100.0, 25) == 125.0. Then run 'python tests/test_calc.py' and make sure everything passes." `
    -Verify {
        & $py -c "import sys; sys.path.insert(0, r'$WorkDir'); from calc import apply_tax; raise SystemExit(0 if apply_tax(100.0,25)==125.0 else 1)" *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
        & $py (Join-Path $WorkDir "tests\test_calc.py") *> $null
        $LASTEXITCODE -eq 0
    }

Write-Host ""
foreach ($r in $results) {
    $color = switch ($r.Verdict) { "PASS" { "Green" } "TIMEOUT" { "Yellow" } default { "Red" } }
    Write-Host ("  {0,-8} {1,-8} {2,5}s   {3}" -f $r.Task, $r.Verdict, $r.Seconds, $r.Log) -ForegroundColor $color
}
$passed = @($results | Where-Object { $_.Verdict -eq "PASS" }).Count
Write-Host ""
Write-Host "  $passed/$($results.Count) passed for $Model" -ForegroundColor Cyan
Write-Host "  Flag a model 'agent' in models.json only on a clean sweep here." -ForegroundColor DarkGray
if (-not $KeepWorkdir) { Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue }
exit $(if ($passed -eq $results.Count) { 0 } else { 1 })
