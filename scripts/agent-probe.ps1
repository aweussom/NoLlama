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

function Write-LfFile {
    # Write text with LF line endings only, UTF-8 without a BOM.
    #
    # Why a helper: Set-Content ends a file with CRLF on Windows, and a mixed
    # file is exactly what an exact-match edit tool trips on (New-Fixture).
    # Operators and cmdlets only, no .NET calls, so it runs in
    # ConstrainedLanguage.
    #
    # In: the text on the pipeline, and a path. Out: the file, ending in one LF.
    param([Parameter(ValueFromPipeline)][string]$Text, [string]$Path)
    (($Text -replace "`r`n", "`n") + "`n") | Set-Content -Path $Path -NoNewline -Encoding utf8
}

function New-Fixture {
    # Write the project the model has to repair.
    #
    # Why a percentage bug rather than a syntax error: it is two characters to
    # fix and impossible to fix by accident, so a pass means the model read the
    # test, understood the intent and edited the right line. The test file is
    # runnable with plain python -- no pytest, because the probe must work on a
    # machine that only has the venv.
    #
    # Line endings are pinned to LF, in the files AND in the fixture repo.
    # Set-Content ended each file with one stray CRLF, and Git for Windows'
    # system core.autocrlf=true made the reset before each task rewrite an
    # edited calc.py as CRLF throughout -- so every feature task after a fix
    # started on a CRLF file, while models write LF in their edit requests and
    # OpenCode's edit tool matches bytes exactly ("Could not find oldString ...
    # line endings") [OBSERVED 2026-09-24, 140V laptop, simulated reset:
    # CRLF=11 LF=0]. Linux never saw it; Windows boxes always did.
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
'@ | Write-LfFile -Path (Join-Path $Dir "calc.py")
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
'@ | Write-LfFile -Path (Join-Path $Dir "tests\test_calc.py")
    Push-Location $Dir
    git init -q 2>$null; git config core.autocrlf false; git add -A 2>$null
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
            # Same limits New-OpenCodeConfig ships, so the probe measures what
            # a user gets. 4096 output was four times tighter than the default
            # and would truncate a model writing a whole file -- it never bound
            # on this fixture (turns run 200-1000 tokens), but a probe that
            # differs from the product measures the wrong thing.
            models  = @{ $ModelId = @{ limit = @{ context = 32000; output = 16384 } } }
        } }
        model     = "probe/$ModelId"
    }
    $cfg | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $Dir "opencode.json") -Encoding utf8
}

function Get-FailureKind {
    # Say WHY a task failed, from the transcript and the filesystem.
    #
    # Why: a bare FAIL confounds two very different defects. Every failure
    # measured on 2026-09-23 was a tool-use failure -- the models diagnosed the
    # percentage bug correctly in prose and then did not act -- but the verdict
    # alone could not show that, and it took reading four transcripts to learn
    # it. The fixture tests comprehension AND action; this separates them.
    #
    # In: the task log and the project dir. Out: one short phrase. Heuristics
    # over a transcript, so it is a label to start from, not evidence.
    #
    # Reads stdout AND stderr: `opencode run` prints its tool lines ($, ✱, ✗)
    # to stderr and only the final reply to stdout [OBSERVED 2026-09-23,
    # opencode on the 140V]. Reading stdout alone labelled LFM2.5 "never called
    # a tool" after five parsed calls, so any verdict from before this fix that
    # carries that label needs its .err re-read before it is believed.
    param([string]$Log, [string]$Dir)
    $text = ""
    foreach ($f in @($Log, "$Log.err")) {
        if (Test-Path $f) { $text += (Get-Content $f -Raw -ErrorAction SilentlyContinue) + "`n" }
    }
    $text = $text -replace "\x1b\[[0-9;]*m", ""
    if (-not $text.Trim()) { return "no output at all (harness or launcher, not the model)" }

    # OpenCode prints a line per tool it runs; a model that only talks prints none.
    $calledTools = $text -match "(?m)^\s*(\$|→|✱|✗|●)\s" -or $text -match "(?m)^\s*(Read|Write|Edit|Glob|Bash)\b"
    $edited = $false
    Push-Location $Dir
    # The harness writes its own logs and opencode.json into this dir; left in,
    # they made the tree look dirty on every run, so "changed nothing" was
    # unreachable and every no-op run read as "wrote somewhere else".
    $dirty = @(git status --porcelain 2>$null | Where-Object { $_ -notmatch '\.log(\.err|\.diff)?$|opencode\.json$|__pycache__' })
    Pop-Location
    if ($dirty | Where-Object { $_ -match "calc\.py" }) { $edited = $true }

    if (-not $calledTools) { return "never called a tool -- wrote prose instead of acting" }
    # OpenCode auto-rejects paths outside the project and the run ends there.
    if ($text -match "permission requested: external_directory") {
        return "called tools, then reached outside the project -- an invented path"
    }
    if (-not $edited -and $dirty.Count -eq 0) { return "called tools but changed nothing" }
    if (-not $edited) { return "wrote somewhere other than calc.py: $($dirty -join ', ')" }
    return "edited calc.py, tests still fail -- a comprehension failure"
}

function Test-OriginalTests {
    # Run the fixture's ORIGINAL tests against whatever calc.py the agent left.
    #
    # Why: the task tells the model to make the tests pass, and rewriting the
    # test file is a way to do that. Running the model's own copy scored
    # Qwen3-14B's feature task PASS after it replaced tests/test_calc.py
    # wholesale with `write` [OBSERVED 2026-09-23] -- and nobody could say
    # afterwards whether the original assertions survived. The pristine file
    # comes from the fixture's own git HEAD, so the model cannot have edited it.
    #
    # In: the project dir. Out: $true only if every original test passes.
    param([string]$Dir)
    $orig = Join-Path $Dir "tests\_probe_original.py"
    git -C $Dir show HEAD:tests/test_calc.py 2>$null | Set-Content -Path $orig -Encoding utf8
    & $py $orig *> $null
    $ok = $LASTEXITCODE -eq 0
    Remove-Item $orig -Force -ErrorAction SilentlyContinue
    return $ok
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
    # Start-Process, not Start-Job: PowerShell jobs are refused outright where
    # an application-control policy pins the system-wide language mode
    # ("Cannot start job. The language mode for this session is incompatible
    # with the system-wide language mode" -- 2026-09-23, 258V laptop), and both
    # test machines run such a policy.
    # Start-Process needs an executable. On Windows `opencode` resolves to an
    # npm .ps1 shim, and launching a script as a process fails with "%1 is not
    # a valid Win32 application" -- which reads like a broken install rather
    # than a wrong launcher, so dispatch through the right interpreter.
    $cmd = Get-Command opencode -ErrorAction Stop
    $started = Get-Date
    $file, $argv = switch -Wildcard ($cmd.Source) {
        "*.ps1" { "pwsh",     @("-NoProfile", "-File", $cmd.Source, "run", $Prompt) }
        "*.cmd" { "cmd.exe",  @("/c", $cmd.Source, "run", $Prompt) }
        "*.bat" { "cmd.exe",  @("/c", $cmd.Source, "run", $Prompt) }
        default { $cmd.Source, @("run", $Prompt) }
    }
    $proc = Start-Process -FilePath $file -ArgumentList $argv `
        -WorkingDirectory $Dir -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    # Cmdlets only, no .NET method calls on the Process object: a machine
    # strict enough to refuse Start-Job may also be in ConstrainedLanguage,
    # where methods on System.Diagnostics.Process are not callable.
    Wait-Process -Id $proc.Id -Timeout $TimeoutSec -ErrorAction SilentlyContinue
    $timedOut = $false
    if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $timedOut = $true
    }
    $elapsed = (Get-Date) - $started
    $verdict = if ($timedOut) { "TIMEOUT" } elseif ((& $Verify) -eq $true) { "PASS" } else { "FAIL" }
    # What the agent actually changed, kept beside the transcript: the
    # transcript says "Wrote file successfully", never what was written, and
    # the next task resets the tree.
    git diff --no-color HEAD 2>$null | Set-Content -Path "$log.diff" -Encoding utf8
    Pop-Location

    # Did the agent stay inside its sandbox? Qwen3-30B-A3B wrote a correct
    # apply_tax() into the NoLlama repo root instead of the fixture directory
    # (2026-09-23) -- the code was fine, the destination was not, and a
    # `git add -A` then committed the model's output. A probe that lets the
    # subject write into the repo it is being run from is a hazard, so this
    # says so loudly rather than leaving it to be noticed in a diff.
    $strays = @(git -C $PSScriptRoot/.. status --porcelain -- calc.py tests/ 2>$null |
                Where-Object { $_ -match 'calc\.py|test_calc\.py' })
    if ($strays) {
        Write-Host "  !! the agent wrote OUTSIDE its workdir, into this repo:" -ForegroundColor Red
        $strays | ForEach-Object { Write-Host "     $_" -ForegroundColor Red }
        Write-Host "     clean these up before committing anything." -ForegroundColor Red
    }
    $why = if ($verdict -eq "FAIL") { Get-FailureKind -Log $log -Dir $Dir } else { "" }
    [PSCustomObject]@{ Task = $Label; Verdict = $verdict; Seconds = [int]$elapsed.TotalSeconds; Log = $log; Why = $why }
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
    -Verify { Test-OriginalTests -Dir $WorkDir }

$results += Invoke-Task -Dir $WorkDir -Label "feature" -TimeoutSec $TimeoutSec `
    -Prompt "Add a function apply_tax(amount, percent) to calc.py that ADDS that percentage to the amount and rounds to two decimals. Add a test for it in tests/test_calc.py asserting apply_tax(100.0, 25) == 125.0. Then run 'python tests/test_calc.py' and make sure everything passes." `
    -Verify {
        & $py -c "import sys; sys.path.insert(0, r'$WorkDir'); from calc import apply_tax; raise SystemExit(0 if apply_tax(100.0,25)==125.0 else 1)" *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
        # The asked-for test must exist, and the model's file must pass...
        if (-not (Select-String -Path (Join-Path $WorkDir "tests\test_calc.py") -Pattern "apply_tax" -Quiet)) { return $false }
        & $py (Join-Path $WorkDir "tests\test_calc.py") *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
        # ...and so must the original assertions it may have rewritten.
        Test-OriginalTests -Dir $WorkDir
    }

Write-Host ""
foreach ($r in $results) {
    $color = switch ($r.Verdict) { "PASS" { "Green" } "TIMEOUT" { "Yellow" } default { "Red" } }
    Write-Host ("  {0,-8} {1,-8} {2,5}s   {3}" -f $r.Task, $r.Verdict, $r.Seconds, $r.Log) -ForegroundColor $color
    if ($r.Why) { Write-Host "           why: $($r.Why)" -ForegroundColor $color }
}
# Keep the transcripts. The first run of this script printed log paths and
# then deleted them with the workdir, so a 0/2 result could not be explained
# afterwards -- and HOW a model fails is the whole output of this probe.
# Two child segments, not one "bench\agent-probe" string: an earlier edit
# wrote that backslash-a as a BEL byte and the directory never existed.
$keepDir = Join-Path (Get-Location) "bench" "agent-probe"
New-Item -ItemType Directory -Path $keepDir -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$safeModel = ($Model -replace '[^A-Za-z0-9._-]', '_')
foreach ($r in $results) {
    if (Test-Path $r.Log) {
        $dest = Join-Path $keepDir "$stamp-$safeModel-$($r.Task).log"
        Copy-Item $r.Log $dest -Force
        # stderr is where OpenCode prints the tool lines -- the evidence.
        if (Test-Path "$($r.Log).err") { Copy-Item "$($r.Log).err" "$dest.err" -Force }
        if (Test-Path "$($r.Log).diff") { Copy-Item "$($r.Log).diff" "$dest.diff" -Force }
        $r.Log = $dest
    }
}

$passed = @($results | Where-Object { $_.Verdict -eq "PASS" }).Count
Write-Host ""
Write-Host "  $passed/$($results.Count) passed for $Model" -ForegroundColor Cyan
Write-Host "  Flag a model 'agent' in models.json only on a clean sweep here." -ForegroundColor DarkGray
if (-not $KeepWorkdir) { Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue }
exit $(if ($passed -eq $results.Count) { 0 } else { 1 })
