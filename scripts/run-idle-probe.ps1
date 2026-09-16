<#
.SYNOPSIS
Run scripts/idle-residency-probe.py under a scheduled task, logging to bench-results.

Why this wrapper exists: a multi-hour probe cannot be launched from an SSH
session. On the B60 box a process started with Start-Process dies the moment
the session ends (docs/dev/machines.md, learned by losing two conversions and
a download), and the laptop that drives these runs is a travelling machine
that sleeps. So the probe runs as a scheduled task instead, and a task action
cannot redirect its own output -- this script is the redirection.

Why Windows PowerShell and not pwsh: PowerShell 7 is a Store app on the B60
box, so it is absent from a task's PATH and the action fails with 0x80070002.
Register the task against powershell.exe by full path and this script runs
under it.

In: a model directory, a tag that names the run, the idle rungs, and the repo
root. Out: two files in bench-results -- <tag>-<stamp>.log and .json -- and
this script's own exit code, which is the probe's. The probe always exits 0,
so a nonzero code here means the wrapper itself failed.
#>
param(
    [Parameter(Mandatory = $true)][string]$ModelDir,
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Rungs = "5,45,120,180",
    [string]$Device = "GPU",
    [string]$RepoRoot = "C:\devel\aweussom\python\NoLlama",
    [int]$SampleSec = 60
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = Join-Path $RepoRoot "bench-results"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$log = Join-Path $outDir "$Tag-$stamp.log"
$json = Join-Path $outDir "$Tag-$stamp.json"
$python = Join-Path $RepoRoot "venv\Scripts\python.exe"
$probe = Join-Path $RepoRoot "scripts\idle-residency-probe.py"

# 2>&1 merges the probe's stderr into the same file: a native abort writes
# there and nowhere else, and a log that silently omits it is how issue #37
# spent a week looking like "the process just stopped".
& $python $probe $ModelDir --device $Device --rungs $Rungs --sample-sec $SampleSec --json $json *>&1 |
    Tee-Object -FilePath $log

exit $LASTEXITCODE
