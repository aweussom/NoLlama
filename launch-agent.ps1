#requires -Version 7.0
<#
.SYNOPSIS
Start NoLlama if it isn't up, place the client config, and launch the agent client.

Why: running an agent against NoLlama is three manual steps people get wrong in
different orders — start the server, wait for a multi-GB model to finish
loading, get opencode.json somewhere the client reads — and the middle one has
no obvious "done" signal, so the usual failure is launching the client against a
server that is still loading and getting a connection error that looks like a
config problem. Ollama's `ollama launch <client>` sets the expectation; this is
the same ergonomic for the one client our issue traffic actually shows
(OpenCode: ktecho #32/#40, dmitriyteteruk #33/#36/#37).

Nothing is written to the user's machine. The config reaches OpenCode through
OPENCODE_CONFIG, which LAYERS onto whatever the user already has rather than
replacing it [OBSERVED 2026-09-13, opencode 1.18.30: with a project
opencode.json defining provider "testmarker" and OPENCODE_CONFIG pointing at
ours, `opencode models` listed both testmarker/* and nollama/*]. Copying a
generated file into someone's project or ~/.config was the alternative, and it
risks clobbering a hand-tuned config — the trimmed tool set for weak hardware
is the documented case.

Deliberately NOT a rewrite of start.ps1: it shells out to the generated
start.ps1 so the server arguments stay in one place (install.ps1 owns them).

In: a project directory, a port, and a client name. Out: an OpenCode session in
that directory, with NoLlama serving on the port. Exits non-zero without
launching anything if the client is missing, the install is incomplete, or the
server never reaches "ready" — a half-started stack is worse than a clear error.
#>
param(
    # Project to open. OpenCode reads opencode.json from the directory it runs
    # in, so this doubles as "where the config goes" unless -Global is given.
    [string] $Path = ".",

    # Only opencode today. Goose and Copilot Chat read different config files
    # and are documented in docs/AGENTS.md; ValidateSet is what makes the
    # not-yet-supported case a clean error instead of a confusing one.
    [ValidateSet("opencode")] [string] $Client = "opencode",

    [int] $Port = 8000,

    # Assume a server is already up (or is being started by hand elsewhere).
    [switch] $NoStart,

    # Do everything except hand over to the client. Why it exists: the client
    # is an interactive TUI, so it cannot run inside a benchmark or a test —
    # this leaves the server warm and the config placed, which is exactly the
    # pre-state a timed arm needs.
    [switch] $NoLaunch,

    # Model load is the long pole — 15 GB on a laptop iGPU is minutes, not
    # seconds, and a cold prefix-cache prewarm adds to it.
    [int] $TimeoutSec = 900
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-ClientAvailable {
    <#
    Locate the agent client's executable, or explain how to install it.

    Why: "opencode is not recognized" arrives from the shell after the server
    is already up and the config is already written, which is the most
    confusing possible moment and leaves a loaded model behind. Checking first
    costs nothing and lets the error name the actual fix.

    In: a client name from the ValidateSet. Out: the resolved path as a string,
    or $null after printing an install hint — the caller decides whether a
    miss is fatal.
    #>
    param([string] $Name)

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    Write-Host "ERROR: '$Name' is not on PATH." -ForegroundColor Red
    switch ($Name) {
        "opencode" {
            Write-Host "  Install it with:  npm install -g opencode-ai" -ForegroundColor Yellow
            Write-Host "  Then re-run this script. Details: docs/AGENTS.md" -ForegroundColor Yellow
        }
    }
    return $null
}

function Get-ServerHealth {
    <#
    One non-throwing probe of NoLlama's /health.

    Why: this is called in a polling loop against a server that is expected to
    be absent at first, so a connection refusal is a normal reading rather than
    an error — letting it throw would turn the expected case into a stack
    trace. The endpoint's own docstring names external launchers as a consumer
    of this field set.

    In: a port. Out: the parsed health object, or $null when the server is not
    listening yet. A returned object's .status is one of ready / loading /
    error / not_configured (nollama.py overall_status).
    #>
    param([int] $P)

    try {
        return Invoke-RestMethod -Uri "http://localhost:$P/health" -TimeoutSec 3 -ErrorAction Stop
    } catch {
        return $null
    }
}

function Wait-ServerReady {
    <#
    Block until /health reports ready, reporting per-slot progress meanwhile.

    Why: the whole point of the script. A fixed sleep is wrong in both
    directions — too short on a cold 15 GB iGPU load, wasted time on a warm
    reuse — and the failure it prevents (client launched against a loading
    server) looks like a config error to the user. Surfacing each slot's status
    also makes a stuck load visible instead of silent.

    Deliberately stricter than /health's own top-level status, which reports
    "ready" as soon as ANY slot is ready (nollama.py overall_status: "a dead
    secondary shouldn't kill the primary"). That is the right answer for a
    liveness probe and the wrong one here: OpenCode's small_model traffic is
    routed to the NPU slot, so handing over while that slot is still loading
    sends the client's first title request at a model that cannot answer it.
    We wait for every slot to leave the loading states, then report.

    In: a port and a timeout in seconds. Out: $true once every slot has
    settled and at least one is usable; $false on timeout or when every slot
    errored. Prints one line per status change, not per poll, so a long load
    does not scroll.
    #>
    param([int] $P, [int] $Seconds)

    $deadline = (Get-Date).AddSeconds($Seconds)
    $lastLine = ""
    # Everything that is not one of these is still in flight (loading,
    # warming_up, and whatever a future slot state is called — treating an
    # unknown state as "still working" is the safe default for a wait loop).
    $settled = @("ready", "idle_unloaded", "error")

    while ((Get-Date) -lt $deadline) {
        $h = Get-ServerHealth -P $P

        if ($null -eq $h) {
            $line = "  waiting for :$P to listen..."
        } elseif ($h.status -ne "error" -and
                  @($h.devices.PSObject.Properties | Where-Object { $_.Value.status -notin $settled }).Count -eq 0) {
            $slots = @($h.devices.PSObject.Properties)
            if (@($slots | Where-Object { $_.Value.status -ne "error" }).Count -eq 0) {
                Write-Host "ERROR: every slot errored - check the server window." -ForegroundColor Red
                return $false
            }
            Write-Host "  server ready (NoLlama $($h.version))" -ForegroundColor Green
            foreach ($d in $slots) {
                $v = $d.Value
                $note = if ($v.kv_pool_gb) { ", kv $($v.kv_pool_gb) GB" } else { "" }
                $colour = if ($v.status -eq "error") { "Yellow" } else { "DarkGray" }
                Write-Host "    $($d.Name.ToUpper()): $($v.model) [$($v.status)$note]" -ForegroundColor $colour
                if ($v.status -eq "error" -and $v.reason) {
                    Write-Host "      reason: $($v.reason)" -ForegroundColor Yellow
                }
            }
            return $true
        } elseif ($h.status -eq "error") {
            Write-Host "ERROR: every slot reports error - check the server window." -ForegroundColor Red
            return $false
        } else {
            $slots = @($h.devices.PSObject.Properties | ForEach-Object { "$($_.Name)=$($_.Value.status)" }) -join " "
            $line = "  loading... $slots"
        }

        if ($line -ne $lastLine) { Write-Host $line -ForegroundColor DarkGray; $lastLine = $line }
        Start-Sleep -Seconds 2
    }

    Write-Host "ERROR: server did not reach 'ready' within $Seconds s." -ForegroundColor Red
    Write-Host "  Raise -TimeoutSec, or watch the server window for a load failure." -ForegroundColor Yellow
    return $false
}

# --- 1. Client first: cheapest check, and the most annoying one to hit late ---
$clientPath = Test-ClientAvailable -Name $Client
if (-not $clientPath) { exit 1 }

$projectDir = (Resolve-Path $Path -ErrorAction SilentlyContinue)?.Path
if (-not $projectDir) {
    Write-Host "ERROR: no such directory: $Path" -ForegroundColor Red
    exit 1
}

$configSrc = Join-Path $ScriptDir "opencode.json"
if (-not (Test-Path $configSrc)) {
    Write-Host "ERROR: no opencode.json in $ScriptDir" -ForegroundColor Red
    Write-Host "  Run .\install.ps1 and pick the 'Coding agent' use-case." -ForegroundColor Yellow
    exit 1
}

Write-Host "=== launch-agent: $Client ===" -ForegroundColor Cyan
Write-Host "  client  : $clientPath"
Write-Host "  project : $projectDir"

# --- 2. Reuse a running server if there is one -------------------------------
# --idle-timeout 0 (what install.ps1 sets for agent use-cases) keeps the model
# resident and the prefix cache warm, so a second launch should cost nothing.
$health = Get-ServerHealth -P $Port

if ($health -and $health.status -eq "ready") {
    Write-Host "  server  : already up on :$Port (reusing, cache stays warm)" -ForegroundColor Green
} elseif ($NoStart) {
    Write-Host "ERROR: nothing ready on :$Port and -NoStart was given." -ForegroundColor Red
    exit 1
} else {
    $startScript = Join-Path $ScriptDir "start.ps1"
    if (-not (Test-Path $startScript)) {
        Write-Host "ERROR: no start.ps1 - run .\install.ps1 first." -ForegroundColor Red
        exit 1
    }

    if ($health) {
        Write-Host "  server  : on :$Port but status '$($health.status)' - waiting" -ForegroundColor Yellow
    } else {
        Write-Host "  server  : starting in a new window..." -ForegroundColor Yellow
        # A separate window, not a background job: the server prints its own
        # device detection and load progress, and that output is the only
        # place a load failure is visible. NOTE over SSH this dies with the
        # session (docs/dev/machines.md, B60 rules) - use -NoStart there and
        # run the server as a scheduled task.
        Start-Process pwsh -ArgumentList @(
            "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $startScript
        ) -WorkingDirectory $ScriptDir | Out-Null
    }

    if (-not (Wait-ServerReady -P $Port -Seconds $TimeoutSec)) { exit 1 }
}

# --- 3. Config by environment, not by copy -----------------------------------
# Layers onto the user's own project/global config instead of replacing it, so
# their providers, agents and keybinds survive. Scoped to this process, so it
# does not leak into the shell the user came from.
$env:OPENCODE_CONFIG = $configSrc
Write-Host "  config  : OPENCODE_CONFIG -> $configSrc (layered, nothing written)" -ForegroundColor DarkGray

# --- 4. Hand over ------------------------------------------------------------
if ($NoLaunch) {
    Write-Host ""
    Write-Host "Ready. -NoLaunch given, so $Client was not started." -ForegroundColor Green
    Write-Host "  To run it yourself with the same config:" -ForegroundColor DarkGray
    Write-Host "    `$env:OPENCODE_CONFIG = '$configSrc'; cd $projectDir; $Client" -ForegroundColor DarkGray
    exit 0
}

Write-Host ""
Write-Host "Starting $Client in $projectDir" -ForegroundColor Green
Write-Host ""
Push-Location $projectDir
try {
    & $clientPath
} finally {
    Pop-Location
}
