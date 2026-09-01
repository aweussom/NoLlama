#requires -Version 7.0
# download-model.ps1 — Download or convert any HuggingFace model for NoLlama
#
# Usage (PowerShell parameters take a SINGLE dash):
#     .\download-model.ps1 OpenVINO/Qwen3-8B-int4-cw-ov          # pre-exported, just download
#     .\download-model.ps1 Qwen/Qwen2.5-VL-3B-Instruct -Convert -Weight int8
#     .\download-model.ps1 Qwen/Qwen2.5-VL-3B-Instruct -Convert -Weight int4 -Trust
#     .\download-model.ps1 HuggingFaceTB/SmolLM3-3B -Convert -Weight int4-cw   # for NPU
#
# Converting for the NPU? Use -Weight int4-cw or int8-cw (channel-wise). The
# default group-quantized int4 produces IRs that crash the NPU driver compiler
# ("Found N duplicated names" / StopLocationVerifierPass, a known vpux bug);
# channel-wise exports compile fine and match Intel's own *-int4-cw-ov models.
# int4-cw vs int8-cw: int8 halves decode speed (SmolLM3-3B on the 285K NPU:
# 23.3 -> 12.3 tok/s) but channel-wise int4 is the lossiest int4 variant, so
# prefer int8-cw for <=3B models when quality matters more than snap.
#     .\download-model.ps1 some-org/gated-model -HfToken hf_xxx  # auth for gated/private models
#
# Downloads to ~/models/<repo-name>/ by default.
# Use -Output to override the target directory.
#
# -HfToken: a HuggingFace access token (https://huggingface.co/settings/tokens).
# Needed for gated/private models; also lifts the unauthenticated rate limit.
# Alternative to a stored 'hf auth login' — same mechanism as install.ps1.

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$HfId,

    [switch]$Convert,

    [string]$Weight = "int4",

    [switch]$Trust,

    [string]$Output = "",

    # Repo branch/tag to download (pre-exported models only). Intel keeps the
    # IR for the current OpenVINO release on a branch while main tracks the
    # next runtime — e.g. OpenVINO/Qwen3.8-27B-int4-ov -Revision 2026.3.1.
    [string]$Revision = "",

    [string]$HfToken,

    # Alternate venv for the conversion (default: <repo>\venv). Lets a scratch
    # venv carry a model-specific stack — e.g. an older transformers for
    # trust-remote-code models written against it (#27) — without touching
    # the venv NoLlama serves from.
    [string]$Venv = "",

    # Catch-all so GNU-style flags (--convert) produce a helpful message
    # instead of PowerShell's cryptic "positional parameter cannot be
    # found" binder error (#19).
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

if ($ExtraArgs) {
    $gnu = @($ExtraArgs | Where-Object { $_ -match '^--' })
    if ($gnu) {
        Write-Host "ERROR: PowerShell flags take a single dash, not '--': $($gnu -join ', ')" -ForegroundColor Red
        Write-Host "  Try:  .\download-model.ps1 $HfId -Convert -Weight int8 -Trust" -ForegroundColor Yellow
    } else {
        Write-Host "ERROR: Unrecognized argument(s): $($ExtraArgs -join ', ')" -ForegroundColor Red
        Write-Host "  Flags: -Convert -Weight <int4|int8> -Trust -Output <dir> -Revision <branch> -HfToken <token> -Venv <dir>" -ForegroundColor Yellow
    }
    exit 1
}
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# huggingface_hub reads HF_TOKEN from the environment, so both 'hf download'
# and optimum-cli pick it up with no further changes. Only set when -HfToken
# was given; otherwise any token stored via 'hf auth login' is used as before.
if ($HfToken) {
    $env:HF_TOKEN = $HfToken
    Write-Host "[+] HF token set for this session (gated/private model auth)" -ForegroundColor DarkGray
}

# Activate venv (Scripts on Windows, bin on POSIX)
$VenvBinDir = if ($IsWindows) { "Scripts" } else { "bin" }
$VenvRoot = if ($Venv) { $Venv } else { Join-Path $ScriptDir "venv" }
$VenvActivate = Join-Path $VenvRoot $VenvBinDir "Activate.ps1"
# A machine set up with 'install.ps1 -Nightly' has venv-nightly/ and no
# venv/. Fall back to it rather than dropping to system Python, which almost
# certainly lacks the 'hf' CLI this script is about to call.
if (-not $Venv -and -not (Test-Path $VenvActivate)) {
    $nightlyRoot = Join-Path $ScriptDir "venv-nightly"
    $nightlyActivate = Join-Path $nightlyRoot $VenvBinDir "Activate.ps1"
    if (Test-Path $nightlyActivate) {
        Write-Host "[i] No venv/ - using venv-nightly/ (nightly install)." -ForegroundColor DarkGray
        $VenvRoot = $nightlyRoot
        $VenvActivate = $nightlyActivate
    }
}
if (Test-Path $VenvActivate) {
    & $VenvActivate
} elseif ($Venv) {
    Write-Host "ERROR: -Venv given but $VenvActivate not found." -ForegroundColor Red
    exit 1
} else {
    Write-Host "WARNING: No venv found. Using system Python." -ForegroundColor Yellow
}

# Determine target directory
$RepoName = ($HfId -split '/')[-1]
if (-not $Output) {
    $Output = Join-Path $HOME "models" $RepoName
}

Write-Host ""
Write-Host "=== NoLlama Model Download ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Model:  $HfId"
Write-Host "  Target: $Output"
if ($Convert) {
    Write-Host "  Mode:   Convert (optimum-cli, $Weight)"
} else {
    Write-Host "  Mode:   Download (pre-exported)"
}
Write-Host ""

if (Test-Path $Output) {
    if (-not $Convert) {
        # Plain downloads RESUME: hf skips complete files and continues
        # partial ones. Deleting here (the old behavior) forced a full
        # re-download after every interrupted attempt.
        Write-Host "Target directory exists — resuming/refreshing download into it." -ForegroundColor Yellow
    } else {
        Write-Host "Target directory already exists: $Output" -ForegroundColor Yellow
        $reply = Read-Host "Overwrite? [y/N]"
        if ($reply -notin @("y", "Y", "yes")) {
            Write-Host "Aborted."
            exit 0
        }
        $item = Get-Item $Output -Force
        if ($item.LinkType) {
            # Remove link without following.
            if ($IsWindows) { cmd /c rmdir "`"$Output`"" | Out-Null }
            else            { Remove-Item -Force $Output }
        } else {
            Remove-Item -Recurse -Force $Output
        }
    }
}

if ($Convert) {
    Write-Host "Converting $HfId to OpenVINO ($Weight)..." -ForegroundColor Cyan
    Write-Host "  This may take 5-30 minutes depending on model size."
    Write-Host ""

    # int4-cw / int8-cw: channel-wise symmetric quantization, the NPU-safe
    # variant (see header note). Maps to optimum-cli's flag spelling.
    if ($Weight -match '^(int[48])-cw$') {
        $args = @("export", "openvino", "--model", $HfId, "--weight-format", $Matches[1],
                  "--group-size", "-1", "--sym", "--ratio", "1.0")
    } else {
        $args = @("export", "openvino", "--model", $HfId, "--weight-format", $Weight)
    }
    if ($Trust) { $args += "--trust-remote-code" }
    $args += $Output

    Write-Host "Running: optimum-cli $($args -join ' ')" -ForegroundColor DarkGray
    Write-Host ""
    # `python -m` rather than the optimum-cli shim, for the same reason as the
    # download path below: application-control policies block generated
    # Scripts\*.exe launchers on managed machines. Same entry point.
    & python -m optimum.commands.optimum_cli @args
    if (-not $?) {
        Write-Host ""
        Write-Host "ERROR: Conversion failed." -ForegroundColor Red
        Write-Host "  Common fixes:" -ForegroundColor Yellow
        Write-Host "    - Add -Trust if the model needs trust-remote-code" -ForegroundColor Yellow
        Write-Host "    - Check that optimum-intel is installed: pip install optimum[openvino]" -ForegroundColor Yellow
        Write-Host "    - Some architectures aren't supported yet by optimum-intel" -ForegroundColor Yellow
        Write-Host "    - 'Maximum required is X, got: Y': transformers is too new for this" -ForegroundColor Yellow
        Write-Host "      architecture's exporter — it needs transformers==X (the version the" -ForegroundColor Yellow
        Write-Host "      error names). Build a scratch venv so the serving venv stays intact:" -ForegroundColor Yellow
        Write-Host "        python -m venv venv-convert" -ForegroundColor Yellow
        Write-Host "        venv-convert\Scripts\pip install `"optimum-intel[openvino]>=1.27`" `"transformers==X`"" -ForegroundColor Yellow
        Write-Host "        rerun this script with:  -Venv venv-convert" -ForegroundColor Yellow
        Write-Host "    - ImportError from the MODEL's own .py files (e.g. 'cannot import name" -ForegroundColor Yellow
        Write-Host "      LossKwargs'): trust-remote-code models ship modeling code written for" -ForegroundColor Yellow
        Write-Host "      an older transformers (LossKwargs died in 4.56). Same scratch-venv" -ForegroundColor Yellow
        Write-Host "      recipe as above with transformers==4.55.*" -ForegroundColor Yellow
        Write-Host "    - 'DefaultCPUAllocator: not enough memory': conversion holds the full-" -ForegroundColor Yellow
        Write-Host "      precision model (MoE models: plus fp32 expert copies) in memory." -ForegroundColor Yellow
        Write-Host "      Raise the Windows pagefile (commit limit) and reboot, then rerun." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "Downloading $HfId..." -ForegroundColor Cyan
    Write-Host ""

    $env:PYTHONIOENCODING = "utf-8"
    $revArgs = @()
    if ($Revision) { $revArgs = @("--revision", $Revision); Write-Host "  Revision: $Revision" }
    # Via python, not the `hf` console script: `hf download` runs
    # venv\Scripts\hf.exe, a generated launcher that Windows application
    # -control policies (WDAC / AppLocker / Smart App Control) block on
    # managed machines while still allowing python.exe. Seen 2026-09-01 on a
    # Win11 Pro workstation — "En programkontrollpolicy har blokkert denne
    # filen" — with the rest of the venv working normally.
    & python (Join-Path $ScriptDir "scripts" "hf_download.py") $HfId $Output @revArgs
    if (-not $?) {
        Write-Host ""
        Write-Host "ERROR: Download failed." -ForegroundColor Red
        Write-Host "  If 401/403: pass -HfToken hf_xxx (or run 'hf auth login' first)" -ForegroundColor Yellow
        exit 1
    }
}

# Verify what actually landed — an interrupted download leaves a truncated
# .bin that fails at load with a cryptic "Empty weights data" error (#17).
# --scan reads the IR's own weight-size records and says so in plain words.
Write-Host ""
$scanOut = & python (Join-Path $ScriptDir "nollama.py") --scan $Output 2>&1
$scanOut | Write-Host
if ($scanOut -match "PROBLEM") {
    Write-Host ""
    Write-Host "ERROR: The downloaded model failed the integrity check (see PROBLEM above)." -ForegroundColor Red
    Write-Host "  Usually an interrupted download. Re-run this script — the download" -ForegroundColor Yellow
    Write-Host "  resumes into the existing directory (complete files are skipped)." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "[OK] Model ready at: $Output" -ForegroundColor Green
Write-Host ""
Write-Host "To use with NoLlama:"
Write-Host "  python nollama.py --model-dir `"$Output`" --device GPU"
Write-Host "  python nollama.py --gpu-model-dir `"$Output`""
Write-Host ""
