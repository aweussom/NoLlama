#requires -Version 7.0
<#
Set up the python venv for NoLlama's optimum backend — the runtime for
brand-new architectures that openvino_genai cannot serve yet (see
docs\MODELS.md "Brand-new architectures"). Serving only: for EXPORTING such
a model see scripts\glimmer-export\.

Note (2026-08-18): Muse Glimmer no longer needs this — its VLM-shaped
exports auto-route to the GenAI path (nightly runtime until 2026.4
releases). This venv remains the path for nemotron_h-class LLM-shaped
architectures, and the '--backend optimum' fallback for Glimmer, which now
requires that flag explicitly (auto-detection picks GenAI).

What lands in the venv, and why this exact order:
  - openvino + openvino-genai + flask + pillow: NoLlama's normal serving deps
  - torch, CPU build: the optimum runtime rides on it; CUDA is dead weight here
  - optimum-intel from git (muse_glimmer support is not in a release yet)
  - transformers from git, LAST: optimum-intel pins transformers<5.6 and pip
    enforces the pin by downgrading whatever is present; Glimmer only exists
    on transformers main, so we override the pin afterwards (pip warns, obeys)

Needs git on PATH (pip installs two packages straight from GitHub).
Idempotent: re-running on a healthy venv exits early.

  .\scripts\New-OptimumVenv.ps1                       # creates ./venv-optimum
  .\scripts\New-OptimumVenv.ps1 -Python python3.12    # pick the python to seed from
  .\scripts\New-OptimumVenv.ps1 -Nightly              # ./venv-optimum-nightly, OpenVINO nightly

-Nightly builds a SECOND venv on the OpenVINO nightly wheels, leaving the
release one intact. That is the setup for re-testing the GPU corruption
(openvinotoolkit/openvino#37419) on a newer runtime: the release venv is the
control that produced the previous verdict, so upgrading it in place would
destroy the comparison. Pin -OptimumIntelRef/-TransformersRef to whatever the
control venv holds ('pip show optimum-intel') so OpenVINO is the only variable.
Then serve with it (--backend optimum: Glimmer's VLM-shaped export would
otherwise auto-route to GenAI, which this venv's release runtime can't run):
  Windows: venv-optimum\Scripts\python.exe nollama.py --model-dir <dir> --backend optimum --device CPU
  Linux:   venv-optimum/bin/python nollama.py --model-dir <dir> --backend optimum --device CPU
#>
param(
    [string]$Python = 'python',
    # Empty = pick the default below, which depends on -Nightly.
    [string]$VenvDir = '',
    # Pin these to a commit/tag if main breaks; defaults track upstream.
    [string]$TransformersRef = 'main',
    [string]$OptimumIntelRef = 'main',
    # Build against the OpenVINO NIGHTLY runtime instead of the current
    # release, into venv-optimum-nightly/ so the release venv survives as a
    # control. This is how you answer "did the new OpenVINO fix the GPU
    # corruption?" (TODONT: re-run the comprehension test on each release)
    # without destroying the runtime that produced the previous verdict.
    [switch]$Nightly,
    [string]$NightlyIndex = 'https://storage.openvinotoolkit.org/simple/wheels/nightly',
    # Catch-all. 'pwsh -File script.ps1 -Unknown' does NOT error — it binds
    # what it recognises, silently drops the rest, and runs the body with
    # exit 0. Passing -Nightly to a checkout that predates it therefore
    # looked like success while building the release venv (2026-08-15).
    # Same guard as download-model.ps1 (#19).
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

if ($ExtraArgs) {
    Write-Host "ERROR: unrecognized argument(s): $($ExtraArgs -join ', ')" -ForegroundColor Red
    Write-Host '  Flags: -Nightly -Python <exe> -VenvDir <dir> -TransformersRef <ref> -OptimumIntelRef <ref>' -ForegroundColor Yellow
    Write-Host '  If you expected one of these to exist, check you are on a branch that has it:' -ForegroundColor Yellow
    Write-Host '    git branch --show-current' -ForegroundColor Yellow
    exit 1
}

if (-not $VenvDir) {
    # Join-Path, not "\": a literal backslash lands in the Linux path
    # (found by the first Fedora user, issue #24).
    $VenvDir = Join-Path $PSScriptRoot ($Nightly ? 'venv-optimum-nightly' : 'venv-optimum')
}
# Step count in the progress labels — the nightly upgrade is one extra pass.
$N = if ($Nightly) { 5 } else { 4 }
function Invoke-Step($desc, [scriptblock]$cmd) {
    Write-Host $desc
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "failed: $desc" }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git is required on PATH (transformers/optimum-intel install from GitHub).'
}

$py = $IsWindows ? "$VenvDir\Scripts\python.exe" : "$VenvDir/bin/python"
if (-not (Test-Path $py)) {
    Invoke-Step "[1/$N] Creating venv at $VenvDir..." { & $Python -m venv $VenvDir }
} else {
    Write-Host "[1/$N] venv exists at $VenvDir - reusing."
}

# The assert mirrors export-glimmer.ps1: muse_glimmer registered proves the
# git transformers survived (a pip run that re-pinned it would drop the arch).
# Under -Nightly it must ALSO be a dev build of OpenVINO, or an early exit
# would hand back a release runtime and quietly invalidate the comparison.
$readyCheck = "import optimum.intel, openvino_genai, flask; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'muse_glimmer' in CONFIG_MAPPING"
if ($Nightly) {
    $readyCheck += "; import openvino; assert 'dev' in openvino.__version__, openvino.__version__"
}
& $py -c $readyCheck 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host 'Venv already serves the optimum backend - nothing to do.'
    exit 0
}

Invoke-Step "[2/$N] Serving deps (openvino, openvino-genai, flask, torch CPU)..." {
    & $py -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) { return }
    & $py -m pip install openvino openvino-genai flask pillow einops accelerate
    if ($LASTEXITCODE -ne 0) { return }
    if ($IsWindows) {
        & $py -m pip install torch   # Windows default wheels are CPU-only
    } else {
        & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    }
}

Invoke-Step "[3/$N] optimum-intel @ $OptimumIntelRef (git)..." {
    & $py -m pip install "optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@$OptimumIntelRef"
}

Invoke-Step "[4/$N] transformers @ $TransformersRef (git) - overrides the <5.6 pin..." {
    & $py -m pip install "transformers @ git+https://github.com/huggingface/transformers.git@$TransformersRef"
}

if ($Nightly) {
    # Last, so it wins over the release wheels step 2 pulled in as
    # optimum-intel's dependencies. Only the three OpenVINO packages get
    # --pre; a blanket pre-release pass would also swap numpy/pillow for
    # release candidates and add variables to an experiment that is meant
    # to have exactly one.
    Invoke-Step "[5/$N] OpenVINO nightly (overrides the release wheels)..." {
        & $py -m pip install --pre -U openvino openvino-tokenizers openvino-genai --extra-index-url $NightlyIndex
    }
}

& $py -c "import openvino, transformers, optimum.intel; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'muse_glimmer' in CONFIG_MAPPING, 'muse_glimmer arch missing - transformers pin was not overridden'; print('openvino', openvino.__version__); print('transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'verification failed - see errors above' }

$pyPath = $IsWindows ? "$VenvDir\Scripts\python.exe" : "$VenvDir/bin/python"
Write-Host ''
Write-Host 'Done. Serve an optimum-backend model with:'
if ($Nightly) {
    Write-Host "  $pyPath nollama.py --model-dir <model-dir> --device GPU"
    Write-Host ''
    Write-Host 'This venv exists to re-test the GPU corruption (openvino#37419) on a'
    Write-Host 'newer runtime, so --device GPU is the point. Run the comprehension test'
    Write-Host 'in TODONT.md before believing any result, and re-run the CPU control in'
    Write-Host 'THIS venv too - transformers/optimum-intel track git main and move'
    Write-Host 'between builds, so only a same-venv control isolates the OpenVINO change.'
} else {
    Write-Host "  $pyPath nollama.py --model-dir <model-dir> --device CPU"
    Write-Host '(--device CPU is deliberate: no current Intel GPU runs Glimmer correctly'
    Write-Host ' - integrated or discrete, verified on the Arc Pro B60. See README.)'
}
