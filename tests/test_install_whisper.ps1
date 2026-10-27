# The installer's speech-to-text step, driven without a console or a download.
#
#     pwsh -NoProfile -File tests\test_install_whisper.ps1
#
# Loads install.ps1's functions from its AST (running the script would build a
# venv and overwrite start.ps1), fakes Read-Host and Install-Model, and checks
# what each hardware shape is offered: with an NPU it is asked before the
# use-case menu, on the NPU; NPU 4 gets small with no further question (medium
# dies there, base is withdrawn); NPU 3 / CPU / GPU choose small or medium. Also the cache check, which used to call a cached Whisper incomplete
# and re-download it on every install.

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$script:failed = 0

function Assert-Equal($expected, $actual, $what) {
    if ("$expected" -ne "$actual") {
        Write-Host "FAIL $what`n  expected: $expected`n  actual:   $actual" -ForegroundColor Red
        $script:failed++
    } else {
        Write-Host "ok   $what" -ForegroundColor DarkGray
    }
}

$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $repo "install.ps1"), [ref]$null, [ref]$null)
foreach ($f in $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    . ([scriptblock]::Create($f.Extent.Text))
}

$Registry = Get-Content (Join-Path $repo "models.json") -Raw | ConvertFrom-Json
$Nightly = $false
$tmp = Join-Path ([IO.Path]::GetTempPath()) "nollama-test-whisper-$PID"
New-Item -ItemType Directory -Force $tmp | Out-Null
$ModelsRoot = Join-Path $tmp "models"
$ScriptDir = $tmp

# Fakes. Functions win over cmdlets, so this Read-Host shadows the real one.
function Read-Host { param($Prompt) if ($script:keys.Count) { $script:keys.Dequeue() } else { "" } }
function Install-Model { param($Selected, $TargetDir) $script:installed = $Selected.HfId; $true }

function Run([string]$arch, [bool]$gpu, [string[]]$keys, [switch]$Upfront) {
    $script:HasNPU = [bool]$arch
    $script:HasGPU = $gpu
    $script:DeviceInfo = if ($arch) { @{ NPU = @{ name = "fake NPU"; arch = $arch } } } else { @{} }
    $script:keys = [System.Collections.Queue]::new()
    foreach ($k in $keys) { $script:keys.Enqueue($k) }
    $script:installed = $null
    $out = Select-SpeechToText -Upfront:$Upfront 6>$null
    return ($out -join " ")
}

# NPU 4, asked upfront: yes -> small on the NPU, no device question, no menu.
Assert-Equal "--whisper-dir whisper-model --whisper-device NPU" (Run "4000" $true @("") -Upfront) "NPU 4: small on the NPU, one question"
Assert-Equal "OpenVINO/whisper-small-int8-ov" $script:installed "NPU 4: the one installed is small"

# NPU 3: medium works there, so there is a choice; 2 = medium.
Assert-Equal "--whisper-dir whisper-model --whisper-device NPU" (Run "3720" $false @("", "2") -Upfront) "NPU 3: menu, pick 2"
Assert-Equal "OpenVINO/whisper-medium-int8-ov" $script:installed "NPU 3: medium offered and taken"

# No NPU, no GPU: asked after the models, CPU only (no device question).
Assert-Equal "--whisper-dir whisper-model --whisper-device CPU" (Run "" $false @("", "1")) "CPU only: no device question, pick 1"
Assert-Equal "OpenVINO/whisper-small-int8-ov" $script:installed "CPU only: 1 is small"

# No NPU, GPU: device question is CPU/GPU; 2 = GPU, then small.
Assert-Equal "--whisper-dir whisper-model --whisper-device GPU" (Run "" $true @("", "2", "1")) "GPU box: pick GPU, then small"

# Base is withdrawn everywhere: the CPU menu has exactly small and medium.
$cpuFit = @(Where-NpuUsable $Registry.whisper "CPU" | Where-Object { -not ($_.PSObject.Properties['withdrawn'] -and $_.withdrawn) })
Assert-Equal "OpenVINO/whisper-small-int8-ov OpenVINO/whisper-medium-int8-ov" (($cpuFit | ForEach-Object hf_id) -join " ") "base is not offered"

Assert-Equal "" (Run "4000" $true @("n") -Upfront) "NPU declined: no args"
Assert-Equal "" (Run "3720" $false @("", "") -Upfront) "Enter at the menu skips: no args"
Assert-Equal $null $script:installed "skipped: nothing installed"

# The upfront call sits before the use-case menu, the fallback after it.
$src = Get-Content (Join-Path $repo "install.ps1") -Raw
$iUp = $src.IndexOf('Select-SpeechToText -Upfront)'); $iMenu = $src.IndexOf('=== What will you use NoLlama for? ==='); $iLate = $src.IndexOf('$SttArgs = @(Select-SpeechToText)')
Assert-Equal $true ($iUp -gt 0 -and $iUp -lt $iMenu -and $iMenu -lt $iLate) "NPU question comes before the use-case menu"

# The cache check: a complete Whisper dir is valid, a truncated one is not.
function New-FakeIr($dir, $base, [int64]$size) {
    $xml = "<net><layers><layer><data offset=`"0`" size=`"$size`"/></layer></layers></net>"
    Set-Content -Path (Join-Path $dir "$base.xml") -Value $xml
    $fs = [IO.File]::Create((Join-Path $dir "$base.bin")); $fs.SetLength($size); $fs.Close()
}
$w = Join-Path $tmp "whisper-cache"; New-Item -ItemType Directory -Force $w | Out-Null
New-FakeIr $w "openvino_encoder_model" 23MB
New-FakeIr $w "openvino_decoder_model" 52MB
Assert-Equal $true (Test-ModelCacheValid -Path $w) "complete Whisper cache is valid"
$fs = [IO.File]::Open((Join-Path $w "openvino_decoder_model.bin"), "Open"); $fs.SetLength(40MB); $fs.Close()
Assert-Equal $false (Test-ModelCacheValid -Path $w 6>$null) "truncated Whisper decoder is not"
Remove-Item (Join-Path $w "openvino_decoder_model.*")
Assert-Equal $false (Test-ModelCacheValid -Path $w) "encoder alone is not"

Remove-Item -Recurse -Force $tmp
if ($script:failed) { Write-Host "$($script:failed) failed" -ForegroundColor Red; exit 1 }
Write-Host "all passed"
