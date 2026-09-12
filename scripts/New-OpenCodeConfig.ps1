#requires -Version 7.0
<#
.SYNOPSIS
Write an opencode.json that points OpenCode at NoLlama — the two-server recipe or dual mode.

Why: the recipe in docs/AGENTS.md is three settings people get wrong by hand
(the model id NoLlama actually advertises, an honest context limit, and the
two five-minute OpenCode timeouts), so install.ps1 writes the file and this
script is the one place that knows the shape. Runs standalone too for a
manual install: give it the names and devices and it prints where it wrote.

In: model DIRECTORY names (what is on disk — the script derives the id the
server advertises the same way nollama.py does), the devices, and a mode.
"two-servers" = coder on :Port, small model on :SmallPort (docs/AGENTS.md,
"Two servers"). "dual" = one NoLlama process serving both, addressed as
<name>@GPU / <name>@NPU. Out: the JSON file at -Out; existing file is
overwritten (it is generated, like start.ps1).
#>
param(
    [Parameter(Mandatory)] [string] $CoderDir,
    [string] $CoderDevice = "GPU",
    [string] $SmallDir = "",
    [string] $SmallDevice = "NPU",
    [ValidateSet("two-servers", "dual")] [string] $Mode = "two-servers",
    [int] $Port = 8000,
    [int] $SmallPort = 8002,
    [int] $CoderContext = 60000,
    [string] $Out = "opencode.json"
)

function Get-NoLlamaDisplayName {
    <#
    Mirror of nollama.py _strip_name_suffixes: the id NoLlama advertises for
    a model directory. Why: the config's model key must equal that id, or
    dual-mode routing (<name>@DEVICE) sends the request to the wrong slot.
    In: a directory name. Out: the name with each of -ov, -openvino, -int8,
    -int4 removed once, in that order, case-insensitively (so
    "Qwen3-8B-int4-cw-ov" -> "Qwen3-8B-int4-cw": "-ov" goes, "-int4" is not
    at the end and stays — exactly what the server does).
    #>
    param([string] $Name)
    foreach ($suffix in @("-ov", "-openvino", "-int8", "-int4")) {
        if ($Name.ToLower().EndsWith($suffix)) { $Name = $Name.Substring(0, $Name.Length - $suffix.Length) }
    }
    return $Name
}

$coder = Get-NoLlamaDisplayName (Split-Path $CoderDir -Leaf)
$small = if ($SmallDir) { Get-NoLlamaDisplayName (Split-Path $SmallDir -Leaf) } else { "" }
# The small model's window: the NPU prompt cap is 4096 tokens; on CPU a
# little more is fine but the side-tasks never need it.
$smallContext = if ($SmallDevice -eq "NPU") { 4096 } else { 8192 }
$timeouts = @{ chunkTimeout = 1800000; headerTimeout = 1800000 }

if ($Mode -eq "two-servers") {
    $providers = [ordered]@{
        "nollama-gpu" = [ordered]@{
            npm = "@ai-sdk/openai-compatible"; name = "NoLlama ($CoderDevice)"
            options = ([ordered]@{ baseURL = "http://localhost:$Port/v1" } + $timeouts)
            models = [ordered]@{ $coder = [ordered]@{ limit = [ordered]@{ context = $CoderContext; output = 16384 } } }
        }
    }
    $model = "nollama-gpu/$coder"
    $smallRef = $null
    if ($small) {
        $providers["nollama-small"] = [ordered]@{
            npm = "@ai-sdk/openai-compatible"; name = "NoLlama small ($SmallDevice)"
            options = [ordered]@{ baseURL = "http://localhost:$SmallPort/v1"; chunkTimeout = 600000 }
            models = [ordered]@{ $small = [ordered]@{ limit = [ordered]@{ context = $smallContext; output = 1024 } } }
        }
        $smallRef = "nollama-small/$small"
    }
} else {
    # Dual mode: one process, two slots, addressed as <name>@DEVICE.
    $models = [ordered]@{ "$coder@$CoderDevice" = [ordered]@{ limit = [ordered]@{ context = $CoderContext; output = 16384 } } }
    if ($small) { $models["$small@$SmallDevice"] = [ordered]@{ limit = [ordered]@{ context = $smallContext; output = 1024 } } }
    $providers = [ordered]@{
        "nollama" = [ordered]@{
            npm = "@ai-sdk/openai-compatible"; name = "NoLlama"
            options = ([ordered]@{ baseURL = "http://localhost:$Port/v1" } + $timeouts)
            models = $models
        }
    }
    $model = "nollama/$coder@$CoderDevice"
    $smallRef = if ($small) { "nollama/$small@$SmallDevice" } else { $null }
}

$cfg = [ordered]@{
    '$schema' = "https://opencode.ai/config.json"
    provider = $providers
    model = $model
}
if ($smallRef) { $cfg["small_model"] = $smallRef }
# The two knobs that keep a weak model's tool fan-out from flooding the pool
# (OPENCODE-PLAN.md arm 0: 27 reads, 1.49 M chars). Free to change.
$cfg["tool_output"] = [ordered]@{ max_lines = 300; max_bytes = 12288 }
$cfg["compaction"] = [ordered]@{ prune = $true }

$cfg | ConvertTo-Json -Depth 6 | Set-Content -Path $Out -Encoding UTF8
Write-Host "[OK] Wrote $Out — model $model$(if ($smallRef) { ", small_model $smallRef" })" -ForegroundColor Green
