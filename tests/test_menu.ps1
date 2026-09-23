# Interaction tests for scripts/Menu.ps1, with no terminal involved.
#
#     pwsh -NoProfile -File tests\test_menu.ps1
#
# Show-Choice takes its keys from -KeyReader and its output through -Writer, so
# a test drives the whole loop with a synthetic key sequence and asserts both
# the returned value and what the user would have seen. The thing this guards
# is the redraw arithmetic and the fallback gate: a menu that assumes a console
# throws on WindowWidth in any redirected session.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\scripts\Menu.ps1"

$script:failed = 0

function Assert-Equal($expected, $actual, $what) {
    if ($expected -ne $actual) {
        Write-Host "FAIL $what`n  expected: $expected`n  actual:   $actual" -ForegroundColor Red
        $script:failed++
    } else {
        Write-Host "ok   $what" -ForegroundColor DarkGray
    }
}

function Assert-True($cond, $what) { Assert-Equal $true ([bool]$cond) $what }

$opts = @(
    (New-ChoiceOption -Label "Coding agent" -Description "OpenCode, tools" -Value "agent"),
    (New-ChoiceOption -Label "Better chat" -Description "bigger model on the GPU" -Value "chat"),
    (New-ChoiceOption -Label "Vision jobs" -Description "VLM on the GPU" -Value "vision")
)

function Run($keyNames) {
    $script:keyQueue = [System.Collections.Queue]::new()
    foreach ($n in $keyNames) {
        $ch  = if ($n.Length -eq 1) { [char]$n } else { [char]0 }
        $key = if ($n.Length -eq 1) { [ConsoleKey]::NoName } else { [ConsoleKey]$n }
        $script:keyQueue.Enqueue([System.ConsoleKeyInfo]::new($ch, $key, $false, $false, $false))
    }
    $script:rendered = @()
    $value = Show-Choice -Question "What is this machine for?" -Options $opts -AllowSkip `
                         -KeyReader { $script:keyQueue.Dequeue() } `
                         -Writer { param($line) $script:rendered += $line }
    return $value
}

Assert-Equal "agent"  (Run @("Enter"))                        "Enter takes the first option"
Assert-Equal "chat"   (Run @("DownArrow", "Enter"))           "down then Enter takes the second"
Assert-Equal "vision" (Run @("UpArrow", "Enter"))             "up from the top wraps to the last"
Assert-Equal "vision" (Run @("End", "Enter"))                 "End jumps to the last"
Assert-Equal "chat"   (Run @("2"))                            "a digit selects directly"
Assert-Equal $null    (Run @("Escape"))                       "Escape skips when allowed"

# The rendered frame is what a user reads, so assert its shape, not just the
# return value: the marker must move even with styling stripped.
$null = Run @("DownArrow", "Enter")
$marked = @($script:rendered | Where-Object { $_ -match "^\s*>" })
Assert-True ($marked.Count -ge 2)                             "every frame marks a row"
Assert-True ($marked[0] -match "Coding agent")                "first frame marks the first option"
Assert-True ($marked[-1] -match "Better chat")                "last frame marks the moved-to option"
Assert-True (@($script:rendered | Where-Object { $_ -match "OpenCode, tools" }).Count -ge 1) `
                                                              "descriptions are rendered"

# The gate that keeps unattended installs alive.
Assert-Equal $false (Test-InteractiveConsole) "no console here, so the numbered fallback is chosen"

# --- the recommendation starts under the cursor -------------------------
function RunDefault($keyNames, $default) {
    $script:keyQueue = [System.Collections.Queue]::new()
    foreach ($n in $keyNames) {
        $ch  = if ($n.Length -eq 1) { [char]$n } else { [char]0 }
        $key = if ($n.Length -eq 1) { [ConsoleKey]::NoName } else { [ConsoleKey]$n }
        $script:keyQueue.Enqueue([System.ConsoleKeyInfo]::new($ch, $key, $false, $false, $false))
    }
    $script:rendered = @()
    return Show-Choice -Question "q" -Options $opts -Default $default `
                       -KeyReader { $script:keyQueue.Dequeue() } `
                       -Writer { param($line) $script:rendered += $line }
}
Assert-Equal "chat"   (RunDefault @("Enter") 1)      "Enter takes the recommended option, not the first"
Assert-Equal "vision" (RunDefault @("Enter") 99)     "an out-of-range default clamps to the last"
Assert-Equal "agent"  (RunDefault @("UpArrow", "Enter") 1) "moving off the default still works"

# --- paging keeps the selection visible ---------------------------------
$many = 1..20 | ForEach-Object { New-ChoiceOption -Label "Model $_" -Value $_ }
$top = Format-ChoiceFrame -Question "q" -Options $many -Selected 0 -WindowSize 8 -Plain
Assert-True  ($top -contains "  > Model 1")          "window starts at the top"
Assert-True  (($top -join "`n") -match "more below")          "and says how many are hidden"
Assert-True  (-not (($top -join "`n") -match "more above"))   "nothing hidden above at the top"

$mid = Format-ChoiceFrame -Question "q" -Options $many -Selected 10 -WindowSize 8 -Plain
Assert-True ($mid -contains "  > Model 11")        "the selected row is inside the window"
Assert-True (($mid -join "`n") -match "more above")           "and both markers show mid-list"
Assert-True (($mid -join "`n") -match "more below")           "..."

$end = Format-ChoiceFrame -Question "q" -Options $many -Selected 19 -WindowSize 8 -Plain
Assert-True ($end -contains "  > Model 20")        "the last row is reachable"
Assert-True (-not (($end -join "`n") -match "more below"))    "nothing hidden below at the end"
$rows = @($end | Where-Object { $_ -match "Model \d+" })
Assert-Equal 8 $rows.Count                                    "the window never grows past its size"

if ($script:failed) { Write-Host "$($script:failed) failure(s)" -ForegroundColor Red; exit 1 }
Write-Host "all passed" -ForegroundColor Green
