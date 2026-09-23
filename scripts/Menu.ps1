# Menu.ps1 — an arrow-key chooser for install.ps1, with no module to install.
#
# Why this is hand-rolled: install.ps1 is the FIRST thing a user runs, so it
# cannot depend on a module that is not there yet. Out-ConsoleGridView needs
# Microsoft.PowerShell.ConsoleGuiTools installed first; Out-GridView is a
# Windows-only GUI popup that dies over SSH and under Linux pwsh, which is how
# Linux users install (there is no install.sh). PromptForChoice is built in but
# is still one keystroke per item. So: ReadKey, ANSI via $PSStyle, ~150 lines.
#
# Dot-source it:  . "$PSScriptRoot\scripts\Menu.ps1"

Set-StrictMode -Version Latest

function Test-InteractiveConsole {
    # True when a real console is attached and can be driven key by key.
    #
    # Why: every other function here touches [Console] members that THROW
    # rather than return a default when there is no console — WindowWidth and
    # TreatControlCAsInput both raise "Ugyldig referanse"/invalid handle in a
    # redirected session (verified 2026-09-23 in a piped pwsh). A menu that
    # assumes a console hangs or crashes an unattended install instead of
    # falling back to the numbered prompt.
    #
    # In: nothing. Out: $true only if stdin is a console AND geometry reads
    # succeed; any failure answers $false, never an exception.
    if ([Console]::IsInputRedirected) { return $false }
    try { $null = [Console]::WindowWidth; return $true } catch { return $false }
}

function Show-NumberedChoice {
    # The fallback: the classic "pick a number" prompt.
    #
    # Why it stays: scripted installs, CI and anything piped into pwsh have no
    # key reader, and a scenario picker that cannot run headless would block
    # the Docker path and every automated rebuild.
    #
    # In: the same shape Show-Choice takes. Out: the chosen option's Value, or
    # $null when skipping is allowed and the user just pressed Enter.
    param(
        [string]$Question,
        [object[]]$Options,
        [switch]$AllowSkip,
        [int]$Default = 0
    )
    Write-Host ""
    Write-Host "  $Question" -ForegroundColor Cyan
    for ($i = 0; $i -lt $Options.Count; $i++) {
        $mark = if ($i -eq $Default) { "*" } else { " " }
        Write-Host ("  {0}{1,2}. {2}" -f $mark, ($i + 1), $Options[$i].Label)
        if ($Options[$i].Description) {
            Write-Host ("       {0}" -f $Options[$i].Description) -ForegroundColor DarkGray
        }
    }
    $star = $Options[$Default].Label
    $prompt = if ($AllowSkip) { "Pick [1-$($Options.Count)], Enter to skip" }
              else            { "Pick [1-$($Options.Count)], Enter for *$star" }
    while ($true) {
        $answer = Read-Host $prompt
        if ([string]::IsNullOrWhiteSpace($answer)) {
            if ($AllowSkip) { return $null }
            return $Options[$Default].Value
        }
        $n = 0
        if ([int]::TryParse($answer, [ref]$n) -and $n -ge 1 -and $n -le $Options.Count) {
            return $Options[$n - 1].Value
        }
        Write-Host "  Enter a number between 1 and $($Options.Count)" -ForegroundColor Red
    }
}

function Format-ChoiceFrame {
    # Render one frame of the menu as an array of lines.
    #
    # Why a pure function: it makes the whole interaction testable without a
    # terminal — a test feeds a cursor position and asserts what the user
    # would see, and the drawing code never has to be trusted by eye alone.
    # It also keeps the redraw logic (how many lines to move up) honest,
    # because the count comes from the same place as the output.
    #
    # In: the options, the selected index, and whether styling is wanted.
    # Out: an array of strings, one per screen line, no trailing newline.
    param(
        [string]$Question,
        [object[]]$Options,
        [int]$Selected,
        [int]$WindowSize = 8,
        [switch]$Plain
    )
    $sel  = if ($Plain) { "" } else { $PSStyle.Reverse }
    $dim  = if ($Plain) { "" } else { $PSStyle.Dim }
    $rst  = if ($Plain) { "" } else { $PSStyle.Reset }

    # Scroll window: a long model list must not push the question off screen,
    # and a list that overflows is also a list nobody reads to the end.
    $first = 0
    if ($Options.Count -gt $WindowSize) {
        $first = [Math]::Max(0, $Selected - [int]($WindowSize / 2))
        $first = [Math]::Min($first, $Options.Count - $WindowSize)
    }
    $last = [Math]::Min($Options.Count - 1, $first + $WindowSize - 1)

    $lines = @("", "  $Question", "")
    if ($first -gt 0) { $lines += "$dim      $first more above$rst" }
    for ($i = $first; $i -le $last; $i++) {
        $marker = if ($i -eq $Selected) { ">" } else { " " }
        # The marker carries the selection as well as the colour: a terminal
        # that drops ANSI (or $PSStyle.OutputRendering = PlainText, which is
        # what redirected output gets) must still show which row is current.
        $label = "  $marker $($Options[$i].Label)"
        $lines += if ($i -eq $Selected) { "$sel$label$rst" } else { $label }
        if ($Options[$i].Description) {
            $lines += "$dim      $($Options[$i].Description)$rst"
        }
    }
    if ($last -lt $Options.Count - 1) {
        $lines += "$dim      $($Options.Count - 1 - $last) more below$rst"
    }
    $lines += ""
    $lines += "$dim  up/down to move, Enter to choose, Esc to skip$rst"
    return $lines
}

function Show-Choice {
    # Ask one question with an arrow-key list, or fall back to numbers.
    #
    # Why: the installer asks several questions in a row (scenario, device,
    # model) and "press 1" makes every one of them a reading exercise. This is
    # the same shape as the question UI people already know: a highlighted row
    # with a one-line description under it.
    #
    # In: a question, and options as objects with Label, optional Description
    # and Value. -AllowSkip lets Esc/Enter return $null. -KeyReader overrides
    # the key source (tests inject a synthetic sequence). Out: the chosen
    # Value, or $null. Restores cursor visibility even when the caller aborts.
    param(
        [Parameter(Mandatory = $true)][string]$Question,
        [Parameter(Mandatory = $true)][object[]]$Options,
        [switch]$AllowSkip,
        [int]$Default = 0,
        [int]$WindowSize = 8,
        [scriptblock]$KeyReader,
        [scriptblock]$Writer
    )
    if (-not $Options -or $Options.Count -eq 0) { return $null }

    $interactive = if ($KeyReader) { $true } else { Test-InteractiveConsole }
    if (-not $interactive) {
        return Show-NumberedChoice -Question $Question -Options $Options `
                                   -AllowSkip:$AllowSkip -Default $Default
    }

    $emit = if ($Writer) { $Writer } else { { param($line) Write-Host $line } }
    $read = if ($KeyReader) { $KeyReader } else { { [Console]::ReadKey($true) } }
    $plain = [bool]$Writer   # a captured render is compared as text, not colour

    # The cursor starts on the recommendation, so Enter alone is the right
    # answer. Anything offered as a default has to be a configuration we have
    # actually run -- see docs/dev/scenarios.md for which ones those are.
    $selected = [Math]::Max(0, [Math]::Min($Default, $Options.Count - 1))
    $drawn = 0
    try {
        if (-not $Writer) { [Console]::CursorVisible = $false }
        while ($true) {
            if ($drawn -gt 0 -and -not $Writer) {
                # Redraw in place: back up over the previous frame so the list
                # does not scroll away down the terminal on every keypress.
                [Console]::Write("`e[${drawn}A`e[0J")
            }
            $frame = Format-ChoiceFrame -Question $Question -Options $Options `
                                        -Selected $selected -WindowSize $WindowSize `
                                        -Plain:$plain
            foreach ($line in $frame) { & $emit $line }
            $drawn = $frame.Count

            $key = & $read
            switch ($key.Key) {
                "UpArrow"   { $selected = ($selected - 1 + $Options.Count) % $Options.Count }
                "DownArrow" { $selected = ($selected + 1) % $Options.Count }
                "Home"      { $selected = 0 }
                "End"       { $selected = $Options.Count - 1 }
                "Enter"     { return $Options[$selected].Value }
                "Escape"    { if ($AllowSkip) { return $null } }
                default {
                    # Digits still work. Not for muscle memory — for the docs
                    # and issues that say "pick 3", and for anyone on a
                    # terminal whose arrow keys arrive as something else.
                    $ch = $key.KeyChar
                    if ($ch -match "[1-9]") {
                        $n = [int]::Parse($ch)
                        if ($n -le $Options.Count) { return $Options[$n - 1].Value }
                    }
                }
            }
        }
    } finally {
        if (-not $Writer) {
            [Console]::CursorVisible = $true
            Write-Host ""
        }
    }
}

function New-ChoiceOption {
    # Build one option for Show-Choice.
    #
    # Why a helper: the three fields are easy to get wrong as a raw hashtable,
    # and Value defaults to Label so the common case stays one argument.
    #
    # In: a label, an optional description, an optional value. Out: an object
    # with Label, Description and Value.
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [string]$Description = "",
        $Value
    )
    if ($null -eq $Value) { $Value = $Label }
    [PSCustomObject]@{ Label = $Label; Description = $Description; Value = $Value }
}
