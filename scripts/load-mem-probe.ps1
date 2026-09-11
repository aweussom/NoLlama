<#
.SYNOPSIS
Load one model under NoLlama and sample memory every few seconds until it is ready, then a while longer.
Why: "where does the memory pressure come from" needs host RAM, commit, the server's own footprint and the
GPU's dedicated vs shared usage on the SAME timeline; Task Manager shows them on four tabs and keeps nothing.
Out: CSV at -Out, and a summary (peak/steady of each column) on stdout. Kills the server at the end.
#>
param(
  [Parameter(Mandatory)] [string] $ModelDir,
  [string] $Repo = "C:\devel\aweussom\python\NoLlama",
  [int] $Port = 8011,
  [int] $CacheGb = 4,
  [string] $Device = "GPU",
  [int] $SettleSeconds = 45,
  [int] $TimeoutMinutes = 20,
  [string] $Out = "$env:TEMP\load-mem-$(Get-Date -Format yyyyMMdd-HHmmss).csv"
)
$ErrorActionPreference = "Continue"
function Sample($proc, $phase) {
  $os = Get-CimInstance Win32_OperatingSystem
  $ded = 0; $sh = 0
  try {
    $c = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -ErrorAction Stop
    foreach ($s in $c.CounterSamples) { if ($s.Path -like '*Dedicated*') { $ded += $s.CookedValue } else { $sh += $s.CookedValue } }
  } catch {}
  $p = if ($proc) { Get-Process -Id $proc.Id -ErrorAction SilentlyContinue } else { $null }
  [pscustomobject]@{
    t = (Get-Date).ToString("HH:mm:ss"); phase = $phase
    free_ram_gb = [math]::Round($os.FreePhysicalMemory/1MB, 2)
    commit_gb = [math]::Round(($os.TotalVirtualMemorySize - $os.FreeVirtualMemory)/1MB, 2)
    proc_ws_gb = if ($p) { [math]::Round($p.WorkingSet64/1GB, 2) } else { 0 }
    proc_private_gb = if ($p) { [math]::Round($p.PrivateMemorySize64/1GB, 2) } else { 0 }
    gpu_dedicated_gb = [math]::Round($ded/1GB, 2); gpu_shared_gb = [math]::Round($sh/1GB, 2)
  }
}
$rows = @()
$rows += Sample $null "baseline"
$log = "$env:TEMP\load-mem-server.log"
$p = Start-Process -FilePath "$Repo\venv\Scripts\python.exe" -ArgumentList @("nollama.py","--port","$Port","--ollama-port","0","--device",$Device,"--model-dir",$ModelDir,"--cache-size-gb","$CacheGb","--idle-timeout","0","--no-prewarm") -WorkingDirectory $Repo -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden
$t0 = Get-Date; $ready = $null; $deadline = $t0.AddMinutes($TimeoutMinutes)
while (-not $ready -and (Get-Date) -lt $deadline -and (Get-Process -Id $p.Id -ErrorAction SilentlyContinue)) {
  $rows += Sample $p "loading"; Start-Sleep 5
  try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.status -eq 'ready') { $ready = Get-Date } } catch {}
}
if ($ready) {
  $settleEnd = (Get-Date).AddSeconds($SettleSeconds)
  while ((Get-Date) -lt $settleEnd) { $rows += Sample $p "ready"; Start-Sleep 5 }
}
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Start-Sleep 8
$rows += Sample $null "after-kill"
$rows | Export-Csv $Out -NoTypeInformation
$size = [math]::Round((Get-ChildItem $ModelDir -Recurse -File | Measure-Object Length -Sum).Sum/1GB, 1)
Write-Output ("model {0} ({1} GB on disk) on {2}; load {3}" -f (Split-Path $ModelDir -Leaf), $size, $Device, $(if ($ready) { "ready after $([int]($ready-$t0).TotalSeconds) s" } else { "NOT ready in $TimeoutMinutes min" }))
Write-Output ("host {0} GB RAM; baseline free {1} GB" -f [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize/1MB,1), $rows[0].free_ram_gb)
foreach ($col in 'free_ram_gb','commit_gb','proc_ws_gb','proc_private_gb','gpu_dedicated_gb','gpu_shared_gb') {
  $load = $rows | Where-Object phase -eq 'loading' | Select-Object -ExpandProperty $col
  $rdy  = $rows | Where-Object phase -eq 'ready'   | Select-Object -ExpandProperty $col
  $ext = if ($col -eq 'free_ram_gb') { ($load | Measure-Object -Minimum).Minimum } else { ($load | Measure-Object -Maximum).Maximum }
  Write-Output ("{0,-18} baseline {1,6}  loading-extreme {2,6}  ready-avg {3,6}  after-kill {4,6}" -f $col, $rows[0].$col, $ext, $(if ($rdy) { [math]::Round(($rdy | Measure-Object -Average).Average,2) } else { 'n/a' }), $rows[-1].$col)
}
Get-Content "$log.err" -ErrorAction SilentlyContinue | Select-Object -Last 3
Write-Output "csv: $Out"
