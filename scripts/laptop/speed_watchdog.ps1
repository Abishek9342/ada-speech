# Offline safety net for run 4 (2026-11-28 15:45). The first run 4 slowly spilled GPU memory into shared
# system RAM and crawled to 182 s/step with no error. If training stays slower than 15 s/step for 30 min,
# kill it: night2.ps1 treats that as a crash and resumes from the last checkpoint (fresh GPU memory).
# night2 allows one such retry; a second failure stops run 4 and the evals still run.
# Ends when no run 4 training is running for 20 minutes (finished or stopped).
$runs = 'C:\Users\ai_en\Projects\ada-speech\runs'
$log  = 'C:\Users\ai_en\Projects\ada-speech\logs\night2.log'
$until = [datetime]'2026-11-29T12:00:00'
function Get-Step {
    $f = Get-ChildItem $runs -Filter 'run4-*.err.txt' | Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $f) { return -1 }
    try {
        $fs = [IO.File]::Open($f.FullName, 'Open', 'Read', 'ReadWrite')
        $sr = New-Object IO.StreamReader($fs)
        if ($fs.Length -gt 200000) { $null = $fs.Seek(-200000, 'End') }
        $c = $sr.ReadToEnd(); $fs.Close()
        $m = [regex]::Matches($c, '(\d+)/6000 \[') | Select-Object -Last 1
        if ($m) { return [int]$m.Groups[1].Value } else { return -1 }
    } catch { return -1 }
}
function Get-TrainPid {
    (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train\.py' -and $_.CommandLine -match 'clean4' } |
        Sort-Object CreationDate | Select-Object -First 1).ProcessId
}
$idle = 0
$hist = New-Object System.Collections.ArrayList
while ((Get-Date) -lt $until) {
    Start-Sleep -Seconds 300
    $tp = Get-TrainPid
    if (-not $tp) { $idle++; $hist.Clear(); if ($idle -ge 4) { break } else { continue } }
    $idle = 0
    [void]$hist.Add(@{ t = Get-Date; s = Get-Step; pid = $tp })
    while ($hist.Count -gt 7) { $hist.RemoveAt(0) }          # 30-minute window (7 samples, 5 min apart)
    $a = $hist[0]; $b = $hist[$hist.Count - 1]
    if ($hist.Count -lt 7 -or $a.pid -ne $b.pid -or $a.s -lt 0 -or $b.s -lt 0) { continue }
    $sec = ($b.t - $a.t).TotalSeconds
    $steps = $b.s - $a.s
    if ($steps -le 0 -or ($sec / $steps) -gt 15) {
        # Evals every 500 steps take a few minutes; 30 min with < 120 steps is far beyond that.
        "$(Get-Date -Format s)  [watchdog] run 4 too slow ($steps steps in $([int]($sec/60)) min): killing PID $tp so night2 resumes from the last checkpoint" |
            Out-File -Append -Encoding utf8 $log
        taskkill /T /F /PID $tp | Out-Null
        $hist.Clear()
    }
}
