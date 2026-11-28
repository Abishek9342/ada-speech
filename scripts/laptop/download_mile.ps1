# Downloads IISc-MILE Tamil (OpenSLR 127) with resume. Only one copy may run (lock file).
$ErrorActionPreference = 'Continue'
$dir  = 'C:\ada-data\raw'
$file = Join-Path $dir 'mile_tamil_asr_corpus.tar.gz'
$log  = Join-Path $dir 'mile_download.log'
$lock = Join-Path $dir 'mile_download.lock'
$url  = 'https://www.openslr.org/resources/127/mile_tamil_asr_corpus.tar.gz'
$expected = 13803410250

function Write-Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 $log }

if (Test-Path $lock) {
    $old = Get-Content $lock
    if (Get-Process -Id $old -ErrorAction SilentlyContinue) { Write-Log "already running as PID $old, exiting"; exit 1 }
}
$PID | Out-File -Encoding ascii $lock
Write-Log "start PID $PID"

for ($i = 1; $i -le 200; $i++) {
    $size = if (Test-Path $file) { (Get-Item $file).Length } else { 0 }
    if ($size -eq $expected) { break }
    if ($size -gt $expected) { Write-Log "ERROR: file larger than expected ($size), stopping"; Remove-Item $lock; exit 2 }
    Write-Log "attempt $i from byte $size"
    curl.exe -sS -L -C - --retry 3 --speed-limit 10000 --speed-time 120 -o $file $url 2>&1 | ForEach-Object { Write-Log "curl: $_" }
    Start-Sleep -Seconds 10
}

$size = (Get-Item $file).Length
Write-Log "final size $size (expected $expected)"
tar -tzf $file > $null 2>&1
if ($LASTEXITCODE -eq 0) { Write-Log "ARCHIVE OK" } else { Write-Log "ARCHIVE BAD (tar exit $LASTEXITCODE)" }
Remove-Item $lock
