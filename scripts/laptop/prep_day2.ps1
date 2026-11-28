# Day 2 re-preparation (fixed code): new folders cv2-ta, kb2-ta, mile2-ta, then combo4-ta + audits.
# Old folders stay untouched (run 3 still reads them through combo3-ta).
$repo = 'C:\Users\ai_en\Projects\ada-speech'
$py   = Join-Path $repo '.venv\Scripts\python.exe'
$log  = Join-Path $repo 'logs\day2_prep.log'
$runs = Join-Path $repo 'runs'
Set-Location $repo
$env:HF_HUB_OFFLINE = '1'
function Write-Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 $log }
function Invoke-Py([string]$name, [string[]]$pyArgs) {
    $out = Join-Path $runs "$name.txt"
    & $py -X utf8 @pyArgs *> $out
    $code = $LASTEXITCODE
    Write-Log "[$name] exit $code"
    Get-Content $out | Where-Object { $_.Trim() -and $_ -notmatch '\d+%\||it/s\]|s/it\]|examples/s\]' } |
        Select-Object -Last 40 | ForEach-Object { "    $_" } | Out-File -Append -Encoding utf8 $log
    return $code
}
$common = @('--drop-digit-clips', '--dedupe-test-sentences', '--min-seconds', '1', '--punctuation', 'none')
Write-Log "=== DAY 2 PREP START (PID $PID)"
foreach ($d in 'cv2-ta', 'kb2-ta', 'mile2-ta', 'combo4-ta') { if (Test-Path "data\stt\$d") { Remove-Item -Recurse -Force "data\stt\$d" } }

$null = Invoke-Py 'cv2-prep' (@('scripts/stt/prepare_data.py', '--format', 'commonvoice', '--src', 'C:\ada-data\raw\cv-corpus-27.0-2026-09-11\ta',
    '--out', 'data/stt/cv2-ta', '--flac16k', '--workers', '6') + $common)
$null = Invoke-Py 'cv2-audit' @('scripts/stt/audit_data.py', '--data', 'data/stt/cv2-ta')
$null = Invoke-Py 'kb2-prep' (@('scripts/stt/prepare_data.py', '--format', 'kathbath', '--src', 'C:\ada-data\raw\kathbath\tamil',
    '--out', 'data/stt/kb2-ta', '--holdout-speakers', '16') + $common)
$null = Invoke-Py 'kb2-audit' @('scripts/stt/audit_data.py', '--data', 'data/stt/kb2-ta')
$null = Invoke-Py 'mile2-prep' (@('scripts/stt/prepare_data.py', '--format', 'mile', '--src', 'C:\ada-data\raw\mile',
    '--out', 'data/stt/mile2-ta', '--holdout-speakers', '16') + $common)
$null = Invoke-Py 'mile2-audit' @('scripts/stt/audit_data.py', '--data', 'data/stt/mile2-ta')

$tests = @('data/stt/cv2-ta/test', 'data/stt/kb2-ta/test', 'data/stt/kb2-ta/test-known',
           'data/stt/mile2-ta/test', 'data/stt/mile2-ta/test-known', 'data/stt/fleurs-ta/test')
$null = Invoke-Py 'combo4-combine' (@('scripts/stt/combine_data.py', '--sources', 'data/stt/cv2-ta', 'data/stt/kb2-ta', 'data/stt/mile2-ta',
    '--out', 'data/stt/combo4-ta', '--max-per-speaker', '5000', '--exclude-sentences-of') + $tests)
$null = Invoke-Py 'combo4-audit' (@('scripts/stt/audit_data.py', '--data', 'data/stt/combo4-ta', '--test-dirs') + $tests)
Write-Log "=== DAY 2 PREP DONE"
