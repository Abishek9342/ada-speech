# Day 2 night pipeline (2026-11-28 -> 29). Runs OFFLINE after about 17:00 (no Claude, no internet).
# Re-entrant: a logon Scheduled Task ("ADA night2 resume") restarts it after a Windows restart; finished
# steps are skipped via marker files in C:\ada-data\night2_state, training always resumes from its checkpoint.
# Steps: wait for prep + run 3 GPU -> eval lists -> run 4 smoke -> run 4 train -> export -> fixed-list evals.
$ErrorActionPreference = 'Continue'
$repo  = 'C:\Users\ai_en\Projects\ada-speech'
$py    = Join-Path $repo '.venv\Scripts\python.exe'
$log   = Join-Path $repo 'logs\night2.log'
$runs  = Join-Path $repo 'runs'
$state = 'C:\ada-data\night2_state'
$lock  = Join-Path $state 'night2.pid'
Set-Location $repo
New-Item -ItemType Directory -Force $runs, $state | Out-Null
$env:HF_HUB_OFFLINE = '1'          # internet goes away this evening; everything is cached
$env:TRANSFORMERS_OFFLINE = '1'

function Write-Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 $log }

# single instance (the logon task and a manual start must not both run)
if (Test-Path $lock) {
    $old = [int](Get-Content $lock -Raw)
    if (Get-Process -Id $old -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq 'powershell' }) { exit 0 }
}
"$PID" | Out-File -Encoding ascii $lock

function Test-Done([string]$step) { Test-Path (Join-Path $state "$step.done") }
function Set-Done([string]$step) { New-Item -ItemType File -Force (Join-Path $state "$step.done") | Out-Null }

function Invoke-Py([string]$name, [string[]]$pyArgs) {
    $out = Join-Path $runs "$name.txt"
    & $py -X utf8 @pyArgs *> $out
    $code = $LASTEXITCODE
    Write-Log "[$name] exit $code"
    Get-Content $out | Where-Object { $_.Trim() -and $_ -notmatch '\d+%\||it/s\]|s/it\]|examples/s\]' } |
        Select-Object -Last 25 | ForEach-Object { "    $_" } | Out-File -Append -Encoding utf8 $log
    return $code
}

function Get-GpuTemp { [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) }

function Test-TrainingRunning {
    [bool](Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'train\.py' })
}

function Invoke-Training([string]$name, [string]$config) {
    # Same as overnight.ps1: heat pause (>=87 C for 3 min -> kill, cool below 65 C, resume), one crash retry.
    $crashes = 0
    while ($true) {
        $a = @('-X', 'utf8', 'scripts/stt/train.py', '--config', $config, '--resume')
        $stamp = Get-Date -Format 'HHmmss'
        $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo -NoNewWindow -PassThru `
            -RedirectStandardOutput (Join-Path $runs "$name-$stamp.out.txt") -RedirectStandardError (Join-Path $runs "$name-$stamp.err.txt")
        $null = $p.Handle
        Write-Log "[$name] training started PID $($p.Id) ($config)"
        $hot = 0; $paused = $false
        while (-not $p.HasExited) {
            Start-Sleep -Seconds 60
            $t = Get-GpuTemp
            "$(Get-Date -Format s),$name,$t" | Out-File -Append -Encoding utf8 (Join-Path $runs 'gpu_temp_night2.csv')
            if ($t -ge 87) { $hot++ } else { $hot = 0 }
            if ($hot -ge 3) {
                Write-Log "[$name] GPU $t C for 3+ min: pausing (kill + resume from last checkpoint)"
                taskkill /T /F /PID $p.Id | Out-Null
                $paused = $true; break
            }
        }
        if ($paused) {
            while ((Get-GpuTemp) -ge 65) { Start-Sleep -Seconds 60 }
            Write-Log "[$name] cooled to $(Get-GpuTemp) C, resuming"; continue
        }
        $p.WaitForExit()
        $code = $p.ExitCode
        $err = Join-Path $runs "$name-$stamp.err.txt"
        Get-Content (Join-Path $runs "$name-$stamp.out.txt"), $err -ErrorAction SilentlyContinue |
            Select-String "Baseline WER|WER before|per-source|Saved to|eval_cer|Traceback|Error" | Select-Object -Last 20 |
            ForEach-Object { "    $($_.Line)" } | Out-File -Append -Encoding utf8 $log
        if ($code -eq 0) { Write-Log "[$name] training finished OK"; return $true }
        $crashes++
        Write-Log "[$name] training exited with code $code (crash $crashes); last error lines:"
        Get-Content $err -Tail 12 | ForEach-Object { "    $_" } | Out-File -Append -Encoding utf8 $log
        if ($crashes -ge 2) { Write-Log "[$name] failed twice: STOPPING run 4"; return $false }
        Start-Sleep -Seconds 30
    }
}

Write-Log "=== NIGHT2 START (PID $PID)"

# ---------------------------------------------------------------- 1. wait for data prep and a free GPU
$prepLog = Join-Path $repo 'logs\day2_prep.log'
while (-not ((Test-Path $prepLog) -and ((Get-Content $prepLog -Raw) -match 'DAY 2 PREP DONE'))) {
    # After a Windows restart the prep script is gone: start it again (it rebuilds its folders from scratch).
    $prepRunning = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'prep_day2\.ps1' }
    if (-not $prepRunning) {
        Write-Log "data prep not running and not done: restarting prep_day2.ps1"
        Start-Process powershell -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'C:\ada-data\prep_day2.ps1' -WindowStyle Hidden | Out-Null
    }
    Start-Sleep -Seconds 120
}
Write-Log "data prep done"
$leakOk = $true
foreach ($a in 'cv2-audit', 'kb2-audit', 'mile2-audit', 'combo4-audit') {
    $txt = Get-Content (Join-Path $runs "$a.txt") -Raw -ErrorAction SilentlyContinue
    $line = ([regex]::Match("$txt", 'LEAK CHECK: [^\r\n]*')).Value
    Write-Log "  ${a}: $line"
    if ($line -notmatch 'PASS') { $leakOk = $false }
}
while (Test-TrainingRunning) { Start-Sleep -Seconds 120 }   # run 3 (or another run) still on the GPU
Write-Log "GPU free (no train.py running)"

# ---------------------------------------------------------------- 2. fixed evaluation lists (committed)
if (-not (Test-Done 'evallists')) {
    $c = Invoke-Py 'eval-lists' @('scripts/stt/make_eval_lists.py',
        'cv_test_500=data/stt/cv2-ta/test:500',
        'kb_unseen_test_500=data/stt/kb2-ta/test:500',
        'kb_known_300=data/stt/kb2-ta/test-known:300',
        'mile_unseen_test_500=data/stt/mile2-ta/test:500',
        'mile_known_300=data/stt/mile2-ta/test-known:300',
        'fleurs_all_300=data/stt/fleurs-ta/test:300',
        'fleurs_nodigits_all=data/stt/fleurs-ta/test:all:nodigits')
    if ($c -eq 0) { Set-Done 'evallists' }
}

# ---------------------------------------------------------------- 3. run 4 (only on leak-free data)
$cfg = 'configs/stt/whisper-base-clean4-8gb.yaml'
$run4ok = $false
if (-not $leakOk) {
    Write-Log "LEAK CHECK failed on the re-prepared data: run 4 NOT trained (evals of runs 1-3 still run)"
} elseif (Test-Done 'run4-train') {
    $run4ok = $true
} else {
    if (-not (Test-Done 'run4-smoke')) {
        $c = Invoke-Py 'run4-smoke' @('scripts/stt/train.py', '--config', $cfg, '--smoke')
        if ($c -ne 0) {
            Write-Log "run 4 smoke failed with workers 4 / no grad checkpointing: falling back to the safe settings"
            $safe = 'configs/stt/whisper-base-clean4-safe-8gb.yaml'
            (Get-Content $cfg) -replace '^gradient_checkpointing: false.*', 'gradient_checkpointing: true' `
                               -replace '^dataloader_num_workers: 4', 'dataloader_num_workers: 0' |
                Out-File -Encoding utf8 $safe
            $cfg = $safe
            "$safe" | Out-File -Encoding ascii (Join-Path $state 'run4-config.txt')
            $c = Invoke-Py 'run4-smoke-safe' @('scripts/stt/train.py', '--config', $cfg, '--smoke')
        }
        if ($c -eq 0) { Set-Done 'run4-smoke' } else { Write-Log "run 4 smoke failed twice: run 4 NOT trained" }
    }
    if (Test-Path (Join-Path $state 'run4-config.txt')) { $cfg = (Get-Content (Join-Path $state 'run4-config.txt') -Raw).Trim() }
    if (Test-Done 'run4-smoke') {
        if (Test-Path 'models\ada-stt-base-clean4-smoke') { Remove-Item -Recurse -Force 'models\ada-stt-base-clean4-smoke' }
        $run4ok = Invoke-Training 'run4' $cfg
        if ($run4ok) { Set-Done 'run4-train' }
    }
}
if ($run4ok -and -not (Test-Done 'run4-export')) {
    $c = Invoke-Py 'run4-export' @('scripts/stt/export_ct2.py', '--model', 'models/ada-stt-base-clean4/final', '--out', 'models/ada-stt-base-clean4-ct2')
    if ($c -eq 0) { Set-Done 'run4-export' }
}

# ---------------------------------------------------------------- 4. fixed-list evals (CPU, after training, so RTF is clean)
# Wait for run 3's own evals (run3_resume.ps1) to finish first: two CPU evals at once distort RTF.
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'evaluate\.py' -and $_.CommandLine -notmatch 'evals[\\/]' }) { Start-Sleep -Seconds 120 }
$results = Join-Path $repo 'logs\day2_results.jsonl'
$models = @()
foreach ($m in 'models/ada-stt-base-clean4-ct2', 'models/ada-stt-base-combo3-ct2', 'models/ada-stt-base-combo-ct2', 'base',
               'small', 'models/ada-stt-base-ct2', 'models/vasista22-ct2') {
    if ($m -in 'base', 'small' -or (Test-Path $m)) { $models += $m }
}
$sets = 'cv_test_500', 'kb_unseen_test_500', 'kb_known_300', 'mile_unseen_test_500', 'mile_known_300', 'fleurs_all_300', 'fleurs_nodigits_all'
foreach ($m in $models) {
    $tag = ($m -replace '^models/', '') -replace '[^\w\-]', '_'
    foreach ($s in $sets) {
        $step = "eval-$tag-$s"
        if (Test-Done $step) { continue }
        $c = Invoke-Py $step @('scripts/stt/evaluate.py', '--list', "evals/$s.txt", '--language', 'ta', '--models', $m, '--json-out', $results)
        if ($c -eq 0) { Set-Done $step }
    }
    foreach ($vad in 'vad', 'novad') {
        $step = "eval-$tag-nonspeech-$vad"
        if (Test-Done $step) { continue }
        $a = @('scripts/stt/evaluate.py', '--list', 'evals/nonspeech.txt', '--language', 'ta', '--models', $m, '--json-out', $results)
        if ($vad -eq 'novad') { $a += '--no-vad' }
        $c = Invoke-Py $step $a
        if ($c -eq 0) { Set-Done $step }
    }
    $null = Invoke-Py 'results-table' @('scripts/stt/results_table.py', '--in', $results, '--out', 'logs/day2_results.md')
    Write-Log "=== evals done for $m"
}
Write-Log "C: free $([math]::Round((Get-PSDrive C).Free/1GB,1)) GB"
Write-Log "=== NIGHT2 ALL DONE"
Remove-Item $lock -Force -ErrorAction SilentlyContinue
