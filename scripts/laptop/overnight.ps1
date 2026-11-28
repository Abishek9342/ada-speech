# Overnight pipeline 2026-11-27 -> 28 (owner away; standing permissions in logs/DAY1_LOG.md).
# Run 2: CV + Kathbath -> run 3 (+ IISc-MILE) if MILE arrives and audits clean.
# Progress log: logs\overnight.log in the repo (committed + pushed after each run).
$ErrorActionPreference = 'Continue'
$repo = 'C:\Users\ai_en\Projects\ada-speech'
$py   = Join-Path $repo '.venv\Scripts\python.exe'
$log  = Join-Path $repo 'logs\overnight.log'
$runs = Join-Path $repo 'runs'
Set-Location $repo
New-Item -ItemType Directory -Force $runs | Out-Null

# Write-Log must not emit to the pipeline: functions below return exit codes / booleans.
function Write-Log($m) { "$(Get-Date -Format s)  $m" | Out-File -Append -Encoding utf8 $log }

function Invoke-Py([string]$name, [string[]]$pyArgs) {
    # Runs python, keeps full output in runs\<name>.txt, copies the last lines into the overnight log.
    $out = Join-Path $runs "$name.txt"
    & $py -X utf8 @pyArgs *> $out
    $code = $LASTEXITCODE
    Write-Log "[$name] exit $code"
    Get-Content $out | Where-Object { $_.Trim() -and $_ -notmatch '\d+%\||it/s\]|s/it\]|examples/s\]' } |
        Select-Object -Last 25 | ForEach-Object { "    $_" } | Out-File -Append -Encoding utf8 $log
    return $code
}

function Test-AuditClean([string]$name) {
    # Leakage lines must all be 0 (test sentences, test speakers, dev speakers).
    $txt = Get-Content (Join-Path $runs "$name.txt") -Raw
    $bad = [regex]::Matches($txt, '(test_sentences_seen_in_train|test_speakers_seen_in_train|dev_speakers_seen_in_train): (\d+)') |
        Where-Object { [int]$_.Groups[2].Value -gt 0 }
    foreach ($b in $bad) { Write-Log "  LEAK: $($b.Value)" }
    return -not $bad
}

function Get-GpuTemp { [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) }

function Invoke-Training([string]$name, [string]$config) {
    # Starts train.py, watches temperature, pauses/resumes on heat, retries once on a crash.
    # Always --resume: with no checkpoint train.py starts fresh; after an owner fix it continues.
    $crashes = 0; $resume = $true
    while ($true) {
        $a = @('-X', 'utf8', 'scripts/stt/train.py', '--config', $config)
        if ($resume) { $a += '--resume' }
        $stamp = Get-Date -Format 'HHmmss'
        $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo -NoNewWindow -PassThru `
            -RedirectStandardOutput (Join-Path $runs "$name-$stamp.out.txt") -RedirectStandardError (Join-Path $runs "$name-$stamp.err.txt")
        $null = $p.Handle   # cache the handle, or ExitCode can come back empty
        Write-Log "[$name] training started PID $($p.Id) resume=$resume"
        $hot = 0; $paused = $false
        while (-not $p.HasExited) {
            Start-Sleep -Seconds 60
            $t = Get-GpuTemp
            "$(Get-Date -Format s),$name,$t" | Out-File -Append -Encoding utf8 (Join-Path $runs 'gpu_temp_overnight.csv')
            if ($t -ge 87) { $hot++ } else { $hot = 0 }
            if ($hot -ge 3) {
                Write-Log "[$name] GPU $t C for 3+ min: pausing (kill + resume from last checkpoint)"
                taskkill /T /F /PID $p.Id | Out-Null
                $paused = $true
                break
            }
        }
        if ($paused) {
            while ((Get-GpuTemp) -ge 65) { Start-Sleep -Seconds 60 }
            Write-Log "[$name] cooled to $(Get-GpuTemp) C, resuming"
            $resume = $true; continue
        }
        $p.WaitForExit()
        $code = $p.ExitCode
        $err = Join-Path $runs "$name-$stamp.err.txt"
        Get-Content (Join-Path $runs "$name-$stamp.out.txt"), $err -ErrorAction SilentlyContinue |
            Select-String "Baseline WER|WER before|Saved to|eval_wer|Traceback|Error" | Select-Object -Last 15 |
            ForEach-Object { "    $($_.Line)" } | Out-File -Append -Encoding utf8 $log
        if ($code -eq 0) { Write-Log "[$name] training finished OK"; return $true }
        $crashes++
        Write-Log "[$name] training exited with code $code (crash $crashes); last error lines:"
        Get-Content $err -Tail 12 | ForEach-Object { "    $_" } | Out-File -Append -Encoding utf8 $log
        if ($crashes -ge 2) { Write-Log "[$name] failed twice: STOPPING this run (needs owner)"; return $false }
        $resume = $true
        Start-Sleep -Seconds 30
    }
}

function Invoke-Evals([string]$tag, [string[]]$models, [bool]$withMile) {
    $m = $models
    Invoke-Py "$tag-eval-cv"        (@('scripts/stt/evaluate.py', '--data', 'data/stt/cv-ta/test', '--language', 'ta', '--limit', '300', '--models') + $m) | Out-Null
    Invoke-Py "$tag-eval-kb"        (@('scripts/stt/evaluate.py', '--data', 'data/stt/kb-ta/test', '--language', 'ta', '--limit', '300', '--models') + $m) | Out-Null
    Invoke-Py "$tag-eval-kbknown"   (@('scripts/stt/evaluate.py', '--data', 'data/stt/kb-ta/test-known', '--language', 'ta', '--limit', '300', '--models') + $m) | Out-Null
    Invoke-Py "$tag-eval-fleurs"    (@('scripts/stt/evaluate.py', '--data', 'data/stt/fleurs-ta/test', '--language', 'ta', '--limit', '150', '--models') + $m) | Out-Null
    Invoke-Py "$tag-eval-fleurs-nd" (@('scripts/stt/evaluate.py', '--data', 'data/stt/fleurs-ta/test', '--language', 'ta', '--limit', '150', '--exclude-digits', '--models') + $m) | Out-Null
    if ($withMile) {
        Invoke-Py "$tag-eval-mile"  (@('scripts/stt/evaluate.py', '--data', 'data/stt/mile-ta/test', '--language', 'ta', '--limit', '300', '--models') + $m) | Out-Null
    }
}

function Push-Logs([string]$msg) {
    git -C $repo add logs/overnight.log 2>&1 | Out-Null
    git -C $repo commit -q -m "$msg`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>&1 | Out-Null
    git -C $repo push -q origin HEAD 2>&1 | Out-Null
    Write-Log "git push exit $LASTEXITCODE ($msg)"
}

function Wait-Owner([string]$step, [string]$kind = 'fix') {
    # Push logs/NEEDS_OWNER.md, then check GitHub every 5 min. A new commit on main (from the owner
    # or from Claude) = pull and retry the step. logs/OWNER_REPLY.md containing SKIP = skip the step.
    $until = [datetime]'2026-11-28T11:00:00'
    $file = if ($kind -eq 'permission') { 'NEEDS_OWNER.md' } else { 'NEEDS_FIX.md' }
    $who  = if ($kind -eq 'permission') { 'Needs owner decision' } else { 'Needs fix (Claude)' }
    $tail = (Get-Content $log -Tail 40) -join "`n"
    $note = "# ${who}: $step`n`nTime: $(Get-Date -Format s)`n`n" +
        "The overnight script (C:\ada-data\overnight.ps1) is paused at this step.`n`n" +
        "- Fix it and push to main: the script pulls within 5 minutes and retries this step.`n" +
        "- Or push logs/OWNER_REPLY.md containing SKIP to skip this step.`n`n" +
        "Full output of each step: runs\<step>.txt on the laptop.`n`n## Last 40 lines of logs/overnight.log`n`n``````n$tail`n```````n"
    [IO.File]::WriteAllText((Join-Path $repo "logs\$file"), $note, (New-Object Text.UTF8Encoding $false))
    git -C $repo add "logs/$file" logs/overnight.log 2>&1 | Out-Null
    git -C $repo commit -q -m "${who}: $step`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>&1 | Out-Null
    git -C $repo push -q origin HEAD 2>&1 | Out-Null
    Write-Log "${who}: $step (pushed logs/$file, push exit $LASTEXITCODE; checking GitHub every 5 min)"
    $head = (git -C $repo rev-parse HEAD).Trim()
    while ((Get-Date) -lt $until) {
        Start-Sleep -Seconds 300
        git -C $repo fetch -q origin 2>&1 | Out-Null
        $remote = (git -C $repo rev-parse origin/main).Trim()
        if ($remote -eq $head) { continue }
        git -C $repo pull -q --rebase --autostash origin main 2>&1 | Out-Null
        Write-Log "new commit on GitHub ($remote), pulled (exit $LASTEXITCODE)"
        $reply = Join-Path $repo 'logs\OWNER_REPLY.md'
        $skip = (Test-Path $reply) -and ((Get-Content $reply -Raw) -match 'SKIP')
        git -C $repo rm -q --ignore-unmatch "logs/$file" logs/OWNER_REPLY.md 2>&1 | Out-Null
        git -C $repo commit -q -m "Overnight: resume after '$step'`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>&1 | Out-Null
        git -C $repo push -q origin HEAD 2>&1 | Out-Null
        if ($skip) { Write-Log "owner said SKIP: skipping '$step'"; return $false }
        Write-Log "retrying '$step'"
        return $true
    }
    Write-Log "no fix arrived for '$step' by ${until}: giving up on it"
    return $false
}

function Invoke-Gated([string]$step, [scriptblock]$body, [string]$kind = 'fix') {
    # $body must return only a boolean. On $false: wait for a fix on GitHub and retry.
    while ($true) {
        $ok = [bool](& $body | Select-Object -Last 1)
        if ($ok) { return $true }
        if (-not (Wait-Owner $step $kind)) { return $false }
    }
}

# ---------------------------------------------------------------- run 2: CV + Kathbath
Write-Log "=== OVERNIGHT START (PID $PID)"
$excl = @('data/stt/cv-ta/test', 'data/stt/kb-ta/test', 'data/stt/kb-ta/test-known', 'data/stt/fleurs-ta/test')
$py2 = 'scripts/stt/evaluate.py'

$run2ok = Invoke-Gated 'Kathbath audit (leakage must be 0)' {
    $null = Invoke-Py 'kb-audit' @('scripts/stt/audit_data.py', '--data', 'data/stt/kb-ta'); Test-AuditClean 'kb-audit' } 'permission'
if ($run2ok) { $run2ok = Invoke-Gated 'combine CV + Kathbath' {
    if (Test-Path data\stt\combo-ta) { Remove-Item -Recurse -Force data\stt\combo-ta }
    (Invoke-Py 'combo-combine' (@('scripts/stt/combine_data.py', '--sources', 'data/stt/cv-ta', 'data/stt/kb-ta',
        '--out', 'data/stt/combo-ta', '--max-per-speaker', '5000', '--exclude-sentences-of') + $excl)) -eq 0 } }
if ($run2ok) { $null = Invoke-Py 'combo-audit' @('scripts/stt/audit_data.py', '--data', 'data/stt/combo-ta')
    $run2ok = Invoke-Gated 'run 2 smoke test' {
        (Invoke-Py 'combo-smoke' @('scripts/stt/train.py', '--config', 'configs/stt/whisper-base-combo-8gb.yaml', '--smoke')) -eq 0 } }
if ($run2ok) { $run2ok = Invoke-Gated 'run 2 training' { Invoke-Training 'run2' 'configs/stt/whisper-base-combo-8gb.yaml' } }
if ($run2ok) { $run2ok = Invoke-Gated 'run 2 export' {
    (Invoke-Py 'run2-export' @('scripts/stt/export_ct2.py', '--model', 'models/ada-stt-base-combo/final', '--out', 'models/ada-stt-base-combo-ct2')) -eq 0 } }
if ($run2ok) { Invoke-Evals 'run2' @('base', 'models/ada-stt-base-ct2', 'models/ada-stt-base-combo-ct2') $false }
Write-Log "=== RUN 2 DONE (ok=$run2ok)"
Push-Logs "Overnight log: run 2 (CV + Kathbath) results"

# ---------------------------------------------------------------- run 3: + IISc-MILE
$mileLog = 'C:\ada-data\raw\mile_download.log'
$deadline = [datetime]'2026-11-28T03:00:00'   # don't start run 3 after 03:00 (fixed, not relative)
while ((Get-Date) -lt $deadline -and -not ((Get-Content $mileLog -Raw) -match 'ARCHIVE OK|ARCHIVE BAD|ERROR')) { Start-Sleep -Seconds 120 }
$mileState = Get-Content $mileLog -Raw
if (-not $run2ok) {
    Write-Log "run 2 did not finish OK: run 3 skipped"
} elseif ($mileState -notmatch 'ARCHIVE OK') {
    Write-Log "MILE not ready/OK by $(Get-Date -Format T): run 3 skipped (not an error; Day 2). Download log:"
    Get-Content $mileLog -Tail 3 | ForEach-Object { "    $_" } | Out-File -Append -Encoding utf8 $log
} else {
    Write-Log "MILE archive OK"
    if (Test-Path 'C:\ada-data\raw\mile_tamil_asr_corpus.tar.gz.bad') {
        Remove-Item 'C:\ada-data\raw\mile_tamil_asr_corpus.tar.gz.bad' -Force; Write-Log "deleted corrupt mile .bad file"
    }
    $run3ok = Invoke-Gated 'MILE extract' {
        New-Item -ItemType Directory -Force 'C:\ada-data\raw\mile' | Out-Null
        tar -xzf 'C:\ada-data\raw\mile_tamil_asr_corpus.tar.gz' -C 'C:\ada-data\raw\mile' 2>&1 | Select-Object -Last 3 | Out-File -Append -Encoding utf8 $log
        $code = $LASTEXITCODE; Write-Log "MILE extracted (tar exit $code)"; $code -eq 0 }
    if ($run3ok) { $run3ok = Invoke-Gated 'MILE prepare + audit (leakage must be 0)' {
        if (Test-Path data\stt\mile-ta) { Remove-Item -Recurse -Force data\stt\mile-ta }
        $c = Invoke-Py 'mile-prep' @('scripts/stt/prepare_data.py', '--format', 'mile', '--src', 'C:\ada-data\raw\mile',
            '--out', 'data/stt/mile-ta', '--drop-digit-clips', '--dedupe-test-sentences', '--min-seconds', '1')
        $null = Invoke-Py 'mile-audit' @('scripts/stt/audit_data.py', '--data', 'data/stt/mile-ta')
        ($c -eq 0) -and (Test-AuditClean 'mile-audit') } 'permission' }
    if ($run3ok) { $run3ok = Invoke-Gated 'combine CV + Kathbath + MILE' {
        if (Test-Path data\stt\combo3-ta) { Remove-Item -Recurse -Force data\stt\combo3-ta }
        $c = Invoke-Py 'combo3-combine' (@('scripts/stt/combine_data.py', '--sources', 'data/stt/cv-ta', 'data/stt/kb-ta', 'data/stt/mile-ta',
            '--out', 'data/stt/combo3-ta', '--max-per-speaker', '5000', '--exclude-sentences-of') + $excl + @('data/stt/mile-ta/test'))
        $null = Invoke-Py 'combo3-audit' @('scripts/stt/audit_data.py', '--data', 'data/stt/combo3-ta'); $c -eq 0 } }
    if ($run3ok) { $run3ok = Invoke-Gated 'run 3 training' { Invoke-Training 'run3' 'configs/stt/whisper-base-combo3-8gb.yaml' } }
    if ($run3ok) { $run3ok = Invoke-Gated 'run 3 export' {
        (Invoke-Py 'run3-export' @('scripts/stt/export_ct2.py', '--model', 'models/ada-stt-base-combo3/final', '--out', 'models/ada-stt-base-combo3-ct2')) -eq 0 } }
    if ($run3ok) { Invoke-Evals 'run3' @('base', 'models/ada-stt-base-combo-ct2', 'models/ada-stt-base-combo3-ct2') $true }
    Write-Log "=== RUN 3 DONE (ok=$run3ok)"
}

# ---------------------------------------------------------------- cleanup (permitted junk only)
foreach ($d in 'models\ada-stt-base-smoke', 'models\ada-stt-base-combo-smoke', 'models\ada-stt-base-combo3-smoke') {
    if (Test-Path $d) { Remove-Item -Recurse -Force $d; Write-Log "deleted $d" }
}
Get-ChildItem models\ada-stt-base -Directory -Filter 'checkpoint-*' -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName; Write-Log "deleted run-1 $($_.Name)" }
Write-Log "C: free $([math]::Round((Get-PSDrive C).Free/1GB,1)) GB"
Write-Log "=== OVERNIGHT ALL DONE"
Push-Logs "Overnight log: final"

