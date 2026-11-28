# Replaced 2026-11-28 12:50 by the Scheduled Task 'ADA git push' (C:\ada-data\push_once.ps1).
exit 0
# Owner request 2026-11-28: from 16:00, commit + push every 30 min so the code and logs can be
# reviewed from Claude Code on the phone. Only logs/, evals/ and already-tracked files (never data,
# models or tokens: those are gitignored and never added). Offline pushes fail quietly and the next
# one catches up. Runs until 2026-11-29 12:00. Single instance via a PID file.
$repo = 'C:\Users\ai_en\Projects\ada-speech'
$lock = 'C:\ada-data\night2_state\push_loop.pid'
$plog = 'C:\ada-data\night2_state\push_loop.log'
New-Item -ItemType Directory -Force 'C:\ada-data\night2_state' | Out-Null
if (Test-Path $lock) {
    $old = [int](Get-Content $lock -Raw)
    if (Get-Process -Id $old -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq 'powershell' }) { exit 0 }
}
"$PID" | Out-File -Encoding ascii $lock
$start = [datetime]'2026-11-28T16:00:00'
$until = [datetime]'2026-11-29T12:00:00'
while ((Get-Date) -lt $start) { Start-Sleep -Seconds 60 }
while ((Get-Date) -lt $until) {
    $t = Get-Date -Format 'yyyy-MM-dd HH:mm'
    git -C $repo add -u 2>&1 | Out-Null                       # tracked files (code, configs, logs)
    git -C $repo add logs evals 2>&1 | Out-Null               # new log / eval-list files
    git -C $repo diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git -C $repo commit -q -m "Auto-push $t (Day 2 night progress)`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>&1 | Out-Null
    }
    git -C $repo push -q origin HEAD 2>&1 | Out-Null
    "$t push exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $plog
    Start-Sleep -Seconds 1800
}
Remove-Item $lock -Force -ErrorAction SilentlyContinue
