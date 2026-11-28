# One commit + push of the ada-speech repo. Run every 30 min by the Scheduled Task "ADA git push"
# (owner request 2026-11-28 12:50: push every 30 min from now on, to review from the phone).
# Only already-tracked files plus logs/ and evals/ are added; data, models, audio and tokens are
# gitignored and never added. Offline pushes fail quietly and the next run catches up.
$repo = 'C:\Users\ai_en\Projects\ada-speech'
$plog = 'C:\ada-data\night2_state\push_loop.log'
New-Item -ItemType Directory -Force 'C:\ada-data\night2_state' | Out-Null
$t = Get-Date -Format 'yyyy-MM-dd HH:mm'
if (Test-Path (Join-Path $repo '.git\index.lock')) { "$t skipped (git busy)" | Out-File -Append -Encoding utf8 $plog; exit 0 }
git -C $repo add -u 2>&1 | Out-Null
git -C $repo add logs evals 2>&1 | Out-Null
git -C $repo diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git -C $repo commit -q -m "Auto-push $t (training progress)`n`nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>&1 | Out-Null
}
git -C $repo push -q origin HEAD 2>&1 | Out-Null
"$t push exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $plog
