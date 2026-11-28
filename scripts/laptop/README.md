# Laptop-only orchestration scripts (copies)

The live copies run from C:\ada-data\ on the owner's laptop (outside the repo, next to the raw data).
These are snapshots so the scripts can be reviewed on GitHub. Machine-specific paths; not part of the package.

- overnight.ps1: Day 1 night (run 2, MILE, run 3)
- run3_resume.ps1: run 3 resumed after the Windows Update restart (02:14, 2026-11-28)
- prep_day2.ps1: Day 2 re-preparation (cv2/kb2/mile2/combo4) with the review fixes
- night2.ps1: Day 2 night (run 4 + fixed-list evals), re-entrant, offline
- push_loop.ps1: commit + push every 30 min from 16:00 (owner request)
- download_mile.ps1: single-instance IISc-MILE downloader
