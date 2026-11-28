# Handoff: ada-speech

## 2026-11-28 (laptop, Day 2)

**What changed**
- Full log in `logs/DAY2_LOG.md`; plan and owner decisions in `logs/DAY2_PLAN.md`.
- Review file fixes (`142c717`): speaker-balanced splits, label cleaning (no punctuation), leakage report on
  every held-out set, CER checkpoint selection, run manifest, fixed eval lists with CI, non-speech set.
- Leak gate fixes (`6a0ca97`): dev deduped against test, combined train deduped against combined dev.
- Re-prepared data `cv2-ta`, `kb2-ta`, `mile2-ta`, `combo4-ta` (202,085 train clips), all audits PASS.

**What was checked**
- 42 tests pass, ruff clean. Offline loading of all eval models verified.
- VAD finding: run 2 hallucinates on 100% of non-speech clips with VAD off (0% with VAD on).

**What is live**
- Nothing published. Running unattended: run 3 finish + export + evals, then `night2.ps1` (run 4 + all
  fixed-list evals), `push_loop.ps1` pushes every 30 min from 16:00 until 2026-11-29 12:00.

- Run 3 done (`models/ada-stt-base-combo3-ct2`): CV CER 5.22, Kathbath unseen 6.16, FLEURS no-digits 10.56.
- Run 4 first attempt spilled VRAM into shared RAM (182 s/step); restarted 15:23 with the safe config
  (grad checkpointing on, workers 0), ~3.1 s/step, ETA about 20:50. `speed_watchdog.ps1` restarts it from
  the last checkpoint if it slows past 15 s/step for 30 min.
- Pushes: Scheduled Task "ADA git push" every 30 min (`C:\ada-data\push_once.ps1`), replaces push_loop.

**Open / next**
- Read `logs/day2_results.md`; check run 4 against the release bar (in `DAY2_PLAN.md`).
- Remove the Scheduled Task "ADA git push" when the owner no longer wants auto-pushes.
- Smoke test should run full-size batches a few minutes and check shared GPU memory (missed the overflow).
- Remove the Scheduled Task "ADA night2 resume" (`Unregister-ScheduledTask -TaskName 'ADA night2 resume'`).
- Run 5: noise / non-speech augmentation (hallucination without VAD).
- Windows Update paused until 2026-12-05; resume it after the runs.

## 2026-11-27 (laptop, Day 1)

**What changed**
- Day 1 of `LAPTOP_DAY1_PLAN.md` done; full log with every command and number in `logs/DAY1_LOG.md`.
- First model `models/ada-stt-base/final` → `models/ada-stt-base-ct2` (int8, 78 MB). Common Voice only.
  CV test CER 27.4 → **8.5**, FLEURS CER 52.5 → **19.3** (no-digits 16.3). vasista22 (244M): CV 5.2,
  FLEURS no-digits 7.7. ADA runs ~9x faster than real time on the laptop CPU.
- Code fixes: audit per-split counts (`0b449ff`), validate on CV dev with 1000-clip cap instead of full test
  (`fcbd7f1`), TensorBoard dir (`7f7c29a`), pyproject SPDX license. Added `CLAUDE.md`, this file.
- Machine: Smart App Control turned OFF; AC sleep/hibernate = Never.

**What was checked**
- 22 tests pass, ruff clean. check_gpu All good (RTX 5050, bf16). FLEURS audit identical to the cloud's.
  CV audit: 0 test sentences / test speakers / dev speakers seen in train.

**What is live**
- Nothing is published. Models are local only (`models/`, gitignored).

**Open / next (Day 2)**
- IISc-MILE re-download running as a standalone process (`C:\ada-data\download_mile.ps1`, log
  `C:\ada-data\raw\mile_download.log`). When it says ARCHIVE OK: delete `mile_tamil_asr_corpus.tar.gz.bad`,
  extract, `prepare_data.py --format mile --drop-digit-clips --dedupe-test-sentences --min-seconds 1`, audit.
  If the laptop was off, just rerun the script: it resumes.
- Kathbath converter (`--format hfparquet`), HF access already granted (Kathbath + IndicVoices).
- Combine CV + MILE + Kathbath, full run, compare with vasista22 and the 74M conformer.
- Decide: cap clips per speaker (one CV speaker = 26% of train); try `dataloader_num_workers > 0`.
