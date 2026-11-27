# Handoff: ada-speech

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
