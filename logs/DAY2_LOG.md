# Day 2 log (2026-11-28, laptop)

Plan and owner decisions: `logs/DAY2_PLAN.md`. Night progress: `logs/night2.log`.
Results table (filled overnight): `logs/day2_results.md`.

## Overnight (27th to 28th)
- Run 2 (CV + Kathbath, `models/ada-stt-base-combo`) finished on the 27th. Old lists (first 300/150 clips),
  WER / CER / RTF: CV 32.99 / 5.49 / 0.11, Kathbath unseen 33.95 / 6.72 / 0.14, Kathbath test-known
  26.03 / 4.77 / 0.13, FLEURS all 38.33 / 14.44 / 0.36, FLEURS no-digits 33.62 / 11.35 / 0.35. Dev WER 29.15.
- IISc-MILE downloaded and prepared. First audit failed: all 430 MILE test speakers were also in train.
  Fixed (`01c2fd7`): official test becomes `test-known`; 16 train-only speakers held out as unseen dev/test.
- Run 3 (CV + Kathbath + MILE, `combo3-ta`: 202,094 clips, 354 h, 766 speakers) started 01:08.
- **Windows Update restarted the laptop at 02:14 / 02:17** and killed run 3 (the update pause had not been
  saved). Resumed from checkpoint-4000 at 08:59 (`C:\ada-data\run3_resume.ps1`).
  Owner paused updates properly at 11:53 (until 2026-12-05) and turned on auto sign-in after updates.

## Review file (`ADA_DAY2_AUDIT_AND_RESEARCH.md`)
Checked each claim against the code. Confirmed and fixed in `142c717`: bugs 1, 2, 3, 4, 6, 7, 9 (11 noted).
Outdated: "MILE dev missing" (run 3 has 868 MILE dev clips). Not in the review: Claude Code needs internet,
which drops around 17:00, so everything after that is scripted and offline (`HF_HUB_OFFLINE=1`; base, small
and openai/whisper-base verified to load offline).

Owner decisions (09:30): no punctuation in labels; run 4 = clean data only (augmentation = run 5);
yes to a logon Scheduled Task; release bar = beats stock Whisper small on every fixed list, FLEURS
no-digits CER <= 12%, hallucination <= 5% on non-speech, RTF <= 0.3.

## Code changes
- `142c717` Day 2 fixes:
  - Speaker-balanced dev/test split (`dev_test_split`).
  - Kathbath duplicate-name check.
  - Train dedupe against dev, test and test-known.
  - `clean_text` (NFC, invisible format chars removed, NBSP, ஶ்ரீ to ஸ்ரீ).
  - `--punctuation none`, `--flac16k` (CV mp3 to 16 kHz FLAC once).
  - Leakage report for every held-out set, including combined sets via `--test-dirs`.
  - Per-source dev sampling, CER checkpoint selection, run manifest.
  - Evaluation on fixed lists with a bootstrap 95% CI.
  - Non-speech hallucination set.
- `794144c` Day 2 plan. `2b877df` laptop scripts snapshot (`scripts/laptop/`).
- `6a0ca97` Leak gate fixes. Prep attempt 1 (`logs/day2_prep_attempt1.log`) failed the gate on small overlaps:
  - CV official dev/test share 3 sentences.
  - MILE held-out dev/test share 5.
  - 8 combo4 train clips repeat a sentence from another corpus's dev.

  The dedupe now also removes test sentences from dev, and combine_data dedupes train against the combined
  dev. Attempt 2 passed all 4 audits at 11:42.
- Tests: 42 pass, ruff clean.

## Finding: VAD hides hallucination
Run 2 on the 32-clip synthetic non-speech set: 0% hallucination with VAD on, **100% with VAD off** (stock
base 90.6%). The model itself invents text on noise; run 5 needs noise / non-speech training.

## Re-prepared data (attempt 2)
- Each source: `--drop-digit-clips --dedupe-test-sentences --min-seconds 1 --punctuation none`.
- `combo4-ta` (cap 5000 clips per speaker, excluding every test sentence):
  - Train 202,085: CV 37,355, Kathbath 89,555, MILE 75,175.
  - Dev 15,455: CV 12,164, Kathbath 2,422, MILE 869.
- Leak check PASS on cv2, kb2, mile2 and combo4. The only nonzero counts are the expected test-known speakers.

## Run 3 progress (dev WER, combo3 dev incl. MILE; not comparable to run 2's dev)
step 2000 46.04, 4000 40.04, 6000 36.23, 8000 34.33, 10000 33.68, 12000 32.13, 14000 31.36.

## Unattended chain (after the owner left at 12:00)
1. `run3_resume.ps1`: finish run 3, export `models/ada-stt-base-combo3-ct2`, old-list evals.
2. `night2.ps1`:
   1. Fixed eval lists into `evals/`.
   2. Run 4 smoke, then run 4 (`configs/stt/whisper-base-clean4-8gb.yaml`, 6000 steps), export
      `models/ada-stt-base-clean4-ct2`.
   3. All models scored on the same fixed lists plus non-speech (VAD on and off), giving
      `logs/day2_results.jsonl` and `logs/day2_results.md`.
3. `push_loop.ps1`: commit and push logs, evals and tracked files every 30 min from 16:00 to tomorrow 12:00.
4. Scheduled Task "ADA night2 resume" restarts 2 and 3 at logon after a restart. **Remove it after 2026-11-29.**
