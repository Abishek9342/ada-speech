# Day 2 plan (2026-11-28)

Input: `ADA_DAY2_AUDIT_AND_RESEARCH.md` (cloud review). Progress goes in `logs/DAY2_LOG.md`.

## Constraints today
- **Internet until about 17:00 to 18:00.** Claude Code needs internet, so after that **nothing can be done
  interactively**. Everything from about 16:30 must run from a script that also works offline
  (`HF_HUB_OFFLINE=1`, every model and package already cached).
- **Owner here until 13:00**, then travelling. Decisions and favours before 13:00.
- **GPU busy with run 3 until about 13:00** (resumed 08:59 from step 4000 after the Windows Update restart),
  then run 3 export and evals on CPU until about 15:30.
- **After 16:00: commit and push every 30 min** (`C:\ada-data\push_loop.ps1`, hidden), so the owner can
  review from the phone. Logs, code and eval lists only; never data, models or tokens.

## Review check (what I verified in the code)
| # | Claim | Verified |
|---|---|---|
| 1 | Kathbath dev = one gender, test = the other (`i % 2` over an F,M,F,M list) | Yes. Fix: pairs, `(i // 2) % 2`, shared with MILE |
| 2 | Audit leak section is empty when there is no `test/` (combined sets) | Yes (`audit.py:133`). The combo3 audit had no leak section |
| 3 | kb-ta prepared before the dedupe fix (01c2fd7) | Yes. Re-prepare |
| 4 | `normalize()` keeps Cf chars (ZWJ/ZWNJ/ZWSP/BOM) | Yes (`text.py` only strips P/S) |
| 6 | Dev set is lopsided | Yes (combo3 dev: 12,165 CV / 2,493 KB / 868 MILE). Note: run 3 *does* have MILE dev |
| 7 | Eval = first N clips | Yes |
| 9 | Checkpoint metric is WER | Yes (`train.py:176`) |
| 11 | GPU idles (CV mp3 decode, workers 0, grad checkpointing on) | Yes in the config |

## Timeline
| Time | GPU | CPU / me | Owner |
|---|---|---|---|
| 09:00 to 13:00 | run 3 training | **Step 1** code fixes + tests (bugs 1, 2, 3, 4, 6, 8, 9, 10). **Step 2** re-prepare into NEW folders `cv2-ta`, `kb2-ta`, `mile2-ta` (run 3 still reads the old ones), CV to 16 kHz FLAC, audit, combine to `combo4-ta`, audit with `--test-dirs`. **Step 3** fixed eval lists in `evals/` + bootstrap CI + synthetic non-speech set | Pause Windows Update, answer the 3 decisions |
| 13:00 to 13:30 | run 3 export, then **run 4 smoke** (workers 4, no grad checkpointing) | | leaves |
| 13:30 | **run 4 starts** (about 5 to 6 h) | run 3 evals (old lists, as planned, for the overnight table) | |
| 15:30 to 16:30 | run 4 | Day 2 log, handoff, launch `night2.ps1` and `push_loop.ps1`, offline test of every step | |
| 16:00 onwards | run 4 | push every 30 min | review on phone |
| about 19:30 | run 4 done, then export | **Step 4** night evals on fixed lists: base, run 1, 2, 3, 4, vasista22 + non-speech hallucination rate | |
| morning | | results table in `logs/overnight-day2.log`, pushed | |

## Run 4 (the first "clean" model)
- Data: re-prepared CV + Kathbath + MILE, text cleanup, dedupe against dev/test/test-known, cap 5,000/speaker,
  exclude every test and test-known set and FLEURS.
- whisper-base, SpecAugment, LR 2.5e-5, **effective batch 64 (16 x 4), 6,000 steps** (about 1.9 epochs, same
  compute as run 3), eval every 500 steps on a per-source dev sample (400 CV + 400 KB + 200 MILE), best by **CER**.
- Speed: CV as 16 kHz FLAC, `dataloader_num_workers: 4`, try `gradient_checkpointing: false` (smoke first;
  fall back if it runs out of memory).
- The night script handles heat pauses, crash retry and resume exactly like `overnight.ps1`, plus a reboot guard:
  a Scheduled Task at logon restarts it with `--resume`.

## Kept for later (not today)
Augmentation (MUSAN, reverb, speed) and empty-target noise clips (run 5), IndicConformer teacher filter,
IndicVoices slice, Vistaar eval, number to words for FLEURS, release kit.

## Owner decisions (2026-11-28, about 09:30)
- Punctuation: **none** (strip all punctuation from training labels).
- Run 4 = **clean data only**; augmentation is run 5.
- **Yes** to a Scheduled Task that resumes the night script at logon (remove it tomorrow).
- Release bar **accepted**: beats stock Whisper small on every fixed list, FLEURS no-digits CER <= 12%, non-speech hallucination rate <= 5%, RTF <= 0.3 on this laptop CPU. Whisper small (CT2 int8) downloaded and checked offline.

