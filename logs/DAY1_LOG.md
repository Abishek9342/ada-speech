# Day 1 log: ADA-STT Tamil

## Summary for the owner

**Day 1 is done except IISc-MILE** (download restarted after corruption; prep/audit moves to Day 2).
The first ADA-STT model exists and clearly beats stock Whisper base, at the same size and speed.

### Results (laptop CPU, int8, beam 5; test clips never seen in training)
| test set | model | size | WER % | CER % | RTF |
|---|---|---|---|---|---|
| Common Voice 27 (300) | stock Whisper base | 74M | 83.67 | 27.38 | 0.13 |
| Common Voice 27 (300) | **ADA-STT base v0.1-dev** | 74M / 78 MB | **44.58** | **8.48** | **0.11** |
| Common Voice 27 (300) | vasista22 whisper-tamil-small | 244M / 241 MB | 30.27 | 5.21 | 0.33 |
| FLEURS, all (150) | stock Whisper base | 74M | 96.33 | 52.53 | 0.39 |
| FLEURS, all (150) | **ADA-STT base** | 74M | **55.90** | **19.31** | 0.35 |
| FLEURS, no digits (150) | stock Whisper base | 74M | 95.64 | 51.79 | 0.40 |
| FLEURS, no digits (150) | **ADA-STT base** | 74M | **52.28** | **16.33** | 0.36 |
| FLEURS, no digits (150) | vasista22 | 244M | 22.91 | 7.67 | 0.92 |

- ADA runs ~9x faster than real time on this CPU and is 78 MB (target 40–80 MB). ✓
- In-domain (CV) CER 27.4 → 8.5. Out-of-domain (FLEURS) CER 52.5 → 19.3 / 16.3.
- vasista22 is still well ahead (and 3x bigger, 2.5–3x slower; likely trained on CV too). The gap is mostly
  data: today's training set is 84 h from **only 22 speakers**. Dev WER was still improving at step 5000.

### What worked
- Setup on Python 3.13 + PyTorch 2.11 cu128; `check_gpu.py` All good (bf16 OK). 22 tests, ruff clean.
- FLEURS prep/audit matches the cloud exactly. Common Voice 27.0 prepared + audited: no leakage.
- GPU smoke test and the full 5000-step run (2.1 h, 7 GB VRAM, max ~79 °C).

### Fixed today (all committed)
- `0b449ff` audit printed all flag counts under `[test]` (969 of 1,051 were train).
- `fcbd7f1` training picked checkpoints on the full **test** set (slow and optimistic) → now CV **dev**,
  1000-clip sample.
- `7f7c29a` TensorBoard logs went to `./runs` (transformers 5 removed `logging_dir`).
- pyproject license format (setuptools deprecation, would break builds Feb 2027).

### Problems hit
- Smart App Control blocked PyTorch → owner turned it off.
- Laptop slept 26 min mid-run → AC sleep/hibernate set to Never.
- IISc-MILE first download corrupted (two resumers after a Claude session restart) → restarted 12:44 as a
  single locked process; ~0.6–0.9 MB/s, ETA this evening.

### Decisions needed (none blocking)
- Day 2: cap clips per speaker? (one CV speaker = 26% of train.)
- Optional speed-up for training: `dataloader_num_workers > 0` (GPU idles while the CPU decodes mp3).

## Machine
- GPU / driver / CUDA (nvidia-smi): NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB, driver 610.74, CUDA 13.3 (OK, >= 12.8)
- CPU / cores / threads: 13th Gen Intel Core i7-13645HX, 14 cores (6P + 8E), 20 threads
- RAM: 15.7 GB
- Free disk (data drive): C: 240 GB free (only drive). `<DATA>` = `C:\ada-data`
- Versions: Python 3.13.13 / PyTorch 2.11.0+cu128 (CUDA 12.8 build) / transformers 5.24.1 / datasets 3.6.0 /
  faster-whisper 1.2.1 / av 18.1.1 / ctranslate2 4.9.1

## Step 1: setup
- [2026-11-27 09:20] Repo cloned to `C:\Users\ai_en\Projects\ada-speech`.
- [09:20] `nvidia-smi` → RTX 5050 listed, CUDA 13.3. Driver OK, no update needed.
- [09:20] `py -0` → only Python 3.13 installed (no 3.12). Decision: use 3.13 (repo needs >=3.10, all deps
  publish cp313 wheels) instead of blocking on a 3.12 install. If a package misbehaves, fall back to 3.12.
- [09:21] venv + `torch torchaudio` (cu128) + `pip install -e ".[train,export,dev]"` started.
- [09:23] Install OK. Python 3.13.13, torch 2.11.0+cu128, torchaudio 2.11.0+cu128, transformers 5.24.1,
  datasets 3.6.0, faster-whisper 1.2.1, av 18.1.1 (< 19 OK), ctranslate2 4.9.1.
- [09:24] `ruff check .` → All checks passed. `python -m pytest` → 20 passed.
  (`pytest.exe` launcher itself is blocked, see below; `python -m pytest` works.)
- check_gpu.py output: **BLOCKED.** `OSError: [WinError 4551] An Application Control policy has blocked this
  file ... torch\lib\shm.dll`. Cause: **Windows Smart App Control is ON** (registry
  `CI\Policy\VerifiedAndReputablePolicyState = 1`); CodeIntegrity log events 3033/3077 show unsigned files
  in `.venv` rejected. PyTorch cannot be imported at all, so no GPU step can run until this is resolved.
  Waiting on owner decision (turn SAC off vs WSL2).
- [09:27] **Owner decided: turn Smart App Control off** (owner does it in Windows Security). Note: SAC can't be
  re-enabled without a Windows reset; Defender antivirus stays on.
- [~10:00] Owner turned SAC off (registry state now 0). `check_gpu.py`:
  ```
  Python 3.13.13 | PyTorch 2.11.0+cu128 | Built for CUDA 12.8
  GPU: RTX 5050 Laptop (compute 12.0, 8.0 GB VRAM) | bf16 support: True | Matmul test: OK
  All good: this machine can train ADA.
  ```

## Step 2: downloads
- IISc-MILE: started 2026-11-27 ~09:21 into `C:\ada-data\raw` (curl resume loop, background).
  Finished 12:41 but **archive BAD**: local 14,118,831,637 bytes vs server Content-Length 13,803,410,250
  (+315 MB). Cause (most likely): after the Claude session restart at ~11:19, the orphaned old curl loop and
  my replacement loop both resumed into the same file. Bad file kept as `mile_tamil_asr_corpus.tar.gz.bad`
  (delete once the new one checks OK).
  Mirrors tested 12:43: www.openslr.org 0.61 MB/s (best), trmal.net 0.50, magicdatatech 0.38, us. (cert
  error), elda (timeout). Re-download from scratch 12:44 via standalone `C:\ada-data\download_mile.ps1`
  (lock file so only one copy runs, exact-size check every attempt, tar check at the end, log in
  `C:\ada-data\raw\mile_download.log`). ETA 3–6 h, so MILE prep/audit likely moves to Day 2.
- FLEURS test: `C:\ada-data\raw\fleurs\data\ta_in` (591 tsv rows, 591 wavs). Downloaded via
  `python -c "huggingface_hub.snapshot_download(...)"` because the `hf.exe` launcher was blocked by SAC.
- Common Voice: 27.0 (cv-corpus-27.0-2026-09-11-ta), owner downloaded from Mozilla Data Collective
  (8.58 GB tar.gz in Downloads, finished 09:57). Extracted in 2.2 min to
  `C:\ada-data\raw\cv-corpus-27.0-2026-09-11\ta`. Rows: train 46,529 / dev 12,168 / test 12,241 /
  validated 136,732 / other 105,237 / invalidated 5,749; 247,718 files in clips\.
  (Only train + test are used by the converter; dev, and the extra validated-but-not-in-train clips, are a
  possible later addition.)
- HF login: **yes** (owner ran `hf auth login` themselves, read token `ada-laptop`, user `abishek-aiml`).
  Kathbath + IndicVoices terms accepted: **yes**, verified by listing both gated repos via the API.

## Step 3: audits
### FLEURS
`prepare_data.py --format fleurs` → 591 test examples in `data/stt/fleurs-ta`.
```
test   clips=591  hours=2.13  unknown_duration=0  median_s=11.7  speakers=0  top_speaker_share=None
[test] 591 clips
      127  contains digits
       51  contains Latin letters
        9  longer than Ns
        3  speaking rate N chars/s vs median N
        2  contains characters from another script
Wrote 192 flags to data\stt\fleurs-ta\audit_flags.csv
```
**Matches the cloud exactly** (591 / 2.13 h / 127 digits / 51 Latin / 9 > 30 s). Test-only set, so no filters
applied. (Cosmetic: the audit prints "Ns"/"N" placeholders instead of the thresholds.)
### Common Voice
Pass 1: `prepare_data.py --format commonvoice --drop-digit-clips` (8.3 min) → train 46,529 / test 12,241.
`--drop-digit-clips` removed **0** clips: CV Tamil train has no digits (confirms "numbers as words").
```
train  clips=46529  hours=84.11  unknown_duration=0  median_s=6.42  speakers=22  top_speaker_share=0.261
test   clips=12241  hours=19.4   unknown_duration=0  median_s=5.3   speakers=596 top_speaker_share=0.021
test_sentences_seen_in_train: 10
test_speakers_seen_in_train: 0
```
**Bug found:** the audit printed all flag counts under `[test]`, but 969 of the 1,051 flags were in **train**
(the counts were pooled across splits). Fixed in `scripts/stt/audit_data.py` (per-split counts), test added
`tests/test_audit_script.py`, commit `0b449ff`.
Train flags (correct split): 884 Latin letters, 36 mostly non-Tamil letters, 24 speaking-rate outliers,
20 repeated transcripts, 5 very short audio.

Actions per plan:
- `test_sentences_seen_in_train` = 10 → re-run with `--dedupe-test-sentences`.
- 5 very short clips → re-run with `--min-seconds 1`.
- `test_speakers_seen_in_train` = 0 ✓.
- `top_speaker_share` = 0.261 (one speaker = 26% of train; only 22 speakers in train, top 5 = 12,131 /
  6,434 / 5,594 / 4,195 / 3,570 clips). Logged; per plan, capping clips per speaker is a later decision.
  Test has 596 speakers, so the test score does measure new voices.
- Latin letters (884 train clips, 1.9%): they are Tamil sentences with English proper names kept in Latin
  script. **Kept for v0.1** (real code-mixing, small share); **owner confirmed: keep them.** 10 examples:
  - இராணுவத் தளபதியாக இருந்த அவர், அயர்லாந்தின் Norman படையெடுப்பை எதிர்த்தார்.
  - அவரது இசை பாணி கேரேஜ் ராக் முதல் R மற்றும் B வரை இருக்கும்.
  - இந்தியாவின் கர்நாடக மாநிலத்தில் உள்ள Tumkur City மற்றும் Bangalore நகரங்களில் அவரது பேரக் குழந்தைகள் வசிக்கின்றனர்.
  - Harding-ல் எட்டு மணிக்கு எனக்கு ஒரு ரெஸ்டாரெண்ட் பதிவு செய்ய வேண்டும்.
  - Madeline Bassett உன்னை ஆக்கிரமித்திடுவார்.
  - ஜூன் மாதத்தில் அவள் "First" என்று பெயரிடப்பட்ட பாப் பாடலை வெளியிட்டாள்.
  - சமீபத்தில் சித்திரக் கதைப் புத்தகங்களில் Karagiozis கதாபாத்திரத்தின் மறு அவதாரங்களை கொண்டு வர பல முயற்சிகள் நடந்தன.
  - Bulletin Board Systems கிட்டத்தட்ட ஒவ்வொரு பொழுதுபோக்கிற்கும் ஆர்வத்திற்கும் காணப்படுகிறது.
  - ஜெர்மன் உற்பத்தியாளர் trautoniks, தனித்து வடிவமைக்கப்பட்ட ட்ரூடோனியங்களை விற்கின்றது.
  - சமூக வரம்பில் Grandview வில் பள்ளிகள் இல்லை.

The 36 "mostly non-Tamil" clips are the same kind (Tamil sentences built around long English names, e.g.
"School of Communication and Academy of Visual Arts-க்காக ..."). Kept, same decision as Latin letters.

**Second issue found before training (commit `fcbd7f1`):** `train.py` validated on the *full test set*
(12,241 clips) every 500 steps and picked the best checkpoint by test WER. That is ~2 h of extra generation
per run and makes the final test score slightly optimistic. Fixed: the commonvoice converter now also writes
`dev/` (CV's own dev.tsv, speaker-disjoint), `train.py` validates on dev when present, on a fixed random
sample of `eval_max_samples: 1000`; test is only used for the final numbers. Audit now covers dev and
reports `dev_speakers_seen_in_train`. 22 tests pass, ruff clean.

Pass 2 (with dev): `--drop-digit-clips --dedupe-test-sentences --min-seconds 1` (13.5 min).
Filters removed 15 of 46,529 train clips (10 test-sentence duplicates + 5 under 1 s).
```
train  clips=46514  hours=84.1   median_s=6.42  speakers=22   top_speaker_share=0.261
dev    clips=12168  hours=19.38  median_s=5.45  speakers=33   top_speaker_share=0.053
test   clips=12241  hours=19.4   median_s=5.3   speakers=596  top_speaker_share=0.021
test_sentences_seen_in_train: 0
test_speakers_seen_in_train: 0
dev_speakers_seen_in_train: 0
[train] 960 flags: 879 Latin, 36 mostly non-Tamil, 20 repeated transcript, 20 speaking rate, 5 other script
[dev]    25 flags: 22 Latin, 2 speaking rate, 1 other script
[test]   82 flags: 53 Latin, 26 speaking rate, 2 mostly non-Tamil, 1 other script
```
**Common Voice ready for training.** No leakage. Remaining flags are the kept Latin/English-name clips and a
few dozen speaking-rate outliers (0.04% of train, left in).
### IISc-MILE
- Decisions (incl. number convention): **Owner decided 2026-11-27: numbers as Tamil words.** Train sets
  prepared with `--drop-digit-clips`; FLEURS reported twice (all clips + `--exclude-digits`).

## Step 4: baseline (laptop CPU)
CPU: i7-13645HX (14C/20T), int8, beam 5, plugged in. 10:33–11:03.

| data | model | clips | WER % | CER % | RTF |
|---|---|---|---|---|---|
| CV 27.0 test | stock Whisper base | 300 | 83.09 | 26.55 | 0.13 |
| CV 27.0 test | stock Whisper small | 300 | 61.73 | 15.42 | 0.41 |
| FLEURS test | stock Whisper base | 150 | 96.33 | 52.53 | 0.39 |

- Laptop CPU is much faster than the cloud CPU: base on FLEURS RTF 0.39 vs 1.04 (cloud). On CV (shorter
  clips) base runs at 0.13 RTF, ~8x faster than real time.
- FLEURS base WER 96.3 vs cloud 102.2: the cloud baseline used the old timestamp setting (duplicated
  transcripts); this run has `without_timestamps=True`. CER close (52.5 vs 54.6).
- Predictions saved to `runs/preds-cv`. vasista22 (optional) not run yet; will run after training if time allows.

## Step 5: training
- Owner approved (~10:20): start the real run (5b) right after the smoke test passes, then export + compare.
- Config: `configs/stt/whisper-base-8gb.yaml` (batch 16 x 1, bf16, grad checkpointing, lr 2.5e-5,
  warmup 500, 5000 steps, eval every 500 on 1000 fixed dev clips).
- Smoke test (11:04, 1.3 min): **PASS on GPU (CUDA + bf16).** `train=64 dev=16`; WER before 76.15 → after
  67.89 (CER 26.55 → 27.84 on 16 clips, noisy) in 20 steps; 1.04 steps/s, 16.6 samples/s; saved
  `models/ada-stt-base-smoke/final`. No OOM, no bf16 errors. Harmless warnings: `gradient_checkpointing`
  deprecated (transformers v6 → `activation_checkpointing`), generation_config + suppress_tokens deprecation,
  "attention mask is not set" (Whisper inputs are fixed 30 s log-mel, so no effect).
- Real run started ~11:07 (`runs/train_base.log`, GPU stats every 60 s in `runs/gpu_monitor.csv`).
  `train=46514 dev=1000`. Baseline (stock base, HF generate) on dev sample: **WER 90.83**.
  11:12 check: GPU 95% util, 7.0 / 8.0 GB, 77 °C.
  **Step 500 eval (dev, 1000 clips): WER 59.01, CER 15.00** (from 90.83 baseline); eval takes 115 s.
  11:19: the Claude Code session restarted and marked its background shells stopped, but the training
  process (PID 35092) and curl kept running; log and TensorBoard events still updating. GPU util swings
  0–99% (CPU-side mp3 decoding with `dataloader_num_workers: 0`), 69–77 °C. Watchdog restarted; temps
  now logged to `runs/gpu_temp.csv` (the nvidia-smi `-f` logger wrote nothing).
  Dev eval so far (WER / CER): 500: 59.01/15.00 · 1000: 52.92/12.85 · 1500: 51.34/12.28 · 2000: 48.07/11.20 ·
  2500: 46.95/11.18 · 3000: 46.22/11.02.
  **Laptop went to sleep** (System log Kernel-Power 42 → resume): 12:08–12:35 (26 min) plus short ones at
  10:47, 11:07, 11:42. Training paused and resumed fine (step 3025 → 3050 took 1,611 s). Fixed with
  `powercfg /change standby-timeout-ac 0` and `hibernate-timeout-ac 0` (sleep/hibernate when plugged in:
  never). Speed when awake: ~1.1 steps/s; each 1000-clip eval ≈ 2 min.
- TensorBoard: transformers 5 wrote events to `./runs/<date>_<host>` instead of `models/ada-stt-base`, so
  `--logdir models/ada-stt-base` showed nothing. This run: use `--logdir runs`. Fixed for future runs:
  transformers 5 removed `logging_dir` from TrainingArguments (checked the signature before committing), so
  `train.py` now sets env `TENSORBOARD_LOGGING_DIR=<output_dir>/runs` (setdefault). Not yet run-tested:
  verify on the next smoke run once the GPU is free. **Verified 13:59** (smoke run wrote events to
  `models/ada-stt-base-smoke/runs`), commit `7f7c29a`.
- Bugs found and fixes (commit hashes): see Step 3 (`0b449ff` audit per-split, `fcbd7f1` dev validation +
  eval cap); TensorBoard log dir via `TENSORBOARD_LOGGING_DIR` (committed at end of day).
- Real run: **done 13:14, no errors.** 5000 steps, 1.72 epochs, train_runtime 7,491 s (2.1 h; ~26 min of it
  the laptop asleep), 0.667 steps/s average (~1.1 when awake), train_loss 0.162. GPU ~7.0 / 8.0 GB,
  65–79 °C (never near 87). Dev (1000 clips) WER / CER per eval:

  | step | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | WER | 90.83 | 59.01 | 52.92 | 51.34 | 48.07 | 46.95 | 46.22 | 44.04 | 44.13 | 43.25 | **42.93** |
  | CER | 38.95 | 15.00 | 12.85 | 12.28 | 11.20 | 11.18 | 11.02 | 10.95 | 10.25 | 10.05 | **10.16** |

  Best = step 5000 (still improving slightly at the end: a longer run or more data should help).
  Saved `models/ada-stt-base/final`.
- Exported size: **77.7 MB** int8 CTranslate2 (`models/ada-stt-base-ct2`), inside the 40–80 MB target.
- Comparison rows (laptop CPU, int8, beam 5, 13:14–13:58):

  | data | model | clips | WER % | CER % | RTF |
  |---|---|---|---|---|---|
  | CV 27.0 test | stock Whisper base | 300 | 83.67 | 27.38 | 0.13 |
  | CV 27.0 test | **ADA-STT base v0.1-dev** | 300 | **44.58** | **8.48** | **0.11** |
  | FLEURS test (all) | stock Whisper base | 150 | 96.33 | 52.53 | 0.39 |
  | FLEURS test (all) | **ADA-STT base** | 150 | **55.90** | **19.31** | 0.35 |
  | FLEURS test, no digits | stock Whisper base | 150* | 95.64 | 51.79 | 0.40 |
  | FLEURS test, no digits | **ADA-STT base** | 150* | **52.28** | **16.33** | 0.36 |
  | FLEURS test (all), cloud | vasista22 whisper-tamil-small (244M) | 150 | 27.61 | 11.42 | (cloud) |
  | CV 27.0 test | vasista22 whisper-tamil-small (244M, int8 241 MB) | 300 | 30.27 | 5.21 | 0.33 |

  vasista22 caveat: its training data likely includes Common Voice Tamil (older release), so some CV test
  sentences/speakers may not be new to it. ADA is 3x smaller and 3x faster.
  | FLEURS test, no digits | vasista22 whisper-tamil-small | 150* | 22.91 | 7.67 | 0.92 |

  *`evaluate.py` filters digits first, then takes the first 150, so this is 150 digit-free clips (not
  the same 150 as the "all" row).
  Stock base on CV: 83.67 here vs 83.09 in the morning baseline on the same 300 clips (small CPU
  non-determinism in beam search; not investigated).
  Reading: on Common Voice (in-domain) ADA cuts CER 27.4 → 8.5 and is *faster* than stock base (it stops
  hallucinating long outputs). On FLEURS (out-of-domain read news, unseen in training) ADA cuts CER 52.5 →
  19.3, but is still well behind vasista22 (11.4 CER, 3x the size). Expected: trained on 84 h from only
  22 speakers. Day 2 data (MILE + Kathbath, ~570 h, many speakers) is the fix.

## Afternoon: run 2 prep (owner: "why wait for tomorrow")
- Checked CV `validated.tsv`: 28,341 extra clips usable, but **all from the same 22 train speakers** (every
  other speaker is in dev/test). Not worth it; speaker diversity must come from other datasets.
- Kathbath Tamil on HF: 40 train + 2 valid parquet files, 21.4 GB, FLAC 16 kHz mono embedded, columns
  fname/text/audio_filepath/lang/duration/gender/speaker_id. HF speed ~11–22 MB/s (vs OpenSLR 0.6).
  Download started ~14:55 to `C:\ada-data\raw\kathbath\tamil`.
- Commit `7fc3919`: `--format kathbath` (valid split by speaker into dev/test), `combine_data.py` (relative
  paths, no copy, `--max-per-speaker`), `spec_augment` config block in train.py, config
  `whisper-base-combo-8gb.yaml` (CV + Kathbath, SpecAugment, 20k steps). 25 tests pass.
- Kathbath download done 15:48 (42 files, 19.94 GiB). Pass 1 prep (13 min, filters removed 4):
  train 95,758 clips / 171.9 h / **138 speakers** / top_speaker_share 0.06; only 138 train flags (124 repeated
  transcripts, 10 speaking rate, 4 > 30 s); no digits, no Latin. **But test_speakers_seen_in_train = 10 and
  dev_speakers_seen_in_train = 10**: Kathbath's official valid reuses train speakers. Dropping those 20 from
  train would cost 36,376 clips / 62.2 h.
  **Owner decided: new unseen-speaker split.** `--format kathbath --holdout-speakers 16`: the 16 train
  speakers closest to the median clip count (alternating gender, excluding valid speakers) move to dev/test
  (alternating); official valid kept as `test-known/` (seen voices, reported as such). Pass 2 running.
- Plan: Kathbath prep + audit → combine → audit → smoke → 20k-step run overnight.
- **Owner away until 2026-11-28 morning. Standing permissions given (~16:30):** auto-start run 2 if audits
  show 0 leakage; commit + push code/logs (never data/models/tokens); delete junk (corrupt MILE .bad only
  after the new archive checks OK, smoke model folders, old checkpoints of finished runs); lid close = do
  nothing on AC. If MILE finishes: prep + audit, and if clean start run 3 (CV + Kathbath + MILE) after run 2.
  Cap train clips per speaker at **5,000**. If GPU ≥ 87 °C for 3+ min: pause, cool below 65 °C, resume
  from the last checkpoint. MILE joins if it is
  ready before the run starts; otherwise it goes into run 3.

## Problems / questions for the owner
- See "Decisions needed" in the summary. Done-list: check_gpu ✓, FLEURS + CV prepared/audited ✓ (MILE
  pending download), number decision ✓, laptop CPU baseline ✓, GPU smoke test ✓, real run + export +
  comparison ✓ (stretch), log committed and pushed ✓.
