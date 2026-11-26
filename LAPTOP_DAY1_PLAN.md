# ADA-STT: Day 1 plan for Claude Code on the laptop

> **Who this is for:** Claude Code running on the owner's Windows laptop (VS Code),
> and the owner following along.
> **How to use it:** the owner opens this repo in VS Code and tells Claude Code:
> *"Read LAPTOP_DAY1_PLAN.md and do Day 1, step by step. Log everything in logs/DAY1_LOG.md."*
>
> Written by the cloud Claude Code session on 26 Nov 2026 and **updated the same night
> after the cloud tested the whole pipeline on CPU** (section A). The cloud has no GPU:
> everything that needs the RTX 5050, the owner's accounts, or data kept on the laptop
> is left for today. All data and models used in the cloud were deleted afterwards;
> **the laptop downloads its own copies** (section 3).

---

## A. Already done in the cloud (don't redo)

| Done | Result |
|---|---|
| Full pipeline run on CPU: `prepare_data` → `audit_data` → `train --smoke` (whisper-tiny, 20 steps) → `export_ct2` → `evaluate` | **Works end to end.** Smoke model: WER 126.6 → 89.2, CER → 39.5 in 20 steps on 64 clips; exported int8 model 43 MB. |
| Bugs found and fixed (all pushed) | `av` 19 breaks faster-whisper audio decoding → pinned `av<19`. `train.py` no longer uses the `audiofolder` loader (rejected valid metadata with new pyarrow). `export_ct2.py` now writes `preprocessor_config.json` (transformers 5 saves `processor_config.json` instead). Inference now uses `without_timestamps=True` (fine-tuned models sometimes printed the transcript twice). |
| Data filters (train only) | `--drop-digit-clips`, `--dedupe-test-sentences`, `--min-seconds` in `prepare_data.py`. |
| Data converters | `--format fleurs` (tested on real FLEURS Tamil), `--format mile` (tested on 3,000 real IISc-MILE clips, reads speaker IDs from file names), `--format commonvoice` (not yet run on real data). |
| Data checks | FLEURS Tamil test: 591 clips, 2.13 h; **21% of transcripts contain digits**, 9% English words. IISc-MILE sample (3,000 clips, 9.1 h, 55 speakers): **no digits, no English letters**, very clean (42 repeated sentences, 6 clips > 30 s). |
| Baseline on FLEURS Tamil test (150 clips, CPU, int8, beam 5) | See table in section B. |
| Configs | `configs/stt/whisper-base-8gb.yaml` (v0.1 target, ready), `configs/stt/smoke-tiny-cpu.yaml` (pipeline check without GPU). |
| Tests | 20 tests pass, `ruff` clean. |

**Never run yet** (today's job): anything on the GPU (CUDA, bf16), `--format commonvoice` on
real data, a training run longer than 20 steps, speed on the laptop's own CPU.

---

## B. Baseline: what ADA has to beat

FLEURS Tamil test, first 150 clips, cloud CPU (4 shared cores, so RTF is pessimistic),
int8, beam 5, ADA's normalization. **RTF < 1 = faster than real time.**

| Model | Params | WER % | CER % | RTF |
|---|---|---|---|---|
| `vasista22/whisper-tamil-small` (int8) | 244M | **27.61** | **11.42** | 1.91 |
| Stock Whisper small | 244M | 77.81 | 32.50 | 1.48 |
| **Stock Whisper base (ADA v0.1 starts here)** | 74M | 102.19 | 54.64 | 1.04 |
| Stock Whisper tiny | 39M | 123.59 | 72.06 | 0.73 |

Details, settings and error examples: [docs/BASELINES.md](docs/BASELINES.md).
**ADA v0.1 should land far below base's 54.6% CER and approach vasista22's 11.4% CER at a
third of its size.**

Why vasista22 scores 27.6% here vs the 9.1% on its model card:
- **Numbers:** FLEURS writes numbers as digits ("1963"), the model writes Tamil words.
  Clips with digits: 39.2% WER; without digits: 24.9%.
- **Word spacing:** e.g. `வேக வைக்கப்பட்ட` vs `வேகவைக்கப்பட்ட`. Each is a whole wrong word for WER
  but only one character for CER. That's why ADA reports **CER** too.
- 3 of 150 clips came out twice (fixed for ADA by `without_timestamps=True`; this run used the old setting).
- Different normalization and int8 quantization also play a part.

Competitor from its model card: `rajagowthamr/tamil-conformer-transducer-74m` (same size as
ADA's target) reports 31.6% WER / 10.5% CER on IndicVoices Tamil valid. Not run here
(custom code).

---

## C. Decision the owner must make first: how numbers are written

| Source | Numbers written as |
|---|---|
| IISc-MILE | Tamil words (no digits found) |
| Common Voice | Words, by Common Voice's sentence rules (confirm with the audit) |
| FLEURS | Digits (21% of clips) |
| vasista22's output | Mostly Tamil words |

**Recommendation for v0.1:** train on **numbers as Tamil words** (that's what the main training
data already does, so no risky text conversion is needed). Report FLEURS results **twice**:
all clips, and `--exclude-digits` (no digit clips), and say so in the model card.
Converting digits ↔ Tamil words properly (with case endings like `1963இல்`) is a later project.

Ask the owner to confirm before step 4.

---

## 0. Context in one minute

- **ADA** = open-source family of small encoder-decoder Transformer models for speech and
  text that run well **on a CPU**. Goal: at least 10 real users.
- **ADA-STT** (speech to text) comes first, for **Tamil**.
- **v0.1 target:** fine-tune **Whisper base (74M)**, export to **int8 CTranslate2** (~40–80 MB),
  faster than real time on a laptop CPU, installable with `pip`, clearly better than stock
  Whisper base, honestly compared with vasista22 and the 74M conformer.
- **Rule from the owner: quality data first.** Nothing is trained on data that hasn't been
  through `scripts/stt/audit_data.py`.
- Background reading: `README.md`, `docs/DATA_TAMIL.md`, `docs/ROADMAP.md`, `docs/SETUP_WINDOWS.md`.

### Hardware
- Windows laptop, **NVIDIA RTX 5050, 8 GB VRAM** (Blackwell), **16 GB RAM**.
- Blackwell needs **PyTorch built for CUDA 12.8 or newer**.

---

## 1. Rules for Claude Code today

1. **Log everything** in `logs/DAY1_LOG.md` (template in section 9), as you go.
2. **Stop and ask the owner** for: browser sign-ups, accepting dataset terms, tokens,
   the number decision (section C), anything costing money, deleting more than a few files.
3. **Never commit** data, audio, models or tokens (`.gitignore` already excludes them).
4. Fix bugs in the scripts when you hit them, add a test where sensible, run `pytest`
   and `ruff check .`, commit with a clear message.
5. Keep the laptop plugged in and awake (see `docs/SETUP_WINDOWS.md`).
6. Use **PowerShell**. Paths are relative to the repo root unless shown otherwise.
7. If a step fails twice for the same reason, log the error and ask the owner.

---

## 2. Step 1: Environment setup (about 30–45 min)

```powershell
nvidia-smi                      # RTX 5050 listed, "CUDA Version" 12.8 or higher
py -0                           # installed Pythons; 3.12 preferred
git --version
Get-PSDrive -PSProvider FileSystem   # free space; need 60+ GB today, 100+ GB later
```
- Driver too old → owner updates it (NVIDIA App), restarts.
- Python 3.12 missing → owner installs from python.org ("Add python.exe to PATH").
- Choose the drive with most space for raw data, e.g. `D:\ada-data`. Below this is called `<DATA>`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train,export,dev]"
python scripts/check_gpu.py           # must end with "All good"; bf16 support: True
pytest                                # 20+ tests pass
ruff check .
```
If `check_gpu.py` says a kernel did not run: uninstall torch/torchaudio and reinstall from the
`cu128` index above. Log PyTorch version and CUDA build.

Also log `pip show av faster-whisper transformers datasets` versions (av must be < 19).

---

## 3. Step 2: Downloads

Nothing from the cloud session is kept: it deleted all its data and models.
**Everything below is downloaded on the laptop.**

### Full download list

| What | Size | How | Login? | When |
|---|---|---|---|---|
| NVIDIA driver (CUDA 12.8+) | ~1 GB | NVIDIA App | no | Step 1, if too old |
| Python 3.12 | ~30 MB | python.org | no | Step 1, if missing |
| PyTorch + torchaudio (cu128) | ~3 GB | `pip` (Step 1) | no | Step 1 |
| ADA's Python packages | ~1 GB | `pip install -e ".[train,export,dev]"` | no | Step 1 |
| **IISc-MILE Tamil** (~150 h) | **13 GB** (+~17 GB extracted) | `curl.exe` loop (3a) | no | Start first, runs in background |
| **FLEURS Tamil test** (591 clips) | 0.4 GB | `huggingface-cli download` (3b) | no | Step 2 |
| **Common Voice 27.0 Tamil** | a few GB | browser, Mozilla Data Collective (3c) | **yes** (free) | Step 2 |
| Stock Whisper tiny/base/small (int8, for baselines) | 40–250 MB each | automatic on first `evaluate.py` | no | Step 4 |
| `openai/whisper-base` (training start point) | ~300 MB | automatic on first `train.py` | no | Step 5 |
| vasista22/whisper-tamil-small (optional competitor) | ~1 GB | Python one-liner (Step 4) | no | Step 4, optional |
| Kathbath Tamil | 21 GB | Hugging Face (accept terms today) | **yes** | **Day 2** |
| IndicVoices Tamil | 50 GB | Hugging Face (accept terms today) | **yes** | **Day 2 or later** |

Disk needed today: about **40–60 GB**. With Day 2 data: **100+ GB**.

### 3a. Start the slow one first: IISc-MILE (13 GB, ~150 h)
The OpenSLR server is slow (the cloud saw 0.1–1 MB/s) and drops connections. `curl.exe -C -`
resumes, so loop it in a **separate PowerShell window** and keep going with other steps:
```powershell
mkdir <DATA>\raw -Force; cd <DATA>\raw
for ($i = 1; $i -le 50; $i++) {
  curl.exe -L -C - -o mile_tamil_asr_corpus.tar.gz https://www.openslr.org/resources/127/mile_tamil_asr_corpus.tar.gz
  if ($LASTEXITCODE -eq 0) { break }
  Start-Sleep -Seconds 10
}
tar -tzf mile_tamil_asr_corpus.tar.gz > $null; if ($?) { "archive OK" }
```
(Final size should be about 13 GB. Licence CC BY 2.0: credit IISc-MILE in the model card.)

### 3b. FLEURS Tamil test (0.4 GB, open, no login)
```powershell
huggingface-cli download google/fleurs --repo-type dataset --include "data/ta_in/test.tsv" "data/ta_in/audio/test.tar.gz" --local-dir <DATA>\raw\fleurs
# newer versions of the tool use:  hf download ... (same arguments)
tar -xzf <DATA>\raw\fleurs\data\ta_in\audio\test.tar.gz -C <DATA>\raw\fleurs\data\ta_in\audio
```

### 3c. Owner does these in the browser (wait for confirmation)
| # | What | Where |
|---|---|---|
| 1 | Free account; download **Common Voice Scripted Speech 27.0 – Tamil** | https://mozilladatacollective.com/datasets/cmu5x7cbc00dqo1071wdidlpf |
| 2 | Free **Hugging Face** account + **read** token | https://huggingface.co/settings/tokens |
| 3 | While logged in, **accept the terms** of Kathbath and IndicVoices (auto-approved; used Day 2) | https://huggingface.co/datasets/ai4bharat/Kathbath , https://huggingface.co/datasets/ai4bharat/IndicVoices |

Then `huggingface-cli login` in the venv (token typed at the prompt, never saved in the repo).

Extract Common Voice: `tar -xzf <download>.tar.gz -C <DATA>\raw`. Find the Tamil folder that has
`clips\`, `train.tsv`, `test.tsv` (e.g. `cv-corpus-27.0-...\ta`). Log the path and row counts.

---

## 4. Step 3: Prepare and audit all three sources

```powershell
python scripts/stt/prepare_data.py --format fleurs --src <DATA>\raw\fleurs\data\ta_in --out data/stt/fleurs-ta
python scripts/stt/prepare_data.py --format commonvoice --src "<DATA>\raw\<cv-folder>\ta" --out data/stt/cv-ta
python scripts/stt/prepare_data.py --format mile --src <DATA>\raw --out data/stt/mile-ta   # after 3a finishes; extract first:
#   tar -xzf <DATA>\raw\mile_tamil_asr_corpus.tar.gz -C <DATA>\raw
python scripts/stt/audit_data.py --data data/stt/fleurs-ta
python scripts/stt/audit_data.py --data data/stt/cv-ta
python scripts/stt/audit_data.py --data data/stt/mile-ta
```
Expected for FLEURS (to confirm the laptop matches the cloud): 591 test clips, 2.13 h,
127 clips with digits, 51 with Latin letters, 9 longer than 30 s.

**Paste every audit summary into the log.** How to act on it:

| Audit line | What to do |
|---|---|
| `test_sentences_seen_in_train` > 0 | Re-run `prepare_data.py` with `--dedupe-test-sentences` (drops train clips whose sentence is in test). Log before/after. |
| `test_speakers_seen_in_train` > 0 | Should be 0. Log it and tell the owner. |
| `speaking rate ... vs median` | Listen to 10 of them (`audit_flags.csv`). If most are wrong, drop all from train. |
| `contains digits` / `Latin letters` in **train** | Apply the owner's decision from section C. For "numbers as words": re-run with `--drop-digit-clips`. Latin letters: log 10 examples for the owner. |
| `very short audio` / `longer than 30s` | Training already skips > 30 s. Re-run with `--min-seconds 1`. |
| `top_speaker_share` > 0.05 | Log; consider capping clips per speaker later. |
| `unknown_duration` > 0 | `soundfile` couldn't read some mp3 headers. If many, fall back to `librosa.get_duration(path=...)` in `_duration()`. |

Filters that already exist (train only; test is never changed): `--drop-digit-clips`,
`--dedupe-test-sentences`, `--min-seconds`. Any new filter goes in as another option of
`prepare_data.py` (reproducible), never hand-edited CSVs. Delete the output folder before re-running.

---

## 5. Step 4: Baseline on the laptop (CPU)

Two goals: numbers on Common Voice (cloud only did FLEURS), and **real laptop CPU speed**.
```powershell
Get-CimInstance Win32_Processor | Select Name, NumberOfCores, NumberOfLogicalProcessors
python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta --limit 300 --save-dir runs/preds-cv --models base small
python scripts/stt/evaluate.py --data data/stt/fleurs-ta/test --language ta --limit 150 --models base
```
The second command repeats the cloud's FLEURS baseline for `base` (same 150 clips); compare RTF.

Optional, strongest competitor (its repo has no `tokenizer.json`, so save it locally first):
```powershell
python -c "from transformers import WhisperForConditionalGeneration as M, WhisperProcessor as P; m='vasista22/whisper-tamil-small'; M.from_pretrained(m).save_pretrained('models/vasista22-hf'); P.from_pretrained(m).save_pretrained('models/vasista22-hf')"
python scripts/stt/export_ct2.py --model models/vasista22-hf --out models/vasista22-ct2
python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta --limit 300 --models models/vasista22-ct2
```

---

## 6. Step 5: First training on the GPU

### 5a. Smoke test (5–10 min)
```powershell
python scripts/stt/train.py --config configs/stt/whisper-base-8gb.yaml --smoke
```
`whisper-base-8gb.yaml` already points to `data/stt/cv-ta`. Pass: prints baseline WER, 20 steps,
eval, saves `models/ada-stt-base-smoke/final`, prints WER/CER before and after.

The pipeline already passed this on CPU, so what's new here is CUDA + bf16:

| Symptom | Fix |
|---|---|
| CUDA out of memory | Halve `per_device_train_batch_size`, double `gradient_accumulation_steps`. |
| bf16 error | Set `bf16: false`, `fp16: true` in the config; log it. |
| Audio decode error on mp3 | `pip install -U soundfile` (needs libsndfile ≥ 1.1); log versions. |
| Windows multiprocessing error | Keep `dataloader_num_workers: 0`. |
| "missing keys ... proj_out.weight" warning | Harmless (tied weights); seen in the cloud, results unaffected. |

### 5b. Real run (Common Voice only, about 1–3 h; owner decides whether to start)
```powershell
python scripts/stt/train.py --config configs/stt/whisper-base-8gb.yaml
tensorboard --logdir models/ada-stt-base       # second terminal → http://localhost:6006
```
Log: steps/s, GPU memory (`nvidia-smi`), temperature, eval WER/CER per `eval_steps`.
Stop early if eval WER doesn't improve twice in a row (best checkpoint is kept).
**Stop if the GPU stays above ~87 °C** and tell the owner.

### 5c. Export and compare on CPU
```powershell
python scripts/stt/export_ct2.py --model models/ada-stt-base/final --out models/ada-stt-base-ct2
python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta --limit 300 --models base models/ada-stt-base-ct2
python scripts/stt/evaluate.py --data data/stt/fleurs-ta/test --language ta --limit 150 --models models/ada-stt-base-ct2
python scripts/stt/evaluate.py --data data/stt/fleurs-ta/test --language ta --limit 150 --exclude-digits --models base models/ada-stt-base-ct2
```
These rows are ADA's first real results. Put them in the log next to the table in section B.

---

## 7. End of day

1. `pytest` and `ruff check .` pass.
2. Commit code changes and `logs/DAY1_LOG.md`; `git push`.
3. Write **"Summary for the owner"** at the top of the log: what worked, results tables,
   open problems, decisions needed.

### Done = all true
- [ ] `check_gpu.py` says `All good`
- [ ] FLEURS + Common Voice prepared and audited (IISc-MILE too if the download finished)
- [ ] Number decision made (section C)
- [ ] Laptop CPU baseline on Common Voice in the log
- [ ] GPU smoke test passes
- [ ] Log committed and pushed

Stretch: real run + export + comparison.

---

## 8. Day 2 preview (don't start today)

- Kathbath converter (`--format hfparquet`): inspect parquet column names on one file first.
  Keep its test-known / test-unknown and noisy sets as test-only.
- Combine Common Voice + IISc-MILE + Kathbath (~570 h), keeping each source's test set separate.
- Full whisper-base run; compare with vasista22 and the 74M conformer on the same clips.

---

## 9. Log template: create `logs/DAY1_LOG.md`

```markdown
# Day 1 log: ADA-STT Tamil

## Summary for the owner
(fill in at the end)

## Machine
- GPU / driver / CUDA (nvidia-smi):
- CPU / cores / threads:
- RAM:
- Free disk (data drive):
- Versions: Python / PyTorch + CUDA build / transformers / datasets / faster-whisper / av / ctranslate2:

## Step 1: setup
- [time] command → result
- check_gpu.py output:

## Step 2: downloads
- IISc-MILE: start time, finish time, final size, archive OK?
- FLEURS test: path
- Common Voice: version, path, train.tsv rows, test.tsv rows
- HF login: yes/no; Kathbath + IndicVoices terms accepted: yes/no

## Step 3: audits
### FLEURS
### Common Voice
### IISc-MILE
- Decisions (incl. number convention):

## Step 4: baseline (laptop CPU)
| data | model | clips | WER % | CER % | RTF |
|---|---|---|---|---|---|

## Step 5: training
- Config:
- Smoke test:
- Bugs found and fixes (commit hashes):
- Real run: steps, time, best eval WER/CER, GPU memory, temperature:
- Exported size:
- Comparison rows:

## Problems / questions for the owner
-
```
