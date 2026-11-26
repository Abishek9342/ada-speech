# ADA-STT: Day 1 plan for Claude Code on the laptop

> **Who this is for:** Claude Code running on the owner's Windows laptop (VS Code),
> and the owner following along.
> **How to use it:** the owner opens this repo in VS Code and tells Claude Code:
> *"Read LAPTOP_DAY1_PLAN.md and do Day 1, step by step. Log everything in logs/DAY1_LOG.md."*
>
> Written by the cloud Claude Code session on 26 Nov 2026. Everything in this
> repo so far was written in the cloud, where there is **no GPU** and no access to
> the laptop. The tests pass, but **`train.py`, `export_ct2.py` and `evaluate.py`
> have never run on real data or a GPU.** Expect small bugs; fixing them is part of today.

---

## 0. Context in one minute

- **ADA** = open-source family of small encoder-decoder Transformer models for speech and
  text that run well **on a CPU**. Goal: at least 10 real users.
- **ADA-STT** (speech to text) comes first, for **Tamil**.
- **v0.1 target:** fine-tune **Whisper base (74M)**, export to **int8 CTranslate2**, so it is
  small, faster than real time on a laptop CPU, and installable with `pip`.
  It must clearly beat stock Whisper base and be honest about how it compares with:
  - `rajagowthamr/tamil-conformer-transducer-74m`: 31.6% WER / 10.5% CER on IndicVoices Tamil valid
  - `vasista22/whisper-tamil-small` (244M): 9.11% WER on FLEURS test
- **Rule from the owner: quality data first.** Nothing gets trained on data that hasn't
  been through `scripts/stt/audit_data.py`.
- Background reading, in this order: `README.md`, `docs/DATA_TAMIL.md`, `docs/ROADMAP.md`,
  `docs/SETUP_WINDOWS.md`.

### Hardware
- Windows laptop, **NVIDIA RTX 5050, 8 GB VRAM** (Blackwell architecture), **16 GB RAM**.
- Blackwell needs **PyTorch built for CUDA 12.8 or newer**. A normal `pip install torch`
  may give a build that installs fine but fails when a GPU kernel runs.

---

## 1. Rules for Claude Code today

1. **Log everything** in `logs/DAY1_LOG.md` (template in section 9): each command,
   whether it worked, key numbers, errors and the fix. Append as you go, don't write it at the end.
2. **Stop and tell the owner** when a step needs them: browser sign-ups, accepting dataset
   terms, entering tokens, anything that costs money, or deleting more than a few files.
3. **Never commit** data, audio, models or tokens. `.gitignore` already excludes
   `data/`, `models/`, `runs/`, `*.wav`, `*.mp3`, `*.safetensors`, `*.bin`, `*.pt`.
   Never put a Hugging Face token in a file in the repo.
4. **Fix bugs in the repo's scripts when you hit them**, add or extend a test in `tests/`
   where it makes sense, run `pytest` and `ruff check .`, then commit with a clear message.
5. **Keep the laptop awake**: plugged in, sleep set to Never while plugged in (see `docs/SETUP_WINDOWS.md`).
6. Use **PowerShell** commands. Paths below are relative to the repo root.
7. If a step fails twice for the same reason, write the error in the log and ask the owner
   instead of guessing.

---

## 2. Step 1: Environment setup (about 30–45 min)

### 1a. Check the basics
```powershell
nvidia-smi                      # must list the RTX 5050; "CUDA Version" must be 12.8 or higher
python --version                # or: py -0  (lists installed Pythons)
git --version
Get-PSDrive -PSProvider FileSystem   # free disk space per drive
```
- If `nvidia-smi` is missing or CUDA < 12.8: the owner updates the driver
  (NVIDIA App or nvidia.com), restarts, then continue.
- **Disk:** need **at least 60 GB free** for today (Common Voice + IISc-MILE + checkpoints),
  **100+ GB** once IndicVoices is added later. Pick the drive with the most space for `data/`.
  If it is not the repo's drive, put data there and pass full paths to the scripts.
- Python **3.12** is preferred. If missing, the owner installs it from python.org
  (tick "Add python.exe to PATH").

### 1b. Virtual environment and packages
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train,export,dev]"
```

### 1c. Verify
```powershell
python scripts/check_gpu.py
pytest
ruff check .
```
**Pass criteria:** `check_gpu.py` ends with `All good`, shows the RTX 5050, about 8 GB VRAM,
and `bf16 support: True`. All tests pass.

**If `check_gpu.py` says a GPU kernel did not run:**
```powershell
pip uninstall -y torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```
Log the PyTorch version and CUDA build it reports.

---

## 3. Step 2: Accounts and downloads (owner does the browser parts)

Ask the owner to do these and wait for confirmation:

| # | What | Where | Why |
|---|---|---|---|
| 1 | Create a free account and download **Common Voice Scripted Speech 27.0, Tamil** | https://mozilladatacollective.com/datasets/cmu5x7cbc00dqo1071wdidlpf | Main training set (CC0, ~235 h validated) |
| 2 | Download **IISc-MILE Tamil ASR corpus** (13 GB) | https://www.openslr.org/127/ (file `mile_tamil_asr_corpus.tar.gz`) | Clean studio speech, 531 speakers (CC BY 2.0) |
| 3 | Create a free **Hugging Face** account and a **read** token | https://huggingface.co/settings/tokens | Needed for the gated AI4Bharat sets |
| 4 | Open these pages while logged in and **accept the terms** (auto-approved) | https://huggingface.co/datasets/ai4bharat/Kathbath and https://huggingface.co/datasets/ai4bharat/IndicVoices | Kathbath is used on Day 2; IndicVoices later |

Then, in the activated venv, the owner runs (token is typed in the prompt, **not** saved in the repo):
```powershell
huggingface-cli login
```

### Extract the archives
Windows 10+ has `tar` built in:
```powershell
mkdir D:\ada-data\raw -Force        # use the drive with the most free space
tar -xzf <path-to-common-voice-download>.tar.gz -C D:\ada-data\raw
tar -xzf <path-to>\mile_tamil_asr_corpus.tar.gz -C D:\ada-data\raw
```
Find the Common Voice **Tamil folder**: the one that contains `clips\` and `train.tsv`,
`test.tsv`, `validated.tsv` (usually `cv-corpus-27.0-...\ta\`). Log its exact path and the
row counts of `train.tsv` and `test.tsv`.

---

## 4. Step 3: Prepare and audit Common Voice (owner's top priority)

```powershell
python scripts/stt/prepare_data.py --format commonvoice --src "D:\ada-data\raw\<cv-folder>\ta" --out data/stt/cv-ta
python scripts/stt/audit_data.py --data data/stt/cv-ta
```
`prepare_data.py` copies the clips. That takes a while and doubles the disk use for
Common Voice; that's fine at this size.

**Write the full audit summary in the log**, then look at it:

| Audit line | What to do |
|---|---|
| `test_sentences_seen_in_train` > 0 | Note the number. Common Voice's official split should keep speakers apart; repeated sentences make the test score look better than it is. Add a `--dedupe-test-sentences` option to `prepare_data.py` that drops train clips whose normalized sentence appears in test, re-run, and log before/after counts. |
| `test_speakers_seen_in_train` > 0 | Should be 0 for Common Voice. If not, log it and tell the owner. |
| `speaking rate ... vs median` | Listen to 10 of these clips (paths are in `data/stt/cv-ta/audit_flags.csv`). If most really are wrong, drop all of them from train; log the count. |
| `contains digits` / `Latin letters` | Count and log them, and show the owner 10 examples. **The owner decides the convention** (e.g. numbers as Tamil words vs digits). Don't change transcripts yet. |
| `very short audio` / `longer than 30s` | Training already filters >30 s. Drop <1 s clips from train. |
| `top_speaker_share` > 0.05 | Log it. Consider capping clips per speaker later. |
| `unknown_duration` > 0 | `soundfile` couldn't read those mp3 headers. Log how many. If it's most of them, fix `_duration()` in `prepare_data.py` (e.g. fall back to `librosa.get_duration(path=...)`). |

If any filtering is added, implement it **as options in `prepare_data.py`** (so it's
reproducible), not by editing CSVs by hand. Commit the changes.

---

## 5. Step 4: Baseline, measuring what ADA has to beat (about 30–60 min, CPU)

```powershell
python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta --limit 300 --models tiny base small
```
- `tiny`, `base`, `small` are stock Whisper, downloaded automatically as int8 CTranslate2.
- The output columns are WER %, CER % and RTF (real-time factor; below 1.0 means faster
  than real time). **Copy the table into the log** together with the CPU model and core count
  (`Get-CimInstance Win32_Processor | Select Name, NumberOfCores`).
- Expected rough range: stock Whisper on Tamil is weak, with WER somewhere around 35–60% or worse
  for base. If WER is near 100% or outputs are English, check `--language ta` was passed
  and log sample outputs.

**Also measure the strongest competitor on the same clips** (useful, not required today):
```powershell
ct2-transformers-converter --model vasista22/whisper-tamil-small --output_dir models/vasista22-small-ct2 --quantization int8 --copy_files tokenizer.json preprocessor_config.json
python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta --limit 300 --models models/vasista22-small-ct2
```
If its repo has no `tokenizer.json`, log that and skip it. Don't spend more than 20 minutes on this.

---

## 6. Step 5: First training, smoke test then a short real run

### 5a. Create the whisper-base config
Copy `configs/stt/whisper-small-8gb.yaml` to `configs/stt/whisper-base-8gb.yaml` and change:
```yaml
base_model: openai/whisper-base
data_dir: data/stt/cv-ta
output_dir: models/ada-stt-base
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
learning_rate: 2.5e-5        # base tolerates a higher rate than small
max_steps: 5000
```
Commit the new config.

### 5b. Smoke test (about 5–10 min)
```powershell
python scripts/stt/train.py --config configs/stt/whisper-base-8gb.yaml --smoke
```
**Pass criteria:** it prints a baseline WER, runs 20 steps, evaluates, saves to
`models/ada-stt-base-smoke/final`, and prints WER before and after. Loss values appear in the output.

`train.py` has **never run before**. Likely issues and fixes:

| Symptom | Likely fix |
|---|---|
| Error about `processing_class` or `eval_strategy` | Installed `transformers` is older than expected. `pip install -U "transformers>=4.46"`. |
| Audio decode error on mp3 | `pip install -U soundfile` (needs libsndfile ≥ 1.1 for mp3). Log the version. |
| `KeyError: 'audio'` in the collator | Check that `remove_unused_columns=False` is in the training arguments. |
| Errors about `forced_decoder_ids` / language tokens | Set language via `model.generation_config.language = "tamil"` only (already done) and make sure the tokenizer prefix tokens are set on the processor. Log the full error. |
| CUDA out of memory | Halve `per_device_train_batch_size`, double `gradient_accumulation_steps`. |
| Windows multiprocessing error | Keep `dataloader_num_workers: 0`. |

Fix in the script, re-run, commit the fix, log it.

### 5c. Short real run (about 1–2 h; owner decides whether to start it today)
```powershell
python scripts/stt/train.py --config configs/stt/whisper-base-8gb.yaml
tensorboard --logdir models/ada-stt-base     # in a second terminal, open http://localhost:6006
```
- Log: steps per second, GPU memory used (`nvidia-smi`), GPU temperature, eval WER/CER at each `eval_steps`.
- If WER on eval stops improving for 2 evaluations in a row, it's fine to stop (Ctrl+C).
  The best checkpoint is kept (`load_best_model_at_end`).
- **Stop if the GPU stays above ~87 °C** and tell the owner (cooling pad, lower batch size).

### 5d. If the real run finishes: export and compare on CPU
```powershell
python scripts/stt/export_ct2.py --model models/ada-stt-base/final --out models/ada-stt-base-ct2
python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta --limit 300 --models base models/ada-stt-base-ct2
```
Log both rows. This is ADA's first real result.

---

## 7. Step 6 (only if time remains): add IISc-MILE

Its layout (from the OpenSLR page): `train/` and `test/`, each with `audio_files/` (16 kHz WAV)
and `trans_files/` (UTF-8 `.txt`, one per audio file, same base name).

Add `--format mile` to `scripts/stt/prepare_data.py`:
- Read `train/` and `test/` separately (keep the official split).
- Match `audio_files/X.wav` with `trans_files/X.txt`.
- Speaker ID: check if the file names contain one (log a few examples); if so, put it in the
  `speaker` column.
- Add a small test with two fake files in `tests/`.

Then:
```powershell
python scripts/stt/prepare_data.py --format mile --src "D:\ada-data\raw\<mile-folder>" --out data/stt/mile-ta
python scripts/stt/audit_data.py --data data/stt/mile-ta
```
Log the audit. **Do not merge datasets for training yet.** That's Day 2, together with Kathbath.

---

## 8. End of day: wrap-up

1. `pytest` and `ruff check .` pass.
2. Commit code changes and `logs/DAY1_LOG.md` (the log is small text; commit it).
3. `git push`.
4. Write a short **"Summary for the owner"** at the top of the log:
   what worked, the baseline table, first training result (if any), open problems,
   and decisions the owner must make (e.g. number convention).

### Done = all of these are true
- [ ] `check_gpu.py` says `All good`
- [ ] Common Voice Tamil prepared and audited; audit summary in the log
- [ ] Baseline WER/CER/RTF for stock tiny/base/small in the log
- [ ] Smoke test passes (and any `train.py` bugs fixed and committed)
- [ ] Log committed and pushed

Stretch: real run finished + exported + compared; IISc-MILE prepared and audited.

---

## 9. Log template: create `logs/DAY1_LOG.md`

```markdown
# Day 1 log: ADA-STT Tamil

## Summary for the owner
(fill in at the end)

## Machine
- GPU / driver / CUDA (nvidia-smi):
- CPU / cores:
- RAM:
- Free disk (drive used for data):
- Python / PyTorch / CUDA build / transformers / datasets / faster-whisper versions:

## Step 1: setup
- [time] command → result
- check_gpu.py output:

## Step 2: downloads
- Common Voice version, path, train.tsv rows, test.tsv rows:
- IISc-MILE path, size:
- HF login: yes/no; Kathbath/IndicVoices terms accepted: yes/no

## Step 3: Common Voice audit
(paste the full audit summary)
- Decisions taken / pending:

## Step 4: baseline (CPU, N clips)
| model | WER % | CER % | RTF |
|---|---|---|---|

## Step 5: training
- Config used:
- Smoke test result:
- Bugs found and fixes (commit hashes):
- Real run: steps, time, best eval WER/CER, GPU memory, temperature:

## Step 6: IISc-MILE
(audit summary)

## Problems / questions for the owner
-
```

---

## 10. Day 2 preview (don't start today)

- Write a Kathbath converter (`--format hfparquet`): first inspect the parquet column
  names on one file, then convert. Keep its *test-known/test-unknown* and noisy sets as test-only.
- Apply the owner's number/English-word convention to all sources.
- Combine Common Voice + Kathbath + IISc-MILE (~570 h) into one training set, keeping
  each source's test set separate so results can be reported per source.
- Full whisper-base training run; compare with the competitors on FLEURS test.
