# ada-speech (ADA-STT)

Open-source family of small, CPU-friendly encoder-decoder speech/text models. First target: **ADA-STT
Tamil v0.1** = Whisper base (74M) fine-tuned, exported to int8 CTranslate2 (~78 MB), faster than real
time on a laptop CPU. GitHub `Abishek9342/ada-speech`. Read `handoff.md` (top entry) before starting.

## Machine (owner's laptop)
- RTX 5050 Laptop 8 GB (Blackwell, needs PyTorch cu128+), i7-13645HX 14C/20T, 16 GB RAM, one drive C:.
- Python 3.13 (no 3.12 installed; works fine). Venv: `.venv` in the repo.
- **Smart App Control was turned OFF** (2026-11-27): it blocked torch DLLs and pip `.exe` launchers.
- Sleep/hibernate on AC set to Never (`powercfg`), or long runs pause.

## Layout
- Raw downloads: `C:\ada-data\raw` (outside the repo). Prepared data: `data/stt/<name>/{train,dev,test}`.
- Models: `models/` (gitignored). TensorBoard: `models/<run>/runs`. Logs of each day: `logs/DAY<n>_LOG.md`.
- `scripts/stt/`: `prepare_data.py` → `audit_data.py` → `train.py` → `export_ct2.py` → `evaluate.py`.

## Rules
- **Quality data first:** nothing is trained on data that hasn't been through `audit_data.py`.
- Filters only touch train; test (and dev) stay as published. New filters = new `prepare_data.py` options.
- Numbers are written as **Tamil words** (owner decision 2026-11-27). Latin-letter English names are kept.
- Validate on dev, never pick checkpoints on test. Report FLEURS both with and without digit clips.
- Never commit data, audio, models or tokens. HF token lives in `~\.cache\huggingface` (owner logged in).
- After code changes: `python -m pytest` and `ruff check .`, then commit.
- Use `python -m ...` or `.venv\Scripts\<tool>.exe` explicitly.
- Long downloads: one process only (see `C:\ada-data\download_mile.ps1`, lock file); two resumers on the
  same file corrupted the first IISc-MILE download.
