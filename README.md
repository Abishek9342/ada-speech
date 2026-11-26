# ADA

**ADA** is an open-source family of lightweight **encoder-decoder Transformer**
models for speech and text, built to run **fast and reliably on a plain CPU**.

| Component | Task | Architecture | Status |
|---|---|---|---|
| **ADA-STT** | Speech → text | Whisper-style encoder-decoder, fine-tuned | 🚧 in progress |
| **ADA-TTS** | Text → speech | VITS/Piper-style | 📋 planned |
| **ADA-T2T** | Text → text (e.g. translation) | T5/mBART-style | 📋 planned |

ADA does not train from scratch. It fine-tunes proven open models on data for a
specific language or domain, then shrinks them (int8, CTranslate2) so they run
well on ordinary laptops.

## Use it (CPU, no GPU needed)

```bash
pip install ada-speech          # once published; for now: pip install -e .
ada-stt meeting.wav --language ta --stats
```

```python
from ada.stt import Transcriber

stt = Transcriber("base")          # or a trained ADA model folder / HF repo id
print(stt.transcribe("meeting.wav", language="ta").text)
```

## Train it (GPU)

See [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) for the full setup.

```bash
python scripts/check_gpu.py                                          # 1. GPU works?
python scripts/stt/prepare_data.py --format commonvoice \
       --src path/to/cv/ta --out data/stt/cv-ta                      # 2. data
python scripts/stt/audit_data.py --data data/stt/cv-ta                # 3. check data quality
python scripts/stt/train.py --config configs/stt/whisper-small-8gb.yaml --smoke   # 4. smoke test
python scripts/stt/train.py --config configs/stt/whisper-small-8gb.yaml           # 5. real run
python scripts/stt/export_ct2.py --model models/ada-stt-small/final \
       --out models/ada-stt-small-ct2                                # 6. make it CPU-fast
python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta \
       --models small models/ada-stt-small-ct2                       # 7. prove it's better
```

## Project layout

```
src/ada/
  text.py            text normalization + WER (shared by training and eval)
  stt/inference.py   CPU inference (faster-whisper, int8)
  stt/audit.py       data-quality checks (leakage, bad transcripts, scripts)
  stt/cli.py         `ada-stt` command
  tts/, t2t/         planned components
scripts/
  check_gpu.py       verify PyTorch + CUDA on the training machine
  stt/prepare_data.py  dataset -> audiofolder layout (+ duration, speaker)
  stt/audit_data.py  data-quality report before training
  stt/train.py       fine-tune Whisper on your GPU
  stt/export_ct2.py  convert to int8 CTranslate2 for CPU
  stt/evaluate.py    WER + real-time factor on CPU
configs/stt/         training configs (8 GB VRAM preset)
docs/                setup guide, roadmap, Tamil data plan (DATA_TAMIL.md)
tests/               fast tests, no GPU or model downloads
```

## Goals for v0.1 (definition of done)

- Beats stock Whisper of the same size on the chosen language's test set (WER).
- Faster than real time (RTF < 1.0) on a 4-core laptop CPU.
- Model download under ~150 MB.
- `pip install` and transcribe in three lines.
- Honest model card: data, licence, WER, known failure cases.

## License

Code: MIT. Each released model also states the licences of its base model and data.
