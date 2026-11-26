# ADA-STT Tamil: baselines

The numbers ADA has to beat. Measured in the cloud session on 26 Nov 2026.

## Setup

- **Test data:** FLEURS Tamil (`google/fleurs`, `ta_in`, CC BY 4.0), **first 150 of 591 test
  clips** (~29 min of audio). Prepared with `prepare_data.py --format fleurs`.
- **Inference:** `ada.stt.Transcriber` (faster-whisper / CTranslate2), CPU, **int8**,
  beam 5, VAD on, `language="ta"`.
- **Scoring:** `ada.text.wer` / `ada.text.cer` (NFC, lowercase, punctuation removed).
- **Machine:** cloud container, 4 shared CPU cores, no GPU, while other jobs were running.
  **RTF is pessimistic**; re-measure on the laptop CPU.
- Command (one model per run):
  `python scripts/stt/evaluate.py --data data/stt/fleurs-ta/test --language ta --limit 150 --save-dir runs/preds --models <model>`

## Results

| Model | Params | WER % | CER % | RTF | Notes |
|---|---|---|---|---|---|
| `vasista22/whisper-tamil-small` (int8) | 244M | **27.61** | **11.42** | 1.91 | Best existing Tamil Whisper. Card claims 9.11% WER on FLEURS. ¹ |
| Stock Whisper small | 244M | 77.81 | 32.50 | 1.48 | ² |
| Stock Whisper base | 74M | 102.19 | 54.64 | 1.04 | **ADA v0.1's starting point.** ¹ |
| Stock Whisper tiny | 39M | 123.59 | 72.06 | 0.73 | ² |

RTF = processing time / audio length; below 1.0 is faster than real time.

¹ Run with timestamps on (old default). 3 of 150 vasista22 outputs were the transcript twice;
with `without_timestamps=True` (now ADA's default) those 3 clips go from 157% to 76% WER.
² Run with `without_timestamps=True`.

## What the errors show

**Stock base hears Tamil but can't spell it.** Its output is almost all Tamil script
(18,451 Tamil vs 1,326 Latin characters) and roughly follows the sounds, but most words
are misspelled or wrong:

```
REF: எங்களுடைய விளையாட்டு வீரர்கள் மற்றும் கிளப்புகளின் நலன்கள் ...
HYP: எங்களுடைய வெளியாட்டு வெளியாட்கள் மற்றும் கலப்புகளின்னாககள் ...
```
This is the kind of error fine-tuning on Tamil text fixes well.

**vasista22's gap to its 9.1% claim** comes from:
1. **Numbers:** FLEURS writes digits (`1963`), the model writes Tamil words
   (`ஆயிரத்து தொள்ளாயிரத்து அறுபத்து மூன்று`). Clips with digits: 39.2% WER; without: 24.9%.
2. **Word spacing:** `வேக வைக்கப்பட்ட` vs `வேகவைக்கப்பட்ட` counts as 3 word errors, 1 character error.
3. Duplicated output on 3 clips (see ¹), plus normalization and int8 differences.

## Implications for ADA

- Report **CER next to WER**: Tamil words are long, so WER exaggerates small differences.
- Report FLEURS **twice**: all clips, and `--exclude-digits`. ADA's training data
  (IISc-MILE, Common Voice) writes numbers as words.
- **v0.1 goal:** whisper-base fine-tuned should land far below stock base's 54.6% CER and
  approach vasista22 (11.4% CER) at a third of the size and ~2× the speed.
