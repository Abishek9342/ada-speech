# ADA-STT Tamil: data and baselines

Research notes from web search (Nov 2026). **Unverified until checked on the
dataset pages and measured ourselves.** Licences especially must be confirmed
on each official page before training on or redistributing anything.

## Candidate training data

| Dataset | Tamil size | Licence | Use for ADA? |
|---|---|---|---|
| [Common Voice 27.0 Tamil](https://mozilladatacollective.com/datasets/cmu5x7cbc00dqo1071wdidlpf) | ~425 h recorded, ~235 h validated, 984 speakers | CC0 | ✅ **Main training set.** Read speech; download needs a Mozilla Data Collective account. |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) (AI4Bharat) | Tamil hours not confirmed (7,348 h over 22 languages) | CC BY 4.0, gated (share contact info) | ✅ **Strong addition.** Natural, spontaneous speech, closer to real users. Needs attribution. |
| [IndicVoices-R](https://ai4bharat.iitm.ac.in/indicvoices_r) | ~99 h, 1,084 speakers | CC BY 4.0 | Better suited to ADA-TTS later. |
| [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) (AI4Bharat) | Tamil hours not confirmed (1,684 h over 12 languages) | CC0 packaging; raw text not owned by AI4Bharat | ⚠️ Probably fine; read the licence note first. |
| [OpenSLR 65](https://huggingface.co/datasets/ylacombe/google-tamil) (Google) | ~7 h, 50 speakers | CC BY-SA 4.0 | ⚠️ Share-alike: a model trained on it may need the same licence. Fine as **test** data. |
| FLEURS `ta_in` (Google) | ~10 h (from memory, not verified) | CC BY 4.0 (from memory, not verified) | ✅ **Standard test set.** Use its test split to compare with others. |
| [Speech-data Tamil](https://huggingface.co/datasets/Speech-data/Tamil-Speech-Dataset) | ~112 h | CC BY-NC-ND 4.0 | ❌ Non-commercial, no derivatives. Skip. |
| [TamilVoiceCorpus](https://huggingface.co/datasets/ragunath-ravi/TamilVoiceCorpus) | ? | Apache 2.0 but scraped from the web, "research only" | ❌ Unclear provenance. Skip. |

**Plan:** start with Common Voice (CC0, simplest), add IndicVoices Tamil
(CC BY) for spontaneous speech, evaluate on Common Voice test **and** FLEURS test.

## What already exists (self-reported WER, not comparable across cards)

| Model | Size | Reported WER |
|---|---|---|
| Stock Whisper small, zero-shot | 244M | ~35% on FLEURS ta_in ([card](https://huggingface.co/sgangireddy/whisper-small-ta/blob/main/README.md)) |
| [vasista22/whisper-tamil-small](https://huggingface.co/vasista22/whisper-tamil-small/raw/refs%2Fpr%2F2/README.md) | 244M | ~9.1% FLEURS, ~8.0% Common Voice 11 |
| [steja/whisper-small-tamil](https://huggingface.co/steja/whisper-small-tamil/blob/main/README.md) | 244M | ~15.0% FLEURS |
| [bhuvanesh25/whis-tam-small](https://huggingface.co/bhuvanesh25/whis-tam-small/blob/main/README.md) | 244M | ~11.2% Common Voice 11 |
| [mahimairaja/whisper-base-tamil](https://huggingface.co/mahimairaja/whisper-base-tamil/blob/main/README.md) | 74M | ~28.2% Common Voice 13 (only 500 steps) |

## What this means for ADA

Good Tamil **small** (244M) fine-tunes already exist, so "another Whisper-small
for Tamil" would not stand out. The gap is **small, fast, CPU-friendly** models:

- The only **base**-size (74M) Tamil model found is weak (~28% WER, barely trained).
  No good **tiny** (39M) Tamil model was found.
- None of these are shipped as int8 CTranslate2 with a one-line CPU install,
  measured for speed on a laptop CPU, or tested for silence/noise hallucination.

**ADA-STT Tamil v0.1 target:** a **whisper-base** fine-tune, int8 on CPU,
clearly better than 28% WER, faster than real time on a 4-core laptop,
with honest benchmarks against stock Whisper and the models above
on the same test sets with the same normalization (`ada.text.normalize`).

A whisper-small version is still worth training as the "accurate" option,
compared head-to-head with vasista22's model.

## Baseline measurement (to do, on the laptop)

The cloud session could not download models or data (Hugging Face is not
reachable from it). Run this on the laptop once data is prepared:

```powershell
python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta --limit 200 `
    --models tiny base small
```

Record WER and RTF here. That is the number ADA has to beat.
