# ADA-STT Tamil: data plan

Research from web search, Nov 2026. This session could not open the dataset
pages or download anything (Hugging Face, arXiv and OpenSLR are blocked from
it), so **every number and licence below is from search-result summaries and
must be confirmed on the official page before use.** Items marked ⚠️ had
conflicting or incomplete information.

## Recommended data, by priority

### Tier 1: start here (clean licences, read speech, simple formats)

| Dataset | Tamil size | Licence | Notes |
|---|---|---|---|
| **Common Voice 27.0 Tamil** ([page](https://mozilladatacollective.com/datasets/cmu5x7cbc00dqo1071wdidlpf)) | ~425 h recorded, **~235 h validated**, 984 speakers | CC0 | Read sentences by volunteers. Download needs a free Mozilla Data Collective account. Use only `validated` clips. Older releases had the same sentence in train and test; `audit_data.py` checks this. |
| **Kathbath Tamil** (AI4Bharat, [paper](https://arxiv.org/pdf/2208.11761), [site](https://ai4bharat.iitm.ac.in/?p=1131)) | **~185 h**, 158 speakers (116 M / 42 F), 171k-word vocabulary | CC0 for AI4Bharat's packaging; AI4Bharat does not own the raw text ⚠️ | Read speech. Comes with ready test sets: *test-unknown* (new speakers, ~3 h) and *test-known* (~5 h), plus noisy versions of both. Noisy test sets are useful for checking reliability. |

Tier 1 total: **~420 hours**. That is plenty for a whisper-base fine-tune.

### Tier 2: add next, for real-world speech

| Dataset | Tamil size | Licence | Notes |
|---|---|---|---|
| **IndicVoices** (AI4Bharat, [HF](https://huggingface.co/datasets/ai4bharat/IndicVoices), [paper](https://arxiv.org/pdf/2403.01926)) | Not confirmed. One community model reports using ~800 h of Tamil from it ⚠️ | CC BY 4.0 (credit required). Possibly gated ⚠️ | **Natural, unscripted and conversational speech** from many districts: closest to how real users talk. Test speakers are kept separate from training speakers. |
| **SPRING-INX Tamil** (IIT Madras, [page](https://asr.iitm.ac.in/dataset), [paper](https://arxiv.org/pdf/2310.14654)) | ~374 h (225.7 h release 1 + 148.4 h release 2) ⚠️ column labels unclear | CC BY 4.0 | Mostly mock conversations and monologues. Good for variety. |

### Tier 3: only after the audit, with care

| Dataset | Tamil size | Licence | Why careful |
|---|---|---|---|
| **Shrutilipi Tamil** (AI4Bharat, [site](https://ai4bharat.iitm.ac.in/?p=1129)) | ~794 h | ⚠️ AI4Bharat says CC0, AIKosh says CC BY 4.0; the All India Radio audio rights are not settled | Audio and text were *matched automatically* from radio news, so some transcripts will not fit the audio. Formal news-reading style only. Use only clips that pass `audit_data.py`. |

### Test-only sets (never train on these)

| Dataset | Licence | Why |
|---|---|---|
| **FLEURS `ta_in`** (Google, [card](https://huggingface.co/datasets/google/fleurs/blob/main/README.md?code=true)) | CC BY 4.0 | The standard Tamil benchmark everyone reports. ~400–586 test clips ⚠️ |
| **Kathbath test-unknown + noisy-test-unknown** | as above | New speakers, clean and noisy. |
| **OpenSLR 65** (Google, [mirror card](https://huggingface.co/datasets/ylacombe/google-tamil)) | CC BY-SA 4.0 | Share-alike licence could bind the model if trained on; fine for testing. ~7 h. |

### Skip

| Dataset | Why |
|---|---|
| Microsoft Speech Corpus / MUCS 2021 Tamil ([MS page](https://www.microsoft.com/en-us/research/event/interspeech-2018-special-session-low-resource-speech-recognition-challenge-indian-languages/tools)) | Research-only, non-commercial. |
| [Speech-data Tamil](https://huggingface.co/datasets/Speech-data/Tamil-Speech-Dataset) | CC BY-NC-ND (no commercial use, no derivatives). |
| [TamilVoiceCorpus](https://huggingface.co/datasets/ragunath-ravi/TamilVoiceCorpus) | Scraped from the web, unclear source. |
| SpeechDataAI conversational | Paid commercial licence. |

## What already exists (ADA's competition)

Self-reported WER from model cards and papers. Test sets and text
normalization differ, so these are **not directly comparable**.

| Model | Size | Reported Tamil WER |
|---|---|---|
| Stock Whisper small, zero-shot | 244M | ~35% FLEURS ([card](https://huggingface.co/sgangireddy/whisper-small-ta/blob/main/README.md)) |
| [vasista22/whisper-tamil-small](https://huggingface.co/vasista22/whisper-tamil-small/raw/refs%2Fpr%2F2/README.md) | 244M | ~9.1% FLEURS, ~8.0% Common Voice 11 |
| [steja/whisper-small-tamil](https://huggingface.co/steja/whisper-small-tamil/blob/main/README.md) | 244M | ~15.0% FLEURS |
| [bhuvanesh25/whis-tam-small](https://huggingface.co/bhuvanesh25/whis-tam-small/blob/main/README.md) | 244M | ~11.2% Common Voice 11 |
| IndicWhisper (AI4Bharat, [Vistaar paper](https://arxiv.org/pdf/2305.15386), [repo](https://github.com/AI4Bharat/vistaar)) | — | ~24% Kathbath, ~29% Common Voice ⚠️ read from a garbled table |
| [mahimairaja/whisper-base-tamil](https://huggingface.co/mahimairaja/whisper-base-tamil/blob/main/README.md) | 74M | ~28.2% Common Voice 13 (only 500 steps) |
| [rajagowthamr/tamil-conformer-transducer-74m](https://huggingface.co/rajagowthamr/tamil-conformer-transducer-74m) | 74M | Not read. Trained on ~970 h (IndicVoices + Kathbath) per search summary ⚠️ **Check this first: it is the closest competitor in ADA's size class.** |

**ADA's opening:** good 244M Tamil models exist; good *small, CPU-fast,
easy-to-install* Tamil models mostly do not. ADA-STT v0.1 targets
**whisper-base (74M), int8 on CPU**, trained on Tier 1 data, measured on
FLEURS + Kathbath test-unknown (clean and noisy) + Common Voice test, against
stock Whisper and the models above, with the same normalization.

## Quality rules for ADA's data

1. **Validated clips only** (Common Voice `validated`/`train.tsv`, Kathbath clean train).
2. **Speakers never shared between train and test.** `prepare_data.py` splits by speaker for CSV input; `audit_data.py` reports any overlap.
3. **No test sentence in train.** Checked by `audit_data.py`.
4. **One convention for numbers and English words.** Tamil speech mixes in English words and numbers. Decide once (e.g. numbers as digits, English words in Latin script as spoken) and check every source follows it. `audit_data.py` flags digits and Latin letters so the sources can be compared.
5. **Drop mismatched transcripts.** Clips with a speaking rate far from the median almost always have a wrong or cut-off transcript.
6. **1–30 s clips only.** Whisper's window is 30 s.
7. **Balance sources and speakers.** Don't let one speaker or one dataset dominate.
8. **Record every source's licence and version** in the model card.

## Disk space on the laptop

- Common Voice Tamil (mp3): a few GB. Kathbath Tamil: likely tens of GB ⚠️.
- Training no longer stores audio features (they are computed per batch), so
  extra space is only needed for checkpoints (~1–3 GB each for whisper-base/small).
- Keep **at least 100 GB free** for Tier 1 + Tier 2.

## Tomorrow's checklist

1. Laptop setup (`docs/SETUP_WINDOWS.md`) and `python scripts/check_gpu.py`.
2. Create a Mozilla Data Collective account and download **Common Voice Tamil**.
3. `python scripts/stt/prepare_data.py --format commonvoice --src <cv-ta folder> --out data/stt/cv-ta`
4. `python scripts/stt/audit_data.py --data data/stt/cv-ta`, and send me the summary.
5. Baseline: `python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta --models tiny base small`
6. Then add Kathbath Tamil and the first smoke-test training run.
