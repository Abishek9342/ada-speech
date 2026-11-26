# ADA roadmap

One component at a time. The next one starts only when the previous one has real users.

## Phase 0: decide (before training)
- [ ] Pick ONE language / accent / domain where stock Whisper is weak.
- [ ] Find data and check its licence (Common Voice is CC0; check every other source).
- [ ] Measure the baseline: run `scripts/stt/evaluate.py` with stock `tiny`, `base`, `small`.
      This shows the gap ADA has to close.

## Phase 1: ADA-STT v0.1 (~6-8 weeks part-time)
- [ ] GPU setup works (`scripts/check_gpu.py`).
- [ ] Data prepared; smoke-test training run passes.
- [ ] Full fine-tune of whisper-base and whisper-small; compare WER.
- [ ] Export to int8 CTranslate2; check WER loss from quantization is small.
- [ ] CPU speed: RTF < 1.0 on a 4-core laptop.
- [ ] Reliability: test silence, music, noise, very short and very long clips.
- [ ] Publish model on Hugging Face with an honest model card.
- [ ] Publish package on PyPI; simple demo (Hugging Face Space).
- [ ] Share it where speakers of that language are; collect feedback.

## Phase 2: ADA-TTS v0.1
- Fine-tune a VITS/Piper voice for the same language (5-20 h of one clean speaker).
- Text normalization for numbers, dates, abbreviations.
- Export to ONNX for CPU.

## Phase 3: ADA-T2T v0.1
- Fine-tune T5/mBART for the chosen task (e.g. translation to/from English).
- Export to CTranslate2 int8 for CPU.

## Later
- Speech-to-speech by chaining STT → T2T → TTS.
- One shared model across tasks, only once the separate ones work well.
