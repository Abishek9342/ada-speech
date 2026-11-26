"""Fine-tune a Whisper encoder-decoder into ADA-STT on your own GPU.

Run:  python scripts/stt/train.py --config configs/stt/whisper-small-8gb.yaml
Watch: tensorboard --logdir models/ada-stt-small

Quick smoke test before a long run (a few minutes, catches setup errors):
      python scripts/stt/train.py --config configs/stt/whisper-small-8gb.yaml --smoke
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ada.text import cer, wer

SAMPLE_RATE = 16_000


@dataclass
class SpeechCollator:
    """Turns raw audio into log-mel features and pads the label tokens."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # Log-mel features are computed here, per batch, instead of being stored:
        # stored features are padded to 30 s, which for a few hundred hours of
        # short clips would need hundreds of GB of disk.
        batch = self.processor.feature_extractor(
            [f["audio"]["array"] for f in features],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        labels = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
        # -100 = "ignore this position" in the loss.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # The model adds the start token itself; drop it if the tokenizer already did.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline works")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch
    from datasets import Audio, load_dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU found; training on CPU will be extremely slow.")

    processor = WhisperProcessor.from_pretrained(
        cfg["base_model"], language=cfg["language"], task=cfg["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"])
    model.generation_config.language = cfg["language"]
    model.generation_config.task = cfg["task"]
    model.generation_config.forced_decoder_ids = None
    if cfg["gradient_checkpointing"]:
        model.config.use_cache = False  # incompatible with checkpointing during training

    # ---- data ---------------------------------------------------------------
    ds = load_dataset("audiofolder", data_dir=cfg["data_dir"])
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    if args.smoke:
        ds["train"] = ds["train"].select(range(min(64, len(ds["train"]))))
        ds["test"] = ds["test"].select(range(min(16, len(ds["test"]))))

    max_seconds = cfg["max_audio_seconds"]
    max_label_tokens = cfg["generation_max_length"]

    # Only the text column is touched here, so no audio is decoded and nothing large is cached.
    ds = ds.map(
        lambda text: {"labels": processor.tokenizer(text).input_ids},
        input_columns=["transcription"],
        desc="tokenizing transcripts",
    )
    ds = ds.filter(lambda labels: len(labels) <= max_label_tokens, input_columns=["labels"])
    if "duration" in ds["train"].column_names:
        # Unknown durations (None/NaN) are kept; the feature extractor truncates at 30 s.
        ds = ds.filter(
            lambda d: d is None or math.isnan(d) or d <= max_seconds, input_columns=["duration"]
        )
    else:
        print("WARNING: no duration column; clips over 30 s are not filtered. "
              "Re-run prepare_data.py to add it.")
    print(f"train={len(ds['train'])}  test={len(ds['test'])}")

    # ---- metrics ------------------------------------------------------------
    def compute_metrics(pred) -> dict:
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        preds = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer(refs, preds), "cer": 100 * cer(refs, preds)}

    # ---- training -----------------------------------------------------------
    smoke = args.smoke
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"] + ("-smoke" if smoke else ""),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=0 if smoke else cfg["warmup_steps"],
        max_steps=20 if smoke else cfg["max_steps"],
        bf16=cfg["bf16"],
        fp16=cfg["fp16"],
        eval_strategy="steps",
        eval_steps=10 if smoke else cfg["eval_steps"],
        save_steps=20 if smoke else cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        logging_steps=5 if smoke else cfg["logging_steps"],
        predict_with_generate=True,
        generation_max_length=cfg["generation_max_length"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        remove_unused_columns=False,  # the collator needs the raw "audio" column
        seed=cfg["seed"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=SpeechCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    baseline = trainer.evaluate()
    print(f"Baseline WER before fine-tuning: {baseline['eval_wer']:.2f}")
    trainer.train()

    final_dir = Path(training_args.output_dir) / "final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    final = trainer.evaluate()
    print(f"WER before: {baseline['eval_wer']:.2f}  after: {final['eval_wer']:.2f}")
    print(f"Saved to {final_dir}. Next: scripts/stt/export_ct2.py to make it CPU-fast.")


if __name__ == "__main__":
    main()
