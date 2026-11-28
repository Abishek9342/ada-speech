"""Fine-tune a Whisper encoder-decoder into ADA-STT on your own GPU.

Run:  python scripts/stt/train.py --config configs/stt/whisper-small-8gb.yaml
Watch: tensorboard --logdir models/ada-stt-small

Quick smoke test before a long run (a few minutes, catches setup errors):
      python scripts/stt/train.py --config configs/stt/whisper-small-8gb.yaml --smoke
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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


def source_of(file_name: str) -> str:
    """'../../kb2-ta/dev/x.flac' -> 'kb2-ta' (combined sets); plain 'x.flac' -> ''."""
    parts = [p for p in Path(file_name).parts if p not in ("..", ".")]
    return parts[0] if len(parts) > 2 else ""


def count_sources(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[source_of(r["file_name"])] = counts.get(source_of(r["file_name"]), 0) + 1
    return counts


def sample_per_source(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Up to n rows from each source, seeded, so every corpus has a say in checkpoint choice."""
    import random

    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(source_of(r["file_name"]), []).append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for src in sorted(by_src):
        group = by_src[src]
        out.extend(rng.sample(group, n) if len(group) > n else group)
    return out


def write_manifest(out_dir: Path, config_path: Path, cfg: dict, sizes: dict[str, int]) -> None:
    """How this model was made: config, git commit, data hashes. Becomes the model card later."""
    import hashlib
    import json
    import subprocess
    import time

    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]

    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(root), *a], check=False, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    data = {}
    for split in ("train", "dev", "test"):
        meta = Path(cfg["data_dir"]) / split / "metadata.csv"
        if meta.exists():
            raw = meta.read_bytes()
            data[split] = {"rows": raw.count(b"\n") - 1, "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_file": str(config_path),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "data": data,
        "used_after_filters": sizes,
    }
    try:
        import torch
        import transformers

        manifest["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    except ImportError:
        pass
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline works")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in output_dir (after a crash or "
                             "a cooling pause); skips the baseline eval")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch
    from datasets import Audio, Dataset, DatasetDict
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
    # SpecAugment (built into Whisper, off by default): masks random time steps / frequency bands
    # of the log-mel input during training only. Helps most when there are few speakers.
    if cfg.get("spec_augment"):
        sa = cfg["spec_augment"]
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = sa.get("mask_time_prob", 0.05)
        model.config.mask_time_length = sa.get("mask_time_length", 10)
        model.config.mask_feature_prob = sa.get("mask_feature_prob", 0.0)
        model.config.mask_feature_length = sa.get("mask_feature_length", 10)
        print(f"SpecAugment on: {sa}")
    if cfg["gradient_checkpointing"]:
        model.config.use_cache = False  # incompatible with checkpointing during training

    # ---- data ---------------------------------------------------------------
    # Read metadata.csv directly (not the "audiofolder" loader, which breaks with some
    # pyarrow versions). Audio is decoded lazily, batch by batch.
    def load_split(split: str, per_source: int | None = None) -> Dataset:
        folder = Path(cfg["data_dir"]) / split
        with open(folder / "metadata.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if per_source:
            rows = sample_per_source(rows, per_source, cfg["seed"])
            print(f"{split}: per-source sample " + ", ".join(
                f"{s}={n}" for s, n in sorted(count_sources(rows).items())))
        return Dataset.from_dict({
            "audio": [str(folder / r["file_name"]) for r in rows],
            "transcription": [r["transcription"] for r in rows],
            "duration": [float(r["duration"]) if r.get("duration") else None for r in rows],
        })

    # Validate on dev when the data has one, so the best checkpoint is never picked on test.
    eval_split = "dev" if (Path(cfg["data_dir"]) / "dev" / "metadata.csv").exists() else "test"
    # eval_per_source: N dev clips from EACH source of a combined set (else the biggest source,
    # Common Voice, decides which checkpoint wins).
    per_source = None if args.smoke else cfg.get("eval_per_source")
    ds = DatasetDict({"train": load_split("train"), eval_split: load_split(eval_split, per_source)})
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    # Generating on a full dev/test set every eval_steps is slow; a fixed random sample is enough.
    eval_max = 16 if args.smoke else cfg.get("eval_max_samples")
    if eval_max and not per_source and len(ds[eval_split]) > eval_max:
        ds[eval_split] = ds[eval_split].shuffle(seed=cfg["seed"]).select(range(eval_max))
    if args.smoke:
        ds["train"] = ds["train"].select(range(min(64, len(ds["train"]))))

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
    print(f"train={len(ds['train'])}  {eval_split}={len(ds[eval_split])} (used for validation)")

    # ---- metrics ------------------------------------------------------------
    def compute_metrics(pred) -> dict:
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        preds = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer(refs, preds), "cer": 100 * cer(refs, preds)}

    # ---- training -----------------------------------------------------------
    smoke = args.smoke
    # transformers 5 dropped `logging_dir` and writes TensorBoard logs to ./runs unless this is set,
    # so `tensorboard --logdir <output_dir>` found nothing. A value set by the user wins.
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR",
                          str(Path(cfg["output_dir"] + ("-smoke" if smoke else "")) / "runs"))
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
        # CER by default: steadier than WER for Tamil (one wrong suffix = one wrong long word).
        metric_for_best_model=cfg.get("metric_for_best_model", "cer"),
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
        eval_dataset=ds[eval_split],
        data_collator=SpeechCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    write_manifest(Path(training_args.output_dir), args.config, cfg,
                   {"train": len(ds["train"]), eval_split: len(ds[eval_split])})
    has_ckpt = any(Path(training_args.output_dir).glob("checkpoint-*"))
    if args.resume and has_ckpt:
        print("Resuming from the latest checkpoint (baseline eval skipped).")
        baseline = None
        trainer.train(resume_from_checkpoint=True)
    else:
        if args.resume:
            print("No checkpoint found; starting from scratch.")
        baseline = trainer.evaluate()
        print(f"Baseline WER before fine-tuning: {baseline['eval_wer']:.2f}  "
              f"CER: {baseline['eval_cer']:.2f}")
        trainer.train()

    final_dir = Path(training_args.output_dir) / "final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    for name in ("run_manifest.json", "config.yaml"):
        src = Path(training_args.output_dir) / name
        if src.exists():
            (final_dir / name).write_bytes(src.read_bytes())
    final = trainer.evaluate()
    before = f"{baseline['eval_wer']:.2f}" if baseline else "n/a (resumed)"
    print(f"WER before: {before}  after: {final['eval_wer']:.2f}  CER after: {final['eval_cer']:.2f}"
          f"  (best checkpoint: {trainer.state.best_model_checkpoint})")
    print(f"Saved to {final_dir}. Next: scripts/stt/export_ct2.py to make it CPU-fast.")


if __name__ == "__main__":
    main()
