"""Measure what end users will actually get: accuracy AND speed on CPU.

Compares one or more models on the same test set and prints WER, CER and
real-time factor (RTF < 1.0 = faster than real time).

Run:
  python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta \
      --models small models/ada-stt-small-ct2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ada.stt.inference import Transcriber
from ada.text import cer, wer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path, help="a split folder with metadata.csv")
    p.add_argument("--models", required=True, nargs="+", help="Whisper sizes, HF ids or CT2 folders")
    p.add_argument("--language", default=None, help="language code, e.g. ta, hi, en")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--limit", type=int, default=200, help="max clips (CPU eval is slow)")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--exclude-digits", action="store_true",
                   help="skip clips whose reference contains digits (models trained on "
                        "number-as-words data are otherwise penalised for a style difference)")
    p.add_argument("--save-dir", type=Path, default=None,
                   help="write each model's transcripts here as <model>.csv, for error analysis")
    args = p.parse_args()

    with open(args.data / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.exclude_digits:
        rows = [r for r in rows if not any(ch.isdigit() for ch in r["transcription"])]
    rows = rows[: args.limit]
    refs = [r["transcription"] for r in rows]
    print(f"Evaluating on {len(rows)} clips from {args.data} ({args.device})\n")

    compute_type = "int8" if args.device == "cpu" else "float16"
    print(f"{'model':45} {'WER %':>7} {'CER %':>7} {'RTF':>6}")
    for name in args.models:
        model = Transcriber(name, device=args.device, compute_type=compute_type)
        preds, audio_s, elapsed_s = [], 0.0, 0.0
        for r in rows:
            res = model.transcribe(
                str(args.data / r["file_name"]), language=args.language, beam_size=args.beam_size
            )
            preds.append(res.text)
            audio_s += res.audio_seconds
            elapsed_s += res.elapsed_seconds
        rtf = elapsed_s / audio_s if audio_s else float("nan")
        print(f"{name:45} {100 * wer(refs, preds):7.2f} {100 * cer(refs, preds):7.2f} {rtf:6.2f}",
              flush=True)
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            safe = name.strip("/\\").replace("/", "__").replace("\\", "__").replace(":", "")
            with open(args.save_dir / f"{safe}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file_name", "reference", "prediction"])
                w.writerows((r["file_name"], ref, pred) for r, ref, pred in zip(rows, refs, preds))


if __name__ == "__main__":
    main()
