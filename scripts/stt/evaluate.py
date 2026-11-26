"""Measure what end users will actually get: accuracy AND speed on CPU.

Compares one or more models on the same test set and prints WER and
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
from ada.text import wer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path, help="a split folder with metadata.csv")
    p.add_argument("--models", required=True, nargs="+", help="Whisper sizes, HF ids or CT2 folders")
    p.add_argument("--language", default=None, help="language code, e.g. ta, hi, en")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--limit", type=int, default=200, help="max clips (CPU eval is slow)")
    args = p.parse_args()

    with open(args.data / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.limit]
    refs = [r["transcription"] for r in rows]
    print(f"Evaluating on {len(rows)} clips from {args.data} ({args.device})\n")

    compute_type = "int8" if args.device == "cpu" else "float16"
    print(f"{'model':45} {'WER %':>7} {'RTF':>6}")
    for name in args.models:
        model = Transcriber(name, device=args.device, compute_type=compute_type)
        preds, audio_s, elapsed_s = [], 0.0, 0.0
        for r in rows:
            res = model.transcribe(str(args.data / r["file_name"]), language=args.language)
            preds.append(res.text)
            audio_s += res.audio_seconds
            elapsed_s += res.elapsed_seconds
        rtf = elapsed_s / audio_s if audio_s else float("nan")
        print(f"{name:45} {100 * wer(refs, preds):7.2f} {rtf:6.2f}")


if __name__ == "__main__":
    main()
