"""Convert a trained ADA-STT model into int8 CTranslate2 format for fast CPU use.

This is what end users download: typically ~4x smaller than the training
checkpoint and several times faster on CPU.

Run:  python scripts/stt/export_ct2.py --model models/ada-stt-small/final --out models/ada-stt-small-ct2
"""

from __future__ import annotations

import argparse
from pathlib import Path

COPY_FILES = ["tokenizer.json", "preprocessor_config.json", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
              "normalizer.json", "generation_config.json"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path, help="folder saved by train.py")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--quantization", default="int8", choices=["int8", "int8_float16", "float16", "float32"])
    p.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = p.parse_args()

    from ctranslate2.converters import TransformersConverter

    copy = [f for f in COPY_FILES if (args.model / f).exists()]
    if "tokenizer.json" not in copy:
        raise SystemExit("tokenizer.json missing: faster-whisper needs it. Re-save the processor.")

    TransformersConverter(str(args.model), copy_files=copy).convert(
        str(args.out), quantization=args.quantization, force=args.force
    )
    size_mb = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1024**2
    print(f"Exported to {args.out} ({size_mb:.0f} MB, {args.quantization}).")
    print(f"Try it:  ada-stt some_audio.wav --model {args.out} --stats")


if __name__ == "__main__":
    main()
