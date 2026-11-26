"""Turn a speech dataset into the simple layout every ADA-STT script reads.

Output layout (Hugging Face "audiofolder"):

    data/stt/<name>/
        train/metadata.csv   + audio files
        test/metadata.csv    + audio files

metadata.csv has two columns: file_name, transcription

Supported inputs:
  --format commonvoice : an extracted Mozilla Common Voice language folder
                         (contains clips/ and train.tsv, test.tsv, ...)
  --format csv         : your own data, a CSV with columns `path,text`
                         (paths relative to the CSV); split into train/test here

Examples:
  python scripts/stt/prepare_data.py --format commonvoice --src D:/cv/ta --out data/stt/cv-ta
  python scripts/stt/prepare_data.py --format csv --src D:/my_audio/labels.csv --out data/stt/mine
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def _write_split(rows: list[tuple[Path, str]], out_dir: Path, max_rows: int | None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_rows:
        rows = rows[:max_rows]
    written = 0
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "transcription"])
        for src, text in rows:
            text = " ".join(text.split())
            if not src.exists() or not text:
                continue
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            writer.writerow([src.name, text])
            written += 1
    return written


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def from_commonvoice(src: Path) -> dict[str, list[tuple[Path, str]]]:
    clips = src / "clips"
    splits = {}
    for split, tsv in (("train", "train.tsv"), ("test", "test.tsv")):
        rows = _read_tsv(src / tsv)
        splits[split] = [(clips / r["path"], r["sentence"]) for r in rows]
    return splits


def from_csv(src: Path, test_fraction: float, seed: int) -> dict[str, list[tuple[Path, str]]]:
    with open(src, encoding="utf-8") as f:
        rows = [(src.parent / r["path"], r["text"]) for r in csv.DictReader(f)]
    # Drop unusable rows BEFORE splitting, so the test set never ends up empty.
    rows = [(p, t) for p, t in rows if t.strip() and p.exists()]
    if len(rows) < 2:
        raise SystemExit(f"Need at least 2 usable rows, found {len(rows)}.")
    random.Random(seed).shuffle(rows)
    n_test = max(1, int(len(rows) * test_fraction))
    return {"test": rows[:n_test], "train": rows[n_test:]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", required=True, choices=["commonvoice", "csv"])
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--test-fraction", type=float, default=0.1, help="csv format only")
    p.add_argument("--max-train", type=int, default=None, help="cap rows, useful for a quick smoke test")
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.format == "commonvoice":
        splits = from_commonvoice(args.src)
    else:
        splits = from_csv(args.src, args.test_fraction, args.seed)

    n_train = _write_split(splits["train"], args.out / "train", args.max_train)
    n_test = _write_split(splits["test"], args.out / "test", args.max_test)
    print(f"Wrote {n_train} train and {n_test} test examples to {args.out}")


if __name__ == "__main__":
    main()
