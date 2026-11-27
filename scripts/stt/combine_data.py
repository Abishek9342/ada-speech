"""Combine several prepared ADA-STT datasets into one training set, without copying audio.

    python scripts/stt/combine_data.py --sources data/stt/cv-ta data/stt/kb-ta --out data/stt/combo-ta

Writes <out>/train/metadata.csv and <out>/dev/metadata.csv whose file_name entries are paths
relative to that folder (e.g. ../../cv-ta/train/x.mp3), so train.py and audit_data.py read them
as usual. Test sets are NOT combined: evaluate each source's own test/ folder, so every number
stays comparable with published results.

Options:
  --max-per-speaker N        cap train clips per speaker (keeps one voice from dominating; seeded)
  --exclude-sentences-of D.. drop train/dev clips whose normalized sentence appears in any of these
                             test folders (sentences can be shared ACROSS corpora)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def read_rows(folder: Path) -> list[dict[str, str]]:
    with open(folder / "metadata.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cap_per_speaker(rows: list[dict[str, str]], cap: int, seed: int) -> list[dict[str, str]]:
    by_spk: dict[str, list[dict[str, str]]] = {}
    for i, r in enumerate(rows):
        by_spk.setdefault(r.get("speaker") or f"__row{i}", []).append(r)
    rng = random.Random(seed)
    kept: list[dict[str, str]] = []
    for spk in sorted(by_spk):
        group = by_spk[spk]
        if len(group) > cap:
            group = rng.sample(group, cap)
        kept.extend(group)
    return kept


def combine(sources: list[Path], out: Path, max_per_speaker: int | None, seed: int,
            exclude_from: list[Path] | None = None) -> dict:
    from ada.text import normalize

    banned = {normalize(r["transcription"]) for d in (exclude_from or []) for r in read_rows(d)}
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "dev"):
        dst = out / split
        rows_out: list[dict[str, str]] = []
        for src in sources:
            folder = src / split
            if not (folder / "metadata.csv").exists():
                continue
            rows = read_rows(folder)
            if banned:
                before = len(rows)
                rows = [r for r in rows if normalize(r["transcription"]) not in banned]
                if len(rows) != before:
                    counts.setdefault("excluded_test_sentences", {})[f"{src.name}/{split}"] = (
                        before - len(rows))
            if split == "train" and max_per_speaker:
                rows = cap_per_speaker(rows, max_per_speaker, seed)
            rel = Path(os.path.relpath(folder, dst))
            for r in rows:
                rows_out.append({**r, "file_name": (rel / r["file_name"]).as_posix()})
            counts.setdefault(split, {})[src.name] = len(rows)
        if not rows_out:
            continue
        dst.mkdir(parents=True, exist_ok=True)
        random.Random(seed).shuffle(rows_out)
        with open(dst / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_name", "transcription", "duration", "speaker"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_out)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", required=True, nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-per-speaker", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-sentences-of", nargs="*", type=Path, default=[])
    args = p.parse_args()
    counts = combine(args.sources, args.out, args.max_per_speaker, args.seed,
                     args.exclude_sentences_of)
    for split, per_src in counts.items():
        print(f"{split}: {sum(per_src.values())}  " + "  ".join(f"{k}={v}" for k, v in per_src.items()))
    print(f"Wrote {args.out}. Run audit_data.py on it before training.")


if __name__ == "__main__":
    main()
