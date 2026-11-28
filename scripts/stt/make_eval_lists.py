"""Write fixed, seeded evaluation clip lists to evals/ (file names only, no audio or text).

Every run is scored on the SAME clips, so differences between runs are real and not
"which 300 rows came first". Lists are committed; audio stays in data/ (not committed).

  python scripts/stt/make_eval_lists.py \
      cv_test_1000=data/stt/cv2-ta/test:1000 \
      kb_unseen_test=data/stt/kb2-ta/test:all \
      fleurs_nodigits=data/stt/fleurs-ta/test:all:nodigits

Spec: <name>=<split folder>:<N or all>[:nodigits]
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def make_list(folder: Path, n: str, nodigits: bool, seed: int) -> list[str]:
    with open(folder / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if nodigits:
        rows = [r for r in rows if not any(ch.isdigit() for ch in r["transcription"])]
    names = sorted(r["file_name"] for r in rows)
    if n != "all" and int(n) < len(names):
        names = sorted(random.Random(seed).sample(names, int(n)))
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+")
    p.add_argument("--out", type=Path, default=Path("evals"))
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.specs:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        # Windows drive letters ("C:\...") would split too; rejoin them.
        if len(parts[0]) == 1 and len(parts) > 2:
            parts = [parts[0] + ":" + parts[1], *parts[2:]]
        folder, n = Path(parts[0]), parts[1]
        nodigits = len(parts) > 2 and parts[2] == "nodigits"
        names = make_list(folder, n, nodigits, args.seed)
        header = [f"# data: {folder.as_posix()}",
                  f"# clips: {len(names)}  sample: {n}  seed: {args.seed}"
                  + ("  digits excluded" if nodigits else "")]
        (args.out / f"{name}.txt").write_text("\n".join(header + names) + "\n", encoding="utf-8")
        print(f"{name}: {len(names)} clips from {folder}")


if __name__ == "__main__":
    main()
