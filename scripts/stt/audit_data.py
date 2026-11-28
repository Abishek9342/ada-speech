"""Check a prepared dataset for quality problems BEFORE training.

Run:  python scripts/stt/audit_data.py --data data/stt/cv-ta
Writes data/stt/cv-ta/audit_flags.csv: the clips worth listening to.

What to do with the results:
  * test sentences/speakers seen in train -> the test score will look better than
    reality; re-split (prepare_data.py splits by speaker for csv input).
  * speaking-rate outliers -> usually a wrong or truncated transcript; listen, then drop.
  * digits / Latin letters -> decide ONE convention (e.g. numbers as Tamil words)
    and apply it to all transcripts, or the model learns to be inconsistent.
  * one speaker with a large share -> the model may overfit to that voice.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ada.stt.audit import audit, is_expected, load_split, real_leaks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path, help="folder with train/ and test/")
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--test-dirs", nargs="*", type=Path, default=[],
                   help="more test folders (with metadata.csv) to check train/dev against, e.g. "
                        "every source's test/ and test-known/ for a combined set, and FLEURS")
    args = p.parse_args()

    clips = {s: load_split(args.data / s, s) for s in ("train", "dev", "test", "test-known") if (args.data / s).exists()}
    extra = {f"{d.parent.name}/{d.name}": load_split(d, f"{d.parent.name}/{d.name}") for d in args.test_dirs}
    report = audit(clips, max_s=args.max_seconds, extra_tests=extra)

    print("== Size ==")
    for split, st in report.stats.items():
        print(f"{split:6} " + "  ".join(f"{k}={v}" for k, v in st.items()))

    if report.leaks:
        print("\n== Train/test leakage ==")
        for k, v in report.leaks.items():
            print(f"{k}: {v}" + ("  (expected)" if is_expected(k) else ""))
        bad = real_leaks(report.leaks)
        print(f"LEAK CHECK: {'FAIL ' + str(sorted(bad)) if bad else 'PASS'}")
    else:
        print("\n== Train/test leakage ==\nno held-out set found: pass --test-dirs\nLEAK CHECK: FAIL [no test set]")

    print("\n== Flags (clip may be counted under several reasons) ==")
    # Count per split: pooling them hid the fact that most flags were in train.
    for split in report.stats:
        n = report.stats[split]["clips"]
        reasons = Counter(re.sub(r"\d+(\.\d+)?", "N", f.reason).split(" (")[0]
                          for f in report.flags if f.clip.split == split)
        print(f"[{split}] {n} clips, {sum(reasons.values())} flags")
        for reason, count in reasons.most_common():
            print(f"  {count:7}  {reason}")

    out = args.data / "audit_flags.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file_name", "reason", "duration", "speaker", "transcription"])
        for fl in report.flags:
            c = fl.clip
            w.writerow([c.split, c.file_name, fl.reason, c.duration, c.speaker, c.text])
    print(f"\nWrote {len(report.flags)} flags to {out}. Listen to a sample before deciding.")


if __name__ == "__main__":
    main()
