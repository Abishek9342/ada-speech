"""Turn evaluate.py --json-out lines into one Markdown table (models x eval sets).

  python scripts/stt/results_table.py --in logs/day2_results.jsonl --out logs/day2_results.md

Cells: CER % [95% CI] / WER %; non-speech sets: hallucination %. The last line of a
(model, set) pair wins, so re-running an eval replaces its cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(lines: list[dict]) -> str:
    cells: dict[tuple[str, str], dict] = {}
    models: list[str] = []
    sets: list[str] = []
    for r in lines:
        s = Path(r["set"]).stem + ("" if r.get("vad", True) else " (no VAD)")
        if r["model"] not in models:
            models.append(r["model"])
        if s not in sets:
            sets.append(s)
        cells[(r["model"], s)] = r
    out = ["| model | " + " | ".join(sets) + " | mean RTF |", "|---" * (len(sets) + 2) + "|"]
    for m in models:
        row, rtfs = [], []
        for s in sets:
            r = cells.get((m, s))
            if not r:
                row.append("")
                continue
            if "hallucination_pct" in r:
                row.append(f"{r['hallucination_pct']:.1f}% halluc")
            else:
                ci = r.get("cer_ci")
                row.append(f"**{r['cer']:.2f}**" + (f" [{ci[0]:.1f}-{ci[1]:.1f}]" if ci else "")
                           + f" / {r['wer']:.1f}")
                rtfs.append(r["rtf"])
        rtf = f"{sum(rtfs) / len(rtfs):.2f}" if rtfs else ""
        out.append(f"| {m} | " + " | ".join(row) + f" | {rtf} |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    lines = [json.loads(ln) for ln in args.inp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    table = build(lines)
    args.out.write_text("# Day 2 fixed-list results\n\nCER % [95% bootstrap CI] / WER %, laptop CPU int8, "
                        "beam 5, VAD on unless marked. Lists in evals/.\n\n" + table + "\n",
                        encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
