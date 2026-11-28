"""Measure what end users will actually get: accuracy AND speed on CPU.

Compares one or more models on the same test set and prints WER, CER and
real-time factor (RTF < 1.0 = faster than real time).

Run:
  python scripts/stt/evaluate.py --data data/stt/cv-ta/test --language ta \
      --models small models/ada-stt-small-ct2
  # fixed, seeded clip list (evals/*.txt, made by make_eval_lists.py) + 95% CI:
  python scripts/stt/evaluate.py --list evals/cv_test_1000.txt --language ta --models base

A set whose references are all empty (evals/nonspeech: silence, noise, hum) reports the
HALLUCINATION RATE instead: the share of clips where the model wrote anything at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ada.text import normalize


def read_list(path: Path) -> tuple[Path, list[str]]:
    """evals/<name>.txt: first line '# data: <split folder>', then one file_name per line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# data:"):
        raise SystemExit(f"{path}: first line must be '# data: <split folder>'")
    data = Path(lines[0].split(":", 1)[1].strip())
    names = [ln.strip() for ln in lines[1:] if ln.strip() and not ln.startswith("#")]
    return data, names


def edit_counts(refs: list[str], preds: list[str], unit: str) -> list[tuple[int, int]]:
    """(errors, reference length) per clip after normalization, for bootstrap CIs."""
    import jiwer

    out = []
    for r, p in zip(refs, preds, strict=True):
        r, p = normalize(r), normalize(p)
        if not r:
            continue
        if unit == "char":
            m = jiwer.process_characters(r, p)
            n = len(r)
        else:
            m = jiwer.process_words(r, p)
            n = len(r.split())
        out.append((m.substitutions + m.deletions + m.insertions, n))
    return out


def rate_with_ci(counts: list[tuple[int, int]], n_boot: int = 1000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Corpus error rate (%) and a 95% bootstrap interval over clips."""
    if not counts:
        return float("nan"), float("nan"), float("nan")
    err = sum(e for e, _ in counts)
    tot = sum(n for _, n in counts)
    if n_boot <= 0:
        return 100 * err / tot, float("nan"), float("nan")
    rng = random.Random(seed)
    k = len(counts)
    boots = []
    for _ in range(n_boot):
        e = n = 0
        for _ in range(k):
            ce, cn = counts[rng.randrange(k)]
            e += ce
            n += cn
        boots.append(100 * e / n if n else 0.0)
    boots.sort()
    return 100 * err / tot, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, help="a split folder with metadata.csv")
    p.add_argument("--list", type=Path, help="evals/<name>.txt fixed clip list (instead of --data)")
    p.add_argument("--models", required=True, nargs="+", help="Whisper sizes, HF ids or CT2 folders")
    p.add_argument("--language", default=None, help="language code, e.g. ta, hi, en")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--limit", type=int, default=None,
                   help="max clips (first N rows; prefer a fixed --list)")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--exclude-digits", action="store_true",
                   help="skip clips whose reference contains digits (models trained on "
                        "number-as-words data are otherwise penalised for a style difference)")
    p.add_argument("--save-dir", type=Path, default=None,
                   help="write each model's transcripts here as <model>.csv, for error analysis")
    p.add_argument("--json-out", type=Path, default=None, help="append one JSON line per model")
    p.add_argument("--no-ci", action="store_true", help="skip bootstrap intervals")
    p.add_argument("--no-vad", action="store_true",
                   help="decode everything (VAD off): the raw model, e.g. for hallucination tests")
    args = p.parse_args()
    if bool(args.data) == bool(args.list):
        raise SystemExit("Give exactly one of --data or --list.")

    from ada.stt.inference import Transcriber

    data = args.data
    names = None
    if args.list:
        data, names = read_list(args.list)
    with open(data / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if names is not None:
        by_name = {r["file_name"]: r for r in rows}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise SystemExit(f"{len(missing)} listed clips not in {data}, e.g. {missing[:3]}")
        rows = [by_name[n] for n in names]
    if args.exclude_digits:
        rows = [r for r in rows if not any(ch.isdigit() for ch in r["transcription"])]
    if args.limit:
        rows = rows[: args.limit]
    refs = [r["transcription"] for r in rows]
    nonspeech = all(not normalize(t) for t in refs)
    label = str(args.list or data)
    print(f"Evaluating on {len(rows)} clips from {label} ({args.device})"
          + ("  [non-speech: hallucination rate]" if nonspeech else "") + "\n")

    compute_type = "int8" if args.device == "cpu" else "float16"
    if nonspeech:
        print(f"{'model':45} {'halluc %':>9} {'RTF':>6}")
    else:
        print(f"{'model':45} {'WER %':>7} {'CER %':>7} {'CER 95% CI':>15} {'RTF':>6}")
    for name in args.models:
        model = Transcriber(name, device=args.device, compute_type=compute_type)
        preds, audio_s, elapsed_s = [], 0.0, 0.0
        for r in rows:
            res = model.transcribe(
                str(data / r["file_name"]), language=args.language, beam_size=args.beam_size,
                vad_filter=not args.no_vad,
            )
            preds.append(res.text)
            audio_s += res.audio_seconds
            elapsed_s += res.elapsed_seconds
        rtf = elapsed_s / audio_s if audio_s else float("nan")
        result = {"set": label, "model": name, "clips": len(rows), "rtf": round(rtf, 3),
                  "vad": not args.no_vad}
        if nonspeech:
            rate = 100 * sum(bool(normalize(pr)) for pr in preds) / max(1, len(preds))
            result["hallucination_pct"] = round(rate, 2)
            print(f"{name:45} {rate:9.2f} {rtf:6.2f}", flush=True)
        else:
            w = rate_with_ci(edit_counts(refs, preds, "word"), n_boot=0 if args.no_ci else 1000)
            c = rate_with_ci(edit_counts(refs, preds, "char"), n_boot=0 if args.no_ci else 1000)
            ci = "" if args.no_ci else f"{c[1]:6.2f}-{c[2]:6.2f}"
            result.update(wer=round(w[0], 2), cer=round(c[0], 2))
            if not args.no_ci:
                result.update(cer_ci=[round(c[1], 2), round(c[2], 2)])
            print(f"{name:45} {w[0]:7.2f} {c[0]:7.2f} {ci:>15} {rtf:6.2f}", flush=True)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            safe = name.strip("/\\").replace("/", "__").replace("\\", "__").replace(":", "")
            with open(args.save_dir / f"{safe}.csv", "w", newline="", encoding="utf-8") as f:
                w_ = csv.writer(f)
                w_.writerow(["file_name", "reference", "prediction"])
                w_.writerows((r["file_name"], ref, pred) for r, ref, pred in zip(rows, refs, preds))


if __name__ == "__main__":
    main()
