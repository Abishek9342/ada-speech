"""Turn a speech dataset into the simple layout every ADA-STT script reads.

Output layout (Hugging Face "audiofolder"):

    data/stt/<name>/
        train/metadata.csv   + audio files
        dev/metadata.csv     + audio files   (only when the source has one, e.g. Common Voice)
        test/metadata.csv    + audio files

metadata.csv columns: file_name, transcription, duration (seconds), speaker
Run scripts/stt/audit_data.py on the output before training.

Supported inputs:
  --format commonvoice : an extracted Mozilla Common Voice language folder
                         (contains clips/ and train.tsv, test.tsv, ...)
  --format fleurs      : a Google FLEURS language folder, e.g. data/ta_in/
                         (contains train/dev/test .tsv and audio/<split>/*.wav;
                         splits whose files are missing are skipped)
  --format mile        : IISc-MILE Tamil ASR corpus (OpenSLR 127): a folder containing
                         train/ and test/, each with audio_files/*.wav and
                         trans_files/*.txt (same base names). The official test reuses
                         train speakers, so it goes to test-known/; --holdout-speakers
                         unseen train speakers become dev/ and test/.
  --format kathbath    : AI4Bharat Kathbath language folder from Hugging Face, e.g.
                         .../Kathbath/tamil (train-*.parquet, valid-*.parquet; FLAC inside).
                         dev/ and test/ = held-out train speakers (unseen voices);
                         official valid (seen speakers) goes to test-known/.
  --format csv         : your own data, a CSV with columns `path,text`
                         and optionally `speaker` (paths relative to the CSV);
                         split into train/test here, keeping each speaker in one split

Examples:
  python scripts/stt/prepare_data.py --format commonvoice --src D:/cv/ta --out data/stt/cv-ta
  python scripts/stt/prepare_data.py --format fleurs --src data/fleurs/data/ta_in --out data/stt/fleurs-ta
  python scripts/stt/prepare_data.py --format mile --src D:/raw/mile_tamil_asr_corpus --out data/stt/mile-ta
  python scripts/stt/prepare_data.py --format csv --src D:/my_audio/labels.csv --out data/stt/mine
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

Row = tuple[Path, str, str]  # (audio path, transcript, speaker id or "")


def _duration(path: Path) -> str:
    """Seconds, read from the file header only (fast). Empty if unreadable."""
    try:
        import soundfile

        return f"{soundfile.info(str(path)).duration:.2f}"
    except (ImportError, RuntimeError, OSError):  # missing lib, unreadable or unsupported file
        return ""


def _write_split(rows: list[Row], out_dir: Path, max_rows: int | None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_rows:
        rows = rows[:max_rows]
    written = 0
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "transcription", "duration", "speaker"])
        for src, text, speaker in rows:
            text = " ".join(text.split())
            if not src.exists() or not text:
                continue
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            writer.writerow([src.name, text, _duration(dst), speaker])
            written += 1
    return written


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def from_commonvoice(src: Path) -> dict[str, list[Row]]:
    clips = src / "clips"
    splits = {}
    # dev is kept as its own split: train.py validates on it, so test is only used at the end.
    for split, tsv in (("train", "train.tsv"), ("dev", "dev.tsv"), ("test", "test.tsv")):
        if split == "dev" and not (src / tsv).exists():
            continue
        rows = _read_tsv(src / tsv)
        splits[split] = [(clips / r["path"], r["sentence"], r.get("client_id", "")) for r in rows]
    return splits


# FLEURS .tsv files have no header. Columns:
# id, file_name, raw_transcription, transcription, characters, num_samples, gender
FLEURS_COLUMNS = ["id", "file_name", "raw_transcription", "transcription", "characters",
                  "num_samples", "gender"]


def from_fleurs(src: Path) -> dict[str, list[Row]]:
    """FLEURS gives no speaker ids, so the speaker column stays empty.

    `raw_transcription` (with punctuation and casing) is used; ADA's normalizer
    removes punctuation before scoring, so results match the plain version.
    """
    splits: dict[str, list[Row]] = {}
    for split in ("train", "test"):
        tsv, audio_dir = src / f"{split}.tsv", src / "audio" / split
        if not tsv.exists() or not audio_dir.exists():
            continue
        with open(tsv, encoding="utf-8") as f:
            rows = csv.DictReader(f, fieldnames=FLEURS_COLUMNS, delimiter="\t",
                                  quoting=csv.QUOTE_NONE)
            splits[split] = [(audio_dir / r["file_name"], r["raw_transcription"], "") for r in rows]
    if not splits:
        raise SystemExit(f"No FLEURS split found in {src} (need <split>.tsv and audio/<split>/).")
    return splits


def _find_split_dir(root: Path, split: str) -> Path | None:
    """The archive may wrap the splits in an extra folder; search a few levels down."""
    for candidate in [root / split, *sorted(root.glob(f"*/{split}")), *sorted(root.glob(f"*/*/{split}"))]:
        if (candidate / "audio_files").is_dir() and (candidate / "trans_files").is_dir():
            return candidate
    return None


def _mile_speaker(stem: str) -> str:
    """IISc-MILE names files <SOURCE>_<SPEAKER>_<UTTERANCE>, e.g. MILE_0000252_0000170."""
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else ""


def from_mile(src: Path, holdout_speakers: int = 16) -> dict[str, list[Row]]:
    """IISc-MILE train/ and test/ folders.

    The published test set reuses train speakers (all 430 test speakers are in train), so it
    cannot measure new voices. Like Kathbath: it is kept as test-known/, and `holdout_speakers`
    train speakers who never appear in the official test (closest to the median clip count)
    are moved out and alternated into dev/ and test/.
    """
    import statistics

    splits: dict[str, list[Row]] = {}
    for split in ("train", "test"):
        folder = _find_split_dir(src, split)
        if folder is None:
            continue
        rows: list[Row] = []
        for wav in sorted((folder / "audio_files").glob("*.wav")):
            txt = folder / "trans_files" / f"{wav.stem}.txt"
            if txt.exists():
                rows.append((wav, txt.read_text(encoding="utf-8").strip(), _mile_speaker(wav.stem)))
        splits[split] = rows
    if not splits:
        raise SystemExit(f"No train/ or test/ with audio_files/ and trans_files/ under {src}.")
    if holdout_speakers <= 0 or "train" not in splits:
        return splits
    official_test = splits.pop("test", [])
    known = {r[2] for r in official_test}
    if official_test:
        splits["test-known"] = official_test
    counts: dict[str, int] = {}
    for r in splits["train"]:
        if r[2]:
            counts[r[2]] = counts.get(r[2], 0) + 1
    candidates = [s for s in counts if s not in known]
    if not candidates:
        return splits
    median = statistics.median(counts[s] for s in candidates)
    held = sorted(candidates, key=lambda s: (abs(counts[s] - median), s))[:holdout_speakers]
    split_of = {s: ("dev" if i % 2 == 0 else "test") for i, s in enumerate(held)}
    print(f"  holding out {len(held)} speakers: " + ", ".join(f"{s}({counts[s]})" for s in held))
    train: list[Row] = []
    for r in splits["train"]:
        dest = split_of.get(r[2])
        (splits.setdefault(dest, []) if dest else train).append(r)
    splits["train"] = train
    return splits


def from_kathbath(src: Path, out: Path, holdout_speakers: int = 16) -> dict[str, list[Row]]:
    """AI4Bharat Kathbath (Hugging Face parquet): train-*.parquet and valid-*.parquet.

    Audio is stored inside the parquet as FLAC bytes; it is written straight into
    <out>/<split>/ (so _write_split finds it there and does not copy it again).
    Kathbath's official valid set reuses speakers from its train set, so it cannot measure
    new voices. Instead, `holdout_speakers` train speakers are moved out (those closest to the
    median clip count, alternating gender, never one of the valid speakers) and alternated into
    dev/ and test/. The official valid set is kept as test-known/ (seen speakers, report as such).
    """
    import statistics

    import pyarrow.parquet as pq

    def speakers_of(pattern: str) -> dict[str, tuple[int, str]]:
        """speaker -> (clip count, gender), from two small columns only (no audio read)."""
        stats: dict[str, list] = {}
        for p in sorted(src.glob(pattern)):
            t = pq.read_table(p, columns=["speaker_id", "gender"])
            for spk, g in zip(t.column("speaker_id").to_pylist(), t.column("gender").to_pylist()):
                s = stats.setdefault(f"kb_{spk}", [0, g or ""])
                s[0] += 1
        return {k: (v[0], v[1]) for k, v in stats.items()}

    def read(pattern: str, split_of) -> dict[str, list[Row]]:
        rows: dict[str, list[Row]] = {}
        files = sorted(src.glob(pattern))
        for n, path in enumerate(files, 1):
            print(f"  {path.name} ({n}/{len(files)})", flush=True)
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=256):
                for r in batch.to_pylist():
                    speaker = f"kb_{r['speaker_id']}"
                    split = split_of(speaker)
                    audio = r["audio_filepath"]
                    name = Path(audio.get("path") or r["fname"]).with_suffix(".flac").name
                    dst = out / split / name
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(audio["bytes"])
                    rows.setdefault(split, []).append((dst, r["text"], speaker))
        return rows

    if not list(src.glob("train-*.parquet")):
        raise SystemExit(f"No train-*.parquet in {src} (point --src at the 'tamil' folder).")
    train_spk = speakers_of("train-*.parquet")
    valid_spk = set(speakers_of("valid-*.parquet"))
    median = statistics.median(c for c, _ in train_spk.values())
    candidates = sorted((s for s in train_spk if s not in valid_spk),
                        key=lambda s: (abs(train_spk[s][0] - median), s))
    by_gender: dict[str, list[str]] = {}
    for s in candidates:
        by_gender.setdefault(train_spk[s][1], []).append(s)
    held: list[str] = []
    queues = [by_gender[g] for g in sorted(by_gender)]
    while len(held) < holdout_speakers and any(queues):
        for q in queues:
            if q and len(held) < holdout_speakers:
                held.append(q.pop(0))
    split_of_held = {s: ("dev" if i % 2 == 0 else "test") for i, s in enumerate(held)}
    print(f"  holding out {len(held)} speakers: "
          + ", ".join(f"{s}({train_spk[s][1][:1]},{train_spk[s][0]})" for s in held))

    splits = read("train-*.parquet", lambda spk: split_of_held.get(spk, "train"))
    splits.update(read("valid-*.parquet", lambda spk: "test-known"))
    return splits


def from_csv(src: Path, test_fraction: float, seed: int) -> dict[str, list[Row]]:
    with open(src, encoding="utf-8") as f:
        rows = [
            (src.parent / r["path"], r["text"], (r.get("speaker") or "").strip())
            for r in csv.DictReader(f)
        ]
    # Drop unusable rows BEFORE splitting, so the test set never ends up empty.
    rows = [(p, t, spk) for p, t, spk in rows if t.strip() and p.exists()]
    if len(rows) < 2:
        raise SystemExit(f"Need at least 2 usable rows, found {len(rows)}.")

    # Split by speaker when known, so test speakers are never heard in training;
    # otherwise the test score overstates how well the model handles new voices.
    groups: dict[str, list[Row]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(row[2] or f"__row{i}", []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, int(len(rows) * test_fraction))
    test: list[Row] = []
    train: list[Row] = []
    for key in keys:
        (test if len(test) < target else train).extend(groups[key])
    if not train:
        raise SystemExit("Every row ended up in test; lower --test-fraction or add speakers.")
    return {"test": test, "train": train}


def apply_filters(splits: dict[str, list[Row]], drop_digit_clips: bool,
                  dedupe_test_sentences: bool, min_seconds: float) -> dict[str, list[Row]]:
    """Quality filters. They only ever touch train; test stays as published."""
    if "train" not in splits:
        return splits
    from ada.text import normalize

    train = splits["train"]
    before = len(train)
    if drop_digit_clips:
        train = [r for r in train if not any(ch.isdigit() for ch in r[1])]
    held_out = [s for s in ("dev", "test", "test-known") if s in splits]
    if dedupe_test_sentences and held_out:
        test_sentences = {normalize(r[1]) for s in held_out for r in splits[s]}
        train = [r for r in train if normalize(r[1]) not in test_sentences]
    if min_seconds > 0:
        def long_enough(row: Row) -> bool:
            d = _duration(row[0])
            return not d or float(d) >= min_seconds  # unknown duration: keep
        train = [r for r in train if long_enough(r)]
    if len(train) != before:
        print(f"Filters removed {before - len(train)} of {before} train clips.")
    return {**splits, "train": train}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", required=True,
                   choices=["commonvoice", "fleurs", "mile", "kathbath", "csv"])
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--test-fraction", type=float, default=0.1, help="csv format only")
    p.add_argument("--max-train", type=int, default=None, help="cap rows, useful for a quick smoke test")
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-speakers", type=int, default=16,
                   help="kathbath/mile: train speakers moved out into dev/test (unseen voices)")
    p.add_argument("--drop-digit-clips", action="store_true",
                   help="train only: drop clips whose transcript contains digits "
                        "(for the 'numbers as Tamil words' convention)")
    p.add_argument("--dedupe-test-sentences", action="store_true",
                   help="train only: drop clips whose sentence also appears in test")
    p.add_argument("--min-seconds", type=float, default=0.0,
                   help="train only: drop clips shorter than this (needs readable durations)")
    args = p.parse_args()

    if args.format == "commonvoice":
        splits = from_commonvoice(args.src)
    elif args.format == "fleurs":
        splits = from_fleurs(args.src)
    elif args.format == "mile":
        splits = from_mile(args.src, args.holdout_speakers)
    elif args.format == "kathbath":
        splits = from_kathbath(args.src, args.out, args.holdout_speakers)
    else:
        splits = from_csv(args.src, args.test_fraction, args.seed)

    splits = apply_filters(splits, args.drop_digit_clips, args.dedupe_test_sentences,
                           args.min_seconds)
    caps = {"train": args.max_train, "dev": args.max_test, "test": args.max_test,
            "test-known": args.max_test}
    counts = {name: _write_split(rows, args.out / name, caps.get(name))
              for name, rows in splits.items()}
    print(f"Wrote {counts} examples to {args.out}")


if __name__ == "__main__":
    main()
