import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("combine_data", ROOT / "scripts" / "stt" / "combine_data.py")
comb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comb)


def _meta(folder: Path, rows):
    folder.mkdir(parents=True)
    with open(folder / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "transcription", "duration", "speaker"])
        w.writerows(rows)
    for name, *_ in rows:
        (folder / name).write_bytes(b"x")


def test_combine_paths_resolve_and_test_is_not_combined(tmp_path):
    _meta(tmp_path / "a" / "train", [("1.mp3", "t1", "2.0", "s1")])
    _meta(tmp_path / "a" / "test", [("9.mp3", "t9", "2.0", "s9")])
    _meta(tmp_path / "b" / "train", [("2.flac", "t2", "2.0", "s2")])
    _meta(tmp_path / "b" / "dev", [("3.flac", "t3", "2.0", "s3")])
    out = tmp_path / "combo"
    counts = comb.combine([tmp_path / "a", tmp_path / "b"], out, None, 42)
    assert counts == {"train": {"a": 1, "b": 1}, "dev": {"b": 1}}
    assert not (out / "test").exists()
    for split in ("train", "dev"):
        for r in comb.read_rows(out / split):
            assert (out / split / r["file_name"]).exists()


def test_exclude_sentences_from_other_test_sets(tmp_path):
    _meta(tmp_path / "a" / "train", [("1.mp3", "வணக்கம் உலகம்", "2.0", "s1"),
                                     ("2.mp3", "நன்றி", "2.0", "s1")])
    _meta(tmp_path / "other" / "test", [("9.wav", "வணக்கம், உலகம்!", "2.0", "s9")])
    counts = comb.combine([tmp_path / "a"], tmp_path / "combo", None, 0, [tmp_path / "other" / "test"])
    assert counts["train"] == {"a": 1}
    assert counts["excluded_test_sentences"] == {"a/train": 1}
    assert [r["transcription"] for r in comb.read_rows(tmp_path / "combo" / "train")] == ["நன்றி"]


def test_cap_per_speaker():
    rows = [{"speaker": "big"} for _ in range(10)] + [{"speaker": "small"}]
    kept = comb.cap_per_speaker(rows, 3, 0)
    assert sum(r["speaker"] == "big" for r in kept) == 3
    assert sum(r["speaker"] == "small" for r in kept) == 1
