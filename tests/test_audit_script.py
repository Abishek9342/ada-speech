"""The audit summary must report flags under the split they belong to."""

import csv
import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stt" / "audit_data.py"


def _write(folder: Path, rows: list[tuple[str, str, str]]) -> None:
    folder.mkdir(parents=True)
    with open(folder / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "transcription", "duration", "speaker"])
        for name, text, dur in rows:
            w.writerow([name, text, dur, "s1"])


def test_flags_are_printed_per_split(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "train", [("a.wav", "இது ok தான்", "3.0"), ("b.wav", "வணக்கம் hello", "3.0")])
    _write(tmp_path / "test", [("c.wav", "வணக்கம் நண்பரே", "3.0")])
    monkeypatch.setattr(sys, "argv", ["audit_data.py", "--data", str(tmp_path)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    out = capsys.readouterr().out
    flags = out.split("== Flags")[1]
    train_part, test_part = flags.split("[test]")
    assert "Latin" in train_part
    assert "Latin" not in test_part
    assert "[test] 1 clips, 0 flags" in out
