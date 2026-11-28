import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    mod_name = f"_day2_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "scripts" / "stt" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod   # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def test_rate_with_ci_matches_corpus_rate_and_brackets_it():
    ev = load("evaluate")
    counts = [(1, 10), (0, 10), (3, 10), (2, 10)] * 25
    rate, lo, hi = ev.rate_with_ci(counts, n_boot=200)
    assert rate == pytest.approx(15.0)
    assert lo < rate < hi
    assert ev.rate_with_ci(counts, n_boot=0)[0] == pytest.approx(15.0)


def test_edit_counts_char_level_after_normalization():
    ev = load("evaluate")
    assert ev.edit_counts(["வணக்கம்!"], ["வணக்கம்"], "char") == [(0, len("வணக்கம்"))]
    assert ev.edit_counts([""], ["x"], "char") == []   # empty refs skipped


def test_eval_list_roundtrip(tmp_path):
    lists = load("make_eval_lists")
    ev = load("evaluate")
    split = tmp_path / "test"
    split.mkdir()
    (split / "metadata.csv").write_text(
        "file_name,transcription,duration,speaker\n"
        + "".join(f"{i}.flac,சொல்,1.0,\n" for i in range(10)) + "x.flac,விலை 5,1.0,\n",
        encoding="utf-8")
    names = lists.make_list(split, "4", nodigits=True, seed=1)
    assert len(names) == 4 and "x.flac" not in names
    assert names == lists.make_list(split, "4", nodigits=True, seed=1)   # seeded = fixed
    f = tmp_path / "l.txt"
    f.write_text(f"# data: {split.as_posix()}\n# c\n" + "\n".join(names), encoding="utf-8")
    data, got = ev.read_list(f)
    assert data == split and got == names


def test_per_source_dev_sample():
    pytest.importorskip("yaml")
    tr = load("train")
    rows = ([{"file_name": f"../../cv2-ta/dev/{i}.flac"} for i in range(50)]
            + [{"file_name": f"../../kb2-ta/dev/{i}.flac"} for i in range(5)])
    out = tr.sample_per_source(rows, 10, seed=0)
    assert tr.count_sources(out) == {"cv2-ta": 10, "kb2-ta": 5}
    assert tr.source_of("x.flac") == ""


def test_results_table_last_line_wins_and_marks_no_vad():
    rt = load("results_table")
    lines = [
        {"set": "evals/cv_test_500.txt", "model": "m1", "rtf": 0.1, "wer": 40, "cer": 9, "cer_ci": [8, 10]},
        {"set": "evals/cv_test_500.txt", "model": "m1", "rtf": 0.1, "wer": 30, "cer": 6, "cer_ci": [5, 7]},
        {"set": "evals/nonspeech.txt", "model": "m1", "rtf": 1.0, "hallucination_pct": 100.0, "vad": False},
    ]
    table = rt.build(lines)
    assert "**6.00**" in table and "**9.00**" not in table
    assert "nonspeech (no VAD)" in table and "100.0% halluc" in table