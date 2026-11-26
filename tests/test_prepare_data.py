import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "stt" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = load("prepare_data")


def wav(path, seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(16000 * seconds), dtype="float32"), 16000)


def test_fleurs_reads_headerless_tsv_and_skips_missing_splits(tmp_path):
    src = tmp_path / "ta_in"
    wav(src / "audio" / "test" / "1.wav")
    (src / "test.tsv").write_text(
        "7\t1.wav\tவணக்கம், உலகம்!\tவணக்கம் உலகம்\tx\t16000\tMALE\n", encoding="utf-8"
    )
    splits = prep.from_fleurs(src)
    assert list(splits) == ["test"]
    (path, text, speaker) = splits["test"][0]
    assert path.name == "1.wav" and text == "வணக்கம், உலகம்!" and speaker == ""


def test_mile_pairs_audio_with_transcripts_in_nested_folder(tmp_path):
    root = tmp_path / "mile_tamil_asr_corpus"
    for split in ("train", "test"):
        wav(root / split / "audio_files" / f"MILE_{split}01_0001.wav")
        wav(root / split / "audio_files" / f"MILE_{split}01_0002.wav")  # no transcript: skipped
        (root / split / "trans_files").mkdir(parents=True)
        (root / split / "trans_files" / f"MILE_{split}01_0001.txt").write_text(
            " நன்றி \n", encoding="utf-8")
    splits = prep.from_mile(tmp_path)  # one level above: must still be found
    assert {k: [r[0].name for r in v] for k, v in splits.items()} == {
        "train": ["MILE_train01_0001.wav"], "test": ["MILE_test01_0001.wav"]}
    assert splits["train"][0][1:] == ("நன்றி", "MILE_train01")


def test_write_split_records_duration(tmp_path):
    wav(tmp_path / "in" / "a.wav", seconds=2.5)
    n = prep._write_split([(tmp_path / "in" / "a.wav", "சரி", "s1")], tmp_path / "out", None)
    assert n == 1
    line = (tmp_path / "out" / "metadata.csv").read_text(encoding="utf-8").splitlines()[1]
    assert line == "a.wav,சரி,2.50,s1"


def test_export_writes_preprocessor_config_from_processor_config(tmp_path):
    export = load("export_ct2")
    model, out = tmp_path / "model", tmp_path / "out"
    model.mkdir()
    out.mkdir()
    (model / "processor_config.json").write_text(
        json.dumps({"feature_extractor": {"feature_size": 128}}), encoding="utf-8")
    export._ensure_preprocessor_config(model, out)
    assert json.loads((out / "preprocessor_config.json").read_text())["feature_size"] == 128


def test_filters_only_touch_train(tmp_path):
    wav(tmp_path / "short.wav", seconds=0.5)
    wav(tmp_path / "ok.wav", seconds=2.0)
    splits = {
        "train": [
            (tmp_path / "ok.wav", "விலை 100 ரூபாய்", ""),    # digits
            (tmp_path / "ok.wav", "வணக்கம், நண்பரே!", ""),   # same sentence as test
            (tmp_path / "short.wav", "சரி", ""),             # too short
            (tmp_path / "ok.wav", "நன்றி", ""),              # kept
        ],
        "test": [(tmp_path / "ok.wav", "வணக்கம் நண்பரே", ""), (tmp_path / "ok.wav", "1 2 3", "")],
    }
    out = prep.apply_filters(splits, drop_digit_clips=True, dedupe_test_sentences=True,
                             min_seconds=1.0)
    assert [r[1] for r in out["train"]] == ["நன்றி"]
    assert out["test"] == splits["test"]
