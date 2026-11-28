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


def test_commonvoice_keeps_dev_as_its_own_split(tmp_path):
    header = "client_id\tpath\tsentence\n"
    for split in ("train", "dev", "test"):
        wav(tmp_path / "clips" / f"{split}.mp3")
        (tmp_path / f"{split}.tsv").write_text(
            header + f"spk_{split}\t{split}.mp3\tவணக்கம்\n", encoding="utf-8"
        )
    splits = prep.from_commonvoice(tmp_path)
    assert list(splits) == ["train", "dev", "test"]
    assert splits["dev"][0][2] == "spk_dev"
    (tmp_path / "dev.tsv").unlink()
    assert list(prep.from_commonvoice(tmp_path)) == ["train", "test"]


def _kathbath_parquet(path, rows):
    import io

    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    recs = []
    for fname, spk, text, *gender in rows:
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="FLAC")
        recs.append({"fname": fname, "text": text, "speaker_id": spk, "duration": 1.0,
                     "gender": gender[0] if gender else "Female",
                     "audio_filepath": {"bytes": buf.getvalue(), "path": fname.replace(".m4a", ".flac")}})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(recs), path)


def test_kathbath_holds_out_unseen_speakers_and_keeps_valid_as_known(tmp_path):
    src, out = tmp_path / "tamil", tmp_path / "out"
    # speaker 1 is also in valid (must stay in train); 2, 5 (F) and 3, 6 (M) are the held-out
    # candidates (1 clip each = the median); 4 has 3 clips (far from the median).
    gender = {1: "Female", 2: "Female", 5: "Female", 3: "Male", 6: "Male", 4: "Male"}
    clips = [(1, "ஒன்று"), (1, "இரண்டு"), (2, "மூன்று"), (3, "நான்கு"), (5, "பத்து"),
             (6, "பதினொன்று"), (4, "ஐந்து"), (4, "ஆறு"), (4, "ஏழு")]
    _kathbath_parquet(src / "train-00000-of-00001.parquet",
                      [(f"c{i}.m4a", spk, text, gender[spk]) for i, (spk, text) in enumerate(clips)])
    _kathbath_parquet(src / "valid-00000-of-00001.parquet", [("v.m4a", 1, "எட்டு", "Female")])
    splits = prep.from_kathbath(src, out, holdout_speakers=4)
    dev, test = {r[2] for r in splits["dev"]}, {r[2] for r in splits["test"]}
    assert dev | test == {"kb_2", "kb_3", "kb_5", "kb_6"}
    # both splits get both genders (the old i % 2 put all women in dev and all men in test)
    assert {gender[int(s[3:])] for s in dev} == {"Female", "Male"}
    assert {gender[int(s[3:])] for s in test} == {"Female", "Male"}
    assert {r[2] for r in splits["train"]} == {"kb_1", "kb_4"}   # unseen voices held out
    assert [r[0].name for r in splits["test-known"]] == ["v.flac"]
    assert (out / "test-known" / "v.flac").exists()
    counts = {s: prep._write_split(rows, out / s, None) for s, rows in splits.items()}
    assert counts == {"train": 5, "dev": 2, "test": 2, "test-known": 1}


def test_kathbath_refuses_repeated_file_names(tmp_path):
    src = tmp_path / "tamil"
    _kathbath_parquet(src / "train-00000-of-00001.parquet",
                      [("a.m4a", 1, "ஒன்று"), ("a.m4a", 2, "இரண்டு")])
    with pytest.raises(SystemExit, match="appears twice"):
        prep.from_kathbath(src, tmp_path / "out", holdout_speakers=0)


def test_dev_test_split_goes_in_pairs():
    assert list(prep.dev_test_split(["f1", "m1", "f2", "m2", "f3"]).values()) == [
        "dev", "dev", "test", "test", "dev"]


def test_mile_pairs_audio_with_transcripts_in_nested_folder(tmp_path):
    root = tmp_path / "mile_tamil_asr_corpus"
    for split in ("train", "test"):
        wav(root / split / "audio_files" / f"MILE_{split}01_0001.wav")
        wav(root / split / "audio_files" / f"MILE_{split}01_0002.wav")  # no transcript: skipped
        (root / split / "trans_files").mkdir(parents=True)
        (root / split / "trans_files" / f"MILE_{split}01_0001.txt").write_text(
            " நன்றி \n", encoding="utf-8")
    splits = prep.from_mile(tmp_path, holdout_speakers=0)  # one level above: must still be found
    assert {k: [r[0].name for r in v] for k, v in splits.items()} == {
        "train": ["MILE_train01_0001.wav"], "test": ["MILE_test01_0001.wav"]}
    assert splits["train"][0][1:] == ("நன்றி", "MILE_train01")


def test_mile_official_test_becomes_known_and_unseen_speakers_are_held_out(tmp_path):
    root = tmp_path / "mile"
    clips = {"train": {"MILE_0000001": 3, "MILE_0000002": 2, "MILE_0000003": 2,
                       "MILE_0000005": 2, "MILE_0000006": 2, "MILE_0000004": 9},
             "test": {"MILE_0000001": 1}}  # speaker 1 is in both: the published test is "known"
    for split, speakers in clips.items():
        (root / split / "trans_files").mkdir(parents=True)
        for spk, n in speakers.items():
            for i in range(n):
                wav(root / split / "audio_files" / f"{spk}_{i:07d}.wav")
                (root / split / "trans_files" / f"{spk}_{i:07d}.txt").write_text(
                    f"வாக்கியம் {spk} {i}", encoding="utf-8")
    splits = prep.from_mile(root, holdout_speakers=4)
    assert sorted(splits) == ["dev", "test", "test-known", "train"]
    assert {r[2] for r in splits["test-known"]} == {"MILE_0000001"}
    held = {r[2] for r in splits["dev"]} | {r[2] for r in splits["test"]}
    # closest to the median, never a known speaker
    assert held == {"MILE_0000002", "MILE_0000003", "MILE_0000005", "MILE_0000006"}
    assert {r[2] for r in splits["train"]} == {"MILE_0000001", "MILE_0000004"}


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


def test_flac16k_converts_resamples_and_repoints_rows(tmp_path):
    src = tmp_path / "in" / "a.wav"
    src.parent.mkdir()
    sf.write(str(src), np.zeros(48000, dtype="float32"), 48000)   # 1 s at 48 kHz
    rows = prep.convert_rows_to_flac([(src, "சரி", "s1")], tmp_path / "out", workers=1)
    assert rows[0][0] == tmp_path / "out" / "a.flac"
    info = sf.info(str(rows[0][0]))
    assert info.samplerate == 16000 and info.channels == 1 and abs(info.duration - 1.0) < 0.05


def test_clean_labels_drops_invisible_chars_and_punctuation():
    rows = [(Path("x.wav"), "க்‍ஷ, \"வணக்கம்\"! don't ஶ்ரீ.", "")]
    assert prep.clean_labels(rows, "keep")[0][1] == "க்ஷ, \"வணக்கம்\"! don't ஸ்ரீ."
    assert prep.clean_labels(rows, "none")[0][1] == "க்ஷ வணக்கம் dont ஸ்ரீ"


def test_dedupe_also_covers_dev_and_test_known():
    p = Path("x.wav")
    splits = {"train": [(p, "ஒன்று", ""), (p, "இரண்டு", ""), (p, "மூன்று", "")],
              "dev": [(p, "ஒன்று", "")], "test-known": [(p, "இரண்டு", "")]}
    out = prep.apply_filters(splits, drop_digit_clips=False, dedupe_test_sentences=True,
                             min_seconds=0)
    assert [r[1] for r in out["train"]] == ["மூன்று"]


def test_dedupe_removes_test_sentences_from_dev_too():
    p = Path("x.wav")
    splits = {"train": [(p, "நான்கு", "")],
              "dev": [(p, "ஒன்று", ""), (p, "ஐந்து", "")],
              "test": [(p, "ஒன்று!", "")], "test-known": [(p, "ஆறு", "")]}
    out = prep.apply_filters(splits, drop_digit_clips=False, dedupe_test_sentences=True,
                             min_seconds=0)
    assert [r[1] for r in out["dev"]] == ["ஐந்து"]
    assert out["test"] == splits["test"]