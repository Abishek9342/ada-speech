from ada.stt.audit import Clip, audit, char_profile


def clip(text, dur=3.0, speaker="s1", split="train", name="a.wav"):
    return Clip(split, name, text, dur, speaker)


def reasons(report):
    return [f.reason for f in report.flags]


def test_char_profile_tamil_latin_digits():
    k = char_profile("வணக்கம் hello 42")
    assert k["tamil"] > 0 and k["latin"] == 5 and k["digit"] == 2


def test_clean_clip_has_no_flags():
    r = audit({"train": [clip("வணக்கம் நண்பரே")] * 1})
    assert reasons(r) == []


def test_flags_digits_latin_short_and_empty():
    r = audit({"train": [
        clip("விலை 100 ரூபாய்"),
        clip("இது ok தான்"),
        clip("வணக்கம்", dur=0.3),
        clip("!!!"),
    ]})
    joined = " | ".join(reasons(r))
    assert "digits" in joined
    assert "Latin" in joined
    assert "very short" in joined
    assert "empty transcript" in joined


def test_speaking_rate_outlier():
    normal = [clip("வணக்கம் நண்பரே எப்படி இருக்கிறீர்கள்", dur=3.0, name=f"{i}.wav") for i in range(9)]
    bad = clip("வணக்கம் நண்பரே எப்படி இருக்கிறீர்கள்", dur=30.0, name="bad.wav")
    r = audit({"train": normal + [bad]})
    flagged = [f.clip.file_name for f in r.flags if "speaking rate" in f.reason]
    assert flagged == ["bad.wav"]


def test_leakage_between_train_and_test():
    train = [clip("ஒன்று இரண்டு", speaker="a"), clip("மூன்று", speaker="b")]
    test = [clip("ஒன்று இரண்டு!", speaker="c", split="test"), clip("நான்கு", speaker="b", split="test")]
    r = audit({"train": train, "test": test})
    assert r.leaks["test_sentences_seen_in_train"] == 1
    assert r.leaks["test_speakers_seen_in_train"] == 1


def test_stats_hours_and_speakers():
    r = audit({"train": [clip("அ", dur=1800, speaker="a"), clip("ஆ", dur=1800, speaker="b")]})
    assert r.stats["train"]["hours"] == 1.0
    assert r.stats["train"]["speakers"] == 2


def test_leakage_on_combined_set_uses_extra_test_dirs():
    from ada.stt.audit import real_leaks

    train = [clip("ஒன்று", speaker="a"), clip("இரண்டு", speaker="b")]
    dev = [clip("இரண்டு", speaker="c", split="dev")]             # sentence seen in train
    ext = {"kb-ta/test": [clip("ஒன்று", speaker="z", split="kb-ta/test")],
           "kb-ta/test-known": [clip("மூன்று", speaker="a", split="kb-ta/test-known")]}
    r = audit({"train": train, "dev": dev}, extra_tests=ext)
    assert r.leaks["dev_sentences_seen_in_train"] == 1
    assert r.leaks["kb-ta/test_sentences_seen_in_train"] == 1
    assert r.leaks["kb-ta/test-known_speakers_seen_in_train"] == 1   # by design for test-known
    bad = real_leaks(r.leaks)
    assert set(bad) == {"dev_sentences_seen_in_train", "kb-ta/test_sentences_seen_in_train"}
    # the in-folder test-known split is expected to share speakers, so it is not a real leak
    r2 = audit({"train": train, "test-known": [clip("நான்கு", speaker="a", split="test-known")]})
    assert real_leaks(r2.leaks) == {}


def test_invisible_chars_are_not_another_script():
    r = audit({"train": [clip("க்\u200dஷ")]})
    assert any("invisible" in x for x in reasons(r))
    assert not any("another script" in x for x in reasons(r))