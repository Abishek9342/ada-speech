import pytest

from ada.text import cer, normalize, wer


def test_normalize_strips_punctuation_and_case():
    assert normalize("  Hello, World!  ") == "hello world"


def test_normalize_keeps_indic_vowel_signs():
    # Tamil vowel signs are combining marks (category M) and must survive.
    assert normalize("வணக்கம்!") == "வணக்கம்"


def test_wer_ignores_formatting_differences():
    assert wer(["Hello, world."], ["hello world"]) == 0.0


def test_wer_counts_errors():
    assert wer(["one two three four"], ["one two tree four"]) == pytest.approx(0.25)


def test_wer_skips_empty_references():
    assert wer(["", "a b"], ["noise", "a b"]) == 0.0


def test_wer_rejects_all_empty():
    with pytest.raises(ValueError):
        wer([""], ["x"])


def test_cer_is_gentler_than_wer_on_suffix_errors():
    # One wrong suffix: the whole word is wrong for WER, a single character for CER.
    ref, hyp = ["நான் வீட்டுக்கு போனேன்"], ["நான் வீட்டுக்கு போனான்"]
    assert wer(ref, hyp) == pytest.approx(1 / 3)
    assert cer(ref, hyp) < 0.1


def test_invisible_chars_do_not_change_cer():
    # ZWJ inside a conjunct used to count as an extra character (CER 0.33).
    assert normalize("க்\u200dஷ") == normalize("க்ஷ")
    assert cer(["க்ஷ"], ["க்\u200dஷ"]) == 0.0
    assert normalize("\ufeffஶ்ரீ\u00a0ராம்") == "ஸ்ரீ ராம்"