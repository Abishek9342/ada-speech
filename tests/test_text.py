import pytest

from ada.text import normalize, wer


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
