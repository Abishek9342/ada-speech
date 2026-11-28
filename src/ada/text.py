"""Text normalization shared by training and evaluation.

WER is only meaningful if the reference and the prediction are normalized the
same way. This normalizer is script-agnostic: it works for Latin as well as
Indic and other scripts (lowercasing is a no-op where a script has no case).

Unlike Whisper's English-centric normalizer it KEEPS combining marks, so Tamil
vowel signs and the virama (pulli) count. Dropping them hides 20-40 WER points
(arXiv 2409.02449).
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
# Two spellings of "sri": SHA (U+0BB6, rare) and SA (U+0BB8, common).
_SRI = ("ஶ்ரீ", "ஸ்ரீ")
_APOSTROPHES = "'’ʼ"


def clean_text(text: str) -> str:
    """Invisible-character cleanup, safe for training labels and for scoring.

    NFC, drop format characters (Unicode Cf: ZWJ, ZWNJ, zero-width space, BOM, ...),
    no-break space to space, one spelling of "sri", collapse whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = text.replace(" ", " ").replace(*_SRI)
    return _WHITESPACE.sub(" ", text).strip()


def strip_punctuation(text: str) -> str:
    """Training-label style 'none' (owner decision 2026-11-28): no punctuation at all.

    Apostrophes inside words are removed without a space (don't -> dont); every other
    punctuation mark becomes a space. Symbols (S*) and case are kept.
    """
    text = "".join(ch for ch in text if ch not in _APOSTROPHES)
    text = "".join(" " if unicodedata.category(ch)[0] == "P" else ch for ch in text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """For scoring: clean_text, lowercase, drop punctuation/symbols, collapse whitespace."""
    text = clean_text(text).lower()
    # Replace punctuation (P*) and symbols (S*) with spaces; keep letters, marks
    # (needed for Indic vowel signs) and numbers.
    text = "".join(" " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text)
    return _WHITESPACE.sub(" ", text).strip()


def _pairs(references: list[str], predictions: list[str]) -> tuple[list[str], list[str]]:
    pairs = [(normalize(r), normalize(p)) for r, p in zip(references, predictions, strict=True)]
    pairs = [(r, p) for r, p in pairs if r]
    if not pairs:
        raise ValueError("No non-empty references to score.")
    refs, preds = zip(*pairs)
    return list(refs), list(preds)


def wer(references: list[str], predictions: list[str]) -> float:
    """Word error rate after normalization. Empty references are skipped."""
    import jiwer

    return jiwer.wer(*_pairs(references, predictions))


def cer(references: list[str], predictions: list[str]) -> float:
    """Character error rate after normalization.

    Report it next to WER for Tamil: words are long and agglutinative, so one
    wrong suffix makes a whole word wrong even when the transcript is close.
    """
    import jiwer

    return jiwer.cer(*_pairs(references, predictions))
