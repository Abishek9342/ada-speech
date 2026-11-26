"""Text normalization shared by training and evaluation.

WER is only meaningful if the reference and the prediction are normalized the
same way. This normalizer is script-agnostic: it works for Latin as well as
Indic and other scripts (lowercasing is a no-op where a script has no case).
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Unicode-normalize, lowercase, drop punctuation/symbols, collapse whitespace."""
    text = unicodedata.normalize("NFC", text).lower()
    # Replace punctuation (P*) and symbols (S*) with spaces; keep letters, marks
    # (needed for Indic vowel signs) and numbers.
    text = "".join(" " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text)
    return _WHITESPACE.sub(" ", text).strip()


def wer(references: list[str], predictions: list[str]) -> float:
    """Word error rate after normalization. Empty references are skipped."""
    import jiwer

    pairs = [(normalize(r), normalize(p)) for r, p in zip(references, predictions, strict=True)]
    pairs = [(r, p) for r, p in pairs if r]
    if not pairs:
        raise ValueError("No non-empty references to score.")
    refs, preds = zip(*pairs)
    return jiwer.wer(list(refs), list(preds))
