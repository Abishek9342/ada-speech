"""Data-quality checks for a prepared ADA-STT dataset (train/ and test/ metadata.csv).

Bad data is the main reason fine-tuning goes wrong, so these checks run
before any training. They read only metadata, never audio, so they are fast
even for hundreds of hours.
"""

from __future__ import annotations

import csv
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ada.text import normalize

TAMIL_BLOCK = (0x0B80, 0x0BFF)


@dataclass
class Clip:
    split: str
    file_name: str
    text: str
    duration: float | None
    speaker: str


@dataclass
class Flag:
    clip: Clip
    reason: str


@dataclass
class Report:
    stats: dict[str, dict] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    leaks: dict[str, int] = field(default_factory=dict)


def load_split(folder: Path, split: str) -> list[Clip]:
    with open(folder / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clips = []
    for r in rows:
        dur = r.get("duration") or ""
        clips.append(
            Clip(
                split=split,
                file_name=r["file_name"],
                text=r["transcription"],
                duration=float(dur) if dur else None,
                speaker=(r.get("speaker") or "").strip(),
            )
        )
    return clips


def char_profile(text: str) -> Counter:
    """Count characters by kind: tamil, latin, digit, other_script, punct, space."""
    kinds: Counter = Counter()
    for ch in text:
        cp = ord(ch)
        if TAMIL_BLOCK[0] <= cp <= TAMIL_BLOCK[1]:
            kinds["tamil"] += 1
        elif ch.isspace():
            kinds["space"] += 1
        elif ch.isdigit():
            kinds["digit"] += 1
        elif ch.isascii() and ch.isalpha():
            kinds["latin"] += 1
        elif unicodedata.category(ch)[0] in ("P", "S"):
            kinds["punct"] += 1
        elif unicodedata.category(ch) == "Cf":  # ZWJ/ZWNJ etc.: invisible, not another script
            kinds["invisible"] += 1
        else:
            kinds["other_script"] += 1
    return kinds


def audit(clips_by_split: dict[str, list[Clip]], min_s: float = 1.0, max_s: float = 30.0,
          extra_tests: dict[str, list[Clip]] | None = None) -> Report:
    report = Report()

    for split, clips in clips_by_split.items():
        durs = [c.duration for c in clips if c.duration is not None]
        speakers = Counter(c.speaker for c in clips if c.speaker)
        report.stats[split] = {
            "clips": len(clips),
            "hours": round(sum(durs) / 3600, 2),
            "unknown_duration": len(clips) - len(durs),
            "median_s": round(statistics.median(durs), 2) if durs else None,
            "speakers": len(speakers),
            "top_speaker_share": round(max(speakers.values()) / len(clips), 3) if speakers else None,
        }

        # Speaking rate: characters (no spaces) per second. Far from the median
        # usually means the transcript does not match the audio.
        rates = [
            len(normalize(c.text).replace(" ", "")) / c.duration
            for c in clips
            if c.duration and c.duration > 0
        ]
        med_rate = statistics.median(rates) if rates else None

        texts = Counter(normalize(c.text) for c in clips)
        for c in clips:
            norm = normalize(c.text)
            kinds = char_profile(c.text)
            letters = kinds["tamil"] + kinds["latin"] + kinds["other_script"]
            if not norm:
                report.flags.append(Flag(c, "empty transcript"))
                continue
            if c.duration is not None and c.duration < min_s:
                report.flags.append(Flag(c, f"very short audio ({c.duration:.2f}s)"))
            if c.duration is not None and c.duration > max_s:
                report.flags.append(Flag(c, f"longer than {max_s:.0f}s ({c.duration:.1f}s)"))
            if med_rate and c.duration:
                rate = len(norm.replace(" ", "")) / c.duration
                if rate > 3 * med_rate or rate < med_rate / 4:
                    report.flags.append(Flag(c, f"speaking rate {rate:.1f} chars/s vs median {med_rate:.1f}"))
            if letters and kinds["tamil"] / letters < 0.5:
                report.flags.append(Flag(c, "mostly non-Tamil letters"))
            elif kinds["latin"]:
                report.flags.append(Flag(c, "contains Latin letters (code-switching?)"))
            if kinds["other_script"]:
                report.flags.append(Flag(c, "contains characters from another script"))
            if kinds["invisible"]:
                report.flags.append(Flag(c, "contains invisible format characters (ZWJ/ZWNJ/BOM)"))
            if kinds["digit"]:
                report.flags.append(Flag(c, "contains digits (check number style is consistent)"))
            if texts[norm] > 1 and split == "train":
                report.flags.append(Flag(c, f"transcript repeated {texts[norm]}x in train"))

    report.leaks = leakage(clips_by_split, extra_tests or {})
    return report


# test-known sets share speakers with train on purpose (that is what "known" means).
EXPECTED_LEAKS = ("test-known_speakers_seen_in_train",)


def leakage(clips_by_split: dict[str, list[Clip]],
            extra_tests: dict[str, list[Clip]]) -> dict[str, int]:
    """Sentence and speaker overlap of every held-out set with train, and of test sets with dev.

    Held-out sets: this folder's dev/, test/, test-known/ plus `extra_tests` (e.g. the test
    folders of every source of a combined set, and FLEURS). Keys look like
    `<set>_sentences_seen_in_<train|dev>`; any value > 0 is a leak except EXPECTED_LEAKS.
    """
    def texts(clips):
        return {normalize(c.text) for c in clips}

    def speakers(clips):
        return {c.speaker for c in clips if c.speaker}

    leaks: dict[str, int] = {}
    refs = {k: clips_by_split[k] for k in ("train", "dev") if k in clips_by_split}
    held = {k: clips_by_split[k] for k in ("dev", "test", "test-known") if k in clips_by_split}
    held.update(extra_tests)
    for ref_name, ref in refs.items():
        ref_texts, ref_spk = texts(ref), speakers(ref)
        for name, clips in held.items():
            if name == ref_name:
                continue
            leaks[f"{name}_sentences_seen_in_{ref_name}"] = sum(
                normalize(c.text) in ref_texts for c in clips)
            leaks[f"{name}_speakers_seen_in_{ref_name}"] = len(speakers(clips) & ref_spk)
    if "test" in clips_by_split:
        leaks["test_clips"] = len(clips_by_split["test"])
    return leaks


def real_leaks(leaks: dict[str, int]) -> dict[str, int]:
    """Leak counts that must be zero before training (test-known speakers are expected)."""
    return {k: v for k, v in leaks.items()
            if "_seen_in_" in k and v > 0 and not is_expected(k)}


def is_expected(key: str) -> bool:
    """Also matches '<source>/test-known_speakers_seen_in_train' from --test-dirs."""
    return any(key == e or key.endswith("/" + e) for e in EXPECTED_LEAKS)
