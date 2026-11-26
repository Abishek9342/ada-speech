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
        else:
            kinds["other_script"] += 1
    return kinds


def audit(clips_by_split: dict[str, list[Clip]], min_s: float = 1.0, max_s: float = 30.0) -> Report:
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
            if kinds["digit"]:
                report.flags.append(Flag(c, "contains digits (check number style is consistent)"))
            if texts[norm] > 1 and split == "train":
                report.flags.append(Flag(c, f"transcript repeated {texts[norm]}x in train"))

    if "train" in clips_by_split and "test" in clips_by_split:
        train, test = clips_by_split["train"], clips_by_split["test"]
        train_texts = {normalize(c.text) for c in train}
        train_spk = {c.speaker for c in train if c.speaker}
        report.leaks = {
            "test_sentences_seen_in_train": sum(normalize(c.text) in train_texts for c in test),
            "test_speakers_seen_in_train": len({c.speaker for c in test if c.speaker} & train_spk),
            "test_clips": len(test),
        }
    return report
