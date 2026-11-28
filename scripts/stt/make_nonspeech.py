"""Synthetic non-speech test set: a model should write NOTHING for these clips.

Whisper-style models are known to "hallucinate" text on silence and noise. This writes
<out>/test/ (16 kHz FLAC + metadata.csv with empty transcriptions) and the list
evals/nonspeech.txt. evaluate.py reports the hallucination rate on it.

  python scripts/stt/make_nonspeech.py --out data/stt/nonspeech

Synthetic only (seeded, so anyone can regenerate it): digital silence, very quiet hiss,
white/pink/brown noise at several levels, mains hum, fan-like low-pass noise, clicks and
pure tones. Real recordings (room noise, traffic, music) are a later addition.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SR = 16_000


def _colored(rng, n: int, exponent: float):
    """Noise with power ~ 1/f^exponent (0 white, 1 pink, 2 brown), unit RMS."""
    import numpy as np

    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(n // 2 + 1)
    f[0] = 1
    spec /= f ** (exponent / 2)
    x = np.fft.irfft(spec, n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def make_clips(seed: int = 7):
    import numpy as np

    rng = np.random.default_rng(seed)
    clips = []

    def add(kind: str, x):
        x = np.clip(x, -1, 1).astype("float32")
        clips.append((f"ns_{len(clips):03d}_{kind}.flac", x))

    for dur in (3, 8, 15):
        add("silence", np.zeros(dur * SR))
        add("hiss", 0.001 * rng.standard_normal(dur * SR))
    for exp, kind in ((0, "white"), (1, "pink"), (2, "brown")):
        for level_db in (-50, -35, -20):
            dur = int(rng.integers(4, 15))
            add(f"{kind}{-level_db}", 10 ** (level_db / 20) * _colored(rng, dur * SR, exp))
    t = np.arange(10 * SR) / SR
    for hz in (50, 100):
        for amp in (0.01, 0.1):
            add(f"hum{hz}", amp * np.sin(2 * np.pi * hz * t) + 0.002 * rng.standard_normal(t.size))
    for i in range(6):  # fan / AC: low-pass filtered noise
        dur = int(rng.integers(5, 15))
        x = _colored(rng, dur * SR, 2.5) * 10 ** (rng.uniform(-40, -20) / 20)
        add(f"fan{i}", x)
    for i in range(4):  # sparse clicks (keyboard, taps)
        x = 0.001 * rng.standard_normal(8 * SR)
        for pos in rng.integers(0, 8 * SR - 200, size=12):
            x[pos:pos + 200] += 0.3 * np.exp(-np.arange(200) / 30) * rng.choice([-1, 1])
        add(f"clicks{i}", x)
    for hz in (440, 1000, 3000):
        add(f"tone{hz}", 0.1 * np.sin(2 * np.pi * hz * np.arange(5 * SR) / SR))
    return clips


def main() -> None:
    import soundfile as sf

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/stt/nonspeech"))
    p.add_argument("--list", type=Path, default=Path("evals/nonspeech.txt"))
    args = p.parse_args()
    test = args.out / "test"
    test.mkdir(parents=True, exist_ok=True)
    clips = make_clips()
    with open(test / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "transcription", "duration", "speaker"])
        for name, x in clips:
            sf.write(str(test / name), x, SR, format="FLAC", subtype="PCM_16")
            w.writerow([name, "", f"{len(x) / SR:.2f}", ""])
    args.list.parent.mkdir(parents=True, exist_ok=True)
    args.list.write_text("\n".join([f"# data: {test.as_posix()}",
                                    f"# clips: {len(clips)}  synthetic non-speech, seed 7"]
                                   + [n for n, _ in clips]) + "\n", encoding="utf-8")
    print(f"Wrote {len(clips)} non-speech clips to {test} and {args.list}")


if __name__ == "__main__":
    main()
