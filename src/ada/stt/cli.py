"""Command line: `ada-stt audio.wav --model path/or/repo --language ta`."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ada-stt", description="ADA speech-to-text (CPU)")
    parser.add_argument("audio", nargs="+", help="audio file(s) to transcribe")
    parser.add_argument("--model", default="base", help="HF repo id, local CT2 folder, or Whisper size")
    parser.add_argument("--language", default=None, help="language code, e.g. en, ta, hi (default: auto)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true", help="disable silence filtering")
    parser.add_argument("--stats", action="store_true", help="print speed stats to stderr")
    args = parser.parse_args(argv)

    from ada.stt.inference import Transcriber

    compute_type = "int8" if args.device == "cpu" else "float16"
    transcriber = Transcriber(args.model, device=args.device, compute_type=compute_type)
    for path in args.audio:
        result = transcriber.transcribe(
            path, language=args.language, beam_size=args.beam_size, vad_filter=not args.no_vad
        )
        prefix = f"{path}\t" if len(args.audio) > 1 else ""
        print(f"{prefix}{result.text}")
        if args.stats:
            print(
                f"[{path}] lang={result.language} audio={result.audio_seconds:.1f}s "
                f"time={result.elapsed_seconds:.1f}s RTF={result.real_time_factor:.2f}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
