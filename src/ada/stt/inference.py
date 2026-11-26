"""CPU-first inference for ADA-STT.

Uses faster-whisper (CTranslate2) with int8 weights, which is typically
several times faster than plain PyTorch on a CPU and needs no GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Result:
    text: str
    language: str
    audio_seconds: float
    elapsed_seconds: float

    @property
    def real_time_factor(self) -> float:
        """< 1.0 means faster than real time."""
        return self.elapsed_seconds / self.audio_seconds if self.audio_seconds else float("inf")


class Transcriber:
    """Load once, transcribe many files.

    Args:
        model: a Hugging Face repo id or a local folder in CTranslate2 format
            (produced by scripts/stt/export_ct2.py). Stock Whisper sizes such as
            "tiny", "base" or "small" also work, which is handy as a baseline.
        device: "cpu" (default, what end users have) or "cuda".
        compute_type: "int8" is the right choice on CPU.
        cpu_threads: 0 lets CTranslate2 decide.
    """

    def __init__(
        self,
        model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Result:
        """Transcribe one audio file.

        vad_filter=True skips silence before decoding. It is on by default because
        Whisper-style models tend to invent text during long silences.
        """
        start = time.perf_counter()
        segments, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=False,  # reduces repetition loops
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Result(
            text=text,
            language=info.language,
            audio_seconds=info.duration,
            elapsed_seconds=time.perf_counter() - start,
        )
