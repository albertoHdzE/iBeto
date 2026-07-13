"""Speech-to-text via faster-whisper (runs on Apple Silicon CPU)."""

import os

import numpy as np
from faster_whisper import WhisperModel


class WhisperSTT:
    def __init__(self, model_size: str = "large-v3-turbo", language: str = "",
                 threads: int = 0):
        # CTranslate2 has no Metal backend, so transcription is CPU-bound. Using
        # many cores is the big speed lever here (large-v3-turbo: ~8s at the
        # default thread count vs ~2.5s at 16-24 threads on the M3 Ultra).
        cpu_threads = threads if threads > 0 else min(os.cpu_count() or 4, 16)
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8",
                                  cpu_threads=cpu_threads)
        # "" = auto-detect the spoken language (reliable on large-v3-turbo, so you
        # can just speak any language). A fixed code ("en", "ja", ...) forces it.
        self.language = language

    def transcribe(self, audio: "np.ndarray | str") -> str:
        """Transcribe a 16 kHz mono float32 array (or an audio file path)."""
        segments, _ = self.model.transcribe(audio, language=self.language or None)
        return "".join(segment.text for segment in segments).strip()
