"""Speech-to-text via faster-whisper (runs on Apple Silicon CPU)."""

import numpy as np
from faster_whisper import WhisperModel


class WhisperSTT:
    def __init__(self, model_size: str = "base", language: str = "en"):
        # CTranslate2 has no Metal backend; int8 on CPU is fast enough on M2.
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        # Fixed language avoids misdetection (e.g. English heard as Japanese).
        # Empty string = auto-detect.
        self.language = language

    def transcribe(self, audio: "np.ndarray | str") -> str:
        """Transcribe a 16 kHz mono float32 array (or an audio file path)."""
        segments, _ = self.model.transcribe(audio, language=self.language or None)
        return "".join(segment.text for segment in segments).strip()
