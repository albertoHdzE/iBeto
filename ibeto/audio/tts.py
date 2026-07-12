"""Text-to-speech: neural Kokoro (default) with a macOS `say` fallback.

Isolated so conversation code only sees an engine with `.speak(text)` and the
streaming `SentenceSpeaker`. Kokoro runs locally via kokoro-onnx (no torch);
the ~315 MB model is fetched once into a cache dir on first use.
"""

import queue
import re
import subprocess
import threading
import urllib.request
from pathlib import Path

_KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/voices-v1.0.bin",
}


def speak(text: str, voice: str = "") -> None:
    """Legacy one-shot macOS `say`. Kept for the `say` fallback and callers/tests."""
    if not text.strip():
        return
    args = ["say"]
    if voice:
        args += ["-v", voice]
    args.append(text)
    subprocess.run(args, check=False)


def _default_model_dir() -> Path:
    return Path.home() / ".cache" / "ibeto" / "kokoro"


def _ensure_models(model_dir: Path) -> tuple[Path, Path]:
    """Download the Kokoro model + voices into model_dir if missing."""
    model_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in _KOKORO_FILES.items():
        path = model_dir / name
        if not path.exists():
            print(f"Downloading Kokoro {name} (one-time, ~300 MB total)...", flush=True)
            urllib.request.urlretrieve(url, path)
        out[name] = path
    return out["kokoro-v1.0.onnx"], out["voices-v1.0.bin"]


class SayTTS:
    """macOS `say`: robotic but zero-dependency. The fallback engine."""

    def __init__(self, voice: str = ""):
        # Kokoro voice ids (bf_/bm_/af_/am_...) are not valid `say` voices.
        self.voice = "" if voice[:3] in ("bf_", "bm_", "af_", "am_") else voice

    def speak(self, text: str) -> None:
        speak(text, self.voice)

    def close(self) -> None:
        pass


class KokoroTTS:
    """Neural Kokoro-82M via kokoro-onnx; plays through sounddevice."""

    def __init__(self, voice: str = "bf_isabella", speed: float = 1.0,
                 lang: str = "en-gb", model_dir: str = ""):
        from kokoro_onnx import Kokoro  # heavy import, kept local
        import sounddevice as sd

        self._sd = sd
        onnx, voices = _ensure_models(Path(model_dir) if model_dir else _default_model_dir())
        self._k = Kokoro(str(onnx), str(voices))
        self.voice = voice
        self.speed = speed
        self.lang = lang

    def synth(self, text: str):
        """Return (float32 samples, sample_rate) for `text`."""
        return self._k.create(text, voice=self.voice, speed=self.speed, lang=self.lang)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        samples, sr = self.synth(text)
        self._sd.play(samples, sr)
        self._sd.wait()

    def close(self) -> None:
        try:
            self._sd.stop()
        except Exception:
            pass


def make_tts(cfg):
    """Build the TTS engine from config, falling back to `say` on any failure."""
    engine = getattr(cfg, "tts_engine", "kokoro")
    if engine == "say":
        return SayTTS(cfg.tts_voice)
    try:
        return KokoroTTS(
            voice=cfg.tts_voice or "bf_isabella",
            speed=getattr(cfg, "tts_speed", 1.0),
            model_dir=getattr(cfg, "tts_model_dir", "") or "",
        )
    except Exception as exc:
        print(f"(Kokoro TTS unavailable: {exc}; using macOS say)", flush=True)
        return SayTTS("")


# Sentence boundary: end punctuation (Latin + CJK) then optional closing quote,
# followed by whitespace or end-of-buffer; or one-or-more newlines.
_SENT_END = re.compile(r'[.!?。！？]+["”\')\]』」]*(?=\s|$)|\n+')


class SentenceSpeaker:
    """Speak a streaming reply sentence-by-sentence on a worker thread.

    feed() buffers deltas and enqueues each complete sentence as it forms, so
    synthesis + playback overlap generation. finish() flushes the tail and
    blocks until everything queued has been spoken.
    """

    def __init__(self, tts):
        self.tts = tts
        self._buf = ""
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            try:
                self.tts.speak(item)
            except Exception:
                pass  # never let a TTS hiccup kill the conversation
            self._q.task_done()

    def feed(self, delta: str) -> None:
        self._buf += delta
        while True:
            m = _SENT_END.search(self._buf)
            if not m:
                return
            sentence = self._buf[: m.end()].strip()
            self._buf = self._buf[m.end():]
            if sentence:
                self._q.put(sentence)

    def finish(self) -> None:
        """Flush the trailing partial sentence and wait for playback to drain."""
        tail = self._buf.strip()
        self._buf = ""
        if tail:
            self._q.put(tail)
        self._q.join()

    def close(self) -> None:
        self._q.put(None)
        self.tts.close()
