"""Text-to-speech: neural, multilingual, streaming.

Each reply is spoken in its own language, routed per sentence by script:
  Arabic  -> Piper (Kokoro has no Arabic voice)
  Chinese -> Kokoro Mandarin voice
  Japanese-> Kokoro Japanese voice
  else    -> the configured Kokoro voice (Latin default)

Everything is isolated behind an engine with `.speak(text)` plus the streaming
SentenceSpeaker, so conversation code stays TTS-agnostic. macOS `say` is the
zero-dependency fallback. Models are fetched once into a cache dir on first use.
"""

import queue
import re
import subprocess
import threading
import urllib.request
from pathlib import Path

import numpy as np

_KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/voices-v1.0.bin",
}
_PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def speak(text: str, voice: str = "") -> None:
    """Legacy one-shot macOS `say`. Kept for the `say` fallback and callers/tests."""
    if not text.strip():
        return
    args = ["say"]
    if voice:
        args += ["-v", voice]
    args.append(text)
    subprocess.run(args, check=False)


# --- language routing -------------------------------------------------------

def detect_lang(text: str) -> str:
    """Pick a TTS language from the dominant non-Latin script in `text`.

    Kana implies Japanese; Han without kana is treated as Chinese; the Arabic
    block is Arabic; anything else falls back to the configured default voice.
    """
    ar = han = kana = 0
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF:            # hiragana + katakana
            kana += 1
        elif 0x3400 <= o <= 0x9FFF:          # CJK Han
            han += 1
        elif (0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F
              or 0x08A0 <= o <= 0x08FF or 0xFB50 <= o <= 0xFDFF
              or 0xFE70 <= o <= 0xFEFF):      # Arabic + presentation forms
            ar += 1
    if kana:
        return "ja"
    if han:
        return "zh"
    if ar:
        return "ar"
    return "default"


# --- engines ----------------------------------------------------------------

def _default_model_dir() -> Path:
    return Path.home() / ".cache" / "ibeto" / "kokoro"


def _default_piper_dir() -> Path:
    return Path.home() / ".cache" / "ibeto" / "piper"


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


def _ensure_piper(voice_id: str, model_dir: Path) -> tuple[Path, Path]:
    """Download a Piper voice (id like 'ar_JO-kareem-medium') if missing."""
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx, cfg = model_dir / f"{voice_id}.onnx", model_dir / f"{voice_id}.onnx.json"
    if not (onnx.exists() and cfg.exists()):
        region, name, quality = voice_id.split("-")   # ar_JO, kareem, medium
        family = region.split("_")[0]                 # ar
        base = f"{_PIPER_BASE}/{family}/{region}/{name}/{quality}/{voice_id}.onnx"
        print(f"Downloading Piper voice {voice_id} (one-time)...", flush=True)
        urllib.request.urlretrieve(base, onnx)
        urllib.request.urlretrieve(base + ".json", cfg)
    return onnx, cfg


class SayTTS:
    """macOS `say`: robotic but zero-dependency. The ultimate fallback."""

    def __init__(self, voice: str = ""):
        # Kokoro voice ids (bf_/bm_/af_/am_...) are not valid `say` voices.
        self.voice = "" if voice[:3] in ("bf_", "bm_", "af_", "am_") else voice

    def speak(self, text: str) -> None:
        speak(text, self.voice)

    def close(self) -> None:
        pass


class KokoroTTS:
    """Neural Kokoro-82M via kokoro-onnx; plays through sounddevice.

    Covers en/es/fr/hi/it/ja/pt/zh. `speak`/`synth` accept a per-call voice+lang
    so one engine serves the configured default plus Chinese and Japanese.
    """

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

    def synth(self, text: str, voice: str | None = None, lang: str | None = None):
        return self._k.create(text, voice=voice or self.voice,
                              speed=self.speed, lang=lang or self.lang)

    def speak(self, text: str, voice: str | None = None, lang: str | None = None) -> None:
        if not text.strip():
            return
        samples, sr = self.synth(text, voice, lang)
        self._sd.play(samples, sr)
        self._sd.wait()

    def close(self) -> None:
        try:
            self._sd.stop()
        except Exception:
            pass


class PiperTTS:
    """Neural Piper voice (onnx, no torch) for languages Kokoro lacks."""

    def __init__(self, voice_id: str = "ar_JO-kareem-medium", model_dir: str = ""):
        from piper import PiperVoice
        import sounddevice as sd

        self._sd = sd
        onnx, cfg = _ensure_piper(voice_id, Path(model_dir) if model_dir else _default_piper_dir())
        self._v = PiperVoice.load(str(onnx), config_path=str(cfg))

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        chunks = list(self._v.synthesize(text))
        if not chunks:
            return
        sr = chunks[0].sample_rate
        samples = np.concatenate(
            [np.asarray(c.audio_float_array, dtype=np.float32) for c in chunks]
        )
        self._sd.play(samples, sr)
        self._sd.wait()

    def close(self) -> None:
        try:
            self._sd.stop()
        except Exception:
            pass


class MultilingualTTS:
    """Route each utterance to the right neural engine by script.

    Kokoro loads eagerly (the common case); Piper loads lazily on the first
    Arabic sentence, so mono-lingual sessions pay nothing for it.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.kokoro = KokoroTTS(
            voice=cfg.tts_voice or "bf_isabella",
            speed=getattr(cfg, "tts_speed", 1.0),
            model_dir=getattr(cfg, "tts_model_dir", "") or "",
        )
        self._piper: PiperTTS | None = None
        self._say: SayTTS | None = None

    def _piper_engine(self) -> PiperTTS:
        if self._piper is None:
            self._piper = PiperTTS(
                getattr(self.cfg, "tts_voice_ar", "ar_JO-kareem-medium"),
                model_dir=getattr(self.cfg, "tts_piper_dir", "") or "",
            )
        return self._piper

    def _say_engine(self) -> SayTTS:
        if self._say is None:
            self._say = SayTTS("")
        return self._say

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        lang = detect_lang(text)
        try:
            if lang == "ar":
                self._piper_engine().speak(text)
            elif lang == "zh":
                self.kokoro.speak(text, voice=getattr(self.cfg, "tts_voice_zh", "zf_xiaobei"),
                                  lang="cmn")
            elif lang == "ja":
                self.kokoro.speak(text, voice=getattr(self.cfg, "tts_voice_ja", "jf_alpha"),
                                  lang="ja")
            else:
                self.kokoro.speak(text)
        except Exception as exc:
            print(f"(TTS for '{lang}' failed: {exc}; using say)", flush=True)
            self._say_engine().speak(text)

    def close(self) -> None:
        self.kokoro.close()
        if self._piper:
            self._piper.close()


def make_tts(cfg):
    """Build the TTS engine from config, falling back to `say` on any failure."""
    if getattr(cfg, "tts_engine", "kokoro") == "say":
        return SayTTS(cfg.tts_voice)
    try:
        return MultilingualTTS(cfg)
    except Exception as exc:
        print(f"(Neural TTS unavailable: {exc}; using macOS say)", flush=True)
        return SayTTS("")


# --- streaming --------------------------------------------------------------

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
