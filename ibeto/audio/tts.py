"""Text-to-speech: neural, multilingual, streaming.

Each reply is spoken in its own language, routed per sentence by script:
  Arabic  -> Piper (Kokoro has no Arabic voice)
  Chinese -> Kokoro Mandarin voice
  Japanese-> Kokoro Japanese voice
  else    -> the configured Kokoro voice (Latin default)

Playback is a two-stage pipeline (synthesize-ahead, then play) so speech flows
without gaps between sentences, and can be interrupted cleanly. Everything is
isolated behind an engine (`.synth(text) -> (samples, sr)` / `.speak(text)`) plus
the streaming SentenceSpeaker, so conversation code stays TTS-agnostic. macOS
`say` is the zero-dependency fallback. Models are fetched once on first use.
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
    """Neural Kokoro-82M via kokoro-onnx.

    Covers en/es/fr/hi/it/ja/pt/zh. `synth`/`speak` accept a per-call voice+lang
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
        pass  # streaming playback is owned by SentenceSpeaker's output stream


class PiperTTS:
    """Neural Piper voice (onnx, no torch) for languages Kokoro lacks."""

    def __init__(self, voice_id: str = "ar_JO-kareem-medium", model_dir: str = ""):
        from piper import PiperVoice
        import sounddevice as sd

        self._sd = sd
        onnx, cfg = _ensure_piper(voice_id, Path(model_dir) if model_dir else _default_piper_dir())
        self._v = PiperVoice.load(str(onnx), config_path=str(cfg))

    def synth(self, text: str):
        chunks = list(self._v.synthesize(text))
        if not chunks:
            return None
        sr = chunks[0].sample_rate
        samples = np.concatenate(
            [np.asarray(c.audio_float_array, dtype=np.float32) for c in chunks]
        )
        return samples, sr

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        audio = self.synth(text)
        if audio is not None:
            self._sd.play(*audio)
            self._sd.wait()

    def close(self) -> None:
        pass  # streaming playback is owned by SentenceSpeaker's output stream


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

    def synth(self, text: str):
        """Return (samples, sr) for `text`, routed by script. None if empty."""
        if not text.strip():
            return None
        lang = detect_lang(text)
        if lang == "ar":
            return self._piper_engine().synth(text)
        if lang == "zh":
            return self.kokoro.synth(text, voice=getattr(self.cfg, "tts_voice_zh", "zf_xiaobei"),
                                     lang="cmn")
        if lang == "ja":
            return self.kokoro.synth(text, voice=getattr(self.cfg, "tts_voice_ja", "jf_alpha"),
                                     lang="ja")
        return self.kokoro.synth(text)

    def speak(self, text: str) -> None:
        """One-shot speak (used for control acks, not the streaming path)."""
        if not text.strip():
            return
        try:
            audio = self.synth(text)
            if audio is not None:
                self.kokoro._sd.play(*audio)
                self.kokoro._sd.wait()
        except Exception as exc:
            print(f"(TTS failed: {exc}; using say)", flush=True)
            if self._say is None:
                self._say = SayTTS("")
            self._say.speak(text)

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

# Sentence boundaries. CJK enders (。！？) split immediately since CJK uses no
# trailing space; Latin enders need following whitespace so "3.14"/"e.g." don't
# split; newlines always split.
_SENT_END = re.compile(
    r'[。！？]+[」』）】]*'          # CJK terminators (+ optional CJK close brackets)
    r'|[.!?]+["”\')\]]*(?=\s|$)'    # Latin terminators before whitespace/end
    r'|\n+'                          # hard line breaks
)
_MAX_LATIN = 180   # backstop chunk cap (chars) for run-on sentences
_MAX_CJK = 60      # CJK packs more phonemes/char; stay well under Kokoro's ~510 limit
_CUT_CHARS = " 、，,;；:"


def _safe_cut(buf: str, maxc: int) -> int:
    """Index to cut a boundary-less run-on: last separator within maxc, else maxc."""
    window = buf[:maxc]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in _CUT_CHARS:
            return i + 1
    return maxc


class SentenceSpeaker:
    """Speak a streaming reply sentence-by-sentence.

    feed() buffers deltas and emits complete sentences (script-aware splitting +
    a length cap so no chunk exceeds the neural model's phoneme limit). If the
    engine exposes synth(), a two-stage pipeline synthesizes the next sentence
    while the current one plays, so speech is gapless; otherwise a single worker
    calls engine.speak() (the `say` fallback). interrupt() stops immediately.
    """

    def __init__(self, tts):
        self.tts = tts
        self._buf = ""
        self._text_q: queue.Queue = queue.Queue()
        self._pipeline = callable(getattr(tts, "synth", None))
        self._stop = threading.Event()
        self._stream = None   # persistent sounddevice OutputStream (pipeline only)
        self._cur_sr = None
        if self._pipeline:
            import sounddevice as sd
            self._sd = sd
            self._audio_q: queue.Queue = queue.Queue(maxsize=8)
            self._synth_t = threading.Thread(target=self._synth_loop, daemon=True)
            self._play_t = threading.Thread(target=self._play_loop, daemon=True)
            self._synth_t.start()
            self._play_t.start()
        else:
            self._sd = None
            self._worker_t = threading.Thread(target=self._say_loop, daemon=True)
            self._worker_t.start()

    # -- worker loops --
    def _synth_loop(self) -> None:
        while True:
            text = self._text_q.get()
            if text is None:
                self._audio_q.put(None)
                self._text_q.task_done()
                return
            try:
                if not self._stop.is_set():
                    audio = self.tts.synth(text)
                    if audio is not None and not self._stop.is_set():
                        self._audio_q.put(audio)
            except Exception as exc:  # never crash on a bad chunk (e.g. phoneme overflow)
                print(f"\n(TTS skipped a chunk: {exc})", flush=True)
            self._text_q.task_done()

    def _play_loop(self) -> None:
        # One persistent output stream: consecutive same-rate sentences are
        # written back-to-back with no gap (no per-sentence stream reopen).
        while True:
            item = self._audio_q.get()
            if item is None:
                self._close_stream()
                self._audio_q.task_done()
                return
            samples, sr = item
            try:
                if not self._stop.is_set():
                    if self._stream is None or self._cur_sr != sr:
                        self._close_stream()
                        self._stream = self._sd.OutputStream(
                            samplerate=sr, channels=1, dtype="float32")
                        self._stream.start()
                        self._cur_sr = sr
                    self._stream.write(np.asarray(samples, dtype=np.float32).reshape(-1, 1))
            except Exception:
                self._close_stream()  # drop a broken stream; next chunk reopens
            self._audio_q.task_done()

    def _close_stream(self) -> None:
        stream, self._stream, self._cur_sr = self._stream, None, None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:
                pass

    def _say_loop(self) -> None:
        while True:
            text = self._text_q.get()
            if text is None:
                self._text_q.task_done()
                return
            try:
                if not self._stop.is_set():
                    self.tts.speak(text)
            except Exception:
                pass
            self._text_q.task_done()

    # -- public API --
    def feed(self, delta: str) -> None:
        self._buf += delta
        while True:
            m = _SENT_END.search(self._buf)
            cap = _MAX_CJK if detect_lang(self._buf[:32]) in ("zh", "ja") else _MAX_LATIN
            if m and m.end() <= cap:
                end = m.end()
            elif len(self._buf) >= cap:
                end = _safe_cut(self._buf, cap)  # boundary-less run-on: hard flush
            else:
                return
            seg = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if seg:
                self._text_q.put(seg)

    def finish(self) -> None:
        """Flush the trailing partial sentence and block until playback drains."""
        tail = self._buf.strip()
        self._buf = ""
        if tail:
            self._text_q.put(tail)
        self._text_q.join()
        if self._pipeline:
            self._audio_q.join()

    def interrupt(self) -> None:
        """Stop speaking now: drop queued speech and abort current playback."""
        self._stop.set()
        for q in (self._text_q, getattr(self, "_audio_q", None)):
            if q is None:
                continue
            try:
                while True:
                    q.get_nowait()
                    q.task_done()
            except queue.Empty:
                pass
        self._close_stream()  # abort current playback immediately
        self._buf = ""
        self._stop.clear()

    def close(self) -> None:
        self._text_q.put(None)
        self.tts.close()
