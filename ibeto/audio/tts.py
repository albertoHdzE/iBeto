"""Text-to-speech: neural, multilingual, streaming, gapless.

Each reply is spoken in its own language(s). Routing per sentence:
  1. clean markdown/emoji  2. split into script runs (Latin / CJK / Arabic)
  3. Latin runs are split into clauses and language-detected (en/de/fr/es/it/pt)
Each language chunk is synthesized by the fastest native engine and resampled to
one shared output stream, so switching languages is seamless (no reopen gaps):

  en/es/fr/it/pt/zh -> Kokoro (kokoro-onnx, no torch)
  de/ar             -> Piper (onnx, no torch)
  ja                -> macOS voice (Kyoko) — neural engines can't read kanji

Engines are isolated behind synth_lang(text, lang) -> (samples, sr); the
SentenceSpeaker owns splitting, playback, resampling and interruption. macOS
`say` is the zero-dependency fallback. Models are fetched once on first use.
"""

import os
import queue
import re
import subprocess
import tempfile
import threading
import urllib.request
import wave
from pathlib import Path

import numpy as np

_KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/voices-v1.0.bin",
}
_PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_TARGET_SR = 24000  # everything is resampled to this so one stream stays open


def speak(text: str, voice: str = "") -> None:
    """Legacy one-shot macOS `say`. Kept for the `say` fallback and callers/tests."""
    if not text.strip():
        return
    args = ["say"]
    if voice:
        args += ["-v", voice]
    args.append(text)
    subprocess.run(args, check=False)


# --- text cleaning ----------------------------------------------------------

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.M)
_MD_HDR = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_MD_BULLET = re.compile(r"^\s{0,3}(?:[*\-+]|\d+[.)])\s+", re.M)
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?", re.M)
_MD_MARKS = re.compile(r"\*\*|\*|__|_|~~|`+")
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff"
    "←-⇿⌀-⏿⬀-⯿]"
)


def clean_for_speech(text: str) -> str:
    """Strip markdown/formatting/emoji so TTS speaks words, not symbols."""
    t = _MD_LINK.sub(r"\1", text)
    t = _MD_RULE.sub(" ", t)
    t = _MD_HDR.sub("", t)
    t = _MD_BULLET.sub("", t)
    t = _MD_QUOTE.sub("", t)
    t = _MD_MARKS.sub("", t)
    t = _EMOJI.sub("", t)
    return re.sub(r"[ \t]+", " ", t).strip()


# --- language routing -------------------------------------------------------

def detect_lang(text: str) -> str:
    """Script-based language: kana->ja, Han->zh, Arabic->ar, else 'default'."""
    ar = han = kana = 0
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF:
            kana += 1
        elif 0x3400 <= o <= 0x9FFF:
            han += 1
        elif (0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF
              or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF):
            ar += 1
    if kana:
        return "ja"
    if han:
        return "zh"
    if ar:
        return "ar"
    return "default"


def _script_class(ch: str) -> str:
    o = ord(ch)
    if 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF:
        return "cjk"
    if (0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF
            or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF):
        return "arab"
    if ch.isalpha():
        return "latin"
    return "neutral"


def split_by_script(text: str) -> list[str]:
    """Split into consecutive runs of one script (Latin / CJK / Arabic)."""
    runs: list[str] = []
    cls = None
    cur: list[str] = []
    for ch in text:
        c = _script_class(ch)
        if c == "neutral" or c == cls:
            cur.append(ch)
        elif cls is None:
            cls = c
            cur.append(ch)
        else:
            runs.append("".join(cur))
            cur = [ch]
            cls = c
    if cur:
        runs.append("".join(cur))
    return [r for r in runs if r.strip()]


# Latin-language detection (lingua): distinguishes en/de/fr/es/it/pt, which
# share the alphabet so script alone cannot tell them apart.
_LATIN_LANGS = ("en", "de", "fr", "es", "it", "pt")
_CLAUSE = re.compile(r'[^,;:—–"“”()\[\]]+[,;:—–"“”()\[\]]*')
_detector = None
_lang_codes: dict = {}
_detector_failed = False


def _latin_detector():
    global _detector, _detector_failed
    if _detector is None and not _detector_failed:
        try:
            from lingua import Language, LanguageDetectorBuilder

            names = {"en": "ENGLISH", "de": "GERMAN", "fr": "FRENCH",
                     "es": "SPANISH", "it": "ITALIAN", "pt": "PORTUGUESE"}
            langs = [getattr(Language, names[c]) for c in _LATIN_LANGS]
            _detector = (LanguageDetectorBuilder.from_languages(*langs)
                         .with_preloaded_language_models().build())
            _lang_codes.update({getattr(Language, names[c]): c for c in _LATIN_LANGS})
        except Exception:
            _detector_failed = True  # no lingua -> everything Latin is English
    return _detector


def _split_latin(text: str) -> list[tuple[str, str]]:
    """Split Latin text into (lang, chunk), detecting each clause and merging
    consecutive clauses of the same language for smooth playback."""
    det = _latin_detector()
    if det is None:
        return [("en", text)]
    out: list[tuple[str, str]] = []
    for clause in _CLAUSE.findall(text):
        if not clause.strip():
            continue
        lang = det.detect_language_of(clause)
        code = _lang_codes.get(lang, "en") if lang is not None else "en"
        if out and out[-1][0] == code:
            out[-1] = (code, out[-1][1] + clause)
        else:
            out.append((code, clause))
    return out or [("en", text)]


def route_text(text: str) -> list[tuple[str, str]]:
    """Route a (cleaned) sentence into (lang, chunk) pieces by script + language."""
    chunks: list[tuple[str, str]] = []
    for run in split_by_script(text):
        d = detect_lang(run)
        if d == "default":
            chunks.extend(_split_latin(run))
        else:
            chunks.append((d, run))
    return chunks


# lang -> (engine, default voice, kokoro lang code). Voices are cfg-overridable.
_ROUTE = {
    "en": ("kokoro", "bf_isabella", "en-gb"),
    "es": ("kokoro", "ef_dora", "es"),
    "fr": ("kokoro", "ff_siwis", "fr-fr"),
    "it": ("kokoro", "if_sara", "it"),
    "pt": ("kokoro", "pf_dora", "pt-br"),
    "zh": ("kokoro", "zf_xiaobei", "cmn"),
    "de": ("piper", "de_DE-thorsten-medium", None),
    "ar": ("piper", "ar_JO-kareem-medium", None),
    "ja": ("macsay", "Kyoko", None),
}


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    """Linear resample to _TARGET_SR (fine for speech; keeps one stream open)."""
    if sr == _TARGET_SR or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * _TARGET_SR / sr))
    xp = np.linspace(0.0, 1.0, len(x), endpoint=False)
    xq = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


# --- engines ----------------------------------------------------------------

def _default_model_dir() -> Path:
    return Path.home() / ".cache" / "ibeto" / "kokoro"


def _default_piper_dir() -> Path:
    return Path.home() / ".cache" / "ibeto" / "piper"


def _ensure_models(model_dir: Path) -> tuple[Path, Path]:
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
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx, cfg = model_dir / f"{voice_id}.onnx", model_dir / f"{voice_id}.onnx.json"
    if not (onnx.exists() and cfg.exists()):
        region, name, quality = voice_id.split("-")
        family = region.split("_")[0]
        base = f"{_PIPER_BASE}/{family}/{region}/{name}/{quality}/{voice_id}.onnx"
        print(f"Downloading Piper voice {voice_id} (one-time)...", flush=True)
        urllib.request.urlretrieve(base, onnx)
        urllib.request.urlretrieve(base + ".json", cfg)
    return onnx, cfg


class SayTTS:
    """macOS `say`: robotic but zero-dependency. The ultimate fallback."""

    def __init__(self, voice: str = ""):
        self.voice = "" if voice[:3] in ("bf_", "bm_", "af_", "am_") else voice

    def speak(self, text: str) -> None:
        speak(text, self.voice)

    def close(self) -> None:
        pass


class MacSayTTS:
    """macOS `say` rendered to samples, for scripts neural engines can't read
    (Japanese kanji). Apple's JA voices read kanji correctly via proper G2P."""

    def __init__(self, voice: str = "Kyoko"):
        self.voice = voice

    def synth(self, text: str):
        if not text.strip():
            return None
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(
                ["say", "-v", self.voice, "-o", path, "--data-format=LEI16@22050", text],
                check=False,
            )
            with wave.open(path, "rb") as w:
                sr = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            return pcm.astype(np.float32) / 32768.0, sr
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def close(self) -> None:
        pass


class KokoroTTS:
    """Neural Kokoro-82M via kokoro-onnx (en/es/fr/it/pt/zh...)."""

    def __init__(self, speed: float = 1.0, model_dir: str = ""):
        from kokoro_onnx import Kokoro
        self._k = Kokoro(*(str(p) for p in _ensure_models(
            Path(model_dir) if model_dir else _default_model_dir())))
        self.speed = speed

    def synth(self, text: str, voice: str, lang: str):
        return self._k.create(text, voice=voice, speed=self.speed, lang=lang)

    def close(self) -> None:
        pass


class PiperTTS:
    """Neural Piper voice (onnx, no torch) for de/ar and other languages."""

    def __init__(self, voice_id: str, model_dir: str = ""):
        from piper import PiperVoice
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

    def close(self) -> None:
        pass


class MultilingualTTS:
    """Synthesize a chunk in a given language with the right native engine.

    Kokoro loads eagerly; Piper voices and macOS-JA load lazily on first use.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.kokoro = KokoroTTS(
            speed=getattr(cfg, "tts_speed", 1.0),
            model_dir=getattr(cfg, "tts_model_dir", "") or "",
        )
        self._pipers: dict[str, PiperTTS] = {}
        self._macsays: dict[str, MacSayTTS] = {}

    def _voice(self, lang: str):
        engine, dvoice, klang = _ROUTE.get(lang, _ROUTE["en"])
        key = "tts_voice" if lang == "en" else f"tts_voice_{lang}"
        return engine, (getattr(self.cfg, key, "") or dvoice), klang

    def _piper(self, voice_id: str) -> PiperTTS:
        if voice_id not in self._pipers:
            self._pipers[voice_id] = PiperTTS(
                voice_id, model_dir=getattr(self.cfg, "tts_piper_dir", "") or "")
        return self._pipers[voice_id]

    def _macsay(self, voice: str) -> MacSayTTS:
        if voice not in self._macsays:
            self._macsays[voice] = MacSayTTS(voice)
        return self._macsays[voice]

    def synth_lang(self, text: str, lang: str):
        """Return (samples, sr) for `text` in `lang`, or None on failure/empty."""
        if not text.strip():
            return None
        engine, voice, klang = self._voice(lang)
        try:
            if engine == "kokoro":
                return self.kokoro.synth(text, voice=voice, lang=klang)
            if engine == "piper":
                return self._piper(voice).synth(text)
            return self._macsay(voice).synth(text)
        except Exception as exc:
            print(f"\n(TTS '{lang}' failed: {exc})", flush=True)
            return None

    def speak(self, text: str) -> None:
        """One-shot speak (control acks): route + play each chunk sequentially."""
        import sounddevice as sd
        for lang, chunk in route_text(clean_for_speech(text)):
            audio = self.synth_lang(chunk, lang)
            if audio is not None:
                sd.play(_resample(np.asarray(audio[0], dtype=np.float32), audio[1]), _TARGET_SR)
                sd.wait()

    def close(self) -> None:
        pass


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

_SENT_END = re.compile(
    r'[。！？]+[」』）】"”\'\*_~`]*'
    r'|[.!?]+["”\')\]\*_~`]*(?=\s|$)'
    r'|\n+'
)
_MAX_LATIN = 180
_MAX_CJK = 60
_CUT_CHARS = " 、，,;；:"


def _safe_cut(buf: str, maxc: int) -> int:
    window = buf[:maxc]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in _CUT_CHARS:
            return i + 1
    return maxc


class SentenceSpeaker:
    """Speak a streaming reply. feed() buffers deltas, emits complete sentences
    (script-aware split + length cap), routes each into (lang, chunk) pieces, and
    a synth-ahead pipeline plays them through one resampled stream so language
    switches are gapless. interrupt() stops immediately.
    """

    def __init__(self, tts):
        self.tts = tts
        self._buf = ""
        self._text_q: queue.Queue = queue.Queue()
        self._pipeline = callable(getattr(tts, "synth_lang", None))
        self._stop = threading.Event()
        self._stream = None
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
            item = self._text_q.get()
            if item is None:
                self._audio_q.put(None)
                self._text_q.task_done()
                return
            lang, text = item
            try:
                if not self._stop.is_set():
                    audio = self.tts.synth_lang(text, lang)
                    if audio is not None and not self._stop.is_set():
                        self._audio_q.put(audio)
            except Exception as exc:
                print(f"\n(TTS skipped a chunk: {exc})", flush=True)
            self._text_q.task_done()

    def _play_loop(self) -> None:
        while True:
            item = self._audio_q.get()
            if item is None:
                self._end_stream(drain=True)
                self._audio_q.task_done()
                return
            samples, sr = item
            try:
                if not self._stop.is_set():
                    if self._stream is None:
                        self._stream = self._sd.OutputStream(
                            samplerate=_TARGET_SR, channels=1, dtype="float32")
                        self._stream.start()
                    self._stream.write(
                        _resample(np.asarray(samples, dtype=np.float32), sr).reshape(-1, 1))
            except Exception:
                self._end_stream(drain=False)
            self._audio_q.task_done()

    def _say_loop(self) -> None:
        while True:
            item = self._text_q.get()
            if item is None:
                self._text_q.task_done()
                return
            _lang, text = item
            try:
                if not self._stop.is_set():
                    self.tts.speak(text)
            except Exception:
                pass
            self._text_q.task_done()

    def _end_stream(self, drain: bool) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop() if drain else stream.abort()
                stream.close()
            except Exception:
                pass

    # -- public API --
    def feed(self, delta: str) -> None:
        self._buf += delta
        while True:
            m = _SENT_END.search(self._buf)
            cap = _MAX_CJK if detect_lang(self._buf[:32]) in ("zh", "ja") else _MAX_LATIN
            if m and m.end() <= cap:
                end = m.end()
            elif len(self._buf) >= cap:
                end = _safe_cut(self._buf, cap)
            else:
                return
            seg = clean_for_speech(self._buf[:end])
            self._buf = self._buf[end:]
            for lang, chunk in route_text(seg):
                if chunk.strip():
                    self._text_q.put((lang, chunk))

    def finish(self) -> None:
        """Flush the trailing sentence and block until playback drains."""
        tail = clean_for_speech(self._buf)
        self._buf = ""
        for lang, chunk in route_text(tail):
            if chunk.strip():
                self._text_q.put((lang, chunk))
        self._text_q.join()
        if self._pipeline:
            self._audio_q.join()
            self._end_stream(drain=True)

    def interrupt(self) -> None:
        """Stop speaking now: drop queued speech and abort playback."""
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
        self._end_stream(drain=False)
        self._buf = ""
        self._stop.clear()

    def close(self) -> None:
        self._text_q.put(None)
        self.tts.close()
