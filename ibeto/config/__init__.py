"""Configuration loading from configs/ibeto.toml with safe defaults."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ibeto.toml"


@dataclass
class Config:
    base_url: str = "http://localhost:1234/v1"
    model: str = "gemma-4-12b-it-mlx"
    temperature: float = 0.7
    enable_thinking: bool = False  # off = fast replies; toggle at runtime with /think
    max_tokens: int = 2048  # hard cap on reply length (seatbelt vs runaway generation)
    system_prompt: str = "assistant"
    history_file: str = "chat_history.json"
    # Voice mode
    whisper_model: str = "base"  # base < small < medium < large-v3 (accuracy vs speed)
    stt_language: str = "en"  # force transcription language; "" = auto-detect
    # "xtts" = one neural voice for every language (needs the `xtts` extra);
    # "kokoro" = fast native per-language voices; "say" = robotic macOS fallback.
    tts_engine: str = "xtts"
    tts_xtts_speaker: str = "Claribel Dervla"  # XTTS built-in speaker (the one voice)
    tts_voice: str = "bf_isabella"  # kokoro engine: default/Latin voice (or `say -v` name)
    tts_speed: float = 1.0  # kokoro speaking rate (0.5-2.0)
    tts_model_dir: str = ""  # kokoro cache dir; "" = ~/.cache/ibeto/kokoro
    # Per-language voices: replies are routed by script (see audio/tts.detect_lang).
    tts_voice_zh: str = "zf_xiaobei"  # kokoro Mandarin voice for Chinese replies
    tts_voice_ja: str = "Kyoko"  # macOS JA voice (Kokoro/espeak can't read kanji)
    tts_voice_ar: str = "ar_JO-kareem-medium"  # piper voice for Arabic replies
    tts_voice_de: str = "de_DE-thorsten-medium"  # piper voice for German (Kokoro lacks de)
    tts_voice_es: str = "ef_dora"  # kokoro Spanish voice
    tts_voice_fr: str = "ff_siwis"  # kokoro French voice
    tts_voice_it: str = "if_sara"  # kokoro Italian voice
    tts_voice_pt: str = "pf_dora"  # kokoro Portuguese voice
    tts_piper_dir: str = ""  # piper cache dir; "" = ~/.cache/ibeto/piper
    sample_rate: int = 16000  # Whisper expects 16 kHz mono
    # Vision
    camera_index: int = 0  # OpenCV camera index (iPhone via Continuity is usually 0 or 1)

    def history_path(self) -> Path:
        """Absolute path to the history file (anchored at the project root)."""
        p = Path(self.history_file)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load config from TOML, ignoring unknown keys. Falls back to defaults."""
    if not path.exists():
        return Config()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in known})
