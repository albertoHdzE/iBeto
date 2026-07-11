"""Configuration loading from configs/ibeto.toml with safe defaults."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ibeto.toml"


@dataclass
class Config:
    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen3.5-4b-instruct-revised"
    temperature: float = 0.7
    enable_thinking: bool = False  # off = fast replies; toggle at runtime with /think
    max_tokens: int = 800  # hard cap on reply length (seatbelt vs runaway generation)
    system_prompt: str = "assistant"
    history_file: str = "chat_history.json"
    # Voice mode
    whisper_model: str = "base"  # base < small < medium < large-v3 (accuracy vs speed)
    stt_language: str = "en"  # force transcription language; "" = auto-detect
    tts_voice: str = ""  # macOS `say` voice; empty = system default
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
