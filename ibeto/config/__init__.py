"""Configuration loading from configs/ibeto.toml with safe defaults."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "ibeto.toml"


@dataclass
class Config:
    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen3.5-4b-instruct-revised"
    temperature: float = 0.7
    system_prompt: str = "assistant"


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load config from TOML, ignoring unknown keys. Falls back to defaults."""
    if not path.exists():
        return Config()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in known})
