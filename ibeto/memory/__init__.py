"""Conversation persistence: save/load dialogue turns as JSON."""

import json
from pathlib import Path


def load_history(path: Path) -> list[dict]:
    """Return prior user/assistant turns, or [] if missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [m for m in data if m.get("role") in ("user", "assistant")]


def save_history(messages: list[dict], path: Path) -> None:
    """Write conversational turns (system prompt excluded) to JSON."""
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
