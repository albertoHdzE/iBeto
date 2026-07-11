"""Model management via the LM Studio `lms` CLI.

On 8 GB only one model fits, so switching = unload all, then load the target.
"""

import json
import subprocess
import urllib.request
from pathlib import Path

LMS = Path.home() / ".lmstudio" / "bin" / "lms"


def lms_available() -> bool:
    return LMS.exists()


def loaded_models(base_url: str = "http://localhost:1234/v1") -> list[str]:
    """Ids of currently-loaded models, via LM Studio's native REST API."""
    native = base_url.rstrip("/").removesuffix("/v1") + "/api/v0/models"
    try:
        with urllib.request.urlopen(native, timeout=5) as resp:
            data = json.load(resp)
        return [m["id"] for m in data.get("data", []) if m.get("state") == "loaded"]
    except Exception:
        return []


def ensure_loaded(key: str, base_url: str = "http://localhost:1234/v1") -> bool:
    """Load `key` if it isn't already loaded. Returns True if a load happened."""
    if not lms_available() or key in loaded_models(base_url):
        return False
    load_model(key)
    return True


def _lms(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(LMS), *args], capture_output=True, text=True, timeout=timeout
    )


def load_model(key: str) -> None:
    """Unload the current model and load `key`. Raises on failure."""
    _lms("unload", "--all")
    result = _lms("load", key, "-y")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "lms load failed")
