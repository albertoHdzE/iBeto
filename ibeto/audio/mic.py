"""Microphone capture: push-to-talk recording into a numpy array."""

import numpy as np
import sounddevice as sd


class NoInputDevice(RuntimeError):
    """Raised when no microphone / input device is available."""


def has_input_device() -> bool:
    """True if PortAudio has at least one usable audio input device."""
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            return int(sd.query_devices(default_in)["max_input_channels"]) > 0
    except Exception:
        pass
    # No valid default: fall back to scanning every device for an input.
    try:
        return any(d["max_input_channels"] > 0 for d in sd.query_devices())
    except Exception:
        return False


def record_until_enter(sample_rate: int = 16000) -> np.ndarray:
    """Record mono audio until the user presses Enter.

    Returns a float32 array at `sample_rate`, ready for Whisper.
    Raises NoInputDevice if there is no microphone to record from.
    """
    if not has_input_device():
        raise NoInputDevice(
            "No microphone found. Connect an input device (a Bluetooth speaker "
            "in output-only mode does not count) and try again."
        )

    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, _status):
        frames.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            input()  # blocks the caller's turn until Enter
    except sd.PortAudioError as exc:
        raise NoInputDevice(f"Could not open the microphone: {exc}") from exc

    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten()
