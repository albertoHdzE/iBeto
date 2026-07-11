"""Microphone capture: push-to-talk recording into a numpy array."""

import numpy as np
import sounddevice as sd


def record_until_enter(sample_rate: int = 16000) -> np.ndarray:
    """Record mono audio until the user presses Enter.

    Returns a float32 array at `sample_rate`, ready for Whisper.
    """
    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        input()  # blocks the caller's turn until Enter

    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten()
