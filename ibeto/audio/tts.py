"""Text-to-speech via the macOS `say` command (fully local, zero deps).

Isolated behind speak() so a neural engine (Kokoro/Piper) can replace it later
without touching the conversation code.
"""

import subprocess


def speak(text: str, voice: str = "") -> None:
    if not text.strip():
        return
    args = ["say"]
    if voice:
        args += ["-v", voice]
    args.append(text)
    subprocess.run(args, check=False)
