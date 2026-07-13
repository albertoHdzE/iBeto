"""Standalone XTTS-v2 worker — runs in its OWN dependency environment.

coqui-tts pins numpy<2 while the main app needs numpy>=2, so XTTS can't live in
the same environment. The main app launches this file with
`uv run --no-project --with coqui-tts ...`, which builds an isolated env, and
talks to it over stdio.

Protocol (one message per line):
  in  (stdin):  {"text": "...", "lang": "de", "speaker": "Claribel Dervla"}
  out (stdout): "READY" once loaded, then "OK <wav_path>" or "ERR <message>"
The wav is 16-bit mono 24 kHz; the main app reads then deletes it.
"""

import json
import os
import sys
import tempfile
import wave

os.environ.setdefault("COQUI_TOS_AGREED", "1")


def main() -> None:
    import numpy as np
    from TTS.api import TTS

    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            wav = model.tts(text=req["text"], language=req["lang"],
                            speaker=req.get("speaker", "Claribel Dervla"))
            pcm = (np.clip(np.asarray(wav, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(pcm.tobytes())
            print("OK " + path, flush=True)
        except Exception as exc:  # never die on one bad request
            print("ERR " + str(exc).replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
