"""Terminal chat for iBeto: `uv run ibeto` / `python -m ibeto`.

Text mode by default; `--voice` enables push-to-talk speech.
"""

import argparse
import sys
import time

from openai import APIConnectionError

from ibeto.config import Config, load_config
from ibeto.core.session import ConversationSession
from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.memory import load_history, save_history
from ibeto.prompts import load_prompt

EXIT_WORDS = {"exit", "quit", ":q"}

_CONN_ERROR = (
    "\n\033[91mCannot reach LM Studio.\033[0m\n"
    "Start the server in LM Studio (Developer tab) and load a model, "
    "then run scripts/setup.sh to verify."
)


def _print_stats(started: float, first_token_at: float | None, backend) -> None:
    now = time.perf_counter()
    ttft = f"{first_token_at - started:.2f}s" if first_token_at else "n/a"
    usage = backend.last_usage
    if usage and first_token_at and now > first_token_at:
        rate = f"{usage.completion_tokens / (now - first_token_at):.0f} tok/s"
    else:
        rate = "n/a"
    print(f"\033[90m[TTFT {ttft} · {rate}]\033[0m")


def _build_session(cfg: Config, resume: bool):
    backend = LMStudioBackend(
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
    )
    history = load_history(cfg.history_path()) if resume else []
    session = ConversationSession(backend, load_prompt(cfg.system_prompt), history=history)
    return backend, session, history


def _stream_and_print(session, backend, user_text: str, stats: bool) -> str | None:
    """Stream one reply to stdout. Returns the reply text, or None on failure."""
    print("Assistant > ", end="", flush=True)
    started = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    try:
        for delta in session.stream(user_text):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(delta, end="", flush=True)
            chunks.append(delta)
    except APIConnectionError:
        print(_CONN_ERROR, file=sys.stderr)
        return None
    print()
    if stats:
        _print_stats(started, first_token_at, backend)
    return "".join(chunks)


def run(stats: bool = False, resume: bool = False) -> int:
    cfg = load_config()
    backend, session, history = _build_session(cfg, resume)

    print("iBeto v0.1")
    print(f"Connected to LM Studio  ·  model: {cfg.model}")
    if resume and history:
        print(f"Resumed {len(history) // 2} previous exchange(s).")
    print("Type 'exit' or Ctrl-D to quit.\n")

    try:
        while True:
            try:
                user = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return 0
            if not user:
                continue
            if user.lower() in EXIT_WORDS:
                print("Bye.")
                return 0
            if _stream_and_print(session, backend, user, stats) is None:
                return 1
            print()
    finally:
        save_history(session.messages, cfg.history_path())


def run_voice(stats: bool = False, resume: bool = False) -> int:
    cfg = load_config()
    backend, session, history = _build_session(cfg, resume)

    # Import audio deps lazily so text mode never loads them.
    from ibeto.audio.mic import record_until_enter
    from ibeto.audio.stt import WhisperSTT
    from ibeto.audio.tts import speak

    print("iBeto v0.1 — voice mode")
    print(f"Connected to LM Studio  ·  model: {cfg.model}")
    print(f"Loading Whisper ({cfg.whisper_model})...", flush=True)
    stt = WhisperSTT(cfg.whisper_model)
    if resume and history:
        print(f"Resumed {len(history) // 2} previous exchange(s).")
    print("Press Enter to start speaking, Enter again to stop. Ctrl-C to quit.\n")

    try:
        while True:
            try:
                input("[Enter to speak] ")
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return 0
            print("Recording... press Enter to stop.", flush=True)
            audio = record_until_enter(cfg.sample_rate)
            if audio.size == 0:
                continue
            print("Transcribing...", flush=True)
            user_text = stt.transcribe(audio)
            if not user_text:
                print("(heard nothing)\n")
                continue
            print(f"You > {user_text}")
            reply = _stream_and_print(session, backend, user_text, stats)
            if reply is None:
                return 1
            speak(reply, cfg.tts_voice)
            print()
    finally:
        save_history(session.messages, cfg.history_path())


def main() -> None:
    parser = argparse.ArgumentParser(prog="ibeto", description="Local-first AI companion.")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Voice mode: push-to-talk speech in, spoken reply out.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show TTFT and tokens/sec after each reply.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the previous conversation from the saved history file.",
    )
    args = parser.parse_args()
    entry = run_voice if args.voice else run
    sys.exit(entry(stats=args.stats, resume=args.resume))
