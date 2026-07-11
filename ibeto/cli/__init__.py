"""Terminal chat for iBeto: `uv run ibeto` / `python -m ibeto`.

Text mode by default; `--voice` enables push-to-talk speech.

Runtime controls (any time, mid-conversation):
  /think on|off    reasoning mode          · say "think harder" (voice)
  /look [question] use the camera          · say "look at this" (voice)
  /model [name]    list or switch model
In voice mode you can also TYPE any of these at the "[Enter to speak]" prompt
(or type a message instead of speaking); press Enter alone to record audio.
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

EXIT_WORDS = {"exit", "quit", ":q", "/exit", "/quit"}
LOOK_CMD = "/look"
THINK_CMD = "/think"
MODEL_CMD = "/model"
DEFAULT_LOOK_PROMPT = "What do you see? Describe it briefly."

# Spoken phrases that switch reasoning on/off in voice mode.
THINK_ON_PHRASES = ("think harder", "turn on thinking", "enable thinking", "start thinking")
THINK_OFF_PHRASES = ("turn off thinking", "disable thinking", "stop thinking", "don't think")
# Spoken phrases that trigger a camera capture for the current turn.
LOOK_TRIGGERS = (
    "look at", "what do you see", "what am i looking at",
    "see this", "can you see", "look here",
)

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
        enable_thinking=cfg.enable_thinking,
        max_tokens=cfg.max_tokens,
    )
    history = load_history(cfg.history_path()) if resume else []
    session = ConversationSession(backend, load_prompt(cfg.system_prompt), history=history)
    return backend, session, history


def _stream_and_print(
    session, backend, user_text: str, stats: bool, image: str | None = None
) -> str | None:
    """Stream one reply to stdout. Returns the reply text, or None on failure."""
    print("Assistant > ", end="", flush=True)
    started = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    try:
        for delta in session.stream(user_text, image=image):
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


def _capture_image(cfg) -> str | None:
    """Capture one camera frame as a data URL, or None on error (message printed)."""
    from ibeto.vision.capture import capture_frame

    print("Looking...", flush=True)
    try:
        return capture_frame(cfg.camera_index)
    except Exception as exc:  # camera errors are expected/recoverable
        print(f"\033[91mCamera error: {exc}\033[0m", file=sys.stderr)
        return None


def _set_thinking(backend, on: bool) -> str:
    backend.enable_thinking = on
    return f"Thinking mode: {'ON (reasoning)' if on else 'OFF (fast)'}"


def _chat_model_ids(backend) -> list[str]:
    """Downloaded models that can chat (exclude embedding models)."""
    return [m.id for m in backend.list_models().data if "embed" not in m.id.lower()]


def _model_command(arg: str, backend) -> str:
    """List models (no arg) or switch to one (by index or name substring)."""
    from ibeto.llm.manager import lms_available, load_model

    ids = _chat_model_ids(backend)
    if not arg:
        lines = [
            f"  {i}. {mid}{'  *' if mid == backend.model else ''}"
            for i, mid in enumerate(ids)
        ]
        return "Models (use /model <number|name>):\n" + "\n".join(lines)

    if arg.isdigit() and int(arg) < len(ids):
        target = ids[int(arg)]
    else:
        matches = [m for m in ids if arg.lower() in m.lower()]
        if len(matches) > 1:
            return "Multiple matches: " + ", ".join(matches)
        if not matches:
            return f"No model matches '{arg}'."
        target = matches[0]

    if target == backend.model:
        return f"Already using {target}."
    if not lms_available():
        backend._model = target
        return f"Model set to {target} (loads on next reply)."

    print(f"Loading {target}... (unloads the current model)", flush=True)
    try:
        load_model(target)
    except Exception as exc:
        return f"Failed to load {target}: {exc}"
    backend._model = target
    return f"Model loaded: {target}"


def _handle_slash(text: str, backend, session, cfg, stats: bool, speak=None) -> str:
    """Handle a /command. Returns: exit | fatal | handled | notcmd."""
    low = text.lower().strip()
    cmd = low.split()[0]

    if cmd in EXIT_WORDS:
        return "exit"

    if cmd == THINK_CMD:
        arg = low[len(THINK_CMD):].strip()
        if arg == "on":
            on = True
        elif arg == "off":
            on = False
        else:
            on = not backend.enable_thinking
        msg = _set_thinking(backend, on)
        print(msg)
        if speak:
            speak(msg)
        return "handled"

    if cmd == MODEL_CMD:
        msg = _model_command(text[len(MODEL_CMD):].strip(), backend)
        print(msg)
        if speak and msg.startswith("Model loaded"):
            speak("Model switched.")
        return "handled"

    if cmd == LOOK_CMD:
        image = _capture_image(cfg)
        if image is None:
            return "handled"
        prompt = text[len(LOOK_CMD):].strip() or DEFAULT_LOOK_PROMPT
        reply = _stream_and_print(session, backend, prompt, stats, image=image)
        if reply is None:
            return "fatal"
        if speak and reply:
            speak(reply)
        return "handled"

    return "notcmd"


def _ensure_model_loaded(cfg, backend) -> None:
    """Load the configured model in LM Studio if it isn't already loaded."""
    from ibeto.llm.manager import lms_available, loaded_models

    if not lms_available() or backend.model in loaded_models(cfg.base_url):
        return
    print(f"Loading {backend.model}... (first launch may take ~15s)", flush=True)
    try:
        from ibeto.llm.manager import load_model

        load_model(backend.model)
    except Exception as exc:
        print(f"(could not preload model: {exc})", file=sys.stderr)


def _startup_banner(cfg, backend, resume, history, extra: str = "") -> None:
    print(f"Connected to LM Studio  ·  model: {backend.model}")
    print(f"Thinking: {'ON' if backend.enable_thinking else 'OFF'}")
    if resume and history:
        print(f"Resumed {len(history) // 2} previous exchange(s).")
    if extra:
        print(extra)


def run(stats: bool = False, resume: bool = False, think: bool | None = None) -> int:
    cfg = load_config()
    backend, session, history = _build_session(cfg, resume)
    if think is not None:
        backend.enable_thinking = think

    print("iBeto v0.1")
    _ensure_model_loaded(cfg, backend)
    _startup_banner(cfg, backend, resume, history,
                    "Commands: /look [q] · /think on|off · /model [name] · exit\n")

    try:
        while True:
            try:
                user = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return 0
            if not user:
                continue
            if user.startswith("/") or user.lower() in EXIT_WORDS:
                status = _handle_slash(user, backend, session, cfg, stats)
                if status == "exit":
                    print("Bye.")
                    return 0
                if status == "fatal":
                    return 1
                if status == "notcmd":
                    print("Unknown command. Try /look, /think, /model, or exit.")
                print()
                continue
            if _stream_and_print(session, backend, user, stats) is None:
                return 1
            print()
    finally:
        save_history(session.messages, cfg.history_path())


def _voice_control(text: str, backend) -> str | None:
    """Handle spoken thinking-toggle commands. Returns a spoken reply, or None."""
    low = text.lower()
    if any(p in low for p in THINK_ON_PHRASES):
        return _set_thinking(backend, True)
    if any(p in low for p in THINK_OFF_PHRASES):
        return _set_thinking(backend, False)
    return None


def run_voice(stats: bool = False, resume: bool = False, think: bool | None = None) -> int:
    cfg = load_config()
    backend, session, history = _build_session(cfg, resume)
    if think is not None:
        backend.enable_thinking = think

    # Import audio deps lazily so text mode never loads them.
    from ibeto.audio.mic import record_until_enter
    from ibeto.audio.stt import WhisperSTT
    from ibeto.audio.tts import speak

    print("iBeto v0.1 — voice mode")
    _ensure_model_loaded(cfg, backend)
    print(f"Loading Whisper ({cfg.whisper_model})...", flush=True)
    stt = WhisperSTT(cfg.whisper_model, cfg.stt_language)
    _startup_banner(cfg, backend, resume, history)
    print(
        "Press Enter to speak, Enter again to stop. Ctrl-C to quit.\n"
        'Say "look at this" / "think harder", or TYPE a command (/model, /think,\n'
        "/look) or a message at the prompt instead of speaking.\n"
    )
    say = lambda t: speak(t, cfg.tts_voice)  # noqa: E731

    try:
        while True:
            try:
                typed = input("[Enter to speak] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return 0

            spoken = False
            if typed:
                if typed.startswith("/") or typed.lower() in EXIT_WORDS:
                    status = _handle_slash(typed, backend, session, cfg, stats, speak=say)
                    if status == "exit":
                        print("Bye.")
                        return 0
                    if status == "fatal":
                        return 1
                    if status == "notcmd":
                        print("Unknown command. Try /look, /think, /model, or exit.")
                    print()
                    continue
                user_text = typed  # typed message instead of speaking
            else:
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
                spoken = True

            image = None
            if spoken:
                control = _voice_control(user_text, backend)
                if control is not None:
                    print(control)
                    say(control)
                    print()
                    continue
                if any(t in user_text.lower() for t in LOOK_TRIGGERS):
                    image = _capture_image(cfg)

            reply = _stream_and_print(session, backend, user_text, stats, image=image)
            if reply is None:
                return 1
            say(reply)
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
        "--think",
        action="store_true",
        help="Start with reasoning mode on (default off for fast replies).",
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
    think = True if args.think else None
    sys.exit(entry(stats=args.stats, resume=args.resume, think=think))
