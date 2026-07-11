"""Terminal chat for iBeto: `uv run ibeto` / `python -m ibeto`."""

import argparse
import sys
import time

from openai import APIConnectionError

from ibeto.config import load_config
from ibeto.core.session import ConversationSession
from ibeto.llm.lmstudio import LMStudioBackend
from ibeto.prompts import load_prompt

EXIT_WORDS = {"exit", "quit", ":q"}


def _print_stats(started: float, first_token_at: float | None, backend) -> None:
    now = time.perf_counter()
    ttft = f"{first_token_at - started:.2f}s" if first_token_at else "n/a"
    usage = backend.last_usage
    if usage and first_token_at and now > first_token_at:
        rate = f"{usage.completion_tokens / (now - first_token_at):.0f} tok/s"
    else:
        rate = "n/a"
    print(f"\033[90m[TTFT {ttft} · {rate}]\033[0m")


def run(stats: bool = False) -> int:
    cfg = load_config()
    backend = LMStudioBackend(
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
    )
    session = ConversationSession(backend, load_prompt(cfg.system_prompt))

    print("iBeto v0.1")
    print(f"Connected to LM Studio  ·  model: {cfg.model}")
    print("Type 'exit' or Ctrl-D to quit.\n")

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

        print("Assistant > ", end="", flush=True)
        started = time.perf_counter()
        first_token_at: float | None = None
        try:
            for delta in session.stream(user):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                print(delta, end="", flush=True)
        except APIConnectionError:
            print(
                "\n\033[91mCannot reach LM Studio at "
                f"{cfg.base_url}.\033[0m\n"
                "Start the server in LM Studio (Developer tab) and load a model, "
                "then run scripts/setup.sh to verify.",
                file=sys.stderr,
            )
            return 1
        print()
        if stats:
            _print_stats(started, first_token_at, backend)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="ibeto", description="Local-first AI companion.")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show TTFT and tokens/sec after each reply.",
    )
    args = parser.parse_args()
    sys.exit(run(stats=args.stats))
