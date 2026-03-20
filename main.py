"""
Terminal chat with gpt-5-nano and get_weather tool. Type 'end' to quit.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from llm_client import MODEL, run_turn

_LOG_FILE = Path(__file__).resolve().parent / "weather_chat.log"


def _configure_audit_logging() -> None:
    """Append JSON audit lines (user query, tool params, weather payload) to weather_chat.log."""
    audit = logging.getLogger("weather_chat.audit")
    if audit.handlers:
        return
    audit.setLevel(logging.INFO)
    audit.propagate = False
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(message)s")
    )
    audit.addHandler(fh)


logging.basicConfig(level=logging.WARNING)


def _require_api_key() -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print(
            "Missing OPENAI_API_KEY. Set it in the environment or in a .env file "
            "(see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather-aware LLM chat (terminal)")
    parser.add_argument(
        "--single",
        metavar="QUERY",
        help="Run one query and exit (non-interactive)",
    )
    args = parser.parse_args()

    _configure_audit_logging()
    _require_api_key()
    client = OpenAI()
    conversation: list[dict] = []

    if args.single is not None:
        try:
            reply = run_turn(client, conversation, args.single)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        print(reply)
        return

    print(f"Weather chat — model {MODEL}. Type 'end' (or 'quit' / 'exit') to stop.\n")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line.casefold() in ("end", "quit", "exit"):
            break
        if not line:
            continue

        try:
            reply = run_turn(client, conversation, line)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            continue
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
