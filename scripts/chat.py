#!/usr/bin/env python3
"""Interactive terminal chat with rolling memory and token streaming."""

import json
import sys
import urllib.request
from collections import deque

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen"  # served model name (see docker-compose.yml)
MAX_HISTORY = 20  # rolling window (pairs of user+assistant turns)


def stream_chat(history: list) -> str:
    payload = json.dumps({
        "model": MODEL,
        "messages": history,
        "stream": True,
    }).encode()

    req = urllib.request.Request(URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    collected = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    collected.append(delta)
            except (KeyError, json.JSONDecodeError):
                pass
    print()
    return "".join(collected)


def main():
    history = deque(maxlen=MAX_HISTORY * 2)  # each turn = 2 messages
    print(f"Chat — model: {MODEL}  (history: last {MAX_HISTORY} turns)")
    print("Type 'exit' or Ctrl-C to quit.\n")

    while True:
        try:
            user_input = input("\033[1mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        history.append({"role": "user", "content": user_input})
        print("\033[1mAssistant:\033[0m ", end="", flush=True)

        try:
            reply = stream_chat(list(history))
        except Exception as e:
            print(f"\n[error: {e}]")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
