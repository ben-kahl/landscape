#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from landscape.hooks.adapters import extract_turns

DEFAULT_HOOK_URL = "http://127.0.0.1:8000/hooks/conversation-turn"


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    client = argv[0] if argv else os.environ.get("LANDSCAPE_HOOK_CLIENT", "unknown")
    hook_url = os.environ.get("LANDSCAPE_HOOK_URL", DEFAULT_HOOK_URL)
    token = os.environ.get("LANDSCAPE_API_TOKEN")

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"landscape hook ignored invalid JSON: {exc}", file=sys.stderr)
        return 0

    turns = extract_turns(client, payload)
    for turn in turns:
        _post_turn(hook_url, token, turn.__dict__)
    return 0


def _post_turn(hook_url: str, token: str | None, body: dict[str, str]) -> None:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        hook_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"landscape hook capture failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
