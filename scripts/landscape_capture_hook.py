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

    event_name = ""
    if isinstance(payload, dict):
        event_name = str(payload.get("hook_event_name") or payload.get("event_name") or "")
    _debug(
        f"--- fired client={client} event={event_name or '?'} "
        f"keys={sorted(payload) if isinstance(payload, dict) else type(payload).__name__}"
    )
    if event_name == "SessionEnd":
        session_id = ""
        if isinstance(payload, dict):
            session_id = str(payload.get("session_id") or payload.get("sessionID") or "")
        if session_id:
            end_url = os.environ.get(
                "LANDSCAPE_SESSION_END_URL",
                DEFAULT_HOOK_URL.replace("/conversation-turn", "/session-end"),
            )
            status = _post_turn(end_url, token, {"client": client, "session_id": session_id})
            _debug(f"session-end session_id={session_id!r} -> {status}")
        else:
            _debug("session-end: no session_id in payload, skipped")
        return 0

    turns = extract_turns(client, payload)
    _debug(
        f"extracted {len(turns)} turn(s): "
        + "; ".join(f"{t.role}/{t.turn_id}={t.text[:50]!r}" for t in turns)
    )
    for turn in turns:
        status = _post_turn(hook_url, token, turn.__dict__)
        _debug(f"posted turn {turn.turn_id} -> {status}")
    return 0


def _debug(message: str) -> None:
    """Append a diagnostic line to $LANDSCAPE_HOOK_DEBUG when set; otherwise no-op."""
    path = os.environ.get("LANDSCAPE_HOOK_DEBUG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except OSError:
        pass


def _post_turn(hook_url: str, token: str | None, body: dict[str, str]) -> str:
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
            preview = response.read().decode("utf-8", "replace")[:160]
            return f"{response.status} {preview}"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:160]
        message = f"HTTP {exc.code} {detail}"
        print(f"landscape hook capture failed: {message}", file=sys.stderr)
        return message
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"landscape hook capture failed: {exc}", file=sys.stderr)
        return f"error {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
