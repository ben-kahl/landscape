from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


async def test_hook_receiver_schedules_normalized_turn(http_client, monkeypatch):
    import landscape.mcp_app as mcp_app

    scheduled = []

    def fake_schedule_auto_ingestion(text, session_id, turn_id, role="user", debug=False):
        scheduled.append(
            {
                "text": text,
                "session_id": session_id,
                "turn_id": turn_id,
                "role": role,
                "debug": debug,
            }
        )
        return object()

    monkeypatch.setattr(mcp_app, "_schedule_auto_ingestion", fake_schedule_auto_ingestion)

    response = await http_client.post(
        "/hooks/conversation-turn",
        json={
            "client": "claude-code",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "role": "User",
            "text": "Alice joined Beacon Labs.",
            "debug": True,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"accepted": True, "scheduled": True}
    assert scheduled == [
        {
            "text": "Alice joined Beacon Labs.",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "role": "user",
            "debug": True,
        }
    ]


async def test_hook_receiver_rejects_ineligible_turn(http_client, monkeypatch):
    import landscape.mcp_app as mcp_app

    scheduled = []

    def fake_schedule_auto_ingestion(text, session_id, turn_id, role="user", debug=False):
        scheduled.append((text, session_id, turn_id, role, debug))
        return object()

    monkeypatch.setattr(mcp_app, "_schedule_auto_ingestion", fake_schedule_auto_ingestion)

    response = await http_client.post(
        "/hooks/conversation-turn",
        json={
            "client": "codex",
            "session_id": "session-1",
            "turn_id": "turn-2",
            "role": "tool",
            "text": "internal tool envelope",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"accepted": False, "scheduled": False}
    assert scheduled == []


def test_adapter_extracts_claude_user_prompt():
    from landscape.hooks.adapters import extract_turns

    turns = extract_turns(
        "claude-code",
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "claude-session",
            "prompt": "Remember that Alice prefers Python.",
        },
    )

    assert [turn.__dict__ for turn in turns] == [
        {
            "client": "claude-code",
            "session_id": "claude-session",
            "turn_id": "claude-code:UserPromptSubmit:519bf3a2cf56",
            "role": "user",
            "text": "Remember that Alice prefers Python.",
        }
    ]


def test_adapter_prefers_prompt_over_transcript_on_user_prompt_submit(tmp_path):
    """Real Claude Code UserPromptSubmit payloads carry transcript_path too. The
    user's typed prompt is the signal for that event and must win over the prior
    assistant message already in the transcript — otherwise typed facts (stated
    in prompts) are silently dropped in favor of re-capturing the last answer."""
    from landscape.hooks.adapters import extract_turns

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "message": {
                    "id": "prev-assistant",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "a previous answer"}],
                }
            }
        ),
        encoding="utf-8",
    )

    turns = extract_turns(
        "claude-code",
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "claude-session",
            "transcript_path": str(transcript),
            "prompt": "I lead the Platform team and we use Neo4j.",
        },
    )

    assert [(turn.role, turn.text) for turn in turns] == [
        ("user", "I lead the Platform team and we use Neo4j.")
    ]


def test_adapter_extracts_assistant_turn_from_transcript(tmp_path):
    from landscape.hooks.adapters import extract_turns

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "message": {
                            "id": "msg-user",
                            "role": "user",
                            "content": [{"type": "text", "text": "Who owns Beacon?"}],
                        }
                    }
                ),
                json.dumps(
                    {
                        "message": {
                            "id": "msg-assistant",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Beacon is owned by Helios Robotics.",
                                }
                            ],
                        }
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    turns = extract_turns(
        "claude-code",
        {
            "hook_event_name": "Stop",
            "session_id": "claude-session",
            "transcript_path": str(transcript),
        },
    )

    assert [turn.__dict__ for turn in turns] == [
        {
            "client": "claude-code",
            "session_id": "claude-session",
            "turn_id": "msg-assistant",
            "role": "assistant",
            "text": "Beacon is owned by Helios Robotics.",
        }
    ]


def test_adapter_extracts_opencode_message_event():
    from landscape.hooks.adapters import extract_turns

    turns = extract_turns(
        "opencode",
        {
            "event": {
                "type": "message.updated",
                "message": {
                    "id": "message-9",
                    "sessionID": "open-session",
                    "role": "assistant",
                    "parts": [{"type": "text", "text": "Alice now works at Zylos."}],
                },
            }
        },
    )

    assert [turn.__dict__ for turn in turns] == [
        {
            "client": "opencode",
            "session_id": "open-session",
            "turn_id": "message-9",
            "role": "assistant",
            "text": "Alice now works at Zylos.",
        }
    ]


def test_hook_examples_reference_shared_adapter():
    root = Path(__file__).resolve().parents[1]

    claude = json.loads((root / "hooks/claude-code/settings.example.json").read_text())
    codex = json.loads((root / "hooks/codex/hooks.example.json").read_text())
    opencode = (root / "hooks/opencode/landscape-conversation.js").read_text()

    assert "scripts/landscape_capture_hook.py" in json.dumps(claude)
    assert "scripts/landscape_capture_hook.py" in json.dumps(codex)
    assert "scripts/landscape_capture_hook.py" in opencode


def test_hook_script_is_importable():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/landscape_capture_hook.py"
    spec = importlib.util.spec_from_file_location("landscape_capture_hook", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DEFAULT_HOOK_URL == "http://127.0.0.1:8000/hooks/conversation-turn"
