import argparse
import io
import json

import pytest

pytestmark = pytest.mark.unit


def _args(path=None, session_id=None, client="claude-code"):
    return argparse.Namespace(
        path=path, session_id=session_id, client=client, debug=False
    )


def test_resolve_input_prefers_path_arg():
    from landscape.cli.transcript import _resolve_input

    path, session_id = _resolve_input(_args(path="/tmp/t.jsonl", session_id="s1"))
    assert str(path) == "/tmp/t.jsonl"
    assert session_id == "s1"


def test_resolve_input_reads_hook_json_from_stdin(monkeypatch):
    from landscape.cli import transcript

    payload = {"transcript_path": "/tmp/hook.jsonl", "session_id": "hook-sess"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    path, session_id = transcript._resolve_input(_args())
    assert str(path) == "/tmp/hook.jsonl"
    assert session_id == "hook-sess"


def test_resolve_input_accepts_camelcase_hook_keys(monkeypatch):
    from landscape.cli import transcript

    # Claude Code hook payloads historically use camelCase keys.
    payload = {"transcriptPath": "/tmp/hook.jsonl", "sessionID": "hook-sess"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    path, session_id = transcript._resolve_input(_args())
    assert str(path) == "/tmp/hook.jsonl"
    assert session_id == "hook-sess"


def test_resolve_input_raises_without_path_or_stdin(monkeypatch):
    from landscape.cli import transcript

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(ValueError, match="transcript"):
        transcript._resolve_input(_args())


# --- handle_ingest_transcript orchestration (monkeypatched, no DB) ---


async def _anoop(*args, **kwargs):
    return None


@pytest.mark.asyncio
async def test_handle_ingest_transcript_missing_path_returns_1(capsys):
    from landscape.cli.transcript import handle_ingest_transcript

    rc = await handle_ingest_transcript(_args(path="/no/such/transcript.jsonl"))
    assert rc == 1
    assert "not found" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_handle_ingest_transcript_no_eligible_turns_returns_0(monkeypatch, tmp_path):
    from landscape.cli import transcript

    f = tmp_path / "t.jsonl"
    f.write_text("{}")
    monkeypatch.setattr("landscape.ingestion.transcript.read_transcript", lambda *a, **k: [])

    rc = await transcript.handle_ingest_transcript(_args(path=str(f)))
    assert rc == 0


@pytest.mark.asyncio
async def test_handle_ingest_transcript_no_session_returns_1(monkeypatch, tmp_path, capsys):
    from landscape.cli import transcript
    from landscape.conversation_ingestion import ConversationTurn

    f = tmp_path / "t.jsonl"
    f.write_text("{}")
    # turns with no session id and no --session-id override -> cannot resolve
    monkeypatch.setattr(
        "landscape.ingestion.transcript.read_transcript",
        lambda *a, **k: [ConversationTurn(session_id="", turn_id="t1", role="user", text="hi")],
    )

    rc = await transcript.handle_ingest_transcript(_args(path=str(f)))
    assert rc == 1
    assert "session id" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_handle_ingest_transcript_happy_path_ingests(monkeypatch, tmp_path):
    from landscape.cli import transcript
    from landscape.conversation_ingestion import ConversationTurn

    f = tmp_path / "t.jsonl"
    f.write_text("{}")
    turns = [ConversationTurn(session_id="s1", turn_id="t1", role="user", text="I lead Landscape.")]
    monkeypatch.setattr("landscape.ingestion.transcript.read_transcript", lambda *a, **k: turns)
    monkeypatch.setattr("landscape.extraction.salience.select_salient", lambda t: ["SAL"])
    monkeypatch.setattr("landscape.embeddings.encoder.load_model", lambda: None)
    monkeypatch.setattr("landscape.storage.qdrant_store.init_collection", _anoop)
    monkeypatch.setattr("landscape.storage.qdrant_store.init_chunks_collection", _anoop)
    # close_runtime is imported at module scope in cli.transcript
    monkeypatch.setattr("landscape.cli.transcript.close_runtime", _anoop)

    calls = {}

    async def _capture(session_id, salient, *, debug=False):
        calls.update(session_id=session_id, salient=salient, debug=debug)

    monkeypatch.setattr(
        "landscape.conversation_ingestion.ingest_conversation_window", _capture
    )

    rc = await transcript.handle_ingest_transcript(_args(path=str(f)))
    assert rc == 0
    assert calls == {"session_id": "s1", "salient": ["SAL"], "debug": False}
