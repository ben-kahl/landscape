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


def test_resolve_input_raises_without_path_or_stdin(monkeypatch):
    from landscape.cli import transcript

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(ValueError, match="transcript"):
        transcript._resolve_input(_args())
