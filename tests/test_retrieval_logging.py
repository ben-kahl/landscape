import json
from datetime import UTC, datetime

import pytest

from landscape.observability.retrieval_logging import (
    create_retrieval_log_context,
    ensure_retrieval_log_sink,
)


@pytest.mark.unit
def test_retrieval_logs_redact_sensitive_fields_by_default(tmp_path):
    log_dir = tmp_path / "logs" / "retrieval"
    log_path = ensure_retrieval_log_sink(log_dir, force=True)

    ctx = create_retrieval_log_context(
        query_text="Project Atlas budget details",
        hops=2,
        limit=10,
        chunk_limit=3,
        reinforce=True,
        session_id="session-secret",
        since=datetime(2026, 4, 25, tzinfo=UTC),
        debug=False,
    )
    ctx.emit_started()
    ctx.emit_completed(
        result_count=1,
        touched_entity_count=1,
        touched_edge_count=0,
        chunk_count=0,
    )

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").strip().splitlines()
    ]

    assert len(events) == 2
    assert all("query_text" not in event for event in events)
    assert all("session_id" not in event for event in events)
    assert all(event["query_text_sha256"] for event in events)
    assert all(event["session_id_sha256"] for event in events)
    assert events[0]["query_length"] == len("Project Atlas budget details")


@pytest.mark.unit
def test_retrieval_logs_include_sensitive_fields_in_debug_mode(tmp_path):
    log_dir = tmp_path / "logs" / "retrieval"
    log_path = ensure_retrieval_log_sink(log_dir, force=True)

    ctx = create_retrieval_log_context(
        query_text="Project Atlas budget details",
        hops=2,
        limit=10,
        chunk_limit=3,
        reinforce=True,
        session_id="session-secret",
        debug=True,
    )
    ctx.emit_started()
    ctx.emit_completed(
        result_count=1,
        touched_entity_count=1,
        touched_edge_count=0,
        chunk_count=0,
    )

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").strip().splitlines()
    ]

    assert len(events) == 2
    assert all(event["query_text"] == "Project Atlas budget details" for event in events)
    assert all(event["session_id"] == "session-secret" for event in events)
