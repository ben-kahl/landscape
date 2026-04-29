"""Tests for the embedded Landscape MCP app.

Uses ``mcp.shared.memory.create_connected_server_and_client_session`` for
in-process testing — no subprocess spawning required.  The MCP SDK wires the
FastMCP server and a ClientSession together via anyio memory streams, so every
``client.call_tool(...)`` call goes through the full JSON-RPC dispatch path.

DB isolation is provided by the autouse ``_isolated_test`` fixture in conftest
(Neo4j wiped + Qdrant collections dropped before each test).
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# MCP app construction
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_mcp_app_registers_expected_tools():
    from landscape.mcp_app import mcp

    tools = {tool.name for tool in await mcp.list_tools()}
    assert {
        "search",
        "remember",
        "capture_turn",
        "add_entity",
        "add_relation",
        "graph_query",
        "status",
        "conversation_history",
    } <= tools


@pytest.mark.smoke
def test_fastapi_app_exposes_mcp_and_oauth_paths():
    from landscape.main import app

    app_paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/mcp" in app_paths
    assert "/healthz" in app_paths


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_fastapi_lifespan_skips_mcp_session_manager_under_pytest(monkeypatch):
    from landscape.embeddings import encoder
    from landscape.main import app
    from landscape.mcp_app import mcp
    from landscape.storage import neo4j_store, qdrant_store

    monkeypatch.setattr(encoder, "load_model", lambda: None)

    async def noop():
        return None

    monkeypatch.setattr(qdrant_store, "init_collection", noop)
    monkeypatch.setattr(qdrant_store, "init_chunks_collection", noop)
    monkeypatch.setattr(qdrant_store, "close_client", noop)
    monkeypatch.setattr(neo4j_store, "close_driver", noop)

    async with app.router.lifespan_context(app):
        assert mcp.session_manager._task_group is None


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_fastapi_lifespan_can_run_repeatedly_under_pytest(monkeypatch):
    from landscape.embeddings import encoder
    from landscape.main import app
    from landscape.mcp_app import mcp
    from landscape.storage import neo4j_store, qdrant_store

    monkeypatch.setattr(encoder, "load_model", lambda: None)

    async def noop():
        return None

    monkeypatch.setattr(qdrant_store, "init_collection", noop)
    monkeypatch.setattr(qdrant_store, "init_chunks_collection", noop)
    monkeypatch.setattr(qdrant_store, "close_client", noop)
    monkeypatch.setattr(neo4j_store, "close_driver", noop)

    async with app.router.lifespan_context(app):
        first_session_manager = mcp.session_manager
        assert first_session_manager._task_group is None

    async with app.router.lifespan_context(app):
        assert mcp.session_manager is first_session_manager
        assert mcp.session_manager._task_group is None


@pytest.mark.smoke
def test_mcp_streamable_http_app_constructs():
    from landscape.mcp_app import mcp

    asgi_app = mcp.streamable_http_app()
    assert asgi_app is not None


@pytest.mark.smoke
def test_mcp_app_has_no_private_ensure_init():
    import landscape.mcp_app as mcp_app

    assert not hasattr(mcp_app, "_ensure_init")


@pytest.mark.asyncio
async def test_search_works_under_fastapi_lifespan_without_mcp_init():
    from landscape.main import app

    async with app.router.lifespan_context(app):
        async with _mcp_client() as client:
            result = await client.call_tool(
                "search",
                {"query": "nobody", "limit": 1, "chunk_limit": 0},
            )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)
    assert "results" in data
    assert "touched_entity_count" in data
    assert "chunks" in data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capture_turn_schedules_auto_ingestion_via_mcp_boundary(monkeypatch):
    """capture_turn should schedule ingestion and return without awaiting it."""
    import landscape.mcp_app as mcp_app

    calls = []

    class _NoAwait:
        def __await__(self):
            raise AssertionError("capture_turn must not await the scheduled task")

    def fake_schedule_auto_ingestion(
        text: str,
        session_id: str,
        turn_id: str,
        role: str = "user",
        debug: bool = False,
    ):
        calls.append(
            {
                "text": text,
                "session_id": session_id,
                "turn_id": turn_id,
                "role": role,
                "debug": debug,
            }
        )
        return _NoAwait()

    monkeypatch.setattr(mcp_app, "_schedule_auto_ingestion", fake_schedule_auto_ingestion)

    async with _mcp_client() as client:
        result = await client.call_tool(
            "capture_turn",
            {
                "session_id": "test-session",
                "turn_id": "turn-1",
                "role": "user",
                "text": "Remember this note for later.",
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert data == {"accepted": True, "scheduled": True}
    assert calls == [
        {
            "text": "Remember this note for later.",
            "session_id": "test-session",
            "turn_id": "turn-1",
            "role": "user",
            "debug": False,
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capture_turn_rejects_ineligible_turns_at_mcp_boundary(monkeypatch):
    """capture_turn should reject blank turns before background scheduling."""
    import landscape.mcp_app as mcp_app

    scheduled = []

    def fake_schedule_auto_ingestion(
        text: str,
        session_id: str,
        turn_id: str,
        role: str = "user",
        debug: bool = False,
    ):
        scheduled.append((text, session_id, turn_id, role, debug))
        raise AssertionError("ineligible turns must not be scheduled")

    monkeypatch.setattr(mcp_app, "_schedule_auto_ingestion", fake_schedule_auto_ingestion)

    async with _mcp_client() as client:
        result = await client.call_tool(
            "capture_turn",
            {
                "session_id": "test-session",
                "turn_id": "turn-1",
                "role": "user",
                "text": "   ",
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert data == {"accepted": False, "scheduled": False}
    assert scheduled == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remember_threads_debug_flag_to_pipeline(monkeypatch):
    from landscape.pipeline import IngestResult

    calls = []

    async def fake_ingest(
        text,
        title,
        source_type="text",
        session_id=None,
        turn_id=None,
        debug=False,
    ):
        calls.append(
            {
                "text": text,
                "title": title,
                "source_type": source_type,
                "session_id": session_id,
                "turn_id": turn_id,
                "debug": debug,
            }
        )
        return IngestResult(
            doc_id="doc-debug",
            already_existed=False,
            entities_created=0,
            entities_reinforced=0,
            relations_created=0,
            relations_reinforced=0,
            relations_superseded=0,
            chunks_created=0,
        )

    monkeypatch.setattr("landscape.pipeline.ingest", fake_ingest)

    async with _mcp_client() as client:
        result = await client.call_tool(
            "remember",
            {
                "text": "Alice joined Beacon Labs.",
                "title": "debug-memory",
                "session_id": "s1",
                "turn_id": "t1",
                "debug": True,
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    assert calls == [
        {
            "text": "Alice joined Beacon Labs.",
            "title": "debug-memory",
            "source_type": "text",
            "session_id": "s1",
            "turn_id": "t1",
            "debug": True,
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capture_turn_threads_debug_flag_to_scheduler(monkeypatch):
    import landscape.mcp_app as mcp_app

    calls = []

    class _NoAwait:
        def __await__(self):
            raise AssertionError("capture_turn must not await the scheduled task")

    def fake_schedule_auto_ingestion(
        text: str,
        session_id: str,
        turn_id: str,
        role: str = "user",
        debug: bool = False,
    ):
        calls.append(
            {
                "text": text,
                "session_id": session_id,
                "turn_id": turn_id,
                "role": role,
                "debug": debug,
            }
        )
        return _NoAwait()

    monkeypatch.setattr(mcp_app, "_schedule_auto_ingestion", fake_schedule_auto_ingestion)

    async with _mcp_client() as client:
        result = await client.call_tool(
            "capture_turn",
            {
                "session_id": "test-session",
                "turn_id": "turn-1",
                "role": "user",
                "text": "Remember this note for later.",
                "debug": True,
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert data == {"accepted": True, "scheduled": True}
    assert calls == [
        {
            "text": "Remember this note for later.",
            "session_id": "test-session",
            "turn_id": "turn-1",
            "role": "user",
            "debug": True,
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capture_turn_skips_turn_already_written_by_remember(monkeypatch):
    """Auto-ingest should not duplicate a turn already written explicitly."""
    import landscape.mcp_app as mcp_app
    from landscape.pipeline import IngestResult

    mcp_app._EXPLICIT_MEMORY_TURN_KEYS.clear()

    async def fake_ingest(
        text,
        title,
        source_type="text",
        session_id=None,
        turn_id=None,
        debug=False,
    ):
        return IngestResult(
            doc_id="explicit-doc",
            already_existed=False,
            entities_created=1,
            entities_reinforced=0,
            relations_created=0,
            relations_reinforced=0,
            relations_superseded=0,
            chunks_created=1,
        )

    def fail_schedule(
        text: str,
        session_id: str,
        turn_id: str,
        role: str = "user",
        debug: bool = False,
    ):
        raise AssertionError("explicitly remembered turns must not be auto-ingested")

    monkeypatch.setattr("landscape.pipeline.ingest", fake_ingest)
    monkeypatch.setattr(mcp_app, "_schedule_auto_ingestion", fail_schedule)

    async with _mcp_client() as client:
        remember_result = await client.call_tool(
            "remember",
            {
                "text": "Alice joined Beacon Labs.",
                "title": "explicit-alice-memory",
                "session_id": "test-session",
                "turn_id": "turn-3",
            },
        )
        capture_result = await client.call_tool(
            "capture_turn",
            {
                "session_id": "test-session",
                "turn_id": "turn-3",
                "role": "assistant",
                "text": "I remembered that Alice joined Beacon Labs.",
            },
        )

    assert not remember_result.isError, f"Tool returned error: {remember_result.content}"
    assert not capture_result.isError, f"Tool returned error: {capture_result.content}"
    assert _parse(capture_result) == {"accepted": False, "scheduled": False}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capture_turn_rejects_tool_role_turns(monkeypatch):
    """Tool-result/protocol turns should not be auto-ingested as memory."""
    import landscape.mcp_app as mcp_app

    def fail_schedule(
        text: str,
        session_id: str,
        turn_id: str,
        role: str = "user",
        debug: bool = False,
    ):
        raise AssertionError("tool-role turns must not be auto-ingested")

    monkeypatch.setattr(mcp_app, "_schedule_auto_ingestion", fail_schedule)

    async with _mcp_client() as client:
        result = await client.call_tool(
            "capture_turn",
            {
                "session_id": "test-session",
                "turn_id": "turn-4",
                "role": "tool",
                "text": '{"doc_id": "explicit-doc", "already_existed": false}',
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    assert _parse(result) == {"accepted": False, "scheduled": False}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capture_turn_logs_background_failure_without_raising(monkeypatch, caplog):
    """capture_turn should return success even if the background task fails."""
    import landscape.mcp_app as mcp_app

    caplog.set_level(logging.ERROR, logger=mcp_app.logger.name)

    class _FailedTask:
        def exception(self):
            return RuntimeError("background boom")

    class _ScheduledTask:
        def add_done_callback(self, callback):
            callback(_FailedTask())

    def fake_create_task(coro):
        coro.close()
        return _ScheduledTask()

    monkeypatch.setattr(mcp_app.asyncio, "create_task", fake_create_task)

    async with _mcp_client() as client:
        result = await client.call_tool(
            "capture_turn",
            {
                "session_id": "test-session",
                "turn_id": "turn-2",
                "role": "user",
                "text": "Keep this note around.",
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)
    assert data == {"accepted": True, "scheduled": True}
    assert "background boom" in caplog.text
    assert "RuntimeError" in caplog.text


# ---------------------------------------------------------------------------
# Fixture: connected MCP client for the landscape server
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _mcp_client(auth=None):
    """Context manager that yields an initialised in-process MCP ClientSession.

    The in-process transport bypasses the FastAPI ASGI auth middleware, so
    we set the request-scoped principal directly. ``auth=None`` defaults to
    an anonymous agent-scoped principal so existing tests keep their behaviour.
    """
    from landscape.auth import AuthContext
    from landscape.mcp_app import mcp
    from landscape.security import _CURRENT_AUTH_CONTEXT

    if auth is None:
        auth = AuthContext(
            client_id="loopback-anonymous",
            client_name="loopback-anonymous",
            token_id="",
            scopes=frozenset({"agent"}),
        )

    token = _CURRENT_AUTH_CONTEXT.set(auth)
    try:
        async with create_connected_server_and_client_session(mcp) as client:
            yield client
    finally:
        _CURRENT_AUTH_CONTEXT.reset(token)


def _parse(result) -> dict:
    """Extract the first text content item and JSON-decode it."""
    assert result.content, "MCP tool returned no content"
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURE_TEXT = (
    "Helios Robotics was founded by Maria Santos in 2021. "
    "Maria leads the Vision Team, which is working on Project Sentinel. "
    "Project Sentinel uses PyTorch for its neural network components."
)

_FIXTURE_TITLE = "helios-test-doc"


async def _seed_document():
    """Ingest the fixture document directly via pipeline (no MCP round-trip)."""
    from landscape.pipeline import ingest

    return await ingest(_FIXTURE_TEXT, _FIXTURE_TITLE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_results_shape(http_client):
    """After ingesting one document, search should return the expected shape."""
    await _seed_document()

    async with _mcp_client() as client:
        result = await client.call_tool("search", {"query": "Maria Santos Vision Team"})

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert "results" in data
    assert "touched_entity_count" in data
    assert isinstance(data["results"], list)
    if data["results"]:
        first = data["results"][0]
        assert "name" in first
        assert "type" in first
        assert "score" in first
        assert "path_edge_types" in first


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_threads_debug_flag_to_retrieve(monkeypatch):
    from landscape.retrieval.query import RetrievalResult

    calls = []

    async def fake_retrieve(
        query_text,
        hops=2,
        limit=10,
        chunk_limit=3,
        weights=None,
        reinforce=True,
        session_id=None,
        since=None,
        debug=False,
        log_context=None,
    ):
        calls.append(
            {
                "query_text": query_text,
                "hops": hops,
                "limit": limit,
                "chunk_limit": chunk_limit,
                "reinforce": reinforce,
                "session_id": session_id,
                "debug": debug,
            }
        )
        return RetrievalResult(
            query=query_text,
            results=[],
            touched_entity_ids=[],
            touched_edge_ids=[],
            chunks=[],
        )

    monkeypatch.setattr("landscape.retrieval.query.retrieve", fake_retrieve)

    async with _mcp_client() as client:
        result = await client.call_tool(
            "search",
            {"query": "Project Atlas", "debug": True},
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    assert calls == [
        {
            "query_text": "Project Atlas",
            "hops": 2,
            "limit": 10,
            "chunk_limit": 3,
            "reinforce": True,
            "session_id": None,
            "debug": True,
        }
    ]


@pytest.mark.asyncio
async def test_remember_creates_entities(http_client):
    """remember() should ingest text and report entities_created > 0."""
    async with _mcp_client() as client:
        result = await client.call_tool(
            "remember",
            {
                "text": "Alice Chen joined the Platform Team in January.",
                "title": "onboarding-note",
                "session_id": "test-session",
                "turn_id": "t1",
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert "doc_id" in data
    assert data["doc_id"]
    assert "entities_created" in data
    assert data["entities_created"] > 0, "Expected at least one entity to be created"
    assert "already_existed" in data
    assert data["already_existed"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_schedule_auto_ingestion_creates_background_task(monkeypatch):
    """_schedule_auto_ingestion() should create, not await, the auto-ingest task."""
    from landscape import mcp_app

    class DummyTask:
        def __init__(self, coro):
            self.coro = coro
            self.callbacks = []

        def add_done_callback(self, callback):
            self.callbacks.append(callback)

    scheduled_coroutines = []

    def fake_create_task(coro):
        scheduled_coroutines.append(coro)
        coro.close()
        return DummyTask(coro)

    monkeypatch.setattr(
        mcp_app, "asyncio", SimpleNamespace(create_task=fake_create_task), raising=False
    )
    task = mcp_app._schedule_auto_ingestion(
        "Alice Chen joined the Platform Team in January.", "test-session", "t1"
    )

    assert len(scheduled_coroutines) == 1, "Expected a background task to be scheduled"
    assert scheduled_coroutines[0].cr_code.co_name == "_auto_ingest_turn"
    assert len(task.callbacks) == 1
    assert task.callbacks[0] is mcp_app._log_auto_ingestion_failure


@pytest.mark.asyncio
async def test_add_entity_returns_canonical_id(http_client):
    """add_entity should return a non-empty entity_id and canonical_name."""
    async with _mcp_client() as client:
        result = await client.call_tool(
            "add_entity",
            {
                "name": "Zylos Corp",
                "entity_type": "Organization",
                "source": "agent:test-session:1",
                "session_id": "test-session",
                "turn_id": "t1",
                "confidence": 0.9,
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert data["entity_id"]
    assert data["canonical_name"]
    assert data["resolved_to_existing"] is False


@pytest.mark.asyncio
async def test_add_relation_creates_or_supersedes(http_client):
    """add_relation should return assertion and memory-fact ids."""
    async with _mcp_client() as client:
        result = await client.call_tool(
            "add_relation",
            {
                "subject": "Bob",
                "subject_type": "Person",
                "object": "Acme",
                "object_type": "Organization",
                "rel_type": "WORKS_FOR",
                "source": "agent:test-session:2",
                "session_id": "test-session",
                "turn_id": "t2",
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert data["outcome"] == "memory_fact"
    assert data["assertion_id"]
    assert data["memory_fact_id"]
    assert data["subject_id"]
    assert data["object_id"]


@pytest.mark.asyncio
async def test_add_relation_supersedes_functional_edge(http_client):
    """Writing WORKS_FOR twice for the same subject should yield distinct memory facts."""
    async with _mcp_client() as client:
        # First write
        r1 = await client.call_tool(
            "add_relation",
            {
                "subject": "Carol",
                "subject_type": "Person",
                "object": "Acme",
                "object_type": "Organization",
                "rel_type": "WORKS_FOR",
                "source": "agent:test-session:3a",
                "session_id": "test-session",
                "turn_id": "t3a",
            },
        )
        assert not r1.isError
        d1 = _parse(r1)
        assert d1["outcome"] == "memory_fact"
        assert d1["assertion_id"]
        assert d1["memory_fact_id"]

        # Second write — different object, same functional rel_type → supersession
        r2 = await client.call_tool(
            "add_relation",
            {
                "subject": "Carol",
                "subject_type": "Person",
                "object": "Beacon Corp",
                "object_type": "Organization",
                "rel_type": "WORKS_FOR",
                "source": "agent:test-session:3b",
                "session_id": "test-session",
                "turn_id": "t3b",
            },
        )
        assert not r2.isError
        d2 = _parse(r2)
        assert d2["outcome"] == "memory_fact"
        assert d2["assertion_id"]
        assert d2["memory_fact_id"]
        assert d2["memory_fact_id"] != d1["memory_fact_id"]


def _graph_query_principal():
    from landscape.auth import AuthContext

    return AuthContext(
        client_id="test-graph-query",
        client_name="test-graph-query",
        token_id="test-token-id",
        scopes=frozenset({"agent", "graph_query"}),
    )


@pytest.mark.asyncio
async def test_graph_query_rejects_writes(http_client):
    """graph_query must refuse CREATE and return an MCP error."""
    async with _mcp_client(auth=_graph_query_principal()) as client:
        result = await client.call_tool(
            "graph_query",
            {"cypher": "CREATE (n:X {name: 'bad'}) RETURN n"},
        )

    # FastMCP converts ValueError to an isError=True tool result
    assert result.isError, "Expected an MCP error for a write query"
    error_text = result.content[0].text if result.content else ""
    assert "CREATE" in error_text or "write" in error_text.lower()


@pytest.mark.asyncio
async def test_graph_query_allows_reads(http_client):
    """graph_query should execute a MATCH and return rows."""
    async with _mcp_client(auth=_graph_query_principal()) as client:
        result = await client.call_tool(
            "graph_query",
            {"cypher": "MATCH (e:Entity) RETURN count(e) AS n"},
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert "rows" in data
    assert isinstance(data["rows"], list)
    # At least one row with the count column
    assert len(data["rows"]) == 1
    assert "n" in data["rows"][0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_loopback_bypass_cannot_call_graph_query():
    """Loopback bypass principal grants only ``agent`` -- ``graph_query`` is forbidden."""
    async with _mcp_client() as client:
        result = await client.call_tool(
            "graph_query",
            {"cypher": "MATCH (n) RETURN count(n)"},
        )

    assert result.isError is True
    error_text = result.content[0].text if result.content else ""
    assert "Forbidden" in error_text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_loopback_bypass_can_call_agent_scope_tools(monkeypatch):
    """Loopback bypass should be able to call ``status`` (agent scope)."""
    from landscape.writeback import StatusSummary

    async def fake_status_summary():
        return StatusSummary(
            entity_count=0,
            document_count=0,
            relation_count=0,
            top_entities=[],
            recent_agent_writes=[],
        )

    monkeypatch.setattr("landscape.writeback.status_summary", fake_status_summary)

    async with _mcp_client() as client:
        result = await client.call_tool("status", {})

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)
    assert "entity_count" in data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_authenticated_client_with_graph_query_scope_can_call_graph_query(monkeypatch):
    """A real authenticated client minted with graph_query scope can run Cypher."""
    from landscape.storage import auth_store

    async def fake_run_cypher(_cypher: str, _params: dict):
        return [{"n": 0}]

    monkeypatch.setattr(
        "landscape.storage.neo4j_store.run_cypher_readonly", fake_run_cypher
    )

    # Build a principal as if an OAuth token was minted with scopes=["agent","graph_query"].
    # We don't go through the SQLite store here -- only the in-memory ContextVar
    # check matters for the scope gate.
    from landscape.auth import AuthContext

    principal = AuthContext(
        client_id="real-client",
        client_name="bench-runner",
        token_id="tok-sec-1",
        scopes=frozenset({"agent", "graph_query"}),
    )

    async with _mcp_client(auth=principal) as client:
        result = await client.call_tool(
            "graph_query",
            {"cypher": "MATCH (n) RETURN count(n) AS n"},
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)
    assert data == {"rows": [{"n": 0}]}
    # Quiet ruff: auth_store is referenced indirectly via fake setup elsewhere.
    assert auth_store is not None


@pytest.mark.asyncio
async def test_status_returns_summary(http_client):
    """status() should return a dict with the expected top-level keys."""
    async with _mcp_client() as client:
        result = await client.call_tool("status", {})

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    for key in (
        "entity_count",
        "document_count",
        "relation_count",
        "top_entities",
        "recent_agent_writes",
    ):
        assert key in data, f"Missing key '{key}' in status response"
    assert isinstance(data["top_entities"], list)
    assert isinstance(data["recent_agent_writes"], list)


@pytest.mark.asyncio
async def test_search_accepts_session_id_and_since_hours(http_client):
    """search() with session_id and since_hours kwargs should not error."""
    await _seed_document()

    async with _mcp_client() as client:
        result = await client.call_tool(
            "search",
            {
                "query": "Maria Santos",
                "hops": 1,
                "limit": 5,
                "session_id": "test-session-search-kwargs",
                "since_hours": 24,
            },
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)
    assert "results" in data
    assert "touched_entity_count" in data


@pytest.mark.asyncio
async def test_conversation_history_round_trips(http_client):
    """conversation_history returns the seeded conversation and turn with entity."""
    from landscape.storage import neo4j_store

    sid = "mcp-ch-test-1"
    turn_eid, _ = await neo4j_store.merge_turn(sid, "t1", turn_number=1, role="user", summary="hi")
    eid = await neo4j_store.merge_entity(
        "ConvEntity", "ORGANIZATION", f"doc-{sid}", 0.9, doc_element_id=None, model="test"
    )
    await neo4j_store.link_entity_to_turn(eid, turn_eid)

    async with _mcp_client() as client:
        result = await client.call_tool(
            "conversation_history",
            {"session_id": sid, "limit": 10},
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert data["conversation"] is not None
    assert data["conversation"]["id"] == sid
    turns = data["turns"]
    assert len(turns) == 1
    assert turns[0]["turn_id"] == "t1"
    entity_names = [e["name"] for e in turns[0]["entities_mentioned"]]
    assert "ConvEntity" in entity_names


@pytest.mark.asyncio
async def test_conversation_history_unknown_session(http_client):
    """conversation_history with an unknown session_id returns null conversation and empty turns."""
    async with _mcp_client() as client:
        result = await client.call_tool(
            "conversation_history",
            {"session_id": "mcp-ch-nonexistent-999", "limit": 10},
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)
    assert data["conversation"] is None
    assert data["turns"] == []


@pytest.mark.asyncio
async def test_search_returns_chunks(http_client):
    """search() with chunk_limit should include a 'chunks' key with the right shape."""
    await _seed_document()

    async with _mcp_client() as client:
        result = await client.call_tool(
            "search",
            {"query": "Maria Santos Vision Team", "chunk_limit": 3},
        )

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    assert "chunks" in data, "'chunks' key missing from search response"
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) <= 3

    for chunk in data["chunks"]:
        for key in ("text", "source_doc", "doc_id", "position", "score"):
            assert key in chunk, f"Chunk missing key '{key}': {chunk}"
