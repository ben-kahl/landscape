"""Tests for src/landscape/mcp_server.py.

Uses ``mcp.shared.memory.create_connected_server_and_client_session`` for
in-process testing — no subprocess spawning required.  The MCP SDK wires the
FastMCP server and a ClientSession together via anyio memory streams, so every
``client.call_tool(...)`` call goes through the full JSON-RPC dispatch path.

DB isolation is provided by the autouse ``_isolated_test`` fixture in conftest
(Neo4j wiped + Qdrant collections dropped before each test).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest
from mcp.shared.memory import create_connected_server_and_client_session


# ---------------------------------------------------------------------------
# Fixture: connected MCP client for the landscape server
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _mcp_client():
    """Context manager that yields an initialised in-process MCP ClientSession."""
    from landscape.mcp_server import mcp

    async with create_connected_server_and_client_session(mcp) as client:
        yield client


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
    """add_relation should report outcome 'created' for a fresh edge."""
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

    assert data["outcome"] in ("created", "reinforced", "superseded")
    assert data["subject_id"]
    assert data["object_id"]
    assert "relation_id" in data


@pytest.mark.asyncio
async def test_add_relation_supersedes_functional_edge(http_client):
    """Writing WORKS_FOR twice for the same subject should supersede the first edge."""
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
        assert d1["outcome"] == "created"

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
        assert d2["outcome"] == "superseded"


@pytest.mark.asyncio
async def test_graph_query_rejects_writes(http_client):
    """graph_query must refuse CREATE and return an MCP error."""
    async with _mcp_client() as client:
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
    async with _mcp_client() as client:
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


@pytest.mark.asyncio
async def test_status_returns_summary(http_client):
    """status() should return a dict with the expected top-level keys."""
    async with _mcp_client() as client:
        result = await client.call_tool("status", {})

    assert not result.isError, f"Tool returned error: {result.content}"
    data = _parse(result)

    for key in ("entity_count", "document_count", "relation_count", "top_entities", "recent_agent_writes"):
        assert key in data, f"Missing key '{key}' in status response"
    assert isinstance(data["top_entities"], list)
    assert isinstance(data["recent_agent_writes"], list)
