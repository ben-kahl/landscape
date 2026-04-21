"""Unit tests for the Cypher read-only validator (no DB required)."""

from __future__ import annotations

import pytest

from landscape.storage.cypher_guard import CypherWriteAttempted, assert_read_only

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def allowed(cypher: str) -> None:
    """Assert that assert_read_only does NOT raise for *cypher*."""
    assert_read_only(cypher)  # should not raise


def rejected(cypher: str) -> str:
    """Assert that assert_read_only raises CypherWriteAttempted and return the message."""
    with pytest.raises(CypherWriteAttempted) as exc_info:
        assert_read_only(cypher)
    return str(exc_info.value)


# ---------------------------------------------------------------------------
# Allowed queries
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_allowed_simple_match():
    """1. Simple MATCH / RETURN / LIMIT."""
    allowed("MATCH (n) RETURN n LIMIT 10")

@pytest.mark.unit
def test_allowed_multi_hop_match():
    """2. Multi-hop relationship pattern."""
    allowed("MATCH (a)-[r:RELATES_TO*1..3]->(b) RETURN a, r, b")

@pytest.mark.unit
def test_allowed_match_with_aggregation():
    """3. MATCH with WITH and aggregations."""
    allowed(
        "MATCH (e:Entity)-[r:RELATES_TO]->(o:Entity) "
        "WITH e, count(r) AS degree "
        "RETURN e.name, degree ORDER BY degree DESC LIMIT 5"
    )

@pytest.mark.unit
def test_allowed_call_db_labels():
    """4a. CALL db.labels() is a read procedure — must be allowed."""
    allowed("CALL db.labels()")

@pytest.mark.unit
def test_allowed_call_db_relationship_types():
    """4b. CALL db.relationshipTypes() is a read procedure — must be allowed."""
    allowed("CALL db.relationshipTypes()")

@pytest.mark.unit
def test_allowed_string_literal_contains_create():
    """5. String literal containing 'CREATE' must not trigger rejection."""
    allowed("MATCH (n) WHERE n.name = 'CREATE foo' RETURN n")

@pytest.mark.unit
def test_allowed_line_comment_contains_delete():
    """6. Line comment containing 'DELETE' must not trigger rejection."""
    allowed("MATCH (n) RETURN n // DELETE later")

@pytest.mark.unit
def test_allowed_block_comment_contains_merge():
    """6b. Block comment containing 'MERGE' must not trigger rejection."""
    allowed("/* MERGE things later */ MATCH (n) RETURN n")


# ---------------------------------------------------------------------------
# Rejected queries
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rejected_create_node():
    """7. CREATE (n:X) RETURN n."""
    msg = rejected("CREATE (n:X) RETURN n")
    assert "CREATE" in msg.upper()

@pytest.mark.unit
def test_rejected_delete():
    """8. MATCH (n) DELETE n."""
    msg = rejected("MATCH (n) DELETE n")
    assert "DELETE" in msg.upper()

@pytest.mark.unit
def test_rejected_detach_delete():
    """9. MATCH (n) DETACH DELETE n."""
    msg = rejected("MATCH (n) DETACH DELETE n")
    assert "DELETE" in msg.upper()

@pytest.mark.unit
def test_rejected_set():
    """10. MATCH (n) SET n.foo = 1."""
    msg = rejected("MATCH (n) SET n.foo = 1")
    assert "SET" in msg.upper()

@pytest.mark.unit
def test_rejected_merge():
    """11. MERGE (n:X {id: 1})."""
    msg = rejected("MERGE (n:X {id: 1})")
    assert "MERGE" in msg.upper()

@pytest.mark.unit
def test_rejected_remove():
    """12. MATCH (n) REMOVE n.foo."""
    msg = rejected("MATCH (n) REMOVE n.foo")
    assert "REMOVE" in msg.upper()

@pytest.mark.unit
def test_rejected_drop():
    """13. DROP INDEX foo IF EXISTS."""
    msg = rejected("DROP INDEX foo IF EXISTS")
    assert "DROP" in msg.upper()

@pytest.mark.unit
def test_rejected_call_apoc_create():
    """14. CALL apoc.create.node([], {}) — admin procedure."""
    msg = rejected("CALL apoc.create.node([], {})")
    assert "apoc" in msg.lower() or "create" in msg.lower()

@pytest.mark.unit
def test_rejected_call_subquery_with_create():
    """15. CALL { CREATE (x) } IN TRANSACTIONS — write inside subquery."""
    msg = rejected("CALL { CREATE (x) } IN TRANSACTIONS")
    assert "CREATE" in msg.upper()

@pytest.mark.unit
def test_rejected_load_csv():
    """16. LOAD CSV FROM 'x' AS row CREATE (n)."""
    msg = rejected("LOAD CSV FROM 'x' AS row CREATE (n)")
    # LOAD is the first write keyword hit
    assert "LOAD" in msg.upper() or "CREATE" in msg.upper()

@pytest.mark.unit
def test_rejected_use():
    """17. USE other.db MATCH (n) RETURN n — database switching."""
    msg = rejected("USE other.db MATCH (n) RETURN n")
    assert "USE" in msg.upper()

@pytest.mark.unit
def test_rejected_lowercase_mixed():
    """18. Case-insensitivity: 'match (n) create (m)' must be rejected."""
    msg = rejected("match (n) create (m)")
    assert "CREATE" in msg.upper()


# ---------------------------------------------------------------------------
# Integration test — requires a running Neo4j instance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_cypher_readonly_returns_rows(neo4j_driver):
    """19. run_cypher_readonly returns correct rows after seeding one entity."""
    from landscape.storage.neo4j_store import run_cypher_readonly

    # Seed a known entity directly so the count is deterministic
    async with neo4j_driver.session() as session:
        await session.run(
            "MERGE (e:Entity {name: '__guard_test__', type: 'TestType'}) "
            "SET e.canonical = true"
        )

    rows = await run_cypher_readonly(
        "MATCH (e:Entity {name: '__guard_test__'}) RETURN count(e) AS c"
    )
    assert rows == [{"c": 1}]
