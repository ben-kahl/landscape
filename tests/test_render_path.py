import pytest

from landscape.retrieval.query import EntityPath, PathEdge, PathNode
from landscape.retrieval.render import render_path


@pytest.mark.unit
def test_render_path_direct_hit():
    path = EntityPath(nodes=[PathNode("Qdrant", "TECHNOLOGY")], edges=[])
    assert render_path(path) == "(Qdrant) [direct match]"


@pytest.mark.unit
def test_render_path_one_hop():
    path = EntityPath(
        nodes=[PathNode("Eric", "PERSON"), PathNode("Netflix", "TECHNOLOGY")],
        edges=[PathEdge(type="DISCUSSION")],
    )
    assert render_path(path) == "(Eric) -[DISCUSSION]-> Netflix [TECHNOLOGY]"


@pytest.mark.unit
def test_render_path_two_hops():
    path = EntityPath(
        nodes=[
            PathNode("Project Aurora", "PROJECT"),
            PathNode("PostgreSQL", "TECHNOLOGY"),
            PathNode("Maya Chen", "PERSON"),
        ],
        edges=[
            PathEdge(type="USES"),
            PathEdge(type="APPROVED_BY"),
        ],
    )
    result = render_path(path)
    assert result == (
        "(Project Aurora) -[USES]-> PostgreSQL [TECHNOLOGY] "
        "-[APPROVED_BY]-> Maya Chen [PERSON]"
    )


@pytest.mark.unit
def test_render_path_negated_edge():
    path = EntityPath(
        nodes=[PathNode("Alice", "PERSON"), PathNode("Acme Corp", "ORGANIZATION")],
        edges=[PathEdge(type="WORKS_FOR", negated=True)],
    )
    assert render_path(path) == "(Alice) -[NOT WORKS_FOR]-> Acme Corp [ORGANIZATION]"


@pytest.mark.unit
def test_render_path_empty_path():
    path = EntityPath(nodes=[], edges=[])
    assert render_path(path) == "(unknown)"


@pytest.mark.unit
def test_render_path_includes_subtype():
    path = EntityPath(
        nodes=[
            PathNode("PhysicsX", "Organization"),
            PathNode("Okta", "Technology"),
        ],
        edges=[PathEdge(type="USES", subtype="for_sso_configuration")],
    )
    assert render_path(path) == (
        "(PhysicsX) -[USES/for_sso_configuration]-> Okta [Technology]"
    )


@pytest.mark.unit
def test_compact_payload_chunks_carry_fetch_keys():
    from types import SimpleNamespace

    from landscape.retrieval.query import RetrievedChunk
    from landscape.retrieval.render import build_compact_payload

    result = SimpleNamespace(
        query="q",
        results=[],
        touched_entity_ids=[],
        chunks=[
            RetrievedChunk(
                chunk_id="4:abc:12:0:h1",
                text="chunk body text",
                doc_id="4:abc:12",
                source_doc="ticket-119987",
                position=0,
                score=0.73,
            )
        ],
    )

    payload = build_compact_payload(result)

    assert payload["chunks"][0]["doc_id"] == "4:abc:12"
    assert payload["chunks"][0]["chunk_id"] == "4:abc:12:0:h1"
    assert payload["chunks"][0]["source"] == "ticket-119987"


@pytest.mark.unit
def test_render_path_negated_with_subtype():
    path = EntityPath(
        nodes=[
            PathNode("PhysicsX", "Organization"),
            PathNode("Domain capture", "Concept"),
        ],
        edges=[PathEdge(type="USES", subtype="signup_flow", negated=True)],
    )
    assert render_path(path) == (
        "(PhysicsX) -[NOT USES/signup_flow]-> Domain capture [Concept]"
    )
