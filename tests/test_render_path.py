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
