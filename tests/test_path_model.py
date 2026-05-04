import pytest
from landscape.retrieval.query import EntityPath, PathEdge, PathNode, RetrievedEntity


@pytest.mark.unit
def test_entity_path_node_edge_invariant():
    path = EntityPath(
        nodes=[PathNode("Aurora", "PROJECT"), PathNode("PostgreSQL", "TECHNOLOGY")],
        edges=[PathEdge(type="USES")],
    )
    assert len(path.nodes) == len(path.edges) + 1


@pytest.mark.unit
def test_entity_path_direct_hit():
    path = EntityPath(nodes=[PathNode("Qdrant", "TECHNOLOGY")], edges=[])
    assert len(path.edges) == 0
    assert path.nodes[0].name == "Qdrant"
    assert path.nodes[0].type == "TECHNOLOGY"


@pytest.mark.unit
def test_path_edge_defaults():
    edge = PathEdge(type="USES")
    assert edge.negated is False
    assert edge.subtype is None
    assert edge.memory_fact_id is None
    assert edge.quantities == {}


@pytest.mark.unit
def test_retrieved_entity_defaults_to_vector_mode():
    entity = RetrievedEntity(
        entity_id="q-id", name="Qdrant", type="TECHNOLOGY",
        distance=0, vector_sim=0.9, reinforcement=0.0,
        edge_confidence=0.0, score=0.9,
    )
    assert entity.retrieval_mode == "vector"
    assert entity.path.nodes == []
    assert entity.path.edges == []


@pytest.mark.unit
def test_retrieved_entity_graph_mode():
    path = EntityPath(
        nodes=[PathNode("Aurora", "PROJECT"), PathNode("PostgreSQL", "TECHNOLOGY")],
        edges=[PathEdge(type="USES")],
    )
    entity = RetrievedEntity(
        entity_id="pg-id", name="PostgreSQL", type="TECHNOLOGY",
        distance=1, vector_sim=0.8, reinforcement=0.0,
        edge_confidence=0.9, score=0.9,
        path=path, retrieval_mode="graph",
    )
    assert entity.retrieval_mode == "graph"
    assert entity.path.edges[0].type == "USES"


@pytest.mark.unit
def test_entity_path_invariant_violated_raises():
    with pytest.raises(ValueError, match="EntityPath invariant violated"):
        EntityPath(
            nodes=[PathNode("Aurora", "PROJECT")],
            edges=[PathEdge(type="USES"), PathEdge(type="LEADS")],
        )
