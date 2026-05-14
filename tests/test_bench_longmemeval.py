"""Unit tests for bench_longmemeval helper functions.

These tests cover the pure-logic helpers that do not need the docker stack.
"""
import os
import sys
from types import SimpleNamespace

import pytest

# The bench script lives in scripts/, not a package. Import its helpers directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _make_result(entities=(), chunks=()):
    """Build a minimal RetrieveResult-like object for testing."""
    return SimpleNamespace(
        results=list(entities),
        chunks=list(chunks),
        touched_entity_ids=[e.entity_id for e in entities if hasattr(e, "entity_id")],
    )


def _entity(name, type_, path_edge_types=(), path_edge_quantities=()):
    """Build a minimal RetrievedEntity-like object for testing.

    path_edge_types maps to PathEdge.type; path_edge_quantities maps to
    PathEdge.quantities (first edge only, for backward-compat with old tests).
    """
    # Build PathNode/PathEdge-like SimpleNamespaces so render_path can traverse them.
    nodes = [SimpleNamespace(name=name, type=type_)]
    edges = []
    for i, et in enumerate(path_edge_types):
        qty = path_edge_quantities[i] if i < len(path_edge_quantities) else {}
        edges.append(SimpleNamespace(
            type=et,
            negated=False,
            subtype=None,
            memory_fact_id=None,
            quantities=qty,
        ))
        nodes.append(SimpleNamespace(name=f"related-{i}", type="Unknown"))

    path = SimpleNamespace(nodes=nodes, edges=edges)

    return SimpleNamespace(
        name=name,
        type=type_,
        score=0.9,
        entity_id=f"{name.lower().replace(' ', '-')}-id",
        retrieval_mode="vector",
        path=path,
        memory_facts=[],
        supporting_assertions=[],
    )


def _chunk(text, source_doc="session-1"):
    return SimpleNamespace(
        text=text,
        source_doc=source_doc,
        doc_id="doc-1",
        position=0,
        score=0.8,
    )


@pytest.mark.unit
def test_format_search_result_includes_entities_and_chunks():
    import json

    from bench_longmemeval import _format_search_result

    result = _make_result(
        entities=[
            _entity("Alice", "Person"),
            _entity("Atlas Corp", "Organization", ["WORKS_FOR"]),
        ],
        chunks=[_chunk("Alice joined Atlas Corp last year.", "session-42")],
    )
    out = json.loads(_format_search_result(result))
    names = [r["name"] for r in out["results"]]
    assert "Alice" in names
    assert "Atlas Corp" in names
    assert out["results"][1]["path"]["edges"][0]["type"] == "WORKS_FOR"
    assert out["results"][1]["retrieval_mode"] == "vector"
    assert "path_str" in out["results"][1]
    assert out["chunks"][0]["text"] == "Alice joined Atlas Corp last year."
    assert out["chunks"][0]["source_doc"] == "session-42"


@pytest.mark.unit
def test_format_search_result_renders_edge_quantities():
    import json

    from bench_longmemeval import _format_search_result

    qty = {"quantity_value": 3, "quantity_unit": None,
           "quantity_kind": "quantity", "time_scope": None}
    result = _make_result(
        entities=[_entity("bike", "Object", ["OWNS"], [qty])],
    )
    out = json.loads(_format_search_result(result))
    assert out["results"][0]["path"]["edges"][0]["quantities"]["quantity_value"] == 3


@pytest.mark.unit
def test_format_search_result_no_chunks_omits_section():
    import json

    from bench_longmemeval import _format_search_result

    result = _make_result(entities=[_entity("Alice", "Person")])
    out = json.loads(_format_search_result(result))
    assert out["chunks"] == []


@pytest.mark.unit
def test_parse_judge_response_valid_json():
    from bench_longmemeval import _parse_judge_response

    raw = '{"judgment": "correct", "reason": "matches gold answer"}'
    result = _parse_judge_response(raw)
    assert result["judgment"] == "correct"
    assert result["reason"] == "matches gold answer"


@pytest.mark.unit
def test_parse_judge_response_markdown_wrapped_json():
    from bench_longmemeval import _parse_judge_response

    raw = '```json\n{"judgment": "incorrect", "reason": "wrong answer"}\n```'
    result = _parse_judge_response(raw)
    assert result["judgment"] == "incorrect"


@pytest.mark.unit
def test_parse_judge_response_plain_code_block():
    from bench_longmemeval import _parse_judge_response

    raw = '```\n{"judgment": "abstained", "reason": "not mentioned"}\n```'
    result = _parse_judge_response(raw)
    assert result["judgment"] == "abstained"


@pytest.mark.unit
def test_parse_judge_response_malformed_returns_incorrect():
    from bench_longmemeval import _parse_judge_response

    result = _parse_judge_response("not json at all")
    assert result["judgment"] == "incorrect"
    assert "parse error" in result["reason"]


@pytest.mark.unit
def test_parse_judge_response_missing_judgment_key():
    from bench_longmemeval import _parse_judge_response

    result = _parse_judge_response('{"verdict": "yes"}')
    assert result["judgment"] == "incorrect"
    assert "parse error" in result["reason"]
