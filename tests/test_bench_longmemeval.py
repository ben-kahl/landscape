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
    return SimpleNamespace(results=list(entities), chunks=list(chunks))


def _entity(name, type_, path_edge_types=(), path_edge_quantities=()):
    return SimpleNamespace(
        name=name,
        type=type_,
        path_edge_types=list(path_edge_types),
        path_edge_quantities=list(path_edge_quantities),
    )


def _chunk(text, source_doc="session-1"):
    return SimpleNamespace(text=text, source_doc=source_doc)


@pytest.mark.unit
def test_format_search_result_includes_entities_and_chunks():
    from bench_longmemeval import _format_search_result

    result = _make_result(
        entities=[
            _entity("Alice", "Person"),
            _entity("Atlas Corp", "Organization", ["WORKS_FOR"]),
        ],
        chunks=[_chunk("Alice joined Atlas Corp last year.", "session-42")],
    )
    out = _format_search_result(result)
    assert "Alice (Person)" in out
    assert "Atlas Corp (Organization)" in out
    assert "WORKS_FOR" in out
    assert "Alice joined Atlas Corp last year." in out
    assert "session-42" in out


@pytest.mark.unit
def test_format_search_result_renders_edge_quantities():
    from bench_longmemeval import _format_search_result

    qty = {"quantity_value": 3, "quantity_unit": None, "quantity_kind": "quantity", "time_scope": None}
    result = _make_result(
        entities=[_entity("bike", "Object", ["OWNS"], [qty])],
    )
    out = _format_search_result(result)
    assert "quantity=3" in out


@pytest.mark.unit
def test_format_search_result_no_chunks_omits_section():
    from bench_longmemeval import _format_search_result

    result = _make_result(entities=[_entity("Alice", "Person")])
    out = _format_search_result(result)
    assert "Source Passages" not in out


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
