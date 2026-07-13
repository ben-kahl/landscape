"""Unit tests for the per-entity adjacent-fact cap in hybrid retrieval."""

from __future__ import annotations

import pytest

from landscape.retrieval.query import _MAX_ADJACENT_FACTS, _top_adjacent_facts

pytestmark = pytest.mark.unit


def _fact(fid: str, confidence: float) -> dict:
    return {"memory_fact_id": fid, "confidence_agg": confidence}


def test_orders_by_confidence_desc():
    facts = [_fact("a", 0.2), _fact("b", 0.9), _fact("c", 0.5)]
    assert [f["memory_fact_id"] for f in _top_adjacent_facts(facts)] == ["b", "c", "a"]


def test_caps_at_limit():
    facts = [_fact(str(i), i / 100) for i in range(_MAX_ADJACENT_FACTS + 10)]
    assert len(_top_adjacent_facts(facts)) == _MAX_ADJACENT_FACTS


def test_handles_missing_confidence():
    facts = [_fact("a", 0.3), {"memory_fact_id": "b", "confidence_agg": None}]
    assert [f["memory_fact_id"] for f in _top_adjacent_facts(facts)] == ["a", "b"]
