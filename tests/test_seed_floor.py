"""Unit tests for the seed-similarity floor in retrieval seed construction.

These cover the pure seed-building logic (`_build_seed_sims`) with no database
or GPU involvement. The floor gates *vector-derived* seeds (Qdrant entity hits
and chunk-propagated entities); direct substring/alias seeds (assigned 1.0) are
never gated.
"""
from types import SimpleNamespace

import pytest

from landscape.retrieval.query import _CHUNK_ENTITY_SIM_DISCOUNT, _build_seed_sims

pytestmark = pytest.mark.unit


def _hit(entity_id: str, score: float):
    """A Qdrant ScoredPoint-like stub: only .payload and .score are read."""
    return SimpleNamespace(payload={"entity_id": entity_id}, score=score)


def test_weak_vector_seed_below_floor_is_dropped():
    seeds = _build_seed_sims(
        direct_seed_ids=[],
        entity_hits=[_hit("strong", 0.8), _hit("weak", 0.1)],
        chunk_entities=[],
        chunk_score_by_id={},
        chunk_ids=[],
        seed_floor=0.3,
    )
    assert "strong" in seeds
    assert "weak" not in seeds


def test_direct_alias_seed_survives_high_floor():
    """Direct substring/alias seeds are authoritative (1.0) and never floored."""
    seeds = _build_seed_sims(
        direct_seed_ids=["alias"],
        entity_hits=[],
        chunk_entities=[],
        chunk_score_by_id={},
        chunk_ids=[],
        seed_floor=0.99,
    )
    assert seeds["alias"] == 1.0


def test_subfloor_chunk_does_not_seed_new_entity():
    """A chunk whose discounted score is below the floor must not introduce a
    new graph seed. 0.3 * 0.7 = 0.21 < 0.3."""
    seeds = _build_seed_sims(
        direct_seed_ids=[],
        entity_hits=[],
        chunk_entities=[{"entity_id": "e1", "chunk_eids": ["c1"]}],
        chunk_score_by_id={"c1": 0.3},
        chunk_ids=["c1"],
        seed_floor=0.3,
    )
    assert "e1" not in seeds


def test_above_floor_chunk_seeds_entity_at_discounted_value():
    """0.8 * 0.7 = 0.56 >= floor, so the entity is seeded at the discount."""
    seeds = _build_seed_sims(
        direct_seed_ids=[],
        entity_hits=[],
        chunk_entities=[{"entity_id": "e1", "chunk_eids": ["c1"]}],
        chunk_score_by_id={"c1": 0.8},
        chunk_ids=["c1"],
        seed_floor=0.3,
    )
    assert abs(seeds["e1"] - 0.8 * _CHUNK_ENTITY_SIM_DISCOUNT) < 1e-9


def test_subfloor_chunk_does_not_lower_existing_seed():
    """A weak chunk propagation must never reduce an entity already seeded above
    the floor by a strong vector hit."""
    seeds = _build_seed_sims(
        direct_seed_ids=[],
        entity_hits=[_hit("e1", 0.9)],
        chunk_entities=[{"entity_id": "e1", "chunk_eids": ["c1"]}],
        chunk_score_by_id={"c1": 0.1},
        chunk_ids=["c1"],
        seed_floor=0.3,
    )
    assert seeds["e1"] == 0.9
