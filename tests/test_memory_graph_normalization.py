from landscape.memory_graph.families import FAMILY_REGISTRY
from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.normalization import normalize_assertion


def test_related_to_is_not_promotable():
    assert "RELATED_TO" not in FAMILY_REGISTRY


def test_benchmark_parity_families_are_explicitly_promotable():
    assert FAMILY_REGISTRY["LEADS"].traversable is True
    assert FAMILY_REGISTRY["LOCATED_IN"].traversable is True
    assert FAMILY_REGISTRY["APPROVED"].traversable is True


def test_normalize_assertion_promotes_known_family():
    payload = AssertionPayload(
        source_kind="document",
        source_id="doc-1",
        raw_subject_text="Alice",
        raw_relation_text="works for",
        raw_object_text="Acme",
        confidence=0.9,
        family_candidate="WORKS_FOR",
    )
    result = normalize_assertion(
        payload,
        subject_entity_id="ent-alice",
        object_entity_id="ent-acme",
    )
    assert result.promotable is True
    assert result.family == "WORKS_FOR"
    assert result.object_entity_id == "ent-acme"


def test_normalize_assertion_keeps_related_to_assertion_only():
    payload = AssertionPayload(
        source_kind="document",
        source_id="doc-1",
        raw_subject_text="Alice",
        raw_relation_text="is connected to",
        raw_object_text="Acme",
        confidence=0.6,
        family_candidate="RELATED_TO",
    )
    result = normalize_assertion(
        payload,
        subject_entity_id="ent-alice",
        object_entity_id="ent-acme",
    )
    assert result.promotable is False
    assert result.family is None
