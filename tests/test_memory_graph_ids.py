import pytest

from landscape.memory_graph.families import FAMILY_REGISTRY
from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.ids import alias_id, assertion_id, fact_key, slot_key


def test_alias_id_is_canonical_entity_scoped():
    first = alias_id("ent-person-123", "Bob")
    second = alias_id("ent-person-123", " bob ")
    other = alias_id("ent-person-999", "Bob")
    assert first == second
    assert first != other


def test_assertion_id_is_source_local_and_deterministic():
    left = assertion_id(
        source_kind="document",
        source_id="doc-1",
        raw_subject_text="Alice",
        raw_relation_text="works at",
        raw_object_text="Acme",
        subtype=None,
        qualifier_payload={},
        chunk_refs=[("doc-1:0:abc", 3, 22)],
    )
    right = assertion_id(
        source_kind="document",
        source_id="doc-1",
        raw_subject_text="Alice",
        raw_relation_text="works at",
        raw_object_text="Acme",
        subtype=None,
        qualifier_payload={},
        chunk_refs=[("doc-1:0:abc", 3, 22)],
    )
    other_source = assertion_id(
        source_kind="document",
        source_id="doc-2",
        raw_subject_text="Alice",
        raw_relation_text="works at",
        raw_object_text="Acme",
        subtype=None,
        qualifier_payload={},
        chunk_refs=[("doc-1:0:abc", 3, 22)],
    )
    assert left == right
    assert left != other_source


def test_family_registry_contains_expected_v1_promotable_families():
    assert FAMILY_REGISTRY["WORKS_FOR"].slot_mode == "subject"
    assert FAMILY_REGISTRY["USES"].slot_mode == "additive"
    assert FAMILY_REGISTRY["HAS_TITLE"].slot_mode == "object"
    assert FAMILY_REGISTRY["HAS_PREFERENCE"].slot_mode == "subtype"
    assert FAMILY_REGISTRY["APPROVED"].traversable is True


def test_slot_and_fact_keys_are_family_specific():
    works_for = FAMILY_REGISTRY["WORKS_FOR"]
    uses = FAMILY_REGISTRY["USES"]
    happened_on = FAMILY_REGISTRY["HAPPENED_ON"]
    assert fact_key(works_for, "ent-a", "ent-b", None) == "WORKS_FOR:ent-a:ent-b"
    assert slot_key(works_for, "ent-a", "ent-b", None) == "WORKS_FOR:ent-a"
    assert fact_key(uses, "ent-a", "ent-b", None) == "USES:ent-a:ent-b"
    assert slot_key(uses, "ent-a", "ent-b", None) == "USES:ent-a:ent-b"
    assert fact_key(
        happened_on,
        "ent-a",
        None,
        None,
        value_time="2026-03-05",
    ) == "HAPPENED_ON:ent-a:value_time=2026-03-05"
    assert slot_key(
        happened_on,
        "ent-a",
        None,
        None,
        value_time="2026-03-05",
    ) == "HAPPENED_ON:ent-a"


def test_created_fact_key_uses_subtype_identity():
    created = FAMILY_REGISTRY["CREATED"]
    assert fact_key(created, "ent-a", None, "diagram") == "CREATED:ent-a:diagram"


def test_assertion_payload_rejects_conflicting_dual_value_fields():
    with pytest.raises(ValueError):
        AssertionPayload(
            source_kind="document",
            source_id="doc-1",
            raw_subject_text="Kickoff",
            raw_relation_text="happened on",
            raw_object_text="2026-03-05",
            confidence=0.9,
            family_candidate="HAPPENED_ON",
            value_time="2026-03-05",
            time_scope="2026-03-04",
        )
