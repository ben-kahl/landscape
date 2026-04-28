from landscape.memory_graph.families import FAMILY_REGISTRY
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
    assert FAMILY_REGISTRY["WORKS_FOR"].single_current is True
    assert FAMILY_REGISTRY["USES"].single_current is False
    assert FAMILY_REGISTRY["APPROVED"].traversable is True


def test_slot_and_fact_keys_are_family_specific():
    works_for = FAMILY_REGISTRY["WORKS_FOR"]
    uses = FAMILY_REGISTRY["USES"]
    assert fact_key(works_for, "ent-a", "ent-b", None) == "WORKS_FOR:ent-a:ent-b"
    assert slot_key(works_for, "ent-a", "ent-b", None) == "WORKS_FOR:ent-a"
    assert fact_key(uses, "ent-a", "ent-b", None) == "USES:ent-a:ent-b"
    assert slot_key(uses, "ent-a", "ent-b", None) == "USES:ent-a:ent-b"


def test_created_fact_key_uses_subtype_identity():
    created = FAMILY_REGISTRY["CREATED"]
    assert fact_key(created, "ent-a", None, "diagram") == "CREATED:ent-a:diagram"
