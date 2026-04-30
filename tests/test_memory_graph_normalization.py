from landscape.memory_graph.families import FAMILY_REGISTRY
from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.normalization import normalize_assertion


def test_related_to_is_not_promotable():
    assert "RELATED_TO" not in FAMILY_REGISTRY


def test_promotable_family_registry_uses_explicit_slot_modes():
    assert FAMILY_REGISTRY["WORKS_FOR"].slot_mode == "subject"
    assert FAMILY_REGISTRY["REPORTS_TO"].slot_mode == "subject"
    assert FAMILY_REGISTRY["HAS_TITLE"].slot_mode == "object"
    assert FAMILY_REGISTRY["HAS_PREFERENCE"].slot_mode == "subtype"
    assert FAMILY_REGISTRY["WORKS_ON"].slot_mode == "additive"
    assert FAMILY_REGISTRY["MAINTAINS"].slot_mode == "additive"
    assert FAMILY_REGISTRY["OWNS"].slot_mode == "additive"
    assert FAMILY_REGISTRY["DEPENDS_ON"].slot_mode == "additive"
    assert FAMILY_REGISTRY["HAPPENED_ON"].object_kind == "value"
    assert FAMILY_REGISTRY["RECOMMENDED"].object_kind == "value"
    assert FAMILY_REGISTRY["DISCUSSED"].object_kind == "value"
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
    assert result.slot_key == "WORKS_FOR:ent-alice"


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


def test_normalize_assertion_promotes_value_family_without_object_entity():
    payload = AssertionPayload(
        source_kind="document",
        source_id="doc-2",
        raw_subject_text="Kickoff",
        raw_relation_text="happened on",
        raw_object_text="2026-03-05",
        confidence=0.95,
        family_candidate="HAPPENED_ON",
        value_time="2026-03-05",
    )
    result = normalize_assertion(
        payload,
        subject_entity_id="ent-kickoff",
        object_entity_id=None,
    )
    assert result.promotable is True
    assert result.family == "HAPPENED_ON"
    assert result.value_time == "2026-03-05"
    assert result.slot_key == "HAPPENED_ON:ent-kickoff"
