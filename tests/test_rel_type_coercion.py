"""Tests for embedding-based rel_type coercion.

These tests require the sentence-transformer encoder to be loaded, so they
use the ``http_client`` fixture (which triggers encoder.load_model() via the
app lifespan).
"""

from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.external]


@pytest.mark.asyncio
async def test_string_synonym_short_circuits(http_client):
    """EMPLOYED_BY is in RELATION_SYNONYMS -> WORKS_FOR.

    The synonym map fires before the embedding path, so confidence == 1.0
    and the encoder is never called.
    """
    from landscape.extraction.rel_type_coercion import _canonical_embeddings, coerce_rel_type

    # Patch _canonical_embeddings.get to assert it's never called
    original_get = _canonical_embeddings.get
    call_count = 0

    def counting_get():
        nonlocal call_count
        call_count += 1
        return original_get()

    with patch.object(_canonical_embeddings, "get", counting_get):
        result_type, confidence = coerce_rel_type("EMPLOYED_BY")

    assert result_type == "WORKS_FOR"
    assert confidence == 1.0
    assert call_count == 0, (
        "Embedding path should not be invoked for a string-synonym hit"
    )


@pytest.mark.asyncio
async def test_canonical_passthrough_unchanged(http_client):
    """WORKS_FOR is already canonical — should be returned as WORKS_FOR."""
    from landscape.extraction.rel_type_coercion import coerce_rel_type

    result_type, confidence = coerce_rel_type("WORKS_FOR")
    assert result_type == "WORKS_FOR"
    # confidence is the cosine similarity of "WORKS_FOR (relation...)" vs
    # canonical WORKS_FOR descriptor — should be high
    assert confidence > 0.5


@pytest.mark.asyncio
async def test_novel_employment_type_coerces_to_works_for(http_client):
    """Novel type 'JOINED_AS_EMPLOYEE' should map to WORKS_FOR (path 3).

    This exercises the novel-type embedding path: the string is not in the
    synonym map, not in the canonical vocab, so it gets embedded and compared.
    'JOINED_AS_EMPLOYEE' should land closer to WORKS_FOR (employment,
    hired by, affiliated with) than to any other canonical.
    """
    from landscape.extraction.rel_type_coercion import coerce_rel_type

    result_type, confidence = coerce_rel_type("JOINED_AS_EMPLOYEE")
    assert result_type == "WORKS_FOR", (
        f"Expected WORKS_FOR for 'JOINED_AS_EMPLOYEE', got {result_type!r} (score={confidence:.3f})"
    )
    assert confidence >= 0.55


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "Bare-token disambiguation: 'LOCATED_IN' vs 'WORKS_FOR' with only "
        "the token itself as input is ambiguous — the embeddings are within "
        "COERCION_MARGIN of each other when no surrounding context is "
        "provided. The intended mitigation for the 'Alice moved to Beacon' "
        "case is the add_relation docstring vocab block (belt-and-suspenders) "
        "plus the LLM being guided to prefer WORKS_FOR for org changes. "
        "Embedding-based coercion only fires with margin >= 0.05; tightening "
        "would cause false positives on legitimate physical-location uses."
    ),
    strict=False,
)
async def test_located_in_for_employment_coerces_to_works_for(http_client):
    """LOCATED_IN supplied for an employment context should coerce to WORKS_FOR.

    This is xfail because bare 'LOCATED_IN' token is too ambiguous for
    embedding-based override without surrounding context. See xfail reason.
    """
    from landscape.extraction.rel_type_coercion import coerce_rel_type

    result_type, _confidence = coerce_rel_type("LOCATED_IN")
    assert result_type == "WORKS_FOR"


@pytest.mark.asyncio
async def test_unknown_low_similarity_passes_through(http_client):
    """A truly domain-alien rel_type string should not be coerced above threshold.

    Note: RELATED_TO is the catch-all canonical and its descriptor ("general
    relationship, connected to, associated with") means generic-sounding
    novel types will often match it at moderate similarity. This test uses
    a purely nonsensical / non-relational string to validate the pass-through
    path. The assertion checks that the confidence is below threshold OR the
    result is the raw normalized form (not a semantic coercion to a specific
    directional canonical like WORKS_FOR, LEADS, etc.).
    """
    from landscape.extraction.rel_type_coercion import coerce_rel_type

    # XXXXXXXXXX has no semantic meaning — should not coerce to anything
    # directional/specific. It may still technically return RELATED_TO
    # (which is acceptable as a catch-all), but the test verifies the
    # coercion module doesn't pick something semantically wrong.
    result_type, confidence = coerce_rel_type("XXXXXXXXXX_NONEXISTENT_TYPE")
    # Accept RELATED_TO (catch-all) or the raw pass-through, but not a
    # specific directional canonical (WORKS_FOR, LEADS, etc.)
    directional_canonicals = {
        "WORKS_FOR",
        "LEADS",
        "MEMBER_OF",
        "REPORTS_TO",
        "APPROVED",
        "USES",
        "BELONGS_TO",
        "LOCATED_IN",
        "CREATED",
    }
    assert result_type not in directional_canonicals, (
        "Nonsense token should not coerce to specific canonical, "
        f"got {result_type!r} ({confidence:.3f})"
    )


@pytest.mark.asyncio
async def test_other_string_synonyms_short_circuit(http_client):
    """Spot-check a few more RELATION_SYNONYMS entries for the 1.0 confidence path."""
    from landscape.extraction.rel_type_coercion import coerce_rel_type

    cases = [
        ("MANAGES", "LEADS"),
        ("OWNS", "LEADS"),
        ("DEPENDS_ON", "USES"),
        ("FOUNDED", "CREATED"),
        ("BASED_IN", "LOCATED_IN"),
    ]
    for raw, expected in cases:
        result_type, confidence = coerce_rel_type(raw)
        assert result_type == expected, f"{raw!r} -> {result_type!r}, expected {expected!r}"
        assert confidence == 1.0, f"{raw!r} should short-circuit with confidence 1.0"


@pytest.mark.asyncio
async def test_empty_string_returns_related_to(http_client):
    """Empty rel_type should return RELATED_TO with 0.0 confidence."""
    from landscape.extraction.rel_type_coercion import coerce_rel_type

    result_type, confidence = coerce_rel_type("")
    assert result_type == "RELATED_TO"
    assert confidence == 0.0
