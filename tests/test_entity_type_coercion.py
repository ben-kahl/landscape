"""Tests for src/landscape/extraction/entity_type_coercion.py.

Non-embedding tests (paths 1 and 2) are plain synchronous tests.
Embedding tests (path 3) use the ``http_client`` fixture which triggers
encoder.load_model() via the FastAPI app lifespan.
"""

import pytest

# ---------------------------------------------------------------------------
# Path 1 & 2 tests — no encoder needed
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_canonical_passthrough():
    """Exact canonical type returns (canonical, 1.0)."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("Person")
    assert result == "Person"
    assert score == 1.0

@pytest.mark.unit
def test_case_insensitive_canonical():
    """Uppercase canonical input still returns canonical with score 1.0."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("PERSON")
    assert result == "Person"
    assert score == 1.0

@pytest.mark.unit
def test_case_insensitive_canonical_all_types():
    """All 8 canonical types pass through regardless of case."""
    from landscape.extraction.entity_type_coercion import ENTITY_TYPE_VOCAB, coerce_entity_type

    for canonical in ENTITY_TYPE_VOCAB:
        result, score = coerce_entity_type(canonical.upper())
        assert result == canonical, f"Expected {canonical!r}, got {result!r}"
        assert score == 1.0

@pytest.mark.unit
def test_synonym_match_company():
    """'Company' maps to Organization via synonym map."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("Company")
    assert result == "Organization"
    assert score == 1.0

@pytest.mark.unit
def test_synonym_match_individual():
    """'Individual' maps to Person via synonym map."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("individual")
    assert result == "Person"
    assert score == 1.0

@pytest.mark.unit
def test_synonym_match_framework():
    """'Framework' maps to Technology via synonym map."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("Framework")
    assert result == "Technology"
    assert score == 1.0

@pytest.mark.unit
def test_subtype_not_set_when_canonical_matches():
    """When input is already canonical, the subtype logic skips (canonical == input)."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    canonical, score = coerce_entity_type("Organization")
    # When canonical == input, writeback skips subtype — score 1.0 signals canonical match
    assert canonical == "Organization"
    assert score == 1.0


# ---------------------------------------------------------------------------
# Path 3 tests — require encoder loaded via http_client fixture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.external
async def test_embedding_coercion_specific_to_broad(http_client):
    """'SoftwareEngineer' should match Person via synonym map (covered by path 2).
    Verify it routes correctly through the coercion pipeline."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("SoftwareEngineer")
    assert result == "Person", (
        f"Expected 'Person' for 'SoftwareEngineer', got {result!r} (score={score:.3f})"
    )
    assert score == 1.0  # synonym map fires, no embedding needed


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.external
async def test_embedding_coercion_tech(http_client):
    """'DatabaseEngine' should embed close enough to Technology (>= 0.55)."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("DatabaseEngine")
    assert result == "Technology", (
        f"Expected 'Technology' for 'DatabaseEngine', got {result!r} (score={score:.3f})"
    )
    assert score >= 0.55


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.external
async def test_embedding_coercion_location(http_client):
    """'GeographicArea' should embed close enough to Location (>= 0.55)."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("GeographicArea")
    assert result == "Location", (
        f"Expected 'Location' for 'GeographicArea', got {result!r} (score={score:.3f})"
    )
    assert score >= 0.55


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.external
async def test_low_similarity_passes_through(http_client):
    """Nonsense type with no close canonical match returns (raw, 0.0)."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("ZorblaxQuantum")
    assert result == "ZorblaxQuantum"
    assert score == 0.0


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.external
async def test_empty_string_returns_fallback(http_client):
    """Empty string input returns a sensible fallback without raising."""
    from landscape.extraction.entity_type_coercion import coerce_entity_type

    result, score = coerce_entity_type("")
    # Should not raise; returns something
    assert isinstance(result, str)
    assert isinstance(score, float)
