"""Unit tests for the ISO temporal parser used by LLM extraction."""


def test_parse_iso_temporal_year_only_from():
    from landscape.extraction.llm import _parse_iso_temporal
    assert _parse_iso_temporal("2015", is_endpoint="from") == "2015-01-01T00:00:00+00:00"


def test_parse_iso_temporal_year_only_until():
    from landscape.extraction.llm import _parse_iso_temporal
    assert _parse_iso_temporal("2015", is_endpoint="until") == "2015-12-31T23:59:59+00:00"


def test_parse_iso_temporal_year_month_from():
    from landscape.extraction.llm import _parse_iso_temporal
    assert _parse_iso_temporal("2015-03", is_endpoint="from") == "2015-03-01T00:00:00+00:00"


def test_parse_iso_temporal_year_month_until_resolves_to_last_day():
    from landscape.extraction.llm import _parse_iso_temporal
    # February non-leap → 28th
    assert _parse_iso_temporal("2015-02", is_endpoint="until") == "2015-02-28T23:59:59+00:00"
    # February leap → 29th
    assert _parse_iso_temporal("2016-02", is_endpoint="until") == "2016-02-29T23:59:59+00:00"
    # April → 30th
    assert _parse_iso_temporal("2015-04", is_endpoint="until") == "2015-04-30T23:59:59+00:00"


def test_parse_iso_temporal_year_month_day_from():
    from landscape.extraction.llm import _parse_iso_temporal
    assert _parse_iso_temporal("2015-06-15", is_endpoint="from") == "2015-06-15T00:00:00+00:00"


def test_parse_iso_temporal_year_month_day_until():
    from landscape.extraction.llm import _parse_iso_temporal
    assert _parse_iso_temporal("2015-06-15", is_endpoint="until") == "2015-06-15T23:59:59+00:00"


def test_parse_iso_temporal_full_datetime_passes_through_utc():
    from landscape.extraction.llm import _parse_iso_temporal
    assert _parse_iso_temporal(
        "2015-06-15T12:30:00Z", is_endpoint="from"
    ) == "2015-06-15T12:30:00+00:00"


def test_parse_iso_temporal_garbage_returns_none(caplog):
    from landscape.extraction.llm import _parse_iso_temporal
    assert _parse_iso_temporal("last quarter", is_endpoint="from") is None
    assert "could not parse" in caplog.text.lower()


def test_parse_iso_temporal_empty_returns_none():
    from landscape.extraction.llm import _parse_iso_temporal
    assert _parse_iso_temporal("", is_endpoint="from") is None
    assert _parse_iso_temporal(None, is_endpoint="from") is None
