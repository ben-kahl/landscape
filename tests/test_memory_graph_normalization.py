from landscape.memory_graph.families import FAMILY_REGISTRY


def test_related_to_is_not_promotable():
    assert "RELATED_TO" not in FAMILY_REGISTRY


def test_benchmark_parity_families_are_explicitly_promotable():
    assert FAMILY_REGISTRY["LEADS"].traversable is True
    assert FAMILY_REGISTRY["LOCATED_IN"].traversable is True
    assert FAMILY_REGISTRY["APPROVED"].traversable is True
