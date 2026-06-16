import pytest
from stack_config import (
    DEFAULT_TEST_NEO4J_URI,
    DEFAULT_TEST_QDRANT_URL,
    LIVE_NEO4J_URI,
    LIVE_QDRANT_URL,
    StackConfig,
    assert_safe_to_wipe,
    resolve_test_stack_config,
)

pytestmark = pytest.mark.unit


def test_test_stack_defaults_use_isolated_ports():
    config = resolve_test_stack_config({})

    assert config.neo4j_uri == DEFAULT_TEST_NEO4J_URI
    assert config.qdrant_url == DEFAULT_TEST_QDRANT_URL


def test_test_stack_config_threads_environment_overrides():
    config = resolve_test_stack_config(
        {
            "LANDSCAPE_TEST_NEO4J_URI": "bolt://example.test:17687",
            "LANDSCAPE_TEST_NEO4J_USER": "tester",
            "LANDSCAPE_TEST_NEO4J_PASSWORD": "secret",
            "LANDSCAPE_TEST_QDRANT_URL": "http://example.test:16333",
        }
    )

    assert config == StackConfig(
        neo4j_uri="bolt://example.test:17687",
        neo4j_auth=("tester", "secret"),
        qdrant_url="http://example.test:16333",
    )


def test_live_stack_wipe_requires_explicit_escape_hatch():
    config = StackConfig(
        neo4j_uri=LIVE_NEO4J_URI,
        neo4j_auth=("neo4j", "landscape-dev"),
        qdrant_url=LIVE_QDRANT_URL,
    )

    with pytest.raises(RuntimeError, match="LANDSCAPE_ALLOW_LIVE_TEST_WIPE=1"):
        assert_safe_to_wipe(config, {})


def test_live_stack_wipe_escape_hatch_is_explicit():
    config = StackConfig(
        neo4j_uri=LIVE_NEO4J_URI,
        neo4j_auth=("neo4j", "landscape-dev"),
        qdrant_url=LIVE_QDRANT_URL,
    )

    assert_safe_to_wipe(config, {"LANDSCAPE_ALLOW_LIVE_TEST_WIPE": "1"})
