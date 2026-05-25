import importlib.util
from pathlib import Path

import pytest


@pytest.mark.unit
def test_ingest_woo_test_defaults_target_test_stack(monkeypatch):
    script_path = Path(__file__).parent.parent / "scripts" / "ingest_woo_test.py"
    spec = importlib.util.spec_from_file_location("ingest_woo_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args = module.parse_args([])
    assert args.path.name == "woo.txt"
    assert args.neo4j_uri == "bolt://localhost:17687"
    assert args.qdrant_url == "http://localhost:16333"

    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    module.configure_environment(args)

    import os

    assert os.environ["NEO4J_URI"] == "bolt://localhost:17687"
    assert os.environ["QDRANT_URL"] == "http://localhost:16333"


@pytest.mark.unit
def test_ingest_woo_test_rejects_live_defaults_without_override():
    script_path = Path(__file__).parent.parent / "scripts" / "ingest_woo_test.py"
    spec = importlib.util.spec_from_file_location("ingest_woo_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args = module.parse_args(["--neo4j-uri", "bolt://localhost:7687"])

    with pytest.raises(SystemExit):
        module.assert_safe_target(args)
