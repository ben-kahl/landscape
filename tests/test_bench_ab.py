"""Unit tests for bench_ab helper behavior."""
import os
import pathlib
import sys

import pytest

# The bench script lives in scripts/, not a package. Import its helpers directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.mark.unit
def test_load_module_from_path_supports_dataclass_modules():
    from bench_ab import _load_module_from_path

    module_path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "fixtures" / (
        "supersession_scenarios.py"
    )

    module = _load_module_from_path("test_supersession_scenarios", module_path)

    assert module.ScenarioResult.__module__ == "test_supersession_scenarios"
    assert callable(module.run_all_scenarios)


@pytest.mark.unit
def test_iter_haystack_sessions_supports_parallel_session_ids():
    from bench_ab import _iter_haystack_sessions

    turns = [
        {"role": "user", "content": "Where did I study?"},
        {"role": "assistant", "content": "You studied in Boston."},
    ]
    question = {
        "haystack_session_ids": ["session-42"],
        "haystack_sessions": [turns],
    }

    entries = list(_iter_haystack_sessions(question, fallback_prefix="q0"))

    assert entries == [("session-42", turns)]


@pytest.mark.unit
def test_iter_haystack_sessions_supports_embedded_session_records():
    from bench_ab import _iter_haystack_sessions

    turns = [
        {"role": "user", "content": "Who is Maya?"},
    ]
    question = {
        "haystack_sessions": [
            {
                "session_id": "session-99",
                "turns": turns,
            }
        ],
    }

    entries = list(_iter_haystack_sessions(question, fallback_prefix="q1"))

    assert entries == [("session-99", turns)]
