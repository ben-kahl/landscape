"""Smoke test for landscape ingest-dir with mixed file types.

Verifies that the default (extension-dispatched) mode picks up markdown,
text, and HTML files; converts each through the right path; and skips
files whose extension has no converter. Uses the fake-runtime pattern
from tests/test_cli.py so no real Neo4j/Qdrant is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from landscape import cli
from landscape.cli import ingest as ingest_cli

pytestmark = pytest.mark.unit


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ingest(
        self,
        text: str,
        title: str,
        source_type: str = "text",
        session_id: str | None = None,
        turn_id: str | None = None,
        debug: bool = False,
        log_context=None,
    ):
        self.calls.append({"title": title, "source_type": source_type})
        return _FakeResult()

    async def ingest_file(self, *args, **kwargs):  # pragma: no cover — unused here
        raise NotImplementedError


class _FakeResult:
    doc_id = "doc-x"
    already_existed = False
    entities_created = 1
    entities_reinforced = 0
    relations_created = 1
    relations_reinforced = 0
    relations_superseded = 0
    chunks_created = 1


class _FakeEncoder:
    def __init__(self) -> None:
        self.loaded = False

    def load_model(self) -> None:
        self.loaded = True


class _FakeQdrant:
    def __init__(self) -> None:
        self.entity_collection_initialized = False
        self.chunk_collection_initialized = False
        self.closed = False

    async def init_collection(self) -> None:
        self.entity_collection_initialized = True

    async def init_chunks_collection(self) -> None:
        self.chunk_collection_initialized = True

    async def close(self) -> None:
        self.closed = True


class _FakeNeo4j:
    def __init__(self) -> None:
        self.closed = False
        self.backfilled = False

    async def backfill_ingest_completed_marker(self) -> None:
        self.backfilled = True

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch):
    pipeline = _FakePipeline()
    encoder = _FakeEncoder()
    neo4j_store = _FakeNeo4j()
    qdrant_store = _FakeQdrant()
    monkeypatch.setattr(
        ingest_cli,
        "_get_runtime",
        lambda: (pipeline, encoder, neo4j_store, qdrant_store),
    )
    return {
        "pipeline": pipeline,
        "encoder": encoder,
        "neo4j_store": neo4j_store,
        "qdrant_store": qdrant_store,
    }


def test_ingest_dir_dispatches_by_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runtime: dict
) -> None:
    (tmp_path / "a.md").write_text("# Alice\n\nAlice leads Atlas.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Bob runs Beta.", encoding="utf-8")
    (tmp_path / "c.html").write_text(
        "<h1>Carol</h1><p>Carol uses Postgres.</p>", encoding="utf-8"
    )
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01binary")

    exit_code = cli.main(["ingest-dir", str(tmp_path)])

    assert exit_code == 0
    calls = fake_runtime["pipeline"].calls
    source_types = {c["source_type"] for c in calls}
    titles = {c["title"] for c in calls}
    assert source_types == {"markdown", "text", "html"}
    assert titles == {"a", "b", "c"}

    out = capsys.readouterr().out
    assert "skip: skip.bin" in out


def test_ingest_dir_glob_mode_restricts_to_pattern(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runtime: dict
) -> None:
    (tmp_path / "a.md").write_text("Alice", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Bob", encoding="utf-8")
    (tmp_path / "c.csv").write_text("name\nCarol\n", encoding="utf-8")

    exit_code = cli.main(["ingest-dir", str(tmp_path), "--glob", "*.md"])

    assert exit_code == 0
    calls = fake_runtime["pipeline"].calls
    assert len(calls) == 1
    assert calls[0]["source_type"] == "markdown"
    assert calls[0]["title"] == "a"

    out = capsys.readouterr().out
    # Glob mode does not print skip lines — caller asked for an explicit subset.
    assert "skip:" not in out


def test_ingest_dir_source_type_override_applies_to_all(
    tmp_path: Path, fake_runtime: dict
) -> None:
    (tmp_path / "a.md").write_text("Alice", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Bob", encoding="utf-8")

    exit_code = cli.main(
        ["ingest-dir", str(tmp_path), "--source-type", "release-notes"]
    )

    assert exit_code == 0
    calls = fake_runtime["pipeline"].calls
    assert all(c["source_type"] == "release-notes" for c in calls)
