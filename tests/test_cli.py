from dataclasses import dataclass

import pytest

from landscape import cli


@dataclass
class FakeIngestResult:
    doc_id: str = "doc-123"
    already_existed: bool = False
    entities_created: int = 2
    entities_reinforced: int = 1
    relations_created: int = 3
    relations_reinforced: int = 1
    relations_superseded: int = 0
    chunks_created: int = 4


class FakeEncoder:
    def __init__(self):
        self.loaded = False

    def load_model(self):
        self.loaded = True


class FakeQdrantStore:
    def __init__(self):
        self.entity_collection_initialized = False
        self.chunk_collection_initialized = False
        self.closed = False

    async def init_collection(self):
        self.entity_collection_initialized = True

    async def init_chunks_collection(self):
        self.chunk_collection_initialized = True

    async def close_client(self):
        self.closed = True


class FakeNeo4jStore:
    def __init__(self):
        self.closed = False

    async def close_driver(self):
        self.closed = True


class FakePipeline:
    def __init__(self, ingest_error=None):
        self.calls = []
        self.ingest_error = ingest_error

    async def ingest(self, text, title, source_type="text", session_id=None, turn_id=None):
        self.calls.append(
            {
                "text": text,
                "title": title,
                "source_type": source_type,
                "session_id": session_id,
                "turn_id": turn_id,
            }
        )
        if self.ingest_error is not None:
            raise self.ingest_error
        return FakeIngestResult()


@pytest.fixture
def fake_runtime(monkeypatch):
    encoder = FakeEncoder()
    qdrant_store = FakeQdrantStore()
    neo4j_store = FakeNeo4jStore()
    pipeline = FakePipeline()
    monkeypatch.setattr(cli, "encoder", encoder)
    monkeypatch.setattr(cli, "qdrant_store", qdrant_store)
    monkeypatch.setattr(cli, "neo4j_store", neo4j_store)
    monkeypatch.setattr(cli, "pipeline", pipeline)
    return {
        "encoder": encoder,
        "qdrant_store": qdrant_store,
        "neo4j_store": neo4j_store,
        "pipeline": pipeline,
    }


def test_ingest_unexpected_failure_closes_runtime(tmp_path, capsys, monkeypatch):
    path = tmp_path / "failure.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    encoder = FakeEncoder()
    qdrant_store = FakeQdrantStore()
    neo4j_store = FakeNeo4jStore()
    pipeline = FakePipeline(ingest_error=RuntimeError("ingest exploded"))
    monkeypatch.setattr(cli, "encoder", encoder)
    monkeypatch.setattr(cli, "qdrant_store", qdrant_store)
    monkeypatch.setattr(cli, "neo4j_store", neo4j_store)
    monkeypatch.setattr(cli, "pipeline", pipeline)

    exit_code = cli.main(["ingest", str(path)])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "ingest exploded" in stderr or "Error:" in stderr
    assert encoder.loaded is True
    assert qdrant_store.entity_collection_initialized is True
    assert qdrant_store.chunk_collection_initialized is True
    assert neo4j_store.closed is True
    assert qdrant_store.closed is True


def test_ingest_uses_file_stem_as_default_title(tmp_path, capsys, fake_runtime):
    path = tmp_path / "architecture-notes.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    exit_code = cli.main(["ingest", str(path)])

    assert exit_code == 0
    assert fake_runtime["pipeline"].calls == [
        {
            "text": "Alice leads Project Atlas.",
            "title": "architecture-notes",
            "source_type": "text",
            "session_id": None,
            "turn_id": None,
        }
    ]
    assert fake_runtime["encoder"].loaded is True
    assert fake_runtime["qdrant_store"].entity_collection_initialized is True
    assert fake_runtime["qdrant_store"].chunk_collection_initialized is True
    assert fake_runtime["neo4j_store"].closed is True
    assert fake_runtime["qdrant_store"].closed is True
    output = capsys.readouterr().out
    assert "doc_id: doc-123" in output
    assert "entities: created=2 reinforced=1" in output
    assert "relations: created=3 reinforced=1 superseded=0" in output
    assert "chunks_created: 4" in output


def test_ingest_accepts_explicit_metadata(tmp_path, fake_runtime):
    path = tmp_path / "input.txt"
    path.write_text("Acme Corp uses PostgreSQL.", encoding="utf-8")

    exit_code = cli.main(
        [
            "ingest",
            str(path),
            "--title",
            "Architecture Notes",
            "--source-type",
            "markdown",
            "--session-id",
            "session-1",
            "--turn-id",
            "turn-1",
        ]
    )

    assert exit_code == 0
    assert fake_runtime["pipeline"].calls == [
        {
            "text": "Acme Corp uses PostgreSQL.",
            "title": "Architecture Notes",
            "source_type": "markdown",
            "session_id": "session-1",
            "turn_id": "turn-1",
        }
    ]


def test_missing_file_exits_before_initialization(tmp_path, fake_runtime):
    path = tmp_path / "missing.md"

    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest", str(path)])

    assert exc.value.code == 2
    assert fake_runtime["pipeline"].calls == []
    assert fake_runtime["encoder"].loaded is False
    assert fake_runtime["qdrant_store"].entity_collection_initialized is False
    assert fake_runtime["qdrant_store"].chunk_collection_initialized is False
    assert fake_runtime["qdrant_store"].closed is False
    assert fake_runtime["neo4j_store"].closed is False


def test_incomplete_provenance_exits_before_initialization(tmp_path, fake_runtime):
    path = tmp_path / "doc.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest", str(path), "--session-id", "session-1"])

    assert exc.value.code == 2
    assert fake_runtime["pipeline"].calls == []
    assert fake_runtime["encoder"].loaded is False
    assert fake_runtime["qdrant_store"].entity_collection_initialized is False
    assert fake_runtime["qdrant_store"].chunk_collection_initialized is False
    assert fake_runtime["qdrant_store"].closed is False
    assert fake_runtime["neo4j_store"].closed is False
