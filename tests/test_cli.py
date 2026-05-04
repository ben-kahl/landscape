import builtins
from dataclasses import dataclass

import pytest

from landscape import cli

pytestmark = pytest.mark.smoke


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
    def __init__(self, close_error=None):
        self.closed = False
        self.close_error = close_error

    async def close_driver(self):
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class FakePipeline:
    def __init__(self, ingest_error=None):
        self.calls = []
        self.ingest_error = ingest_error

    async def ingest(
        self,
        text,
        title,
        source_type="text",
        session_id=None,
        turn_id=None,
        debug=False,
    ):
        self.calls.append(
            {
                "text": text,
                "title": title,
                "source_type": source_type,
                "session_id": session_id,
                "turn_id": turn_id,
                "debug": debug,
            }
        )
        if self.ingest_error is not None:
            raise self.ingest_error
        return FakeIngestResult()


def assert_runtime_untouched(fake_runtime):
    assert fake_runtime["pipeline"].calls == []
    assert fake_runtime["encoder"].loaded is False
    assert fake_runtime["qdrant_store"].entity_collection_initialized is False
    assert fake_runtime["qdrant_store"].chunk_collection_initialized is False
    assert fake_runtime["qdrant_store"].closed is False
    assert fake_runtime["neo4j_store"].closed is False


def test_cli_process_defaults_use_host_service_urls(monkeypatch):
    for name in (
        "NEO4J_URI",
        "QDRANT_URL",
        "OLLAMA_URL",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising=False)

    cli._apply_cli_process_defaults()

    assert cli.os.environ["NEO4J_URI"] == "bolt://localhost:7687"
    assert cli.os.environ["QDRANT_URL"] == "http://localhost:6333"
    assert cli.os.environ["OLLAMA_URL"] == "http://localhost:11434"
    assert cli.os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_cli_process_defaults_preserve_explicit_environment(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://custom-neo4j:7687")
    monkeypatch.setenv("QDRANT_URL", "http://custom-qdrant:6333")
    monkeypatch.setenv("OLLAMA_URL", "http://custom-ollama:11434")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    cli._apply_cli_process_defaults()

    assert cli.os.environ["NEO4J_URI"] == "bolt://custom-neo4j:7687"
    assert cli.os.environ["QDRANT_URL"] == "http://custom-qdrant:6333"
    assert cli.os.environ["OLLAMA_URL"] == "http://custom-ollama:11434"
    assert cli.os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_top_level_help_lists_operator_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "Landscape local memory CLI" in output
    assert "ingest" in output
    assert "ingest-dir" in output
    assert "query" in output
    assert "graph" in output
    assert "status" in output
    assert "seed" in output
    assert "wipe" in output


def test_top_level_help_does_not_import_runtime_heavy_modules(monkeypatch, capsys):
    blocked_prefixes = (
        "landscape.pipeline",
        "landscape.embeddings",
        "landscape.storage",
        "landscape.retrieval",
        "ollama",
        "neo4j",
        "qdrant_client",
    )
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(blocked_prefixes):
            raise AssertionError(f"help imported runtime-heavy module {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    assert "Landscape local memory CLI" in capsys.readouterr().out


def test_graph_help_lists_nested_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["graph", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "counts" in output
    assert "entity" in output
    assert "neighbors" in output


def test_seed_killer_demo_requires_confirm(capsys):
    exit_code = cli.main(["seed", "killer-demo"])

    assert exit_code == 2
    output = capsys.readouterr().out
    assert "without --confirm" in output


def test_seed_killer_demo_threads_debug_flag(monkeypatch, tmp_path, capsys):
    from landscape.cli import seed as seed_cli

    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    (demo_dir / "01_alpha.md").write_text("Alice leads Alpha.", encoding="utf-8")
    (demo_dir / "02_beta.md").write_text("Bob leads Beta.", encoding="utf-8")

    class FakeSeedIngest:
        def __init__(self):
            self.calls = []

        async def __call__(
            self,
            text,
            title,
            source_type="text",
            session_id=None,
            turn_id=None,
            debug=False,
        ):
            self.calls.append(
                {
                    "text": text,
                    "title": title,
                    "source_type": source_type,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "debug": debug,
                }
            )
            return FakeIngestResult()

    fake_ingest = FakeSeedIngest()
    encoder = FakeEncoder()
    qdrant_store = FakeQdrantStore()
    neo4j_store = FakeNeo4jStore()

    async def fake_wipe_state():
        return None

    monkeypatch.setattr(seed_cli, "wipe_state", fake_wipe_state)
    monkeypatch.setattr(
        seed_cli,
        "_get_runtime",
        lambda: (encoder, fake_ingest, neo4j_store, qdrant_store),
    )
    monkeypatch.setattr(
        seed_cli.resources,
        "files",
        lambda package: demo_dir,
    )

    exit_code = cli.main(["seed", "killer-demo", "--confirm", "--debug"])

    assert exit_code == 0
    assert fake_ingest.calls == [
        {
            "text": "Alice leads Alpha.",
            "title": "killer-demo:01_alpha",
            "source_type": "text",
            "session_id": "seed-killer-demo",
            "turn_id": "t1",
            "debug": True,
        },
        {
            "text": "Bob leads Beta.",
            "title": "killer-demo:02_beta",
            "source_type": "text",
            "session_id": "seed-killer-demo",
            "turn_id": "t2",
            "debug": True,
        },
    ]
    assert encoder.loaded is True
    assert qdrant_store.entity_collection_initialized is True
    assert qdrant_store.chunk_collection_initialized is True
    assert neo4j_store.closed is True
    assert qdrant_store.closed is True
    output = capsys.readouterr().out
    assert "Step 3/3  Ingesting 2 docs..." in output


def test_query_command_threads_debug_flag(monkeypatch, capsys):
    from landscape.cli import query as query_cli
    from landscape.retrieval.query import RetrievalResult, RetrievedEntity

    class FakeQueryRetrieve:
        def __init__(self):
            self.calls = []

        async def __call__(
            self,
            query_text,
            hops=2,
            limit=10,
            chunk_limit=3,
            weights=None,
            reinforce=True,
            session_id=None,
            since=None,
            debug=False,
            include_historical=False,
            log_context=None,
        ):
            self.calls.append(
                {
                    "query_text": query_text,
                    "hops": hops,
                    "limit": limit,
                    "reinforce": reinforce,
                    "debug": debug,
                    "include_historical": include_historical,
                }
            )
            return RetrievalResult(
                query=query_text,
                results=[
                    RetrievedEntity(
                        entity_id="atlas-id",
                        name="Project Atlas",
                        type="PROJECT",
                        distance=0,
                        vector_sim=0.9,
                        reinforcement=0.0,
                        edge_confidence=0.0,
                        score=1.0,
                    )
                ],
                touched_entity_ids=["atlas-id"],
                touched_edge_ids=[],
                chunks=[],
            )

    fake_retrieve = FakeQueryRetrieve()
    encoder = FakeEncoder()
    qdrant_store = FakeQdrantStore()
    neo4j_store = FakeNeo4jStore()

    monkeypatch.setattr(
        query_cli,
        "_get_runtime",
        lambda: (encoder, fake_retrieve, neo4j_store, qdrant_store),
    )

    exit_code = cli.main(["query", "Project Atlas", "--debug"])

    assert exit_code == 0
    assert fake_retrieve.calls == [
        {
            "query_text": "Project Atlas",
            "hops": 2,
            "limit": 10,
            "reinforce": True,
            "debug": True,
            "include_historical": False,
        }
    ]
    output = capsys.readouterr().out
    assert "1. Project Atlas [PROJECT]" in output


@pytest.fixture
def fake_runtime(monkeypatch):
    from landscape.cli import ingest as ingest_cli

    encoder = FakeEncoder()
    qdrant_store = FakeQdrantStore()
    neo4j_store = FakeNeo4jStore()
    pipeline = FakePipeline()
    monkeypatch.setattr(
        ingest_cli,
        "_get_runtime",
        lambda: (pipeline, encoder, neo4j_store, qdrant_store),
    )
    return {
        "encoder": encoder,
        "qdrant_store": qdrant_store,
        "neo4j_store": neo4j_store,
        "pipeline": pipeline,
    }


def test_ingest_unexpected_failure_closes_runtime(tmp_path, capsys, monkeypatch):
    from landscape.cli import ingest as ingest_cli

    path = tmp_path / "failure.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    encoder = FakeEncoder()
    qdrant_store = FakeQdrantStore()
    neo4j_store = FakeNeo4jStore()
    pipeline = FakePipeline(ingest_error=RuntimeError("ingest exploded"))
    monkeypatch.setattr(
        ingest_cli,
        "_get_runtime",
        lambda: (pipeline, encoder, neo4j_store, qdrant_store),
    )

    exit_code = cli.main(["ingest", str(path)])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "ingest exploded" in stderr or "Error:" in stderr


def test_ingest_command_threads_debug_flag(tmp_path, fake_runtime, capsys):
    path = tmp_path / "debug.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    exit_code = cli.main(["ingest", str(path), "--debug"])

    assert exit_code == 0
    assert fake_runtime["pipeline"].calls == [
        {
            "text": "Alice leads Project Atlas.",
            "title": "debug",
            "source_type": "text",
            "session_id": None,
            "turn_id": None,
            "debug": True,
        }
    ]
    assert "doc_id: doc-123" in capsys.readouterr().out
    assert fake_runtime["encoder"].loaded is True
    assert fake_runtime["qdrant_store"].entity_collection_initialized is True
    assert fake_runtime["qdrant_store"].chunk_collection_initialized is True
    assert fake_runtime["neo4j_store"].closed is True
    assert fake_runtime["qdrant_store"].closed is True


def test_ingest_cleanup_warning_does_not_override_success(tmp_path, capsys, monkeypatch):
    from landscape.cli import ingest as ingest_cli

    path = tmp_path / "success.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    encoder = FakeEncoder()
    qdrant_store = FakeQdrantStore()
    neo4j_store = FakeNeo4jStore(close_error=RuntimeError("neo4j close exploded"))
    pipeline = FakePipeline()
    monkeypatch.setattr(
        ingest_cli,
        "_get_runtime",
        lambda: (pipeline, encoder, neo4j_store, qdrant_store),
    )

    exit_code = cli.main(["ingest", str(path)])

    assert exit_code == 0
    assert qdrant_store.closed is True
    stderr = capsys.readouterr().err
    assert "Warning:" in stderr
    assert "neo4j close exploded" in stderr


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
            "debug": False,
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
            "debug": False,
        }
    ]


def test_missing_file_exits_before_initialization(tmp_path, fake_runtime):
    path = tmp_path / "missing.md"

    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest", str(path)])

    assert exc.value.code == 2
    assert_runtime_untouched(fake_runtime)


def test_incomplete_provenance_exits_before_initialization(tmp_path, fake_runtime):
    path = tmp_path / "doc.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest", str(path), "--session-id", "session-1"])

    assert exc.value.code == 2
    assert_runtime_untouched(fake_runtime)


@pytest.mark.parametrize(
    "session_id, turn_id",
    [
        ("", "turn-1"),
        ("  ", "\t"),
    ],
)
def test_blank_provenance_exits_before_initialization(
    tmp_path, fake_runtime, session_id, turn_id
):
    path = tmp_path / "doc.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "ingest",
                str(path),
                "--session-id",
                session_id,
                "--turn-id",
                turn_id,
            ]
        )

    assert exc.value.code == 2
    assert_runtime_untouched(fake_runtime)


def test_directory_path_exits_before_initialization(tmp_path, fake_runtime):
    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest", str(tmp_path)])

    assert exc.value.code == 2
    assert_runtime_untouched(fake_runtime)


def test_read_error_exits_before_initialization(tmp_path, monkeypatch, fake_runtime):
    path = tmp_path / "doc.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    def raise_oserror(self, *args, **kwargs):
        raise OSError("read failed")

    monkeypatch.setattr("pathlib.Path.read_text", raise_oserror)

    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest", str(path)])

    assert exc.value.code == 2
    assert_runtime_untouched(fake_runtime)


def test_unicode_decode_error_exits_before_initialization(tmp_path, monkeypatch, fake_runtime):
    path = tmp_path / "doc.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    def raise_unicode_decode_error(self, *args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"x", 0, 1, "bad byte")

    monkeypatch.setattr("pathlib.Path.read_text", raise_unicode_decode_error)

    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest", str(path)])

    assert exc.value.code == 2
    assert_runtime_untouched(fake_runtime)
