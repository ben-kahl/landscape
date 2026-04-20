# CLI Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `landscape ingest /path/to/document` for direct local ingestion through the existing pipeline.

**Architecture:** Add a focused standard-library `argparse` CLI in `src/landscape/cli.py`. The command validates and reads one local file, initializes the same encoder and Qdrant collections used by FastAPI/MCP, calls `pipeline.ingest(...)`, prints a compact result summary, and closes storage clients before exit.

**Tech Stack:** Python 3.12+, argparse, asyncio, pytest, monkeypatch, existing Landscape pipeline/storage modules.

---

## File Structure

- Create `src/landscape/cli.py`: command parsing, validation, local file reading, runtime initialization, ingestion call, result formatting, cleanup, and console-script `main`.
- Modify `pyproject.toml`: add `landscape = "landscape.cli:main"` while keeping `landscape-mcp`.
- Create `tests/test_cli.py`: mocked unit tests for CLI behavior. These tests must not contact Neo4j, Qdrant, Ollama, or load real embedding models.

## Task 1: CLI Unit Tests

**Files:**
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py` with:

```python
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
    def __init__(self):
        self.calls = []

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


def test_incomplete_provenance_exits_before_initialization(tmp_path, fake_runtime):
    path = tmp_path / "doc.md"
    path.write_text("Alice leads Project Atlas.", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest", str(path), "--session-id", "session-1"])

    assert exc.value.code == 2
    assert fake_runtime["pipeline"].calls == []
    assert fake_runtime["encoder"].loaded is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_cli.py -q
```

Expected: FAIL during collection or test execution because `landscape.cli` does not exist yet.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add tests/test_cli.py
git commit -m "test: specify cli ingest behavior"
```

## Task 2: CLI Implementation

**Files:**
- Create: `src/landscape/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add the console entry point**

Modify the existing `[project.scripts]` section in `pyproject.toml` to:

```toml
[project.scripts]
landscape = "landscape.cli:main"
landscape-mcp = "landscape.mcp_server:main"
```

- [ ] **Step 2: Add the minimal CLI implementation**

Create `src/landscape/cli.py` with:

```python
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from landscape import pipeline
from landscape.embeddings import encoder
from landscape.storage import neo4j_store, qdrant_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="landscape")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a local text document")
    ingest_parser.add_argument("path", type=Path, help="Path to a text or Markdown document")
    ingest_parser.add_argument("--title", help="Document title. Defaults to the file stem.")
    ingest_parser.add_argument("--source-type", default="text", help="Source type label")
    ingest_parser.add_argument("--session-id", help="Conversation session id for provenance")
    ingest_parser.add_argument("--turn-id", help="Conversation turn id for provenance")
    ingest_parser.set_defaults(func=_run_ingest_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(args.func(args, parser))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


async def _run_ingest_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    path: Path = args.path
    if not path.exists():
        parser.error(f"file does not exist: {path}")
    if not path.is_file():
        parser.error(f"path is not a file: {path}")
    if bool(args.session_id) != bool(args.turn_id):
        parser.error("--session-id and --turn-id must be provided together")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        parser.error(f"could not read file {path}: {exc}")
    except UnicodeDecodeError as exc:
        parser.error(f"could not decode file {path} as UTF-8: {exc}")

    title = args.title or path.stem

    try:
        encoder.load_model()
        await qdrant_store.init_collection()
        await qdrant_store.init_chunks_collection()
        result = await pipeline.ingest(
            text,
            title,
            args.source_type,
            session_id=args.session_id,
            turn_id=args.turn_id,
        )
        _print_ingest_result(result)
        return 0
    finally:
        await neo4j_store.close_driver()
        await qdrant_store.close_client()


def _print_ingest_result(result: pipeline.IngestResult) -> None:
    print(f"doc_id: {result.doc_id}")
    print(f"already_existed: {result.already_existed}")
    print(
        "entities: "
        f"created={result.entities_created} reinforced={result.entities_reinforced}"
    )
    print(
        "relations: "
        f"created={result.relations_created} "
        f"reinforced={result.relations_reinforced} "
        f"superseded={result.relations_superseded}"
    )
    print(f"chunks_created: {result.chunks_created}")
```

- [ ] **Step 3: Run CLI tests to verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_cli.py -q
```

Expected: PASS, 4 tests.

- [ ] **Step 4: Commit the implementation**

Run:

```bash
git add pyproject.toml src/landscape/cli.py tests/test_cli.py
git commit -m "feat: add local ingest cli"
```

## Task 3: Verification And Documentation

**Files:**
- Modify: `README.md`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add README usage**

Add this short usage block near the setup or API/MCP usage section in `README.md`:

```markdown
### CLI ingest

Ingest a local Markdown or text file directly through the Landscape pipeline:

```bash
uv run landscape ingest /path/to/document.md
uv run landscape ingest /path/to/document.md --title "Architecture Notes" --source-type markdown
```

The command expects the local Neo4j, Qdrant, and Ollama services to be reachable through the usual `LANDSCAPE_*`, `NEO4J_*`, `QDRANT_URL`, and `OLLAMA_URL` settings.
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run --extra dev pytest tests/test_cli.py tests/test_ingest.py::test_ingest_idempotent -q
```

Expected: PASS for all selected tests.

- [ ] **Step 3: Run lint**

Run:

```bash
uv run --extra dev ruff check src/landscape/cli.py tests/test_cli.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit README and any verification fixes**

Run:

```bash
git add README.md src/landscape/cli.py tests/test_cli.py pyproject.toml
git commit -m "docs: document cli ingest"
```

If there are no README or verification changes left after Task 2, skip this commit and report that no documentation commit was needed.

## Self-Review

- Spec coverage: The plan covers direct local ingest, metadata flags, provenance validation, mocked tests, console script registration, compact output, cleanup, and README usage.
- Placeholder scan: No placeholder steps are left; each task has exact files, commands, and expected results.
- Type consistency: The tests and implementation both use `cli.main(argv)`, `pipeline.ingest(text, title, source_type, session_id=..., turn_id=...)`, and the existing `IngestResult` field names.
