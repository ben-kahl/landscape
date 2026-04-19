# CLI Ingest Design

## Goal

Add a first-party `landscape ingest /path/to/document` command that ingests a local text or Markdown file through the existing Landscape ingestion pipeline.

## User Experience

The first supported command is:

```bash
uv run landscape ingest /path/to/document.md
```

The command reads exactly one local file, derives a default document title from the filename stem, and prints a compact ingestion summary. Operators can override the title and source type:

```bash
uv run landscape ingest /path/to/document.md --title "Architecture Notes" --source-type markdown
```

The command also accepts `--session-id` and `--turn-id` so callers can attach document ingestion to Landscape's existing conversation provenance model. Both values must be provided together; a single provenance flag is an input error.

## Architecture

Add a focused `src/landscape/cli.py` module using Python standard-library `argparse`. The CLI should not call the FastAPI server. It initializes the same local runtime dependencies used by FastAPI and MCP:

1. Load the embedding model with `landscape.embeddings.encoder.load_model()`.
2. Initialize Qdrant entity and chunk collections.
3. Call `landscape.pipeline.ingest(...)` directly.
4. Close Neo4j and Qdrant clients before exit.

Expose the command through a new `landscape = "landscape.cli:main"` entry in `pyproject.toml`.

## Error Handling

The CLI exits with status `2` for argument and input errors, matching `argparse` conventions. This includes missing paths, directory paths, unreadable files, and incomplete provenance flags.

Unexpected ingestion failures should print a short error to stderr and exit with status `1`. Successful ingestion exits with status `0`.

## Testing

Add unit tests for the CLI module that mock the expensive dependencies. The tests should not require Neo4j, Qdrant, Ollama, or the embedding model.

Coverage must include:

- successful local ingest with default title
- explicit `--title`, `--source-type`, `--session-id`, and `--turn-id`
- missing file path exits nonzero before initialization
- incomplete provenance flags exits nonzero before initialization

## Out Of Scope

Remote HTTP ingestion, directory ingestion, glob ingestion, JSON output, stdin input, and file-type detection are intentionally deferred. The command shape leaves room for these additions later without changing the direct local-ingest path.
