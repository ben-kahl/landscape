"""Unit tests for the pipeline.ingest_file() wrapper.

The wrapper is a thin pass-through: convert path → markdown, hand off to
ingest(). Real ingestion is covered by the existing pipeline tests and the
killer-demo integration. Here we verify the wrapper's contract: it forwards
the converted text, picks the right source_type (with override semantics),
and picks the right title (with override semantics).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from landscape import pipeline

pytestmark = pytest.mark.unit


class _IngestSpy:
    """Stand-in for pipeline.ingest that records the call args."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(
        self,
        text: str,
        title: str,
        source_type: str = "text",
        session_id: str | None = None,
        turn_id: str | None = None,
        debug: bool = False,
        log_context=None,
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
        return "sentinel-result"


@pytest.fixture
def ingest_spy(monkeypatch: pytest.MonkeyPatch) -> _IngestSpy:
    spy = _IngestSpy()
    monkeypatch.setattr(pipeline, "ingest", spy)
    return spy


async def test_ingest_file_forwards_text_for_markdown(
    tmp_path: Path, ingest_spy: _IngestSpy
) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Heading\n\nbody\n", encoding="utf-8")

    result = await pipeline.ingest_file(p)

    assert result == "sentinel-result"
    assert len(ingest_spy.calls) == 1
    call = ingest_spy.calls[0]
    assert call["text"] == "# Heading\n\nbody\n"
    assert call["source_type"] == "markdown"
    assert call["title"] == "doc"  # falls back to stem


async def test_ingest_file_passes_title_override(
    tmp_path: Path, ingest_spy: _IngestSpy
) -> None:
    p = tmp_path / "doc.md"
    p.write_text("body", encoding="utf-8")

    await pipeline.ingest_file(p, title="My Custom Title")

    assert ingest_spy.calls[0]["title"] == "My Custom Title"


async def test_ingest_file_source_type_override_wins(
    tmp_path: Path, ingest_spy: _IngestSpy
) -> None:
    # File is .md (would infer "markdown"); caller forces "release_notes".
    p = tmp_path / "notes.md"
    p.write_text("body", encoding="utf-8")

    await pipeline.ingest_file(p, source_type="release_notes")

    assert ingest_spy.calls[0]["source_type"] == "release_notes"


async def test_ingest_file_forwards_session_and_turn(
    tmp_path: Path, ingest_spy: _IngestSpy
) -> None:
    p = tmp_path / "doc.txt"
    p.write_text("text body", encoding="utf-8")

    await pipeline.ingest_file(
        p, session_id="sess-42", turn_id="turn-7", debug=True
    )

    call = ingest_spy.calls[0]
    assert call["session_id"] == "sess-42"
    assert call["turn_id"] == "turn-7"
    assert call["debug"] is True


async def test_ingest_file_uses_html_source_type(
    tmp_path: Path, ingest_spy: _IngestSpy
) -> None:
    p = tmp_path / "page.html"
    p.write_text("<h1>Hello</h1><p>world</p>", encoding="utf-8")

    await pipeline.ingest_file(p)

    call = ingest_spy.calls[0]
    assert call["source_type"] == "html"
    # Markitdown converted the HTML to something markdown-ish:
    assert "Hello" in call["text"]
    assert "world" in call["text"]


async def test_ingest_file_unknown_extension_falls_back_to_text(
    tmp_path: Path, ingest_spy: _IngestSpy
) -> None:
    p = tmp_path / "server.log"
    p.write_text("startup ok\n", encoding="utf-8")

    await pipeline.ingest_file(p)

    call = ingest_spy.calls[0]
    assert call["source_type"] == "text"
    assert "startup ok" in call["text"]
    assert call["title"] == "server"
