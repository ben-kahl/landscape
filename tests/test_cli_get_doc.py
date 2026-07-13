"""CLI get-doc subcommand tests (unit; storage monkeypatched)."""

from __future__ import annotations

import argparse

import pytest

pytestmark = pytest.mark.unit


def _noop_close():
    async def noop(*args):
        return None

    return noop


@pytest.mark.asyncio
async def test_handle_get_doc_prints_full_text(monkeypatch, capsys):
    from landscape.cli import query as cli_query
    from landscape.storage import neo4j_store

    async def fake_get(doc_id: str):
        return {
            "doc_id": doc_id,
            "title": "Ticket 119987",
            "source_type": "text",
            "ingested_at": "2026-07-12T00:00:00+00:00",
            "chunks": [
                {"chunk_id": "d:0:h1", "position": 0, "text": "first chunk"},
                {"chunk_id": "d:1:h1", "position": 1, "text": "second chunk"},
            ],
            "sessions": ["sess-1"],
        }

    monkeypatch.setattr(neo4j_store, "get_document_with_chunks", fake_get)
    monkeypatch.setattr(cli_query, "close_runtime", _noop_close())

    args = argparse.Namespace(doc_id="4:abc:12")
    rc = await cli_query.handle_get_doc(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert "Ticket 119987" in out
    assert "sess-1" in out
    assert "first chunk" in out
    assert "second chunk" in out


@pytest.mark.asyncio
async def test_handle_get_doc_missing_returns_nonzero(monkeypatch, capsys):
    from landscape.cli import query as cli_query
    from landscape.storage import neo4j_store

    async def fake_get(doc_id: str):
        return None

    monkeypatch.setattr(neo4j_store, "get_document_with_chunks", fake_get)
    monkeypatch.setattr(cli_query, "close_runtime", _noop_close())

    args = argparse.Namespace(doc_id="4:nope:0")
    rc = await cli_query.handle_get_doc(args)

    assert rc == 1
    assert "No document" in capsys.readouterr().out
