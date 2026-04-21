"""Regression tests for the LangChain demo agent script."""

from __future__ import annotations

from importlib import util
from pathlib import Path

import pytest
from langchain_core.documents import Document


@pytest.mark.asyncio
async def test_search_memory_separates_graph_facts_from_chunk_context(monkeypatch):
    """_search_memory() should distinguish authoritative graph facts from
    secondary chunk context when both kinds of docs are returned."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "demo_langchain_agent.py"
    spec = util.spec_from_file_location("demo_langchain_agent", module_path)
    assert spec and spec.loader
    demo_langchain_agent = util.module_from_spec(spec)
    spec.loader.exec_module(demo_langchain_agent)

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, query: str):
            return [
                Document(
                    page_content="Alice now works for Atlas Corp.",
                    metadata={"kind": "entity", "neo4j_id": "entity-1"},
                ),
                Document(
                    page_content="Alice moved to Austin last month.",
                    metadata={"kind": "chunk", "chunk_neo4j_id": "chunk-1"},
                ),
            ]

    monkeypatch.setattr(
        "landscape.retrieval.langchain_retriever.LandscapeRetriever",
        FakeRetriever,
    )

    output = await demo_langchain_agent._search_memory("where does Alice work?")

    assert "Current-state graph evidence (authoritative):" in output
    assert "Historical chunk context (secondary):" in output
    assert "  - [GRAPH] Alice now works for Atlas Corp." in output
    assert "  - [CHUNK] Alice moved to Austin last month." in output
