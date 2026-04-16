"""Integration test for the LangChain BaseRetriever wrapper.

Confirms that LandscapeRetriever plugs into LangChain's ainvoke protocol,
returns Document objects, and preserves the numeric signals (score,
distance, reinforcement, path_edge_types) in metadata so downstream
agents can re-rank or filter.

Uses a tiny 1-doc synthetic corpus instead of the killer-demo fixture so
it runs in the default test pass without the @pytest.mark.retrieval
opt-in."""
import pytest

from landscape.retrieval.langchain_retriever import LandscapeRetriever

RETRIEVER_DOC = (
    "Helios Robotics is a robotics company in Boston. "
    "Nadia Khan leads the Helios platform team. "
    "The Helios platform team uses PostgreSQL for telemetry storage."
)


@pytest.mark.asyncio
async def test_langchain_retriever_returns_documents(http_client):
    """ainvoke() should return a non-empty list of Documents with
    Landscape-specific metadata populated."""
    r = await http_client.post(
        "/ingest", json={"text": RETRIEVER_DOC, "title": "langchain-retriever-test"}
    )
    assert r.status_code == 200

    retriever = LandscapeRetriever(hops=2, limit=10, reinforce=False)
    docs = await retriever.ainvoke("Who leads the Helios platform team?")

    assert docs, "retriever should return at least one Document"
    for d in docs:
        # page_content is human-readable and always includes the entity name
        # and parenthesized type
        assert "(" in d.page_content and ")" in d.page_content, (
            f"page_content missing type annotation: {d.page_content}"
        )
        # metadata carries every numeric signal the scoring layer produced
        for key in (
            "neo4j_id",
            "name",
            "type",
            "distance",
            "score",
            "vector_sim",
            "reinforcement",
            "edge_confidence",
            "path_edge_types",
        ):
            assert key in d.metadata, (
                f"metadata missing {key!r}: {d.metadata}"
            )
        assert isinstance(d.metadata["score"], float)
        assert isinstance(d.metadata["distance"], int)


@pytest.mark.asyncio
async def test_langchain_retriever_sync_path_raises(http_client):
    """The sync _get_relevant_documents path must refuse to run — Landscape
    is async-only and a sync wrapper would either block the loop or spawn
    a new one. Calling invoke() from an already-running loop triggers
    _get_relevant_documents, which we assert blows up loudly."""
    retriever = LandscapeRetriever(hops=1, limit=5)
    with pytest.raises(NotImplementedError, match="async-only"):
        retriever.invoke("anything")
