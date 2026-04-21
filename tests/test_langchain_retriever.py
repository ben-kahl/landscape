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

@pytest.mark.unit
def test_entity_document_renders_quantity_qualifiers():
    from landscape.retrieval.langchain_retriever import _entity_to_document
    from landscape.retrieval.query import RetrievedEntity

    doc = _entity_to_document(
        RetrievedEntity(
            neo4j_id="netflix-id",
            name="Netflix",
            type="TECHNOLOGY",
            distance=1,
            vector_sim=0.9,
            reinforcement=0.0,
            edge_confidence=0.9,
            score=1.0,
            path_edge_ids=["rel-1"],
            path_edge_types=["DISCUSSED"],
            path_edge_subtypes=["watched"],
            path_edge_quantities=[
                {
                    "quantity_value": 10,
                    "quantity_unit": "hour",
                    "quantity_kind": "duration",
                    "time_scope": "last_month",
                }
            ],
        )
    )

    assert "DISCUSSED[watched]" in doc.page_content
    assert "duration=10 hour" in doc.page_content
    assert "scope=last_month" in doc.page_content
    assert doc.metadata["path_edge_quantities"][0]["quantity_value"] == 10

@pytest.mark.asyncio
@pytest.mark.unit
async def test_langchain_retriever_returns_chunk_documents(monkeypatch):
    from landscape.retrieval import langchain_retriever
    from landscape.retrieval.query import RetrievalResult, RetrievedChunk

    async def fake_retrieve(**kwargs):
        return RetrievalResult(
            query=kwargs["query_text"],
            results=[],
            touched_entity_ids=[],
            touched_edge_ids=[],
            chunks=[
                RetrievedChunk(
                    chunk_neo4j_id="chunk-1",
                    text="I spent 10 hours last month watching documentaries on Netflix.",
                    doc_id="doc-1",
                    source_doc="longmemeval:test",
                    position=0,
                    score=0.88,
                )
            ],
        )

    monkeypatch.setattr(langchain_retriever, "retrieve", fake_retrieve)

    retriever = LandscapeRetriever(chunk_limit=1)
    docs = await retriever.ainvoke("How many hours on Netflix?")

    assert len(docs) == 1
    assert docs[0].page_content == (
        "I spent 10 hours last month watching documentaries on Netflix."
    )
    assert docs[0].metadata["kind"] == "chunk"
    assert docs[0].metadata["chunk_neo4j_id"] == "chunk-1"


@pytest.mark.asyncio
@pytest.mark.integration
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
    entity_docs = [d for d in docs if d.metadata.get("kind") == "entity"]
    chunk_docs = [d for d in docs if d.metadata.get("kind") == "chunk"]
    assert entity_docs, "retriever should return at least one entity Document"
    assert chunk_docs, "retriever should include raw chunk Documents"
    for d in entity_docs:
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
    for d in chunk_docs:
        assert d.page_content
        for key in (
            "chunk_neo4j_id",
            "doc_id",
            "source_doc",
            "position",
            "score",
        ):
            assert key in d.metadata, f"chunk metadata missing {key!r}: {d.metadata}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_langchain_retriever_sync_path_raises(http_client):
    """The sync _get_relevant_documents path must refuse to run — Landscape
    is async-only and a sync wrapper would either block the loop or spawn
    a new one. Calling invoke() from an already-running loop triggers
    _get_relevant_documents, which we assert blows up loudly."""
    retriever = LandscapeRetriever(hops=1, limit=5)
    with pytest.raises(NotImplementedError, match="async-only"):
        retriever.invoke("anything")
