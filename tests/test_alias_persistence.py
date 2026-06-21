"""LLM-extracted aliases must be persisted as :Alias/SAME_AS in Neo4j.

The extraction prompt asks the model to consolidate alternate surface forms into
ExtractedEntity.aliases. This is the *only* alias signal for a single ingest
(the resolver path only fires on a second, differently-spelled mention). If the
pipeline drops .aliases, a big text yields zero aliases in Neo4j — which is the
bug this guards.
"""
import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def _clean(neo4j_driver):
    names = ["Maya Chen Alias-Test"]
    aliases = ["Maya Alias-Test"]
    async with neo4j_driver.session() as session:
        for n in names:
            await session.run("MATCH (e:Entity {name: $n}) DETACH DELETE e", n=n)
        for a in aliases:
            await session.run("MATCH (a:Alias {name: $a}) DETACH DELETE a", a=a)
    yield names[0], aliases[0]


@pytest.mark.asyncio
async def test_llm_extracted_alias_persisted_as_same_as(http_client, neo4j_driver, _clean):
    from landscape import pipeline
    from landscape.extraction.chunker import Chunk
    from landscape.extraction.schema import ExtractedEntity, Extraction

    canonical_name, alias_name = _clean
    title = "alias-persistence-integration"

    async def fake_resolve_entity(name, entity_type, vector, source_doc):
        # Always-new so the resolver's own alias path can't fire — this isolates
        # the LLM-extracted-alias persistence path, which is the bug under test.
        return None, True, None

    async def noop_upsert(**kwargs):
        return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(pipeline.resolver, "resolve_entity", fake_resolve_entity)
    monkeypatch.setattr(
        pipeline.encoder, "embed_documents", lambda texts: [[0.1, 0.2] for _ in texts]
    )
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", noop_upsert)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_entity", noop_upsert)
    monkeypatch.setattr(pipeline, "chunk_text", lambda text: [Chunk(index=0, text=text)])
    monkeypatch.setattr(
        pipeline.llm,
        "extract",
        lambda text: Extraction(
            entities=[
                ExtractedEntity(
                    name=canonical_name,
                    type="PERSON",
                    confidence=0.95,
                    aliases=[alias_name],
                ),
            ],
            relations=[],
        ),
    )

    try:
        resp = await http_client.post(
            "/ingest", json={"text": f"{canonical_name} did a thing.", "title": title}
        )
        assert resp.status_code == 200, resp.text
    finally:
        monkeypatch.undo()

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Alias)-[:SAME_AS]->(e:Entity {name: $canonical})
            RETURN collect(a.name) AS aliases
            """,
            canonical=canonical_name,
        )
        record = await result.single()

    aliases = record["aliases"] if record else []
    assert alias_name in aliases, (
        f"LLM-extracted alias {alias_name!r} was not persisted as a :Alias/SAME_AS "
        f"on {canonical_name!r}; got {aliases}"
    )
