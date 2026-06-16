from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "sample_transcript.jsonl"


@pytest.mark.asyncio
async def test_ingest_transcript_writes_turns_to_graph(http_client, neo4j_driver):
    from landscape.conversation_ingestion import ingest_conversation_window
    from landscape.extraction.salience import select_salient
    from landscape.ingestion.transcript import read_transcript
    from landscape.storage import qdrant_store

    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()

    turns = read_transcript(FIXTURE, session_id="e2e-sess")
    salient = select_salient(turns)
    assert salient, "fixture should produce at least one salient turn"
    await ingest_conversation_window("e2e-sess", salient, debug=False)

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (c:Conversation {id: 'e2e-sess'})-[:HAS_TURN]->(t:Turn) "
            "RETURN count(t) AS cnt"
        )
        record = await result.single()
    assert record["cnt"] >= 1
