import json
import pathlib

import pytest

from landscape.conversation_buffer import SessionBuffer
from landscape.conversation_ingestion import ConversationTurn, ingest_conversation_window
from landscape.extraction.salience import select_salient
from landscape.retrieval.query import retrieve

pytestmark = [pytest.mark.integration, pytest.mark.external]

FIXTURE = (
    pathlib.Path(__file__).parent
    / "fixtures"
    / "conversation_capture"
    / "session_alpha.json"
)


@pytest.mark.asyncio
async def test_multihop_answer_from_captured_conversation(neo4j_driver, qdrant_client):
    from landscape.embeddings import encoder
    from landscape.storage import qdrant_store

    # The autouse isolation fixture drops Qdrant collections and the app
    # lifespan never runs in-process, so recreate the collections and load the
    # encoder before driving the capture path directly (mirrors the in-process
    # external tests in test_retrieval_basic.py).
    existing = await qdrant_store.get_client().get_collections()
    names = {c.name for c in existing.collections}
    if qdrant_store.COLLECTION not in names:
        await qdrant_store.init_collection()
    if qdrant_store.CHUNKS_COLLECTION not in names:
        await qdrant_store.init_chunks_collection()
    encoder.load_model()

    data = json.loads(FIXTURE.read_text())
    turns = [
        ConversationTurn(data["session_id"], turn["turn_id"], turn["role"], turn["text"])
        for turn in data["turns"]
    ]

    buf = SessionBuffer(session_id=data["session_id"])
    for turn in turns:
        buf.append(turn)
    window = buf.take_window(overlap_turns=0)
    salient = select_salient(window)
    await ingest_conversation_window(data["session_id"], salient)

    result = await retrieve(
        "Who approved the database that the Atlas project uses?",
        hops=3,
        limit=10,
    )
    names = {item.name for item in result.results}
    assert any("Priya" in name for name in names), (
        f"expected Priya via multi-hop, got {names}"
    )
