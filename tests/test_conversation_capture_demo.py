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
