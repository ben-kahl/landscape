import pytest

from landscape.extraction.salience import SalientItem

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_ingest_window_joins_doc_and_links_all_turns(monkeypatch):
    import landscape.conversation_ingestion as ci

    calls = {"ingest": [], "merge_turn": [], "links": []}

    async def fake_ingest(text, title, *, session_id, turn_id, debug=False, log_context=None):
        calls["ingest"].append({"text": text, "session_id": session_id, "turn_id": turn_id})
        from landscape.pipeline import IngestResult

        return IngestResult(
            doc_id="doc-1",
            already_existed=False,
            entities_created=2,
            entities_reinforced=0,
            relations_created=1,
            relations_reinforced=0,
            relations_superseded=0,
            chunks_created=1,
        )

    async def fake_merge_turn(session_id, turn_id):
        calls["merge_turn"].append((session_id, turn_id))
        return (f"elem::{turn_id}", True)

    async def fake_link(doc_id, turn_element_id):
        calls["links"].append((doc_id, turn_element_id))

    monkeypatch.setattr(ci, "ingest", fake_ingest)
    monkeypatch.setattr(ci.neo4j_store, "merge_turn", fake_merge_turn)
    monkeypatch.setattr(ci.neo4j_store, "link_document_to_turn", fake_link)

    salient = [
        SalientItem(turn_id="t1", text="I work at Acme.", category="identity"),
        SalientItem(turn_id="t3", text="We chose Postgres.", category="decision"),
    ]
    result = await ci.ingest_conversation_window("s1", salient)

    assert len(calls["ingest"]) == 1
    assert calls["ingest"][0]["turn_id"] == "t1"
    assert "I work at Acme." in calls["ingest"][0]["text"]
    assert "We chose Postgres." in calls["ingest"][0]["text"]
    assert ("s1", "t3") in calls["merge_turn"]
    assert ("doc-1", "elem::t3") in calls["links"]
    assert result.entities_created == 2


@pytest.mark.asyncio
async def test_ingest_window_noop_when_no_salient_items():
    import landscape.conversation_ingestion as ci

    result = await ci.ingest_conversation_window("s1", [])
    assert result is None
