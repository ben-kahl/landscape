"""Tests for get_conversation_detail storage primitive."""
import asyncio

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_get_conversation_detail_orders_turns_by_timestamp(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    await neo4j_store.merge_turn("conv-ch1", "t1", turn_number=1, role="user", summary="first")
    await asyncio.sleep(0.01)
    await neo4j_store.merge_turn(
        "conv-ch1",
        "t2",
        turn_number=2,
        role="assistant",
        summary="second",
    )
    await asyncio.sleep(0.01)
    await neo4j_store.merge_turn("conv-ch1", "t3", turn_number=3, role="user", summary="third")

    detail = await neo4j_store.get_conversation_detail("conv-ch1")

    assert detail["conversation"] is not None
    turns = detail["turns"]
    assert len(turns) == 3

    turn_ids = [t["turn_id"] for t in turns]
    assert turn_ids == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_get_conversation_detail_includes_entities_mentioned(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    turn_eid, _ = await neo4j_store.merge_turn("conv-ch2", "t1", turn_number=1)
    eid = await neo4j_store.merge_entity(
        "DetailEntity", "ORGANIZATION", "doc-ch2", 0.9, doc_element_id=None, model="test"
    )
    await neo4j_store.link_entity_to_turn(eid, turn_eid)

    detail = await neo4j_store.get_conversation_detail("conv-ch2")

    assert len(detail["turns"]) == 1
    entities = detail["turns"][0]["entities_mentioned"]
    assert len(entities) == 1
    assert entities[0]["name"] == "DetailEntity"


@pytest.mark.asyncio
async def test_get_conversation_detail_unknown_session(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    detail = await neo4j_store.get_conversation_detail("conv-ch-nonexistent-999")

    assert detail["conversation"] is None
    assert detail["turns"] == []


@pytest.mark.asyncio
async def test_get_conversation_detail_respects_turn_limit(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    for i in range(1, 6):
        await neo4j_store.merge_turn(
            "conv-ch3", f"t{i}", turn_number=i, role="user", summary=f"turn {i}"
        )
        await asyncio.sleep(0.01)

    detail = await neo4j_store.get_conversation_detail("conv-ch3", turn_limit=2)

    turns = detail["turns"]
    assert len(turns) == 2
    # Should be the 2 earliest (ASC order)
    assert turns[0]["turn_id"] == "t1"
    assert turns[1]["turn_id"] == "t2"
