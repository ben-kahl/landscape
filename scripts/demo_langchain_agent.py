"""LangChain agent demo for Landscape.

Proves the framework-integration story: any LangChain agent can use Landscape
as a memory backend -- LandscapeRetriever for read, writeback.add_relation for
write.

Same scenario as demo_mcp_session.py (Alice -> Atlas -> Beacon supersession)
but driven by an LLM choosing tool calls rather than a hardcoded sequence.

Run:
    uv run python scripts/demo_langchain_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Make the src tree importable when running as a plain script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "landscape-dev")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

os.environ.setdefault("NEO4J_URI", NEO4J_URI)
os.environ.setdefault("NEO4J_USER", NEO4J_USER)
os.environ.setdefault("NEO4J_PASSWORD", NEO4J_PASSWORD)
os.environ.setdefault("QDRANT_URL", QDRANT_URL)
os.environ.setdefault("OLLAMA_URL", os.environ.get("OLLAMA_URL", "http://localhost:11434"))

# Avoid CUDA contention with the Docker stack.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------------------------------------------------------------------------
# Step 0: reset helper
# ---------------------------------------------------------------------------


async def reset_state() -> None:
    """Wipe Neo4j + drop Qdrant collections so each run is reproducible."""
    from neo4j import AsyncGraphDatabase
    from qdrant_client import AsyncQdrantClient

    from landscape.storage import qdrant_store

    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    finally:
        await driver.close()

    qclient = AsyncQdrantClient(url=QDRANT_URL)
    try:
        existing = await qclient.get_collections()
        names = {c.name for c in existing.collections}
        for coll in (qdrant_store.COLLECTION, qdrant_store.CHUNKS_COLLECTION):
            if coll in names:
                await qclient.delete_collection(coll)
    finally:
        await qclient.close()


# ---------------------------------------------------------------------------
# Tool implementations (async)
# ---------------------------------------------------------------------------


async def _search_memory(query: str) -> str:
    """Search Landscape's hybrid memory and return a text summary."""
    from landscape.retrieval.langchain_retriever import LandscapeRetriever

    retriever = LandscapeRetriever(hops=2, limit=5)
    docs = await retriever.ainvoke(query)
    if not docs:
        return "No results found in memory."
    lines = ["Found the following in memory:"]
    for doc in docs:
        lines.append(f"  - {doc.page_content}")
    return "\n".join(lines)


async def _record_relation(subject: str, object: str, rel_type: str) -> str:
    """Record a relationship between two entities in Landscape memory."""
    from landscape import writeback

    result = await writeback.add_relation(
        subject=subject,
        object_=object,
        rel_type=rel_type,
        source="agent:langchain-demo:auto",
        session_id="langchain-demo",
        turn_id="auto",
    )
    return (
        f"Relation recorded: {subject} --[{rel_type}]--> {object} "
        f"(outcome={result.outcome})"
    )


# ---------------------------------------------------------------------------
# Build agent
# ---------------------------------------------------------------------------


def build_agent():
    """Construct the LangChain agent graph with Landscape tools."""
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool
    from langchain_ollama import ChatOllama
    from pydantic import BaseModel, Field

    from landscape.config import settings

    model_name = settings.llm_model or "llama3.1:8b"
    ollama_url = settings.ollama_url

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0,
    )

    # Pydantic input schemas for structured tool calls.
    class SearchInput(BaseModel):
        query: str = Field(description="The search query to look up in memory.")

    class RecordRelationInput(BaseModel):
        subject: str = Field(description="The entity that is the subject of the relation.")
        object: str = Field(  # noqa: A002 -- tool param name must match description
            description="The entity that is the object of the relation."
        )
        rel_type: str = Field(
            description=(
                "The relation type. Use canonical forms like WORKS_FOR, LEADS, "
                "MEMBER_OF, REPORTS_TO, APPROVED, USES, BELONGS_TO, LOCATED_IN, "
                "CREATED, or RELATED_TO."
            )
        )

    # Sync wrappers -- the agent graph calls tools synchronously in its
    # executor thread; async is handled via coroutine= kwarg for ainvoke.
    def _search_sync(query: str) -> str:
        return asyncio.get_event_loop().run_until_complete(_search_memory(query))

    def _record_sync(subject: str, object: str, rel_type: str) -> str:  # noqa: A002
        return asyncio.get_event_loop().run_until_complete(
            _record_relation(subject, object, rel_type)
        )

    search_tool = StructuredTool.from_function(
        func=_search_sync,
        coroutine=_search_memory,
        name="search_memory",
        description=(
            "Search Landscape's hybrid knowledge-graph memory. "
            "Use this to look up facts previously learned."
        ),
        args_schema=SearchInput,
    )

    record_tool = StructuredTool.from_function(
        func=_record_sync,
        coroutine=_record_relation,
        name="record_relation",
        description=(
            "Record a new relationship between two entities in memory. "
            "Use this when the user shares new information to update what is known."
        ),
        args_schema=RecordRelationInput,
    )

    system_prompt = (
        "You are a helpful assistant with persistent memory. "
        "You have two memory tools: search_memory (to recall facts) and "
        "record_relation (to save new facts the user shares). "
        "When a user tells you something new, call record_relation to save it. "
        "When a user asks about something, call search_memory first to look it up. "
        "Be concise and direct."
    )

    agent = create_agent(llm, [search_tool, record_tool], system_prompt=system_prompt)
    return agent


# ---------------------------------------------------------------------------
# Helpers to extract info from agent output messages
# ---------------------------------------------------------------------------


def _extract_final_answer(messages: list) -> str:
    """Get the last AI message text that isn't a tool call."""
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return str(msg.content)
    return ""


def _extract_tool_calls(messages: list) -> list[dict]:
    """Collect all tool call results from ToolMessages."""
    from langchain_core.messages import AIMessage, ToolMessage

    calls = []
    # Pair AIMessage tool_calls with subsequent ToolMessages.
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append({"name": tc["name"], "args": tc.get("args", {}), "result": None})
        elif isinstance(msg, ToolMessage):
            # Associate with the most recent call that has no result yet.
            for c in reversed(calls):
                if c["result"] is None:
                    c["result"] = msg.content
                    break
    return calls


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


async def run_demo() -> None:  # noqa: C901
    # ------------------------------------------------------------------ #
    # Step 0: Reset                                                        #
    # ------------------------------------------------------------------ #
    print("==== Step 0: Reset ====")
    try:
        await reset_state()
        print("Reset complete -- Neo4j wiped, Qdrant collections dropped.")
    except Exception as exc:
        print(f"WARNING: reset failed: {exc}")
    print()

    # ------------------------------------------------------------------ #
    # Step 1: Initialize services + seed the graph                        #
    # ------------------------------------------------------------------ #
    print("==== Step 1: Seed graph ====")
    try:
        from landscape.embeddings import encoder
        from landscape.storage import qdrant_store

        encoder.load_model()
        await qdrant_store.init_collection()
        await qdrant_store.init_chunks_collection()

        from landscape.pipeline import ingest

        seed_text = "Alice works for Atlas Corp as a senior engineer."
        result = await ingest(seed_text, title="Personnel record")
        print(f'Ingested seed: "{seed_text}"')
        print(
            f"  entities_created={result.entities_created}, "
            f"relations_created={result.relations_created}, "
            f"relations_superseded={result.relations_superseded}"
        )
    except Exception as exc:
        print(f"EXCEPTION during seed ingest: {exc}")
        return
    print()

    # ------------------------------------------------------------------ #
    # Step 2: Build the LangChain agent                                   #
    # ------------------------------------------------------------------ #
    print("==== Step 2: Building LangChain agent ====")
    try:
        agent = build_agent()
        print("Agent built successfully.")
    except Exception as exc:
        print(f"EXCEPTION building agent: {exc}")
        return
    print()

    # ------------------------------------------------------------------ #
    # Turn tracking state                                                  #
    # ------------------------------------------------------------------ #
    turn2_recorded = False
    turn2_outcome: str | None = None
    turn3_answer: str | None = None

    # ------------------------------------------------------------------ #
    # Turn 1: read -- who does Alice work for?                            #
    # ------------------------------------------------------------------ #
    print("==== Turn 1: Read (who does Alice work for?) ====")
    user1 = "Who does Alice work for?"
    print(f"USER: {user1}")
    try:
        response = await agent.ainvoke({"messages": [("human", user1)]})
        messages = response.get("messages", [])
        answer = _extract_final_answer(messages)
        tool_calls = _extract_tool_calls(messages)
        for tc in tool_calls:
            print(f"  [TOOL] {tc['name']}({tc['args']}) -> {str(tc['result'])[:150]}")
        print(f"AGENT: {answer}")
    except Exception as exc:
        print(f"EXCEPTION in turn 1: {exc}")
    print()

    # ------------------------------------------------------------------ #
    # Turn 2: write -- Alice moved to Beacon                              #
    # ------------------------------------------------------------------ #
    print("==== Turn 2: Write (Alice moved to Beacon) ====")
    user2 = "Update: Alice has moved to Beacon."
    print(f"USER: {user2}")
    try:
        response = await agent.ainvoke({"messages": [("human", user2)]})
        messages = response.get("messages", [])
        answer = _extract_final_answer(messages)
        tool_calls = _extract_tool_calls(messages)
        for tc in tool_calls:
            obs = str(tc["result"])
            print(f"  [TOOL] {tc['name']}({tc['args']}) -> {obs[:200]}")
            if tc["name"] == "record_relation":
                turn2_recorded = True
                if "superseded" in obs.lower():
                    turn2_outcome = "superseded"
                elif "created" in obs.lower():
                    turn2_outcome = "created"
                else:
                    turn2_outcome = obs
        print(f"AGENT: {answer}")
        if not turn2_recorded:
            print("[WARN] Agent did not call record_relation on turn 2.")
    except Exception as exc:
        print(f"EXCEPTION in turn 2: {exc}")
    print()

    # ------------------------------------------------------------------ #
    # Turn 3: verify -- who does Alice work for now?                      #
    # ------------------------------------------------------------------ #
    print("==== Turn 3: Verify (who does Alice work for now?) ====")
    user3 = "Who does Alice work for now?"
    print(f"USER: {user3}")
    try:
        response = await agent.ainvoke({"messages": [("human", user3)]})
        messages = response.get("messages", [])
        answer = _extract_final_answer(messages)
        tool_calls = _extract_tool_calls(messages)
        for tc in tool_calls:
            print(f"  [TOOL] {tc['name']}({tc['args']}) -> {str(tc['result'])[:200]}")
        print(f"AGENT: {answer}")
        turn3_answer = answer
    except Exception as exc:
        print(f"EXCEPTION in turn 3: {exc}")
    print()

    # ------------------------------------------------------------------ #
    # Final status                                                         #
    # ------------------------------------------------------------------ #
    print("==== Final Status ====")

    if turn2_recorded:
        if turn2_outcome == "superseded":
            print("Turn 2 -- PASS: record_relation called, outcome=superseded.")
        elif turn2_outcome == "created":
            print(
                "Turn 2 -- NOTE: record_relation called, outcome=created "
                "(no prior WORKS_FOR edge found; supersession requires a prior edge)."
            )
        else:
            print(f"Turn 2 -- NOTE: record_relation called, outcome={turn2_outcome}.")
    else:
        print("Turn 2 -- FAIL: agent did not call record_relation.")

    if turn3_answer and "beacon" in turn3_answer.lower():
        print("Turn 3 -- PASS: agent answer mentions Beacon.")
    elif turn3_answer and "atlas" in turn3_answer.lower():
        print(
            "Turn 3 -- FAIL: agent answer mentions Atlas "
            "(supersession may not have fired or retrieval is not temporally filtering)."
        )
    else:
        print(f"Turn 3 -- UNKNOWN: agent answer: {turn3_answer!r}")

    print()
    print("==== Demo complete ====")


if __name__ == "__main__":
    asyncio.run(run_demo())
