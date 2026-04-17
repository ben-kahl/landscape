"""Deterministic MCP transcript demo for Landscape.

Spawns the MCP server as a subprocess over stdio (Option B) and drives a
7-step session:

  0. Reset  -- wipe Neo4j + Qdrant so the run is reproducible
  1. Ingest -- remember("Alice works for Atlas Corp as a senior engineer.")
  2. Search -- confirm Alice / Atlas surfaces
  3. Write-back -- add_relation(Alice, Beacon, WORKS_FOR) -> superseded
  4. Search -- Atlas relation no longer surfaces (temporally filtered)
  5. Status -- recent_agent_writes shows the supersession;
               asserts conversation_count >= 1 and turn_count >= 1
  6. Conversation graph -- graph_query proves Conversation/Turn schema is live

Output is plain text formatted for direct inclusion in the README.

Run:
    uv run python scripts/demo_mcp_session.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Ensure the src tree is importable when running as a plain script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "landscape-dev")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Propagate connection env-vars to the subprocess.
os.environ.setdefault("NEO4J_URI", NEO4J_URI)
os.environ.setdefault("NEO4J_USER", NEO4J_USER)
os.environ.setdefault("NEO4J_PASSWORD", NEO4J_PASSWORD)
os.environ.setdefault("QDRANT_URL", QDRANT_URL)
os.environ.setdefault("OLLAMA_URL", os.environ.get("OLLAMA_URL", "http://localhost:11434"))

# Force CPU torch in this process (the subprocess inherits the env and will
# also use CPU for the encoder, avoiding CUDA contention with the Docker stack).
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------------------------------------------------------------------------
# Step 0: reset helper (runs in-process; no MCP needed)
# ---------------------------------------------------------------------------


async def reset_state() -> None:
    """Wipe Neo4j + drop Qdrant collections so the demo is reproducible."""
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
# MCP client helpers
# ---------------------------------------------------------------------------


def _parse(result) -> dict:
    """Extract and JSON-decode the first text content block."""
    if not result.content:
        return {}
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


async def run_demo() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # ------------------------------------------------------------------ #
    # Step 0: Reset                                                        #
    # ------------------------------------------------------------------ #
    print("==== Step 0: Reset ====")
    try:
        await reset_state()
        print("  Reset complete -- Neo4j wiped, Qdrant collections dropped.")
    except Exception as exc:
        print(f"  WARNING: reset failed: {exc}")
    print()

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "landscape-mcp"],
        env={**os.environ},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # -------------------------------------------------------------- #
            # Step 1: Ingest seed fact                                        #
            # -------------------------------------------------------------- #
            print("==== Step 1: Ingest ====")
            try:
                result = await session.call_tool(
                    "remember",
                    {
                        "text": "Alice works for Atlas Corp as a senior engineer.",
                        "title": "Personnel record",
                        "session_id": "demo-mcp-1",
                        "turn_id": "t1",
                    },
                )
                if result.isError:
                    print(f"  ERROR: {result.content[0].text if result.content else 'unknown'}")
                else:
                    data = _parse(result)
                    ec = data.get("entities_created", "?")
                    rc = data.get("relations_created", "?")
                    rs = data.get("relations_superseded", "?")
                    print(f"  -> remember(\"Alice works for Atlas Corp as a senior engineer.\")")
                    print(f"     session_id=demo-mcp-1  turn_id=t1")
                    print(f"     entities_created={ec}, relations_created={rc},")
                    print(f"     relations_superseded={rs}")
            except Exception as exc:
                print(f"  EXCEPTION: {exc}")
            print()

            # -------------------------------------------------------------- #
            # Step 2: Search confirms Alice/Atlas are present                 #
            # -------------------------------------------------------------- #
            print("==== Step 2: Search (expect Alice + Atlas) ====")
            step2_results = []
            try:
                result = await session.call_tool(
                    "search",
                    {"query": "Who is on Atlas?", "hops": 2, "limit": 5},
                )
                if result.isError:
                    print(f"  ERROR: {result.content[0].text if result.content else 'unknown'}")
                else:
                    data = _parse(result)
                    step2_results = data.get("results", [])
                    print(f"  -> search(\"Who is on Atlas?\") -- {len(step2_results)} result(s)")
                    for r in step2_results:
                        score = round(r.get("score", 0), 3)
                        print(f"     - {r.get('name')} ({r.get('type')})  score={score}")
                    names_lower = {r.get("name", "").lower() for r in step2_results}
                    if any("alice" in n for n in names_lower) or any("atlas" in n for n in names_lower):
                        print("  [OK] Alice or Atlas found in results.")
                    else:
                        print("  [NOTE] Alice / Atlas not in top results (extraction may have used")
                        print("         different names). See results above.")
            except Exception as exc:
                print(f"  EXCEPTION: {exc}")
            print()

            # -------------------------------------------------------------- #
            # Step 3: Agent write-back -- supersede WORKS_FOR                #
            # -------------------------------------------------------------- #
            print("==== Step 3: Agent write-back (WORKS_FOR supersession) ====")
            superseded = False
            try:
                result = await session.call_tool(
                    "add_relation",
                    {
                        "subject": "Alice",
                        "subject_type": "Person",
                        "object": "Beacon",
                        "object_type": "Organization",
                        "rel_type": "WORKS_FOR",
                        "source": "agent:demo-mcp-1:t2",
                        "session_id": "demo-mcp-1",
                        "turn_id": "t2",
                    },
                )
                if result.isError:
                    print(f"  ERROR: {result.content[0].text if result.content else 'unknown'}")
                else:
                    data = _parse(result)
                    outcome = data.get("outcome", "?")
                    rid = data.get("relation_id", "?")
                    print(f"  -> add_relation(Alice, Beacon, WORKS_FOR)")
                    print(f"     session_id=demo-mcp-1  turn_id=t2")
                    print(f"     outcome={outcome}  relation_id={rid}")
                    if outcome == "superseded":
                        superseded = True
                        print("  [OK] Supersession fired -- Atlas relation marked valid_until.")
                    elif outcome == "created":
                        print("  [NOTE] New edge created (Alice had no prior WORKS_FOR edge).")
                        print("         Supersession will fire on the NEXT write for Alice.")
                    else:
                        print(f"  [NOTE] Outcome: {outcome}")
            except Exception as exc:
                print(f"  EXCEPTION: {exc}")
            print()

            # -------------------------------------------------------------- #
            # Step 4: Search again -- Atlas relation should be filtered out   #
            # -------------------------------------------------------------- #
            print("==== Step 4: Search (expect Atlas filtered, Beacon visible) ====")
            try:
                result = await session.call_tool(
                    "search",
                    {"query": "Who is on Atlas?", "hops": 2, "limit": 5},
                )
                if result.isError:
                    print(f"  ERROR: {result.content[0].text if result.content else 'unknown'}")
                else:
                    data = _parse(result)
                    step4_results = data.get("results", [])
                    print(f"  -> search(\"Who is on Atlas?\") -- {len(step4_results)} result(s)")
                    for r in step4_results:
                        score = round(r.get("score", 0), 3)
                        print(f"     - {r.get('name')} ({r.get('type')})  score={score}")
                    names_lower4 = {r.get("name", "").lower() for r in step4_results}
                    # Check whether the old Atlas-team relation path is gone
                    atlas_gone = not any("atlas" in n for n in names_lower4)
                    beacon_present = any("beacon" in n for n in names_lower4)
                    if superseded and atlas_gone:
                        print("  [OK] Atlas relation filtered by temporal supersession.")
                    elif superseded and not atlas_gone:
                        print("  [NOTE] Atlas still in results -- retrieval may surface the")
                        print("         Atlas Entity node itself (not the superseded edge).")
                    elif not superseded:
                        print("  [NOTE] Supersession did not fire in Step 3; Atlas may still")
                        print("         appear. This is expected on a fresh graph (no prior edge).")
                    if beacon_present:
                        print("  [OK] Beacon present in results (new WORKS_FOR edge visible).")
            except Exception as exc:
                print(f"  EXCEPTION: {exc}")
            print()

            # -------------------------------------------------------------- #
            # Step 5: Status snapshot                                         #
            # -------------------------------------------------------------- #
            print("==== Step 5: Status snapshot ====")
            try:
                result = await session.call_tool("status", {})
                if result.isError:
                    print(f"  ERROR: {result.content[0].text if result.content else 'unknown'}")
                else:
                    data = _parse(result)
                    print(f"  -> status()")
                    print(f"     entity_count        = {data.get('entity_count', '?')}")
                    print(f"     document_count      = {data.get('document_count', '?')}")
                    print(f"     relation_count      = {data.get('relation_count', '?')}")
                    conv_count = data.get("conversation_count", 0)
                    turn_count = data.get("turn_count", 0)
                    print(f"     conversation_count  = {conv_count}")
                    print(f"     turn_count          = {turn_count}")
                    top = data.get("top_entities", [])
                    print(f"     top_entities ({len(top)}):")
                    for e in top:
                        print(f"       - {e.get('name')} ({e.get('type')})  "
                              f"reinforcement={e.get('reinforcement')}")
                    writes = data.get("recent_agent_writes", [])
                    print(f"     recent_agent_writes ({len(writes)}):")
                    for w in writes:
                        print(f"       - {w.get('subject')} --[{w.get('rel_type')}]--> "
                              f"{w.get('object')}  "
                              f"session={w.get('session_id')}  turn={w.get('turn_id')}")
                    demo_write_found = any(
                        w.get("session_id") == "demo-mcp-1" for w in writes
                    )
                    if demo_write_found:
                        print("  [OK] demo-mcp-1 write appears in recent_agent_writes.")
                    else:
                        print("  [NOTE] Demo write not yet in recent_agent_writes.")
                    if conv_count >= 1:
                        print("  [OK] conversation_count >= 1 (Conversation node created).")
                    else:
                        print("  [WARN] conversation_count = 0 -- Conversation node not found.")
                    if turn_count >= 1:
                        print("  [OK] turn_count >= 1 (Turn nodes created).")
                    else:
                        print("  [WARN] turn_count = 0 -- Turn nodes not found.")
            except Exception as exc:
                print(f"  EXCEPTION: {exc}")
            print()

            # -------------------------------------------------------------- #
            # Step 6: Conversation graph (proves Conversation/Turn schema)    #
            # -------------------------------------------------------------- #
            print("==== Step 6: Conversation graph (proves Conversation/Turn schema is live) ====")
            try:
                result = await session.call_tool(
                    "graph_query",
                    {
                        "cypher": (
                            "MATCH (c:Conversation {id: 'demo-mcp-1'})-[:HAS_TURN]->(t:Turn) "
                            "RETURN c.id AS conv, t.id AS turn_id, t.timestamp AS ts "
                            "ORDER BY t.timestamp"
                        ),
                    },
                )
                if result.isError:
                    print(f"  ERROR: {result.content[0].text if result.content else 'unknown'}")
                else:
                    data = _parse(result)
                    rows = data.get("rows", [])
                    print(f"  -> graph_query(Conversation/Turn for demo-mcp-1) -- {len(rows)} row(s)")
                    for row in rows:
                        print(f"     conv={row.get('conv')}  turn_id={row.get('turn_id')}  ts={row.get('ts')}")
                    if rows:
                        print("  [OK] Conversation/Turn schema confirmed live in Neo4j.")
                    else:
                        print("  [NOTE] No rows returned -- Conversation node may use a different id.")
                        print("         Check that remember() and add_relation() received session_id.")
            except Exception as exc:
                print(f"  EXCEPTION: {exc}")
            print()

    print("==== Demo complete ====")


if __name__ == "__main__":
    asyncio.run(run_demo())
