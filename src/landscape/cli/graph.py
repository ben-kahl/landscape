from __future__ import annotations

import argparse

from landscape.cli.runtime import close_runtime


def _get_runtime():
    from landscape.storage import neo4j_store, qdrant_store

    return neo4j_store, qdrant_store


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "graph",
        help="Inspect graph entities, counts, and neighborhoods",
    )
    graph_subparsers = parser.add_subparsers(dest="graph_command", required=True)

    counts = graph_subparsers.add_parser("counts", help="Show graph object counts")
    counts.set_defaults(func=handle_counts)

    entity = graph_subparsers.add_parser("entity", help="Show one entity by name")
    entity.add_argument("name")
    entity.set_defaults(func=handle_entity)

    neighbors = graph_subparsers.add_parser("neighbors", help="Show entity neighborhood")
    neighbors.add_argument("name")
    neighbors.add_argument("--hops", type=int, default=2)
    neighbors.add_argument("--limit", type=int, default=25)
    neighbors.set_defaults(func=handle_neighbors)


async def handle_counts(args: argparse.Namespace) -> int:
    neo4j_store, qdrant_store = _get_runtime()
    driver = neo4j_store.get_driver()
    try:
        async with driver.session() as session:
            result = await session.run(
                """
                CALL () { MATCH (d:Document) RETURN count(d) AS documents }
                CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
                CALL () { MATCH (c:Chunk) RETURN count(c) AS chunks }
                CALL () {
                  MATCH ()-[live:RELATES_TO]->()
                  WHERE live.valid_until IS NULL
                  RETURN count(live) AS live_relations
                }
                CALL () {
                  MATCH ()-[stale:RELATES_TO]->()
                  WHERE stale.valid_until IS NOT NULL
                  RETURN count(stale) AS superseded_relations
                }
                RETURN documents, entities, chunks, live_relations, superseded_relations
                """
            )
            row = await result.single()
        print(f"Documents: {row['documents']}")
        print(f"Entities: {row['entities']}")
        print(f"Chunks: {row['chunks']}")
        print(
            "Relations: "
            f"{row['live_relations']} live / {row['superseded_relations']} superseded"
        )
        return 0
    finally:
        await close_runtime(neo4j_store, qdrant_store)


async def handle_entity(args: argparse.Namespace) -> int:
    neo4j_store, qdrant_store = _get_runtime()
    driver = neo4j_store.get_driver()
    try:
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower($name)
                RETURN e.name AS name, e.type AS type, elementId(e) AS id,
                       e.source_doc AS source_doc, e.confidence AS confidence
                LIMIT 5
                """,
                name=args.name,
            )
            rows = [dict(r) async for r in result]
        if not rows:
            print("No matching entity.")
            return 0
        for row in rows:
            print(
                f"{row['name']} [{row['type']}] id={row['id']} "
                f"source={row['source_doc']} confidence={row['confidence']}"
            )
        return 0
    finally:
        await close_runtime(neo4j_store, qdrant_store)


async def handle_neighbors(args: argparse.Namespace) -> int:
    neo4j_store, qdrant_store = _get_runtime()
    driver = neo4j_store.get_driver()
    try:
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (start:Entity)
                WHERE toLower(start.name) = toLower($name)
                MATCH path = (start)-[:RELATES_TO*1..3]-(other:Entity)
                WHERE length(path) <= $hops
                  AND all(r IN relationships(path) WHERE r.valid_until IS NULL)
                RETURN other.name AS name, other.type AS type, length(path) AS distance,
                       [r IN relationships(path) | r.relationship_type] AS rel_types
                ORDER BY distance, name
                LIMIT $limit
                """,
                name=args.name,
                hops=args.hops,
                limit=args.limit,
            )
            rows = [dict(r) async for r in result]
        if not rows:
            print("No neighbors found.")
            return 0
        for row in rows:
            print(
                f"{row['name']} [{row['type']}] distance={row['distance']} "
                f"path={' -> '.join(row['rel_types'])}"
            )
        return 0
    finally:
        await close_runtime(neo4j_store, qdrant_store)
