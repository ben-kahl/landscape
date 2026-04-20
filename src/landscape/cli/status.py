from __future__ import annotations

import argparse
import json

from landscape.cli.runtime import close_runtime


def _get_runtime():
    import ollama

    from landscape.config import settings
    from landscape.storage import neo4j_store, qdrant_store

    return ollama, settings, neo4j_store, qdrant_store


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "status",
        help="Check local service, config, and storage state",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.set_defaults(func=handle_status)


async def _neo4j_counts() -> dict:
    _ollama, _settings, neo4j_store, _qdrant_store = _get_runtime()
    driver = neo4j_store.get_driver()
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
    return dict(row)


async def handle_status(args: argparse.Namespace) -> int:
    ollama, settings, neo4j_store, qdrant_store = _get_runtime()
    data = {
        "config": {
            "neo4j_uri": settings.neo4j_uri,
            "qdrant_url": settings.qdrant_url,
            "ollama_url": settings.ollama_url,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        },
        "services": {},
        "storage": {},
    }
    try:
        try:
            data["storage"] = await _neo4j_counts()
            data["services"]["neo4j"] = "ok"
        except Exception as exc:
            data["services"]["neo4j"] = f"error: {exc}"

        try:
            collections = await qdrant_store.get_client().get_collections()
            data["services"]["qdrant"] = "ok"
            data["storage"]["qdrant_collections"] = [c.name for c in collections.collections]
        except Exception as exc:
            data["services"]["qdrant"] = f"error: {exc}"

        try:
            client = ollama.Client(host=settings.ollama_url)
            models = client.list()
            model_names = {model.model for model in models.models}
            data["services"]["ollama"] = "ok"
            data["models"] = {
                "configured_llm_present": settings.llm_model in model_names,
                "ollama_models": sorted(model_names) if args.verbose else [],
            }
        except Exception as exc:
            data["services"]["ollama"] = f"error: {exc}"

        if args.as_json:
            print(json.dumps(data, indent=2, sort_keys=True))
            return 0

        print("Landscape status")
        print()
        print("Config")
        print(f"  Neo4j:  {data['config']['neo4j_uri']}")
        print(f"  Qdrant: {data['config']['qdrant_url']}")
        print(f"  Ollama: {data['config']['ollama_url']}")
        print(f"  LLM:    {data['config']['llm_model']}")
        print(f"  Embed:  {data['config']['embedding_model']}")
        print()
        print("Services")
        print(f"  Neo4j:  {data['services'].get('neo4j', 'unknown')}")
        print(f"  Qdrant: {data['services'].get('qdrant', 'unknown')}")
        print(f"  Ollama: {data['services'].get('ollama', 'unknown')}")
        if data["storage"]:
            print()
            print("Storage")
            for key, value in data["storage"].items():
                print(f"  {key}: {value}")
        if args.verbose:
            print()
            print("Environment")
            print("  CLI defaults use host-reachable service URLs unless env overrides them.")
        return 0
    finally:
        await close_runtime(neo4j_store, qdrant_store)
