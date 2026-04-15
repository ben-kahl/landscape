from landscape.storage import neo4j_store, qdrant_store

SIMILARITY_THRESHOLD = 0.85


async def resolve_entity(
    name: str,
    entity_type: str,
    vector: list[float],
    source_doc: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[str | None, bool, float | None]:
    """
    Returns (canonical_neo4j_id, is_new, similarity).
    is_new=True means no match found — caller must create the entity.
    """
    candidates = await qdrant_store.search_similar_entities(
        vector=vector,
        entity_type=entity_type,
        limit=5,
    )

    if not candidates or candidates[0].score < threshold:
        return (None, True, None)

    best = candidates[0]
    canonical_id: str = best.payload["neo4j_node_id"]
    canonical = await neo4j_store.find_entity_by_element_id(canonical_id)

    if canonical is None:
        # Qdrant has a stale entry; treat as new
        return (None, True, None)

    if name.lower() != canonical["name"].lower() and name not in canonical["aliases"]:
        await neo4j_store.add_alias(canonical_id, name, source_doc, best.score)

    return (canonical_id, False, best.score)
