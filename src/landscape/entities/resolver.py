from landscape.storage import neo4j_store, qdrant_store

SIMILARITY_THRESHOLD = 0.85
# Stricter threshold for cross-type resolution: when the caller passes
# entity_type="Unknown" we can't filter Qdrant by type, so we widen the
# search but require a higher cosine score to avoid false-positive collapses
# between unrelated entities that share a name prefix.
UNKNOWN_TYPE_THRESHOLD = 0.90


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

    When entity_type is "Unknown" (the default for agent-authored relation
    endpoints), we search across *all* types with a stricter 0.90 threshold
    so an existing ``Alice (PERSON)`` is found before a duplicate
    ``Alice (Unknown)`` is created.
    """
    if entity_type == "Unknown":
        candidates = await qdrant_store.search_entities_any_type(
            vector=vector,
            limit=5,
        )
        effective_threshold = UNKNOWN_TYPE_THRESHOLD
    else:
        candidates = await qdrant_store.search_similar_entities(
            vector=vector,
            entity_type=entity_type,
            limit=5,
        )
        effective_threshold = threshold

    if not candidates or candidates[0].score < effective_threshold:
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


async def resolve_existing_entity_id(name: str) -> str | None:
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity)
            WHERE e.name = $name OR $name IN coalesce(e.aliases, [])
            RETURN elementId(e) AS eid
            ORDER BY CASE WHEN coalesce(e.canonical, false) THEN 0 ELSE 1 END,
                     CASE WHEN e.name = $name THEN 0 ELSE 1 END
            LIMIT 1
            """,
            name=name,
        )
        record = await result.single()
        if record is None:
            return None
        return record["eid"]
