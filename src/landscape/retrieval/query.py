import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from landscape.embeddings import encoder
from landscape.retrieval.scoring import (
    ScoringWeights,
    parse_neo4j_datetime,
    reinforcement_score,
    score_candidate,
)
from landscape.storage import neo4j_store, qdrant_store


@dataclass
class RetrievedEntity:
    neo4j_id: str
    name: str
    type: str
    distance: int
    vector_sim: float
    reinforcement: float
    edge_confidence: float
    score: float
    path_edge_ids: list[str] = field(default_factory=list)
    path_edge_types: list[str] = field(default_factory=list)
    path_edge_subtypes: list[str | None] = field(default_factory=list)
    path_edge_quantities: list[dict[str, object | None]] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    chunk_neo4j_id: str  # Neo4j elementId of the :Chunk node
    text: str
    doc_id: str          # Document node ID from the chunk payload
    source_doc: str      # Human-readable document title/filename slug
    position: int
    score: float


@dataclass
class RetrievalResult:
    query: str
    results: list[RetrievedEntity]
    touched_entity_ids: list[str]
    touched_edge_ids: list[str]
    chunks: list[RetrievedChunk] = field(default_factory=list)


async def retrieve(
    query_text: str,
    hops: int = 2,
    limit: int = 10,
    chunk_limit: int = 3,
    weights: ScoringWeights | None = None,
    reinforce: bool = True,
    session_id: str | None = None,
    since: datetime | None = None,
) -> RetrievalResult:
    """Hybrid retrieval: seed by vector similarity, expand by graph BFS,
    score and rank, optionally reinforce touched elements."""
    w = weights or ScoringWeights.from_settings()
    now = datetime.now(UTC)

    query_vector = encoder.embed_query(query_text)

    # Seed searches: independent Qdrant queries against two collections.
    entity_hits, chunk_hits = await asyncio.gather(
        qdrant_store.search_entities_any_type(query_vector, limit=5),
        qdrant_store.search_chunks(query_vector, limit=5),
    )

    seed_sims: dict[str, float] = {}
    for hit in entity_hits:
        payload = hit.payload or {}
        neo4j_id = payload.get("neo4j_node_id")
        if not neo4j_id:
            continue
        seed_sims[neo4j_id] = max(seed_sims.get(neo4j_id, 0.0), float(hit.score))

    chunk_ids: list[str] = []
    chunk_score_by_id: dict[str, float] = {}
    retrieved_chunks: list[RetrievedChunk] = []
    for hit in chunk_hits:
        payload = hit.payload or {}
        cid = payload.get("chunk_neo4j_id")
        if not cid:
            continue
        chunk_ids.append(cid)
        chunk_score_by_id[cid] = float(hit.score)
        retrieved_chunks.append(
            RetrievedChunk(
                chunk_neo4j_id=cid,
                text=payload.get("text", ""),
                doc_id=payload.get("doc_id", ""),
                source_doc=payload.get("source_doc", ""),
                position=int(payload.get("position", 0)),
                score=float(hit.score),
            )
        )

    chunk_entities = await neo4j_store.get_entities_from_chunks(chunk_ids)
    for ent in chunk_entities:
        eid = ent["eid"]
        src_chunk_ids = (
            ent["chunk_eids"] if ent.get("chunk_eids") is not None else chunk_ids
        )
        best = max(
            (chunk_score_by_id.get(cid, 0.0) for cid in src_chunk_ids),
            default=0.0,
        )
        seed_sims[eid] = max(seed_sims.get(eid, 0.0), best)

    if not seed_sims:
        return RetrievalResult(
            query=query_text,
            results=[],
            touched_entity_ids=[],
            touched_edge_ids=[],
            chunks=retrieved_chunks[:chunk_limit],
        )

    # 3. Hydrate seed entity info (name, type, access_count, last_accessed).
    seed_rows = await _hydrate_entities(list(seed_sims.keys()))

    candidates: dict[str, RetrievedEntity] = {}

    # Seeds as distance-0 candidates.
    for row in seed_rows:
        eid = row["eid"]
        r = reinforcement_score(
            row["access_count"],
            parse_neo4j_datetime(row["last_accessed"]),
            now,
            w,
        )
        s = score_candidate(
            vector_sim=seed_sims.get(eid, 0.0),
            graph_distance=0,
            edge_confidence=0.0,
            reinforcement=r,
            weights=w,
        )
        candidates[eid] = RetrievedEntity(
            neo4j_id=eid,
            name=row["name"],
            type=row["type"],
            distance=0,
            vector_sim=seed_sims.get(eid, 0.0),
            reinforcement=r,
            edge_confidence=0.0,
            score=s,
        )

    # 4. Graph expansion from the seeds that survived hydration (so we don't
    #    waste BFS work on entities whose edges are all superseded anyway).
    live_seed_ids = [row["eid"] for row in seed_rows]
    expansions = await neo4j_store.bfs_expand(live_seed_ids, max_hops=hops)

    for row in expansions:
        target_id = row["target_id"]
        # Use the best (highest) seed similarity we've found for reaching this target
        seed_id = row["seed_id"]
        inherited_sim = seed_sims.get(seed_id, 0.0)

        # Edge signals: average confidence along the path, worst-case reinforcement
        edge_confs = row["edge_confidences"] or []
        avg_conf = sum(edge_confs) / len(edge_confs) if edge_confs else 0.0

        edge_access_counts = row["edge_access_counts"] or []
        edge_last_accesseds = [
            parse_neo4j_datetime(x) for x in (row["edge_last_accessed"] or [])
        ]
        edge_reinforcements = [
            reinforcement_score(c, la, now, w)
            for c, la in zip(edge_access_counts, edge_last_accesseds, strict=False)
        ]
        # Take the max edge reinforcement along the path — if any edge on the
        # path has been heavily reinforced, that's the relevant signal.
        path_reinforcement = max(edge_reinforcements) if edge_reinforcements else 0.0

        # Also fold in the target entity's own reinforcement
        target_reinforcement = reinforcement_score(
            row["target_access_count"],
            parse_neo4j_datetime(row["target_last_accessed"]),
            now,
            w,
        )
        combined_reinforcement = min(
            max(path_reinforcement, target_reinforcement),
            w.reinforcement_cap,
        )

        s = score_candidate(
            vector_sim=inherited_sim,
            graph_distance=row["distance"],
            edge_confidence=avg_conf,
            reinforcement=combined_reinforcement,
            weights=w,
        )

        existing = candidates.get(target_id)
        if existing is None or s > existing.score:
            candidates[target_id] = RetrievedEntity(
                neo4j_id=target_id,
                name=row["target_name"],
                type=row["target_type"],
                distance=row["distance"],
                vector_sim=inherited_sim,
                reinforcement=combined_reinforcement,
                edge_confidence=avg_conf,
                score=s,
                path_edge_ids=list(row["edge_ids"] or []),
                path_edge_types=list(row["edge_types"] or []),
                path_edge_subtypes=list(row.get("edge_subtypes") or []),
                path_edge_quantities=list(row.get("edge_quantities") or []),
            )

    # 5. Session/time allowlist filtering (post-search intersection per spec).
    #    Vector search runs against the full index; we narrow candidates after.
    if session_id is not None or since is not None:
        if session_id is not None and since is not None:
            conv_ids, since_ids, conv_cids, since_cids = await asyncio.gather(
                neo4j_store.get_entities_in_conversation(session_id),
                neo4j_store.get_entities_since(since),
                neo4j_store.get_chunks_in_conversation(session_id),
                neo4j_store.get_chunks_since(since),
            )
            allowlist = set(conv_ids) & set(since_ids)
            chunk_allowlist = set(conv_cids) & set(since_cids)
        elif session_id is not None:
            ents, chunks_in_conv = await asyncio.gather(
                neo4j_store.get_entities_in_conversation(session_id),
                neo4j_store.get_chunks_in_conversation(session_id),
            )
            allowlist = set(ents)
            chunk_allowlist = set(chunks_in_conv)
        else:
            assert since is not None
            ents, chunks_since = await asyncio.gather(
                neo4j_store.get_entities_since(since),
                neo4j_store.get_chunks_since(since),
            )
            allowlist = set(ents)
            chunk_allowlist = set(chunks_since)

        retrieved_chunks = [
            c for c in retrieved_chunks if c.chunk_neo4j_id in chunk_allowlist
        ]

        if not allowlist:
            return RetrievalResult(
                query=query_text,
                results=[],
                touched_entity_ids=[],
                touched_edge_ids=[],
                chunks=retrieved_chunks[:chunk_limit],
            )

        candidates = {k: v for k, v in candidates.items() if k in allowlist}

    ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)[:limit]

    touched_entity_ids = [c.neo4j_id for c in ranked]
    touched_edge_ids: list[str] = []
    seen_edges: set[str] = set()
    for c in ranked:
        for eid in c.path_edge_ids:
            if eid not in seen_edges:
                seen_edges.add(eid)
                touched_edge_ids.append(eid)

    if reinforce:
        now_iso = now.isoformat()
        await asyncio.gather(
            neo4j_store.touch_entities(touched_entity_ids, now_iso),
            neo4j_store.touch_relations(touched_edge_ids, now_iso),
        )

    return RetrievalResult(
        query=query_text,
        results=ranked,
        touched_entity_ids=touched_entity_ids,
        touched_edge_ids=touched_edge_ids,
        chunks=retrieved_chunks[:chunk_limit],
    )


async def _hydrate_entities(element_ids: list[str]) -> list[dict]:
    """Hydrate canonical entities and drop any whose edges are *all* superseded.

    An entity with zero edges is kept (newly ingested, not orphaned). An entity
    with at least one currently-valid edge is kept. Only entities that had
    edges and have had *all* of them superseded are dropped — they're stale
    facts masquerading as live seeds."""
    if not element_ids:
        return []
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity) WHERE elementId(e) IN $ids AND e.canonical = true
            OPTIONAL MATCH (e)-[r:RELATES_TO]-()
            WITH e,
                 count(r) AS total_edges,
                 sum(CASE WHEN r.valid_until IS NULL THEN 1 ELSE 0 END) AS valid_edges
            WHERE total_edges = 0 OR valid_edges > 0
            RETURN elementId(e) AS eid,
                   e.name AS name,
                   e.type AS type,
                   coalesce(e.access_count, 0) AS access_count,
                   e.last_accessed AS last_accessed
            """,
            ids=element_ids,
        )
        return [dict(r) async for r in result]
