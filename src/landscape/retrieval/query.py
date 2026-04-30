import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter

from landscape.embeddings import encoder
from landscape.observability import RetrievalLogContext, create_retrieval_log_context
from landscape.retrieval.scoring import (
    ScoringWeights,
    parse_neo4j_datetime,
    reinforcement_score,
    score_candidate,
)
from landscape.storage import neo4j_store, qdrant_store


@dataclass
class RetrievedEntity:
    entity_id: str
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
    path_memory_fact_ids: list[str] = field(default_factory=list)
    memory_facts: list[dict[str, object]] = field(default_factory=list)
    supporting_assertions: list[dict[str, object]] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    chunk_id: str
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


def _top_results_for_logging(results: list[RetrievedEntity], *, max_items: int = 5) -> list[dict]:
    return [
        {
            "name": item.name,
            "type": item.type,
            "score": round(item.score, 6),
            "distance": item.distance,
        }
        for item in results[:max_items]
    ]


async def _hydrate_memory_path_details(
    memory_fact_ids: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return await neo4j_store.get_memory_fact_details_batch(memory_fact_ids)


async def _hydrate_current_non_traversable_entity_memory(
    entity_ids: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return await neo4j_store.get_current_non_traversable_fact_details_for_entities(entity_ids)


async def retrieve(
    query_text: str,
    hops: int = 2,
    limit: int = 10,
    chunk_limit: int = 3,
    weights: ScoringWeights | None = None,
    reinforce: bool = True,
    session_id: str | None = None,
    since: datetime | None = None,
    debug: bool = False,
    include_historical: bool = False,
    log_context: RetrievalLogContext | None = None,
) -> RetrievalResult:
    """Hybrid retrieval: seed by vector similarity, expand by graph BFS,
    score and rank, optionally reinforce touched elements."""
    w = weights or ScoringWeights.from_settings()
    now = datetime.now(UTC)
    log = log_context or create_retrieval_log_context(
        query_text=query_text,
        hops=hops,
        limit=limit,
        chunk_limit=chunk_limit,
        reinforce=reinforce,
        session_id=session_id,
        since=since,
        debug=debug,
    )
    log.emit_started()

    try:
        stage_started_at = log.set_stage("query_embedding_completed")
        query_vector = encoder.embed_query(query_text)
        log.emit(
            "query_embedding_completed",
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        stage_started_at = log.set_stage("alias_seed_resolution_completed")
        direct_seed_ids = await neo4j_store.resolve_seed_entity_ids(query_text)
        log.emit(
            "alias_seed_resolution_completed",
            alias_seed_count=len(direct_seed_ids),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        # Seed searches: independent Qdrant queries against two collections.
        stage_started_at = log.set_stage("seed_search_completed")
        entity_hits, chunk_hits = await asyncio.gather(
            qdrant_store.search_entities_any_type(query_vector, limit=5),
            qdrant_store.search_chunks(query_vector, limit=5),
        )
        log.emit(
            "seed_search_completed",
            entity_hit_count=len(entity_hits),
            chunk_hit_count=len(chunk_hits),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        seed_sims: dict[str, float] = {}
        for entity_id in direct_seed_ids:
            seed_sims[entity_id] = max(seed_sims.get(entity_id, 0.0), 1.0)
        for hit in entity_hits:
            payload = hit.payload or {}
            entity_id = payload.get("entity_id")
            if not entity_id:
                continue
            seed_sims[entity_id] = max(seed_sims.get(entity_id, 0.0), float(hit.score))

        chunk_ids: list[str] = []
        chunk_score_by_id: dict[str, float] = {}
        retrieved_chunks: list[RetrievedChunk] = []
        for hit in chunk_hits:
            payload = hit.payload or {}
            cid = payload.get("chunk_id")
            if not cid:
                continue
            chunk_ids.append(cid)
            chunk_score_by_id[cid] = float(hit.score)
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=payload.get("text", ""),
                    doc_id=payload.get("doc_id", ""),
                    source_doc=payload.get("source_doc", ""),
                    position=int(payload.get("position", 0)),
                    score=float(hit.score),
                )
            )

        stage_started_at = log.set_stage("chunk_entity_propagation_completed")
        chunk_entities = await neo4j_store.get_entities_from_chunks(chunk_ids)
        for ent in chunk_entities:
            entity_id = ent["entity_id"]
            src_chunk_ids = (
                ent["chunk_eids"] if ent.get("chunk_eids") is not None else chunk_ids
            )
            best = max(
                (chunk_score_by_id.get(cid, 0.0) for cid in src_chunk_ids),
                default=0.0,
            )
            seed_sims[entity_id] = max(seed_sims.get(entity_id, 0.0), best)
        log.emit(
            "chunk_entity_propagation_completed",
            propagated_entity_count=len(chunk_entities),
            seed_entity_count=len(seed_sims),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        if not seed_sims:
            result = RetrievalResult(
                query=query_text,
                results=[],
                touched_entity_ids=[],
                touched_edge_ids=[],
                chunks=retrieved_chunks[:chunk_limit],
            )
            log.emit_completed(
                result_count=0,
                touched_entity_count=0,
                touched_edge_count=0,
                chunk_count=len(result.chunks),
                top_results=[],
            )
            return result

        # 3. Hydrate seed entity info (name, type, access_count, last_accessed).
        stage_started_at = log.set_stage("seed_hydration_completed")
        seed_rows = await _hydrate_entities(list(seed_sims.keys()))
        log.emit(
            "seed_hydration_completed",
            hydrated_seed_count=len(seed_rows),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        candidates: dict[str, RetrievedEntity] = {}

        # Seeds as distance-0 candidates.
        for row in seed_rows:
            entity_id = row["entity_id"]
            r = reinforcement_score(
                row["access_count"],
                parse_neo4j_datetime(row["last_accessed"]),
                now,
                w,
            )
            s = score_candidate(
                vector_sim=seed_sims.get(entity_id, 0.0),
                graph_distance=0,
                edge_confidence=0.0,
                reinforcement=r,
                weights=w,
            )
            candidates[entity_id] = RetrievedEntity(
                entity_id=entity_id,
                name=row["name"],
                type=row["type"],
                distance=0,
                vector_sim=seed_sims.get(entity_id, 0.0),
                reinforcement=r,
                edge_confidence=0.0,
                score=s,
            )

        # 4. Graph expansion from the seeds that survived hydration (so we don't
        #    waste BFS work on entities whose edges are all superseded anyway).
        live_seed_ids = [row["entity_id"] for row in seed_rows]
        stage_started_at = log.set_stage("graph_expansion_completed")
        expansions = await neo4j_store.bfs_expand_memory_rel(live_seed_ids, max_hops=hops)
        log.emit(
            "graph_expansion_completed",
            expansion_count=len(expansions),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        for row in expansions:
            target_id = row["target_id"]
            # Use the best (highest) seed similarity we've found for reaching this target
            seed_id = row["seed_id"]
            inherited_sim = seed_sims.get(seed_id, 0.0)

            # Edge signals: average confidence along the path, worst-case reinforcement
            edge_confs = row.get("edge_confidences") or []
            avg_conf = sum(edge_confs) / len(edge_confs) if edge_confs else 0.0

            edge_access_counts = row.get("edge_access_counts") or []
            edge_last_accesseds = [
                parse_neo4j_datetime(x) for x in (row.get("edge_last_accessed") or [])
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
                row.get("target_access_count", 0),
                parse_neo4j_datetime(row.get("target_last_accessed")),
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

            path_memory_fact_ids = list(row.get("path_memory_fact_ids") or [])
            path_edge_types = list(row.get("path_edge_types") or [])

            existing = candidates.get(target_id)
            if existing is None or s > existing.score:
                candidates[target_id] = RetrievedEntity(
                    entity_id=target_id,
                    name=row["target_name"],
                    type=row["target_type"],
                    distance=row["distance"],
                    vector_sim=inherited_sim,
                    reinforcement=combined_reinforcement,
                    edge_confidence=avg_conf,
                    score=s,
                    path_edge_ids=list(row.get("edge_ids") or []),
                    path_edge_types=path_edge_types,
                    path_edge_subtypes=list(row.get("edge_subtypes") or []),
                    path_edge_quantities=list(row.get("edge_quantities") or []),
                    path_memory_fact_ids=path_memory_fact_ids,
                )

        # 5. Session/time allowlist filtering (post-search intersection per spec).
        #    Vector search runs against the full index; we narrow candidates after.
        stage_started_at = log.set_stage("filter_completed")
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
                c for c in retrieved_chunks if c.chunk_id in chunk_allowlist
            ]

            if not allowlist:
                result = RetrievalResult(
                    query=query_text,
                    results=[],
                    touched_entity_ids=[],
                    touched_edge_ids=[],
                    chunks=retrieved_chunks[:chunk_limit],
                )
                log.emit(
                    "filter_completed",
                    candidate_count=0,
                    chunk_count=len(result.chunks),
                    duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
                )
                log.emit_completed(
                    result_count=0,
                    touched_entity_count=0,
                    touched_edge_count=0,
                    chunk_count=len(result.chunks),
                    top_results=[],
                )
                return result

            candidates = {k: v for k, v in candidates.items() if k in allowlist}
        log.emit(
            "filter_completed",
            candidate_count=len(candidates),
            chunk_count=len(retrieved_chunks[:chunk_limit]),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        stage_started_at = log.set_stage("ranking_completed")
        ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)[:limit]
        log.emit(
            "ranking_completed",
            ranked_count=len(ranked),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        path_fact_ids = [
            fact_id
            for item in ranked
            for fact_id in item.path_memory_fact_ids
        ]
        if path_fact_ids:
            stage_started_at = log.set_stage("path_hydration_completed")
            memory_facts, supporting_assertions = await _hydrate_memory_path_details(
                path_fact_ids
            )
            facts_by_id = {
                fact["memory_fact_id"]: fact
                for fact in memory_facts
                if fact.get("memory_fact_id") is not None
            }
            assertions_by_fact_id: dict[str, list[dict[str, object]]] = {}
            for assertion in supporting_assertions:
                fact_id = assertion.get("memory_fact_id")
                if fact_id is None:
                    continue
                assertions_by_fact_id.setdefault(str(fact_id), []).append(assertion)
            for item in ranked:
                item.memory_facts = [
                    facts_by_id[fact_id]
                    for fact_id in item.path_memory_fact_ids
                    if fact_id in facts_by_id
                ]
                item.supporting_assertions = [
                    assertion
                    for fact_id in item.path_memory_fact_ids
                    for assertion in assertions_by_fact_id.get(fact_id, [])
                ]
            log.emit(
                "path_hydration_completed",
                hydrated_fact_count=len(memory_facts),
                hydrated_assertion_count=len(supporting_assertions),
                duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
            )

        current_scalar_facts, current_scalar_assertions = (
            await _hydrate_current_non_traversable_entity_memory([item.entity_id for item in ranked])
        )
        if current_scalar_facts:
            facts_by_entity_id: dict[str, list[dict[str, object]]] = {}
            for fact in current_scalar_facts:
                entity_id = fact.get("subject_entity_id")
                if entity_id is None:
                    continue
                facts_by_entity_id.setdefault(str(entity_id), []).append(fact)
            assertions_by_fact_id: dict[str, list[dict[str, object]]] = {}
            for assertion in current_scalar_assertions:
                fact_id = assertion.get("memory_fact_id")
                if fact_id is None:
                    continue
                assertions_by_fact_id.setdefault(str(fact_id), []).append(assertion)
            for item in ranked:
                seen_fact_ids = {str(fact["memory_fact_id"]) for fact in item.memory_facts}
                extra_facts = [
                    fact
                    for fact in facts_by_entity_id.get(item.entity_id, [])
                    if str(fact["memory_fact_id"]) not in seen_fact_ids
                ]
                item.memory_facts.extend(extra_facts)
                for fact in extra_facts:
                    fact_id = str(fact["memory_fact_id"])
                    item.supporting_assertions.extend(assertions_by_fact_id.get(fact_id, []))

        touched_entity_ids = [c.entity_id for c in ranked]
        touched_edge_ids: list[str] = []
        seen_edges: set[str] = set()
        for c in ranked:
            for eid in c.path_edge_ids:
                if eid not in seen_edges:
                    seen_edges.add(eid)
                    touched_edge_ids.append(eid)

        if reinforce:
            stage_started_at = log.set_stage("reinforcement_completed")
            now_iso = now.isoformat()
            await asyncio.gather(
                neo4j_store.touch_entities(touched_entity_ids, now_iso),
                neo4j_store.touch_relations(touched_edge_ids, now_iso),
            )
            log.emit(
                "reinforcement_completed",
                touched_entity_count=len(touched_entity_ids),
                touched_edge_count=len(touched_edge_ids),
                duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
            )

        result = RetrievalResult(
            query=query_text,
            results=ranked,
            touched_entity_ids=touched_entity_ids,
            touched_edge_ids=touched_edge_ids,
            chunks=retrieved_chunks[:chunk_limit],
        )
        log.emit_completed(
            result_count=len(result.results),
            touched_entity_count=len(result.touched_entity_ids),
            touched_edge_count=len(result.touched_edge_ids),
            chunk_count=len(result.chunks),
            top_results=_top_results_for_logging(result.results),
        )
        return result
    except Exception as exc:
        log.emit_failed(exc)
        raise


async def _hydrate_entities(element_ids: list[str]) -> list[dict]:
    """Hydrate canonical entities by stable app id and drop stale-only nodes.

    An entity with zero edges is kept (newly ingested, not orphaned). An entity
    with at least one currently-valid edge is kept. Only entities that had
    edges and have had *all* of them superseded are dropped — they're stale
    facts masquerading as live seeds."""
    return await neo4j_store.get_rankable_entities(element_ids)
