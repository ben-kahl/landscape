import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from landscape.config import settings
from landscape.embeddings import encoder
from landscape.entities import resolver
from landscape.extraction import llm
from landscape.extraction.chunker import chunk_text
from landscape.extraction.entity_type_coercion import coerce_entity_type
from landscape.extraction.rel_type_coercion import coerce_rel_type
from landscape.extraction.schema import normalize_subtype
from landscape.observability import IngestLogContext, create_ingest_log_context
from landscape.storage import neo4j_store, qdrant_store


@dataclass
class IngestResult:
    doc_id: str
    already_existed: bool
    entities_created: int
    entities_reinforced: int
    relations_created: int
    relations_reinforced: int
    relations_superseded: int
    chunks_created: int


async def ingest(
    text: str,
    title: str,
    source_type: str = "text",
    session_id: str | None = None,
    turn_id: str | None = None,
    debug: bool = False,
    log_context: IngestLogContext | None = None,
) -> IngestResult:
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    log = log_context or create_ingest_log_context(
        title=title,
        source_type=source_type,
        session_id=session_id,
        turn_id=turn_id,
        debug=debug,
    )
    log.emit_started(content_hash=content_hash, text_length=len(text))

    try:
        stage_started_at = log.set_stage("document_merged")
        doc_id, created = await neo4j_store.merge_document(content_hash, title, source_type)
        log.emit(
            "document_merged",
            doc_id=doc_id,
            already_existed=not created,
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )
        if not created:
            result = IngestResult(
                doc_id=doc_id,
                already_existed=True,
                entities_created=0,
                entities_reinforced=0,
                relations_created=0,
                relations_reinforced=0,
                relations_superseded=0,
                chunks_created=0,
            )
            log.emit_completed(
                doc_id=result.doc_id,
                already_existed=result.already_existed,
                entities_created=result.entities_created,
                entities_reinforced=result.entities_reinforced,
                relations_created=result.relations_created,
                relations_reinforced=result.relations_reinforced,
                relations_superseded=result.relations_superseded,
                chunks_created=result.chunks_created,
            )
            return result

        # Ensure both Qdrant collections exist before any chunk/entity writes or
        # entity-resolution lookups. Some test setups spin up an empty Qdrant.
        await asyncio.gather(
            qdrant_store.init_collection(),
            qdrant_store.init_chunks_collection(),
        )

        # Step 1b: wire Document to Turn when conversation context is provided
        turn_element_id: str | None = None
        if session_id is not None and turn_id is not None:
            stage_started_at = log.set_stage("turn_linked")
            turn_element_id, _ = await neo4j_store.merge_turn(session_id, turn_id)
            await neo4j_store.link_document_to_turn(doc_id, turn_element_id)
            log.emit(
                "turn_linked",
                doc_id=doc_id,
                turn_element_id=turn_element_id,
                duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
            )

        # Step 2: chunk + embed chunks (batched)
        stage_started_at = log.set_stage("chunking_completed")
        chunks = chunk_text(text)
        log.emit(
            "chunking_completed",
            chunk_count=len(chunks),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )
        chunks_created = 0
        if chunks:
            chunk_ids: list[str] = []
            for chunk in chunks:
                chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
                chunk_ids.append(
                    await neo4j_store.create_chunk(doc_id, chunk.index, chunk.text, chunk_hash)
                )

            stage_started_at = log.set_stage("chunk_embeddings_completed")
            chunk_vectors = encoder.embed_documents([c.text for c in chunks])
            log.emit(
                "chunk_embeddings_completed",
                chunk_count=len(chunks),
                duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
            )

            stage_started_at = log.set_stage("chunk_upserts_completed")
            await asyncio.gather(
                *(
                    qdrant_store.upsert_chunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        source_doc=title,
                        position=chunk.index,
                        text=chunk.text,
                        vector=vec,
                    )
                    for chunk, cid, vec in zip(chunks, chunk_ids, chunk_vectors, strict=True)
                )
            )
            chunks_created = len(chunks)
            log.emit(
                "chunk_upserts_completed",
                chunk_count=chunks_created,
                duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
            )

        # Step 3: extract entities + relations from full text
        stage_started_at = log.set_stage("extraction_completed")
        extraction = llm.extract(text)
        log.emit(
            "extraction_completed",
            entities_extracted=len(extraction.entities),
            relations_extracted=len(extraction.relations),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        now = datetime.now(UTC).isoformat()
        entities_created = 0
        entities_reinforced = 0

        # Step 4: entity resolution + write
        stage_started_at = log.set_stage("entity_grouping_completed")
        # Pre-group mentions by (lowercased name, canonical type) so duplicate
        # mentions in the same doc resolve once. Preserves the dedupe the serial
        # loop used to get "for free" from ordered resolution.
        grouped: dict[tuple[str, str], dict] = {}
        for entity in extraction.entities:
            canonical_entity_type, _ = coerce_entity_type(entity.type)
            etype_subtype = entity.type if canonical_entity_type != entity.type else None
            key = (entity.name.strip().lower(), canonical_entity_type)
            # First-mention wins for subtype/confidence within a group. The old
            # serial loop had last-wins (each merge overwrote the node); this
            # differs in principle but not in killer-demo or test outcomes.
            if key not in grouped:
                grouped[key] = {
                    "name": entity.name.strip(),
                    "canonical_entity_type": canonical_entity_type,
                    "subtype": etype_subtype,
                    "confidence": entity.confidence,
                    "encode_text": f"{entity.name.strip()} ({canonical_entity_type})",
                }
        log.emit(
            "entity_grouping_completed",
            grouped_entities=len(grouped),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        group_keys = list(grouped.keys())
        stage_started_at = log.set_stage("entity_resolution_completed")
        if group_keys:
            vectors = encoder.embed_documents(
                [grouped[k]["encode_text"] for k in group_keys]
            )
            resolutions = await asyncio.gather(
                *(
                    resolver.resolve_entity(
                        name=grouped[k]["name"],
                        entity_type=grouped[k]["canonical_entity_type"],
                        vector=vectors[i],
                        source_doc=title,
                    )
                    for i, k in enumerate(group_keys)
                )
            )
        else:
            vectors = []
            resolutions = []
        log.emit(
            "entity_resolution_completed",
            resolved_entities=len(group_keys),
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        stage_started_at = log.set_stage("entity_writes_completed")
        for key, vector, (canonical_id, is_new, _sim) in zip(
            group_keys, vectors, resolutions, strict=True
        ):
            g = grouped[key]
            if is_new:
                canonical_id = await neo4j_store.merge_entity(
                    name=g["name"],
                    entity_type=g["canonical_entity_type"],
                    source_doc=title,
                    confidence=g["confidence"],
                    doc_element_id=doc_id,
                    model=settings.llm_model,
                    session_id=session_id,
                    turn_id=turn_id,
                    subtype=g["subtype"],
                )
                await qdrant_store.upsert_entity(
                    neo4j_element_id=canonical_id,
                    name=g["name"],
                    entity_type=g["canonical_entity_type"],
                    source_doc=title,
                    timestamp=now,
                    vector=vector,
                )
                entities_created += 1
            else:
                await neo4j_store.link_entity_to_doc(
                    entity_element_id=canonical_id,
                    doc_element_id=doc_id,
                    model=settings.llm_model,
                )
                entities_reinforced += 1

            if turn_element_id is not None:
                await neo4j_store.link_entity_to_turn(canonical_id, turn_element_id)
        log.emit(
            "entity_writes_completed",
            entities_created=entities_created,
            entities_reinforced=entities_reinforced,
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        # Step 5: relation upsert with supersession
        stage_started_at = log.set_stage("relation_upserts_completed")
        relations_created = 0
        relations_reinforced = 0
        relations_superseded = 0
        for relation in extraction.relations:
            canonical_rel_type, _coerce_score = coerce_rel_type(relation.relation_type)
            # If coercion changed the rel type, preserve the LLM's original
            # phrasing as the subtype — captures nuance even when the LLM didn't
            # produce an explicit subtype field. Explicit subtype (step 3 schema)
            # still wins when present.
            raw_upper = (relation.relation_type or "").strip().upper().replace(" ", "_")
            subtype_source = relation.subtype or (
                raw_upper if raw_upper and raw_upper != canonical_rel_type else None
            )
            canonical_subtype = normalize_subtype(subtype_source)
            outcome, _ = await neo4j_store.upsert_relation(
                subject_name=relation.subject,
                object_name=relation.object,
                relation_type=canonical_rel_type,
                confidence=relation.confidence,
                source_doc=title,
                session_id=session_id,
                turn_id=turn_id,
                subtype=canonical_subtype,
                quantity_value=relation.quantity_value,
                quantity_unit=relation.quantity_unit,
                quantity_kind=relation.quantity_kind,
                time_scope=relation.time_scope,
            )
            if outcome == "created":
                relations_created += 1
            elif outcome == "reinforced":
                relations_reinforced += 1
            elif outcome == "superseded":
                relations_superseded += 1
        log.emit(
            "relation_upserts_completed",
            relations_created=relations_created,
            relations_reinforced=relations_reinforced,
            relations_superseded=relations_superseded,
            duration_ms=round((perf_counter() - stage_started_at) * 1000, 3),
        )

        result = IngestResult(
            doc_id=doc_id,
            already_existed=False,
            entities_created=entities_created,
            entities_reinforced=entities_reinforced,
            relations_created=relations_created,
            relations_reinforced=relations_reinforced,
            relations_superseded=relations_superseded,
            chunks_created=chunks_created,
        )
        log.emit_completed(
            doc_id=result.doc_id,
            already_existed=result.already_existed,
            entities_created=result.entities_created,
            entities_reinforced=result.entities_reinforced,
            relations_created=result.relations_created,
            relations_reinforced=result.relations_reinforced,
            relations_superseded=result.relations_superseded,
            chunks_created=result.chunks_created,
        )
        return result
    except Exception as exc:
        log.emit_failed(exc)
        raise
