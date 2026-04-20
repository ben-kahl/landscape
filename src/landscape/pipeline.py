import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from landscape.config import settings
from landscape.embeddings import encoder
from landscape.entities import resolver
from landscape.extraction import llm
from landscape.extraction.chunker import chunk_text
from landscape.extraction.entity_type_coercion import coerce_entity_type
from landscape.extraction.rel_type_coercion import coerce_rel_type
from landscape.extraction.schema import normalize_subtype
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
) -> IngestResult:
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    doc_id, created = await neo4j_store.merge_document(content_hash, title, source_type)
    if not created:
        return IngestResult(
            doc_id=doc_id,
            already_existed=True,
            entities_created=0,
            entities_reinforced=0,
            relations_created=0,
            relations_reinforced=0,
            relations_superseded=0,
            chunks_created=0,
        )

    # Step 1b: wire Document to Turn when conversation context is provided
    turn_element_id: str | None = None
    if session_id is not None and turn_id is not None:
        turn_element_id, _ = await neo4j_store.merge_turn(session_id, turn_id)
        await neo4j_store.link_document_to_turn(doc_id, turn_element_id)

    # Step 2: chunk + embed chunks (batched)
    chunks = chunk_text(text)
    chunks_created = 0
    if chunks:
        chunk_neo4j_ids: list[str] = []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
            chunk_neo4j_ids.append(
                await neo4j_store.create_chunk(doc_id, chunk.index, chunk.text, chunk_hash)
            )
        chunk_vectors = encoder.embed_documents([c.text for c in chunks])
        await asyncio.gather(
            *(
                qdrant_store.upsert_chunk(
                    chunk_neo4j_id=cid,
                    doc_id=doc_id,
                    source_doc=title,
                    position=chunk.index,
                    text=chunk.text,
                    vector=vec,
                )
                for chunk, cid, vec in zip(chunks, chunk_neo4j_ids, chunk_vectors, strict=True)
            )
        )
        chunks_created = len(chunks)

    # Step 3: extract entities + relations from full text
    extraction = llm.extract(text)

    now = datetime.now(UTC).isoformat()
    entities_created = 0
    entities_reinforced = 0

    # Step 4: entity resolution + write
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

    group_keys = list(grouped.keys())
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

    # Step 5: relation upsert with supersession
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
        )
        if outcome == "created":
            relations_created += 1
        elif outcome == "reinforced":
            relations_reinforced += 1
        elif outcome == "superseded":
            relations_superseded += 1

    return IngestResult(
        doc_id=doc_id,
        already_existed=False,
        entities_created=entities_created,
        entities_reinforced=entities_reinforced,
        relations_created=relations_created,
        relations_reinforced=relations_reinforced,
        relations_superseded=relations_superseded,
        chunks_created=chunks_created,
    )
