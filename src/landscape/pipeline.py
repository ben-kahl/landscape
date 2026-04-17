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

    # Step 2: chunk + embed chunks
    chunks = chunk_text(text)
    chunks_created = 0
    for chunk in chunks:
        chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
        chunk_neo4j_id = await neo4j_store.create_chunk(doc_id, chunk.index, chunk.text, chunk_hash)
        chunk_vector = encoder.encode(chunk.text)
        await qdrant_store.upsert_chunk(
            chunk_neo4j_id=chunk_neo4j_id,
            doc_id=doc_id,
            source_doc=title,
            position=chunk.index,
            text=chunk.text,
            vector=chunk_vector,
        )
        chunks_created += 1

    # Step 3: extract entities + relations from full text
    extraction = llm.extract(text)

    now = datetime.now(UTC).isoformat()
    entities_created = 0
    entities_reinforced = 0

    # Step 4: entity resolution + write
    for entity in extraction.entities:
        # Coerce LLM-extracted entity type to the canonical vocab.
        canonical_entity_type, _etype_score = coerce_entity_type(entity.type)
        etype_subtype: str | None = entity.type if canonical_entity_type != entity.type else None

        vector = encoder.encode(f"{entity.name} ({canonical_entity_type})")
        canonical_id, is_new, sim = await resolver.resolve_entity(
            name=entity.name,
            entity_type=canonical_entity_type,
            vector=vector,
            source_doc=title,
        )
        if is_new:
            canonical_id = await neo4j_store.merge_entity(
                name=entity.name,
                entity_type=canonical_entity_type,
                source_doc=title,
                confidence=entity.confidence,
                doc_element_id=doc_id,
                model=settings.llm_model,
                session_id=session_id,
                turn_id=turn_id,
                subtype=etype_subtype,
            )
            await qdrant_store.upsert_entity(
                neo4j_element_id=canonical_id,
                name=entity.name,
                entity_type=canonical_entity_type,
                source_doc=title,
                timestamp=now,
                vector=vector,
            )
            entities_created += 1
        else:
            # Resolved to an existing canonical entity — still record that
            # this document mentions it, so provenance stays complete.
            await neo4j_store.link_entity_to_doc(
                entity_element_id=canonical_id,
                doc_element_id=doc_id,
                model=settings.llm_model,
            )
            entities_reinforced += 1

        # Tag every entity touched by this ingest with a :MENTIONED_IN edge
        # when conversation context is available.
        if turn_element_id is not None:
            await neo4j_store.link_entity_to_turn(canonical_id, turn_element_id)

    # Step 5: relation upsert with supersession
    relations_created = 0
    relations_reinforced = 0
    relations_superseded = 0
    for relation in extraction.relations:
        canonical_rel_type, _coerce_score = coerce_rel_type(relation.relation_type)
        outcome, _ = await neo4j_store.upsert_relation(
            subject_name=relation.subject,
            object_name=relation.object,
            relation_type=canonical_rel_type,
            confidence=relation.confidence,
            source_doc=title,
            session_id=session_id,
            turn_id=turn_id,
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
