import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from landscape.config import settings

COLLECTION = "entities"
CHUNKS_COLLECTION = "chunks"


def _dims() -> int:
    return settings.embedding_dims

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def init_collection() -> None:
    client = get_client()
    existing = await client.get_collections()
    names = {c.name for c in existing.collections}
    if COLLECTION not in names:
        await client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=_dims(), distance=Distance.COSINE),
        )


async def init_chunks_collection() -> None:
    client = get_client()
    existing = await client.get_collections()
    names = {c.name for c in existing.collections}
    if CHUNKS_COLLECTION not in names:
        await client.create_collection(
            collection_name=CHUNKS_COLLECTION,
            vectors_config=VectorParams(size=_dims(), distance=Distance.COSINE),
        )


async def upsert_entity(
    neo4j_element_id: str,
    name: str,
    entity_type: str,
    source_doc: str,
    timestamp: str,
    vector: list[float],
) -> None:
    client = get_client()
    # Deterministic UUID so re-ingestion of the same entity doesn't duplicate points
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, neo4j_element_id))
    await client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "neo4j_node_id": neo4j_element_id,
                    "name": name,
                    "type": entity_type,
                    "source_doc": source_doc,
                    "timestamp": timestamp,
                },
            )
        ],
    )


async def search_similar_entities(
    vector: list[float],
    entity_type: str,
    limit: int = 5,
) -> list[ScoredPoint]:
    client = get_client()
    result = await client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=entity_type))]
        ),
        limit=limit,
        with_payload=True,
    )
    return result.points


async def search_entities_any_type(
    vector: list[float],
    limit: int = 10,
) -> list[ScoredPoint]:
    """Entity search without the type filter — for retrieval where the
    caller doesn't know the entity type in advance."""
    client = get_client()
    result = await client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    return result.points


async def search_chunks(
    vector: list[float],
    limit: int = 10,
) -> list[ScoredPoint]:
    client = get_client()
    result = await client.query_points(
        collection_name=CHUNKS_COLLECTION,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    return result.points


async def upsert_chunk(
    chunk_neo4j_id: str,
    doc_id: str,
    source_doc: str,
    position: int,
    text: str,
    vector: list[float],
) -> None:
    client = get_client()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_neo4j_id))
    await client.upsert(
        collection_name=CHUNKS_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_neo4j_id": chunk_neo4j_id,
                    "doc_id": doc_id,
                    "source_doc": source_doc,
                    "position": position,
                    "text": text,
                },
            )
        ],
    )
