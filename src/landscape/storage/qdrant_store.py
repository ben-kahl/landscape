import asyncio
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
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


async def _wait_for_collection_ready(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.1,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_exc: Exception | None = None
    while True:
        try:
            await client.get_collection(collection_name)
            return
        except Exception as exc:  # pragma: no cover - exercised through init callers
            last_exc = exc
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for Qdrant collection {collection_name!r} to become ready"
            ) from last_exc
        await asyncio.sleep(poll_interval_s)


async def _ensure_collection(
    collection_name: str,
) -> None:
    client = get_client()
    try:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=_dims(), distance=Distance.COSINE),
        )
    except UnexpectedResponse as exc:
        if exc.status_code != 409:
            raise
    await _wait_for_collection_ready(client, collection_name)


async def init_collection() -> None:
    await _ensure_collection(COLLECTION)


async def init_chunks_collection() -> None:
    await _ensure_collection(CHUNKS_COLLECTION)


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
    chunk_id: str,
    doc_id: str,
    source_doc: str,
    position: int,
    text: str,
    vector: list[float],
) -> None:
    client = get_client()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
    await client.upsert(
        collection_name=CHUNKS_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_id": chunk_id,
                    "chunk_neo4j_id": chunk_id,
                    "doc_id": doc_id,
                    "source_doc": source_doc,
                    "position": position,
                    "text": text,
                },
            )
        ],
    )
