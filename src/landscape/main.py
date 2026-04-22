from contextlib import asynccontextmanager

from fastapi import FastAPI

from landscape.api.ingest import router as ingest_router
from landscape.api.query import router as query_router
from landscape.embeddings import encoder
from landscape.mcp_app import mcp
from landscape.storage import neo4j_store, qdrant_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    encoder.load_model()
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()
    yield
    await neo4j_store.close_driver()
    await qdrant_store.close_client()


app = FastAPI(title="Landscape", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(query_router)
app.mount("/mcp", mcp.streamable_http_app())


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
