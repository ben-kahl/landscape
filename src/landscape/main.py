import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from landscape.api.ingest import router as ingest_router
from landscape.api.query import router as query_router
from landscape.embeddings import encoder
from landscape.mcp_app import mcp
from landscape.storage import auth_store, neo4j_store, qdrant_store

mcp_http_app = mcp.streamable_http_app()


def _refresh_mcp_http_session_manager() -> None:
    """Refresh the mounted MCP session manager after a completed lifespan.

    The MCP SDK's streamable HTTP session manager is a one-run object. Uvicorn
    only runs the app lifespan once, but tests enter the FastAPI lifespan many
    times in-process.
    """
    mcp._session_manager = None
    fresh_mcp_http_app = mcp.streamable_http_app()
    mcp_http_app.routes[0].endpoint.session_manager = (
        fresh_mcp_http_app.routes[0].endpoint.session_manager
    )


def _should_start_mcp_http_session_manager() -> bool:
    """Avoid MCP HTTP lifespan side effects in pytest's in-process app fixtures."""
    return "PYTEST_CURRENT_TEST" not in os.environ


async def _startup_storage() -> None:
    encoder.load_model()
    await auth_store.ensure_schema()
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()


async def _shutdown_storage() -> None:
    await neo4j_store.close_driver()
    await qdrant_store.close_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_mcp_http = _should_start_mcp_http_session_manager()
    if start_mcp_http:
        _refresh_mcp_http_session_manager()
    try:
        if start_mcp_http:
            async with mcp_http_app.router.lifespan_context(mcp_http_app):
                await _startup_storage()
                yield
                await _shutdown_storage()
        else:
            await _startup_storage()
            yield
            await _shutdown_storage()
    finally:
        if start_mcp_http:
            _refresh_mcp_http_session_manager()


app = FastAPI(title="Landscape", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


app.router.routes.extend(mcp_http_app.routes)
