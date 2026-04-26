import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from landscape.api.ingest import router as ingest_router
from landscape.api.query import router as query_router
from landscape.config import settings
from landscape.embeddings import encoder
from landscape.mcp_app import mcp
from landscape.security import mcp_auth_middleware
from landscape.storage import auth_store, neo4j_store, qdrant_store

logger = logging.getLogger(__name__)

_LOOPBACK_BIND_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})


def _enforce_loopback_bypass_safety(host: str, bypass_enabled: bool) -> None:
    """Refuse to start when the dev-only loopback bypass is paired with a
    non-loopback bind host. Pure helper so tests can drive it directly."""
    if not bypass_enabled:
        return
    if host in _LOOPBACK_BIND_HOSTS:
        return
    raise RuntimeError(
        "Refusing to start: allow_unauthenticated_loopback=True is dev-only "
        f"and incompatible with non-loopback bind host '{host}'. "
        "Set LANDSCAPE_ALLOW_UNAUTHENTICATED_LOOPBACK=false or bind to 127.0.0.1."
    )


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
    _enforce_loopback_bypass_safety(
        settings.api_host, settings.allow_unauthenticated_loopback
    )
    encoder.load_model()
    await auth_store.ensure_schema()
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()
    if settings.allow_unauthenticated_loopback:
        logger.warning(
            "TRANSITIONAL SECURITY BYPASS ENABLED: localhost requests may "
            "access agent scope without credentials. Disable "
            "allow_unauthenticated_loopback for remote/cloud deployments."
        )


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


for _mcp_route in mcp_http_app.routes:
    # Wrap each MCP transport route's dispatched ASGI app so per-request auth
    # resolution populates the request-scoped principal before any tool runs.
    # We leave `endpoint` alone so `_refresh_mcp_http_session_manager` can
    # still hot-swap the underlying StreamableHTTPASGIApp's session_manager.
    if hasattr(_mcp_route, "app"):
        _mcp_route.app = mcp_auth_middleware(_mcp_route.app)
    app.router.routes.append(_mcp_route)
