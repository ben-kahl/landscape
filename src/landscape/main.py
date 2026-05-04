import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend
from starlette.middleware.authentication import AuthenticationMiddleware

from landscape.api.ingest import router as ingest_router
from landscape.api.query import router as query_router
from landscape.embeddings import encoder
from landscape.mcp_app import mcp
from landscape.middleware.token_counter import TokenCounterMiddleware, metrics_router
from landscape.security import mcp_oauth_scope_middleware
from landscape.storage import auth_store, neo4j_store, qdrant_store

logger = logging.getLogger(__name__)

mcp_http_app = mcp.streamable_http_app()


def _find_streamable_app(http_app):
    """Return the StreamableHTTPASGIApp that owns the session_manager.

    With OAuth enabled FastMCP adds OAuth middleware routes ahead of /mcp, and
    wraps the /mcp endpoint in RequireAuthMiddleware, so routes[0] no longer
    points directly at the StreamableHTTPASGIApp. Search by path and walk one
    level of middleware instead of using a hardcoded index.
    """
    for route in http_app.routes:
        if getattr(route, "path", None) != "/mcp":
            continue
        ep = getattr(route, "endpoint", None)
        if ep is None:
            continue
        if hasattr(ep, "session_manager"):
            return ep
        inner = getattr(ep, "app", None)
        if inner is not None and hasattr(inner, "session_manager"):
            return inner
    raise RuntimeError("Cannot locate StreamableHTTPASGIApp in MCP routes")


def _refresh_mcp_http_session_manager() -> None:
    """Refresh the mounted MCP session manager after a completed lifespan.

    The MCP SDK's streamable HTTP session manager is a one-run object. Uvicorn
    only runs the app lifespan once, but tests enter the FastAPI lifespan many
    times in-process.
    """
    mcp._session_manager = None
    fresh_mcp_http_app = mcp.streamable_http_app()
    _find_streamable_app(mcp_http_app).session_manager = (
        _find_streamable_app(fresh_mcp_http_app).session_manager
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

app.add_middleware(TokenCounterMiddleware)
app.include_router(metrics_router)

app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# FastMCP's streamable_http_app() wraps its Starlette app with
# AuthenticationMiddleware so scope["user"] is set before RequireAuthMiddleware
# checks it. By extracting routes and appending them to FastAPI we lose that
# outer middleware layer. Re-add it here so bearer validation populates
# scope["user"] for every request, including /mcp.
if mcp._token_verifier is not None:
    app.add_middleware(AuthenticationMiddleware, backend=BearerAuthBackend(mcp._token_verifier))


_mcp_path = mcp.settings.streamable_http_path  # "/mcp"

for _mcp_route in mcp_http_app.routes:
    # Only wrap the /mcp transport route with our ContextVar middleware.
    # OAuth protocol routes (/register, /authorize, /token, /revoke) don't
    # need it and wrapping them is unnecessary overhead.
    if getattr(_mcp_route, "path", None) == _mcp_path and hasattr(_mcp_route, "app"):
        _mcp_route.app = mcp_oauth_scope_middleware(_mcp_route.app)
    app.router.routes.append(_mcp_route)
