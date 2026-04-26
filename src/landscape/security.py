"""Request-time auth resolution for the FastAPI surface and MCP transport.

The FastAPI side uses ``Depends`` chains: ``resolve_request_auth`` produces an
``AuthContext`` (or 401), and ``require_scope`` checks that context for a
specific scope (or 403).

The MCP transport is ASGI-only -- FastAPI dependency injection does not run
there. We resolve auth in custom ASGI middleware and stash the principal in a
``ContextVar`` so per-tool calls can pull it via ``require_current_scope`` and
fail with ``ValueError("Forbidden")`` when the caller is missing scope.

TRANSITIONAL SECURITY BYPASS: when ``settings.allow_unauthenticated_loopback``
is true and the request originates from a loopback interface with no bearer
credential, we synthesize a principal limited to the ``agent`` scope. This is
strictly a developer-laptop convenience and must be disabled for any remote
or cloud deployment. The bypass principal is intentionally NOT granted
``graph_query`` -- raw Cypher requires a real authenticated client.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from landscape.auth import AuthContext
from landscape.config import settings
from landscape.storage.auth_store import authenticate_bearer_token

bearer_scheme = HTTPBearer(auto_error=False)

_CURRENT_AUTH_CONTEXT: ContextVar[AuthContext | None] = ContextVar(
    "landscape_auth_context",
    default=None,
)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback_request(request: Request) -> bool:
    client = request.scope.get("client") if request.scope else None
    if client is None and request.client is not None:
        client = (request.client.host, request.client.port)
    if not client:
        return False
    host = client[0]
    return host in _LOOPBACK_HOSTS


def _loopback_bypass_context() -> AuthContext:
    return AuthContext(
        client_id="loopback-anonymous",
        client_name="loopback-anonymous",
        secret_id=None,
        scopes=frozenset({"agent"}),
        is_loopback_bypass=True,
    )


async def resolve_request_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> AuthContext:
    if credentials is None:
        if settings.allow_unauthenticated_loopback and _is_loopback_request(request):
            return _loopback_bypass_context()
        raise HTTPException(status_code=401, detail="Unauthorized")
    auth = await authenticate_bearer_token(credentials.credentials)
    if auth is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return auth


def require_scope(*required: str):
    """Build a FastAPI dependency that asserts every named scope is granted."""
    needed = frozenset(required)

    async def _dependency(
        auth: AuthContext = Depends(resolve_request_auth),
    ) -> AuthContext:
        if not needed.issubset(auth.scopes):
            raise HTTPException(status_code=403, detail="Forbidden")
        return auth

    return _dependency


_require_agent_scope = require_scope("agent")

AgentPrincipal = Annotated[AuthContext, Depends(_require_agent_scope)]


def require_current_scope(scope: str) -> AuthContext:
    """MCP-tool helper: read the request-scoped principal and check ``scope``.

    Raises ``ValueError("Forbidden")`` if no principal is set or the scope
    is missing. FastMCP turns the ValueError into an ``isError`` tool result.
    """
    auth = _CURRENT_AUTH_CONTEXT.get()
    if auth is None or scope not in auth.scopes:
        raise ValueError("Forbidden")
    return auth


async def _resolve_asgi_auth(scope: dict) -> AuthContext:
    """Resolve auth for an MCP/Starlette request without FastAPI's Depends graph.

    The ASGI middleware path runs outside the FastAPI router, so we manually
    crack the Authorization header out of ``scope['headers']`` and reuse the
    same loopback / bearer rules as ``resolve_request_auth``.
    """
    headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }
    raw = headers.get("authorization")
    bearer_token: str | None = None
    if raw and raw.lower().startswith("bearer "):
        bearer_token = raw.split(" ", 1)[1].strip() or None

    if bearer_token is None:
        client = scope.get("client")
        host = client[0] if client else None
        if settings.allow_unauthenticated_loopback and host in _LOOPBACK_HOSTS:
            return _loopback_bypass_context()
        raise PermissionError("Unauthorized")

    auth = await authenticate_bearer_token(bearer_token)
    if auth is None:
        raise PermissionError("Unauthorized")
    return auth


def mcp_auth_middleware(asgi_app):
    """Wrap an ASGI app so each HTTP request resolves auth into the ContextVar.

    Non-HTTP scopes (lifespan, websocket) pass through untouched. HTTP scopes
    that fail auth get a 401 response and never reach the wrapped app.
    """

    async def _wrapped(scope, receive, send):
        if scope.get("type") != "http":
            await asgi_app(scope, receive, send)
            return

        try:
            auth = await _resolve_asgi_auth(scope)
        except PermissionError:
            await _send_plain_response(send, 401, b"Unauthorized")
            return

        token = _CURRENT_AUTH_CONTEXT.set(auth)
        try:
            await asgi_app(scope, receive, send)
        finally:
            _CURRENT_AUTH_CONTEXT.reset(token)

    return _wrapped


async def _send_plain_response(send, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
