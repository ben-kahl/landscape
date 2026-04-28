"""Request-time auth resolution for the FastAPI surface and MCP transport.

The FastAPI side uses ``Depends`` chains: ``resolve_request_auth`` produces an
``AuthContext`` (or 401), and ``require_scope`` checks that context for a
specific scope (or 403).

The MCP transport: FastMCP's ``AuthenticationMiddleware`` + ``BearerAuthBackend``
validates the bearer by calling ``LandscapeOAuthProvider.load_access_token`` and
stores the result in ``scope["user"]`` as an ``AuthenticatedUser`` instance.
``mcp_oauth_scope_middleware`` then reads ``scope["user"]`` and populates
``_CURRENT_AUTH_CONTEXT`` so per-tool scope checks work unchanged.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser

from landscape.auth import AuthContext
from landscape.storage import auth_store

bearer_scheme = HTTPBearer(auto_error=False)

_CURRENT_AUTH_CONTEXT: ContextVar[AuthContext | None] = ContextVar(
    "landscape_auth_context",
    default=None,
)


async def resolve_request_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> AuthContext:
    """FastAPI dependency: validate OAuth bearer, return AuthContext or 401."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    row = await auth_store.load_oauth_token_by_access(credentials.credentials)
    if row is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return AuthContext(
        client_id=row["client_id"],
        client_name=row["client_name"],
        token_id=row["token_id"],
        scopes=frozenset(row["scopes"]),
    )


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


def mcp_oauth_scope_middleware(asgi_app):
    """Wrap an MCP route's inner ASGI app to populate the auth ContextVar.

    FastMCP's AuthenticationMiddleware + BearerAuthBackend runs on the outer
    Starlette app and validates the bearer via ``load_access_token``, storing
    the result in ``scope["user"]`` as an ``AuthenticatedUser``. By the time
    this middleware runs, the bearer is already validated. We just translate
    it into our ``AuthContext`` ContextVar so tool code can call
    ``require_current_scope``.

    Non-HTTP scopes (lifespan, websocket) pass through untouched.
    """
    async def _wrapped(scope, receive, send):
        if scope.get("type") != "http":
            await asgi_app(scope, receive, send)
            return

        user = scope.get("user")
        if isinstance(user, AuthenticatedUser):
            access_token = user.access_token
            auth = AuthContext(
                client_id=access_token.client_id,
                client_name=getattr(access_token, "client_name", access_token.client_id),
                token_id=getattr(access_token, "token_id", ""),
                scopes=frozenset(access_token.scopes),
            )
            ctx_token = _CURRENT_AUTH_CONTEXT.set(auth)
            try:
                await asgi_app(scope, receive, send)
            finally:
                _CURRENT_AUTH_CONTEXT.reset(ctx_token)
        else:
            await asgi_app(scope, receive, send)

    return _wrapped
