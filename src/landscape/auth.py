"""Auth context type shared by the API surface, MCP server, and CLI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthContext:
    """The authenticated identity attached to a request."""

    client_id: str
    client_name: str
    token_id: str
    scopes: frozenset[str]
