"""Compatibility shim for the legacy stdio MCP entry point."""

from __future__ import annotations

from landscape.mcp_app import mcp


def main() -> None:
    raise RuntimeError(
        "landscape.mcp_server is deprecated. Run the FastAPI app and connect to /mcp instead."
    )
