"""Compatibility shim for the retired standalone MCP runtime."""


def main() -> None:
    raise RuntimeError(
        "Standalone MCP server entrypoint has been removed. "
        "Run the FastAPI app and connect to /mcp instead."
    )
