from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from landscape.config import settings

_driver: AsyncDriver | None = None


def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def run_cypher_readonly(
    cypher: str, params: dict | None = None
) -> list[dict]:
    """Execute a read-only Cypher query."""
    from landscape.storage.cypher_guard import assert_read_only

    assert_read_only(cypher)

    driver = get_driver()
    params = params or {}

    async def _work(tx: Any) -> list[dict]:
        result = await tx.run(cypher, **params)
        return [dict(record) async for record in result]

    async with driver.session() as session:
        return await session.execute_read(_work)
