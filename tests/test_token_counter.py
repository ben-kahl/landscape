import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def token_app():
    from landscape.middleware.token_counter import (
        TokenCounterMiddleware,
        reset_counters,
    )

    app = FastAPI()
    app.add_middleware(TokenCounterMiddleware)

    @app.post("/query")
    async def _query():
        return {"results": [{"name": "Alice"}]}

    @app.post("/ingest")
    async def _ingest():
        return {"doc_id": "test-doc"}

    @app.get("/healthz")
    async def _health():
        return {"status": "ok"}

    reset_counters()
    return app


@pytest.mark.unit
async def test_x_response_tokens_header_on_query(token_app):
    from landscape.middleware.token_counter import reset_counters

    reset_counters()
    async with AsyncClient(
        transport=ASGITransport(app=token_app), base_url="http://test"
    ) as c:
        r = await c.post("/query", json={})
    assert r.status_code == 200
    assert "x-response-tokens" in r.headers
    assert int(r.headers["x-response-tokens"]) > 0


@pytest.mark.unit
async def test_x_response_tokens_header_on_ingest(token_app):
    from landscape.middleware.token_counter import reset_counters

    reset_counters()
    async with AsyncClient(
        transport=ASGITransport(app=token_app), base_url="http://test"
    ) as c:
        r = await c.post("/ingest", json={})
    assert "x-response-tokens" in r.headers


@pytest.mark.unit
async def test_no_header_on_unmonitored_path(token_app):
    async with AsyncClient(
        transport=ASGITransport(app=token_app), base_url="http://test"
    ) as c:
        r = await c.get("/healthz")
    assert "x-response-tokens" not in r.headers


@pytest.mark.unit
async def test_accumulator_increments_per_request(token_app):
    from landscape.middleware.token_counter import get_usage, reset_counters

    reset_counters()
    async with AsyncClient(
        transport=ASGITransport(app=token_app), base_url="http://test"
    ) as c:
        await c.post("/query", json={})
        await c.post("/query", json={})
    usage = get_usage()
    assert usage["endpoints"]["/query"]["request_count"] == 2
    assert usage["endpoints"]["/query"]["total_response_tokens"] > 0


@pytest.mark.unit
async def test_metrics_endpoint_structure(token_app):
    from landscape.middleware.token_counter import metrics_router, reset_counters

    token_app.include_router(metrics_router)
    reset_counters()
    async with AsyncClient(
        transport=ASGITransport(app=token_app), base_url="http://test"
    ) as c:
        await c.post("/query", json={})
        r = await c.get("/metrics/token-usage")
    assert r.status_code == 200
    body = r.json()
    assert "since" in body
    assert "endpoints" in body
    assert "ollama" in body
    ep = body["endpoints"]["/query"]
    assert ep["request_count"] == 1
    assert "avg_response_tokens" in ep


@pytest.mark.unit
async def test_increment_ollama_tokens():
    from landscape.middleware.token_counter import (
        get_usage,
        increment_ollama_tokens,
        reset_counters,
    )

    reset_counters()
    increment_ollama_tokens(prompt_tokens=100, completion_tokens=50)
    usage = get_usage()
    assert usage["ollama"]["total_prompt_tokens"] == 100
    assert usage["ollama"]["total_completion_tokens"] == 50


@pytest.mark.unit
async def test_reset_clears_all_state():
    from landscape.middleware.token_counter import (
        get_usage,
        increment_ollama_tokens,
        reset_counters,
    )

    increment_ollama_tokens(prompt_tokens=999, completion_tokens=999)
    reset_counters()
    usage = get_usage()
    assert usage["endpoints"] == {}
    assert usage["ollama"]["total_prompt_tokens"] == 0
