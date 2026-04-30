from __future__ import annotations

import threading
from datetime import UTC, datetime

import tiktoken
from fastapi import APIRouter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_ENCODING = tiktoken.get_encoding("cl100k_base")
_STARTUP_TIME = datetime.now(UTC).isoformat()

_MONITORED_PATHS = {"/query", "/ingest"}

_lock = threading.Lock()
_usage: dict[str, dict[str, int]] = {}
_ollama: dict[str, int] = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
}


class TokenCounterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path not in _MONITORED_PATHS:
            return response

        chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        body = b"".join(chunks)

        n_tokens = len(_ENCODING.encode(body.decode("utf-8", errors="replace")))

        with _lock:
            ep = _usage.setdefault(
                path, {"request_count": 0, "total_response_tokens": 0}
            )
            ep["request_count"] += 1
            ep["total_response_tokens"] += n_tokens

        headers = dict(response.headers)
        headers["x-response-tokens"] = str(n_tokens)

        return Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )


def increment_ollama_tokens(*, prompt_tokens: int, completion_tokens: int) -> None:
    with _lock:
        _ollama["total_prompt_tokens"] += prompt_tokens
        _ollama["total_completion_tokens"] += completion_tokens


def get_usage() -> dict:
    with _lock:
        endpoints: dict[str, dict] = {}
        for path, data in _usage.items():
            count = data["request_count"]
            total = data["total_response_tokens"]
            endpoints[path] = {
                "request_count": count,
                "total_response_tokens": total,
                "avg_response_tokens": round(total / count, 1) if count else 0.0,
            }
        return {
            "since": _STARTUP_TIME,
            "endpoints": endpoints,
            "ollama": dict(_ollama),
        }


def reset_counters() -> None:
    with _lock:
        _usage.clear()
        _ollama["total_prompt_tokens"] = 0
        _ollama["total_completion_tokens"] = 0


metrics_router = APIRouter()


@metrics_router.get("/metrics/token-usage")
async def token_usage():
    return get_usage()
