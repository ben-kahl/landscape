"""Lazy Weave (W&B) tracing for the extraction pipeline.

`traced` is a no-op decorator unless `settings.weave_project` is set AND the
`weave` package is installed. Keeps Weave a strictly optional dep.

Enable with:
    LANDSCAPE_WEAVE_PROJECT=landscape-extraction uv run ...
"""

from __future__ import annotations

import logging
from functools import wraps
from threading import Lock
from typing import Any, Callable, TypeVar

from landscape.config import settings

_log = logging.getLogger(__name__)
_F = TypeVar("_F", bound=Callable[..., Any])

_init_lock = Lock()
_init_state: dict[str, Any] = {"done": False, "weave": None}


def _get_weave() -> Any | None:
    if _init_state["done"]:
        return _init_state["weave"]
    with _init_lock:
        if _init_state["done"]:
            return _init_state["weave"]
        _init_state["done"] = True
        project = settings.weave_project
        if not project:
            return None
        try:
            import weave  # type: ignore[import-not-found]
        except ImportError:
            _log.warning(
                "LANDSCAPE_WEAVE_PROJECT=%r set but `weave` is not installed. "
                "Install with: uv sync --extra observability",
                project,
            )
            return None
        try:
            weave.init(project)
        except Exception as exc:  # noqa: BLE001 — auth/network failures must not break extraction
            _log.warning("weave.init(%r) failed: %s — tracing disabled", project, exc)
            return None
        _init_state["weave"] = weave
        _log.info("Weave tracing enabled for project %r", project)
        return weave


def traced(name: str | None = None) -> Callable[[_F], _F]:
    """Record this call in Weave if configured, else no-op. Lazy on first call."""

    def decorator(fn: _F) -> _F:
        wrapped: Callable[..., Any] | None = None

        @wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            nonlocal wrapped
            weave = _get_weave()
            if weave is None:
                return fn(*args, **kwargs)
            if wrapped is None:
                wrapped = weave.op(name=name)(fn) if name else weave.op()(fn)
            return wrapped(*args, **kwargs)

        return inner  # type: ignore[return-value]

    return decorator
