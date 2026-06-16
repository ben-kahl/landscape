from __future__ import annotations

import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from landscape.config import LLM_PROFILES, LLMProfile, settings

_T = TypeVar("_T", bound=BaseModel)


def active_profile() -> LLMProfile:
    return LLM_PROFILES[settings.llm_profile]


def resolve_key(profile: LLMProfile) -> str:
    """Cloud profiles read a real key from the named env var; local profiles
    (api_key_env is None) get a placeholder llama-server ignores."""
    if profile.api_key_env is None:
        return "sk-noauth"
    key = os.environ.get(profile.api_key_env)
    if not key:
        raise RuntimeError(
            f"LLM profile {settings.llm_profile!r} needs the API key in "
            f"${profile.api_key_env}, but it is unset or empty."
        )
    return key


def get_client() -> OpenAI:
    p = active_profile()
    return OpenAI(base_url=p.base_url, api_key=resolve_key(p))


def _messages(prompt: str) -> list[dict]:
    if active_profile().no_think:
        prompt = "/no_think\n" + prompt
    return [{"role": "user", "content": prompt}]


def complete_structured(
    prompt: str, *, model_cls: type[_T], schema_name: str
) -> _T:
    """One structured-output call that works against llama-server and cloud.

    Sends the pydantic schema as an OpenAI json_schema response_format
    (grammar-constrained on llama-server, strict on cloud), validates with
    pydantic, and retries once before raising. Token usage is captured
    automatically by weave's native OpenAI integration when tracing is enabled.
    """
    client = get_client()
    p = active_profile()
    last_err: Exception | None = None
    for _ in range(2):
        resp = client.chat.completions.create(
            model=p.model,
            messages=_messages(prompt),
            temperature=p.temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": model_cls.model_json_schema(),
                    "strict": True,
                },
            },
        )
        content = resp.choices[0].message.content or ""
        try:
            return model_cls.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            last_err = exc
    raise last_err  # type: ignore[misc]
