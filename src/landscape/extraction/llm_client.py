from __future__ import annotations

import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from landscape.config import LLM_PROFILES, LLMProfile, settings

_T = TypeVar("_T", bound=BaseModel)

_token_totals = {"prompt_tokens": 0, "completion_tokens": 0}


def get_token_totals() -> dict[str, int]:
    return dict(_token_totals)


def reset_token_totals() -> None:
    _token_totals["prompt_tokens"] = 0
    _token_totals["completion_tokens"] = 0


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
    base_url = settings.llm_base_url or p.base_url
    # max_retries=0: complete_structured does its own bounded retry. The SDK
    # default (2 silent retries) turned one slow call into a 3x-timeout storm
    # that, behind llama-server's --parallel 1, never converged.
    return OpenAI(
        base_url=base_url,
        api_key=resolve_key(p),
        timeout=p.request_timeout,
        max_retries=0,
    )


def _messages(prompt: str) -> list[dict]:
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
    # llama.cpp extensions ride in extra_body. enable_thinking=False is the actual
    # fix for the hang: with thinking on, the model emits its whole budget into the
    # reasoning channel and returns empty content. repeat_penalty guards greedy
    # decoding against runaway loops. Cloud profiles send neither.
    extra_body: dict[str, object] = {}
    if p.no_think:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if p.repeat_penalty is not None:
        extra_body["repeat_penalty"] = p.repeat_penalty
    messages = _messages(prompt)
    last_err: Exception | None = None
    for _ in range(2):
        resp = client.chat.completions.create(
            model=p.model,
            messages=messages,
            temperature=p.temperature,
            max_tokens=p.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": model_cls.model_json_schema(),
                    "strict": True,
                },
            },
            extra_body=extra_body or None,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            _token_totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            _token_totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        content = resp.choices[0].message.content or ""
        try:
            return model_cls.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            last_err = exc
            # Informed retry: show the model its failed output and the error
            # instead of blind-resending the same prompt.
            messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous output failed validation: "
                        f"{str(exc)[:2000]}\n"
                        "Return corrected JSON matching the schema exactly."
                    ),
                },
            ]
    raise last_err  # type: ignore[misc]
