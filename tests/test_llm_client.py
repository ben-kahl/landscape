from types import SimpleNamespace

import pytest

from landscape.config import LLMProfile
from landscape.extraction import llm_client
from landscape.extraction.schema import Extraction

pytestmark = pytest.mark.unit


def _fake_client(content: str, recorder: list[dict]):
    def create(**kwargs):
        recorder.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _fake_client_seq(contents: list[str], recorder: list[dict]):
    def create(**kwargs):
        recorder.append(kwargs)
        content = contents[min(len(recorder) - 1, len(contents) - 1)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_retry_feeds_validation_error_back(monkeypatch):
    recorder: list[dict] = []
    monkeypatch.setattr(
        llm_client,
        "get_client",
        lambda: _fake_client_seq(["not json", '{"entities":[],"relations":[]}'], recorder),
    )
    monkeypatch.setattr(
        llm_client, "active_profile",
        lambda: LLMProfile(base_url="http://x/v1", model="m", no_think=False),
    )

    result = llm_client.complete_structured(
        "p", model_cls=Extraction, schema_name="Extraction"
    )

    assert isinstance(result, Extraction)
    assert len(recorder) == 2
    retry_msgs = recorder[1]["messages"]
    assert retry_msgs[0] == {"role": "user", "content": "p"}
    assert retry_msgs[1]["role"] == "assistant"
    assert retry_msgs[1]["content"] == "not json"
    assert retry_msgs[2]["role"] == "user"
    assert "failed validation" in retry_msgs[2]["content"]


def test_resolve_key_local_uses_placeholder():
    p = LLMProfile(base_url="http://x/v1", model="m", api_key_env=None)
    assert llm_client.resolve_key(p) == "sk-noauth"


def test_resolve_key_cloud_reads_named_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-real")
    p = LLMProfile(base_url="http://x/v1", model="m", api_key_env="MY_KEY")
    assert llm_client.resolve_key(p) == "sk-real"


def test_resolve_key_missing_cloud_key_raises(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    p = LLMProfile(base_url="http://x/v1", model="m", api_key_env="MISSING_KEY")
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        llm_client.resolve_key(p)


def test_complete_structured_sends_json_schema_and_validates(monkeypatch):
    recorder: list[dict] = []
    monkeypatch.setattr(
        llm_client, "get_client", lambda: _fake_client('{"entities":[],"relations":[]}', recorder)
    )
    monkeypatch.setattr(
        llm_client, "active_profile",
        lambda: LLMProfile(base_url="http://x/v1", model="m", no_think=False),
    )

    result = llm_client.complete_structured(
        "prompt text", model_cls=Extraction, schema_name="Extraction"
    )

    assert isinstance(result, Extraction)
    rf = recorder[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Extraction"
    assert recorder[0]["messages"][0]["content"] == "prompt text"


def test_complete_structured_disables_thinking_via_extra_body(monkeypatch):
    recorder: list[dict] = []
    monkeypatch.setattr(
        llm_client, "get_client", lambda: _fake_client('{"entities":[],"relations":[]}', recorder)
    )
    monkeypatch.setattr(
        llm_client, "active_profile",
        lambda: LLMProfile(base_url="http://x/v1", model="m", no_think=True),
    )

    llm_client.complete_structured("prompt", model_cls=Extraction, schema_name="Extraction")

    # The prompt is sent verbatim — no "/no_think" prefix (a no-op in llama.cpp).
    assert recorder[0]["messages"][0]["content"] == "prompt"
    assert recorder[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_complete_structured_caps_tokens_and_sends_repeat_penalty(monkeypatch):
    recorder: list[dict] = []
    monkeypatch.setattr(
        llm_client, "get_client", lambda: _fake_client('{"entities":[],"relations":[]}', recorder)
    )
    monkeypatch.setattr(
        llm_client, "active_profile",
        lambda: LLMProfile(
            base_url="http://x/v1", model="m", max_tokens=4096, repeat_penalty=1.1
        ),
    )

    llm_client.complete_structured("prompt", model_cls=Extraction, schema_name="Extraction")

    assert recorder[0]["max_tokens"] == 4096
    assert recorder[0]["extra_body"]["repeat_penalty"] == 1.1


def test_complete_structured_retries_once_then_raises(monkeypatch):
    recorder: list[dict] = []
    monkeypatch.setattr(
        llm_client, "get_client", lambda: _fake_client("not json", recorder)
    )
    monkeypatch.setattr(
        llm_client, "active_profile",
        lambda: LLMProfile(base_url="http://x/v1", model="m", no_think=False),
    )

    with pytest.raises(ValueError):
        llm_client.complete_structured("p", model_cls=Extraction, schema_name="Extraction")
    assert len(recorder) == 2  # one retry


def test_get_client_uses_base_url_override(monkeypatch):
    from landscape.config import settings
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        llm_client, "active_profile",
        lambda: LLMProfile(base_url="http://llama-server:8080/v1", model="m", api_key_env=None),
    )
    monkeypatch.setattr(settings, "llm_base_url", "http://localhost:8080/v1")

    llm_client.get_client()
    assert captured["base_url"] == "http://localhost:8080/v1"


def test_get_client_falls_back_to_profile_base_url(monkeypatch):
    from landscape.config import settings
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        llm_client, "active_profile",
        lambda: LLMProfile(base_url="http://llama-server:8080/v1", model="m", api_key_env=None),
    )
    monkeypatch.setattr(settings, "llm_base_url", None)

    llm_client.get_client()
    assert captured["base_url"] == "http://llama-server:8080/v1"


def test_token_totals_accumulate_and_reset(monkeypatch):
    recorder: list[dict] = []
    monkeypatch.setattr(
        llm_client, "get_client", lambda: _fake_client('{"entities":[],"relations":[]}', recorder)
    )
    monkeypatch.setattr(
        llm_client, "active_profile",
        lambda: LLMProfile(base_url="http://x/v1", model="m", no_think=False),
    )
    llm_client.reset_token_totals()
    llm_client.complete_structured("p", model_cls=Extraction, schema_name="Extraction")
    llm_client.complete_structured("p", model_cls=Extraction, schema_name="Extraction")
    totals = llm_client.get_token_totals()
    assert totals["prompt_tokens"] == 6   # _fake_client returns usage 3/2 per call
    assert totals["completion_tokens"] == 4
    llm_client.reset_token_totals()
    assert llm_client.get_token_totals() == {"prompt_tokens": 0, "completion_tokens": 0}
