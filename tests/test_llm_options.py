from types import SimpleNamespace

import pytest

from landscape.config import LLMProfile
from landscape.extraction import llm

pytestmark = pytest.mark.unit


def _fake_client(recorder: list[dict]):
    def create(**kwargs):
        recorder.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"entities":[],"relations":[]}'))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_extract_uses_profile_model_and_json_schema(monkeypatch):
    recorder: list[dict] = []
    monkeypatch.setattr(llm.llm_client, "get_client", lambda: _fake_client(recorder))
    monkeypatch.setattr(
        llm.llm_client, "active_profile",
        lambda: LLMProfile(base_url="http://x/v1", model="qwen-test", no_think=False),
    )

    result = llm.extract("Maya leads the Platform Team.")

    assert result.entities == []
    assert recorder[0]["model"] == "qwen-test"
    assert recorder[0]["response_format"]["json_schema"]["name"] == "Extraction"


def test_extract_prepends_no_think_when_profile_requests_it(monkeypatch):
    recorder: list[dict] = []
    monkeypatch.setattr(llm.llm_client, "get_client", lambda: _fake_client(recorder))
    monkeypatch.setattr(
        llm.llm_client, "active_profile",
        lambda: LLMProfile(base_url="http://x/v1", model="m", no_think=True),
    )

    llm.extract("Maya leads the Platform Team.")

    assert recorder[0]["messages"][0]["content"].startswith("/no_think\n")
