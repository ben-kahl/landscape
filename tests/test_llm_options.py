import json
from types import SimpleNamespace

import pytest

from landscape.config import LLMProfile
from landscape.extraction import llm

pytestmark = pytest.mark.retrieval


class RecordingClient:
    calls: list[dict] = []

    def __init__(self, host: str):
        self.host = host

    def chat(self, **kwargs):
        self.calls.append(
            {
                "model": kwargs.get("model"),
                "think": kwargs.get("think", "MISSING"),
            }
        )
        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps({"entities": [], "relations": []})
            )
        )


def test_extract_passes_think_false_for_no_think_profile(monkeypatch):
    RecordingClient.calls = []
    monkeypatch.setattr(llm.ollama, "Client", RecordingClient)
    monkeypatch.setitem(
        llm.LLM_PROFILES,
        "test_no_think",
        LLMProfile(ollama_tag="qwen3:14b", thinking=False),
    )
    monkeypatch.setattr(llm.settings, "llm_profile", "test_no_think")
    monkeypatch.setattr(llm.settings, "llm_model", "qwen3:14b")

    llm.extract("Maya leads the Platform Team.")

    assert RecordingClient.calls[0]["think"] is False


def test_extract_passes_think_true_for_thinking_profile(monkeypatch):
    RecordingClient.calls = []
    monkeypatch.setattr(llm.ollama, "Client", RecordingClient)
    monkeypatch.setitem(
        llm.LLM_PROFILES,
        "test_thinking",
        LLMProfile(ollama_tag="qwen3:14b", thinking=True),
    )
    monkeypatch.setattr(llm.settings, "llm_profile", "test_thinking")
    monkeypatch.setattr(llm.settings, "llm_model", "qwen3:14b")

    llm.extract("Maya leads the Platform Team.")

    assert RecordingClient.calls[0]["think"] is True
