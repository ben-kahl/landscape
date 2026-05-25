import json
from types import SimpleNamespace

import pytest

from landscape.extraction import llm
from landscape.observability import weave_tracing

pytestmark = pytest.mark.unit


def test_attach_usage_writes_weave_usage_shape():
    """_attach_usage_to_call writes the per-model usage block Weave's trace
    server reads (summary['usage'][model] with prompt/completion/total)."""
    call = SimpleNamespace(summary={})

    weave_tracing._attach_usage_to_call(
        call, "llama3.1:8b", prompt_tokens=120, completion_tokens=45
    )

    assert call.summary["usage"] == {
        "llama3.1:8b": {
            "requests": 1,
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
        }
    }


def test_record_token_usage_is_noop_when_weave_disabled(monkeypatch):
    """With Weave disabled, record_token_usage must not raise."""
    monkeypatch.setattr(weave_tracing, "_get_weave", lambda: None)

    weave_tracing.record_token_usage(
        "llama3.1:8b", prompt_tokens=10, completion_tokens=5
    )


def test_record_token_usage_is_noop_outside_traced_op(monkeypatch):
    """With no active call, record_token_usage must not raise."""
    fake_weave = SimpleNamespace(get_current_call=lambda: None)
    monkeypatch.setattr(weave_tracing, "_get_weave", lambda: fake_weave)

    weave_tracing.record_token_usage(
        "llama3.1:8b", prompt_tokens=10, completion_tokens=5
    )


def test_record_token_usage_attaches_to_active_call(monkeypatch):
    """record_token_usage routes counts onto the current Weave call."""
    call = SimpleNamespace(summary={})
    fake_weave = SimpleNamespace(get_current_call=lambda: call)
    monkeypatch.setattr(weave_tracing, "_get_weave", lambda: fake_weave)

    weave_tracing.record_token_usage(
        "llama3.1:8b", prompt_tokens=7, completion_tokens=3
    )

    assert call.summary["usage"]["llama3.1:8b"]["total_tokens"] == 10


class _UsageReportingClient:
    """Ollama client stub that returns Ollama-style token counts."""

    def __init__(self, host: str):
        self.host = host

    def chat(self, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps({"entities": [], "relations": []})
            ),
            prompt_eval_count=120,
            eval_count=45,
        )


def test_extract_records_token_usage_to_weave(monkeypatch):
    """extract() forwards the Ollama response token counts to Weave."""
    recorded: list[tuple[str, int, int]] = []

    def _fake_record(model, *, prompt_tokens, completion_tokens):
        recorded.append((model, prompt_tokens, completion_tokens))

    monkeypatch.setattr(llm.ollama, "Client", _UsageReportingClient)
    monkeypatch.setattr(llm, "record_token_usage", _fake_record)
    monkeypatch.setattr(llm.settings, "llm_model", "llama3.1:8b")

    llm.extract("Maya leads the Platform Team.")

    assert recorded == [("llama3.1:8b", 120, 45)]
