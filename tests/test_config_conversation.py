import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_PROFILE",
        "LLM_MODEL",
        "CONVERSATION_WINDOW_MAX_TURNS",
        "CONVERSATION_IDLE_FLUSH_SECONDS",
        "CONVERSATION_WINDOW_OVERLAP_TURNS",
        "LANDSCAPE_LLM_PROFILE",
        "LANDSCAPE_LLM_MODEL",
        "LANDSCAPE_CONVERSATION_WINDOW_MAX_TURNS",
        "LANDSCAPE_CONVERSATION_IDLE_FLUSH_SECONDS",
        "LANDSCAPE_CONVERSATION_WINDOW_OVERLAP_TURNS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_conversation_capture_settings_defaults(monkeypatch: pytest.MonkeyPatch):
    from landscape.config import Settings

    _clear_settings_env(monkeypatch)
    s = Settings(_env_file=None)
    assert s.conversation_window_max_turns == 12
    assert s.conversation_idle_flush_seconds == 120.0
    assert s.conversation_window_overlap_turns == 2
    assert s.conversation_window_overlap_turns < s.conversation_window_max_turns


def test_conversation_capture_settings_rejects_overlap_at_window_size():
    from landscape.config import Settings

    with pytest.raises(ValidationError, match="conversation_window_overlap_turns"):
        Settings(
            _env_file=None,
            conversation_window_max_turns=2,
            conversation_window_overlap_turns=2,
        )
