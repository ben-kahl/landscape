import pytest

pytestmark = pytest.mark.unit


def test_conversation_capture_settings_defaults():
    from landscape.config import Settings

    s = Settings()
    assert s.conversation_window_max_turns == 12
    assert s.conversation_idle_flush_seconds == 120.0
    assert s.conversation_window_overlap_turns == 2
    assert s.conversation_window_overlap_turns < s.conversation_window_max_turns
