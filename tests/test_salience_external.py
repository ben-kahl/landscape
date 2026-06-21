"""Real-LLM salience checks.

Kept out of test_salience.py so the module-level ``unit`` marker there does not
also tag these tests — otherwise the CI slice (``-m "unit or smoke"``) would try
to run them without a live LLM (llama-server). These require the local stack and
are ``external`` only.
"""

import pytest

from landscape.conversation_ingestion import ConversationTurn

pytestmark = pytest.mark.external


def _turn(turn_id, role, text):
    return ConversationTurn(session_id="s1", turn_id=turn_id, role=role, text=text)


def test_select_salient_real_llm_keeps_facts_drops_chitchat():
    """Sanity check against a real LLM: durable facts kept, greetings dropped."""
    import landscape.extraction.salience as salience

    turns = [
        _turn("g", "user", "hey good morning!"),
        _turn("f", "user", "I lead the Platform team and we use Neo4j."),
    ]
    items = salience.select_salient(turns)
    kept = {item.turn_id for item in items}
    assert "f" in kept
    assert "g" not in kept
