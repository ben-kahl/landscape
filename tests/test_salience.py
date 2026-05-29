import pytest

from landscape.conversation_ingestion import ConversationTurn

pytestmark = pytest.mark.unit


def _turn(turn_id, role, text):
    return ConversationTurn(session_id="s1", turn_id=turn_id, role=role, text=text)


def test_select_salient_maps_indices_to_turn_provenance(monkeypatch):
    import landscape.extraction.salience as salience

    turns = [
        _turn("t1", "user", "Hi there!"),
        _turn("t2", "user", "I just started as a backend engineer at Acme."),
        _turn("t3", "assistant", "Congrats! What stack?"),
        _turn("t4", "user", "We decided to standardize on Postgres."),
    ]

    def fake_call(prompt: str):
        return salience.SalienceSelection(
            selected=[
                salience.SalienceSelectionItem(turn_index=2, category="identity"),
                salience.SalienceSelectionItem(turn_index=4, category="decision"),
            ]
        )

    monkeypatch.setattr(salience, "_call_salience_model", fake_call)

    items = salience.select_salient(turns)

    assert [i.turn_id for i in items] == ["t2", "t4"]
    assert [i.text for i in items] == [
        "I just started as a backend engineer at Acme.",
        "We decided to standardize on Postgres.",
    ]
    assert [i.category for i in items] == ["identity", "decision"]


def test_select_salient_drops_heuristic_noise_before_llm(monkeypatch):
    import landscape.extraction.salience as salience

    turns = [
        _turn("t1", "tool", "<tool output 4kb>"),
        _turn("t2", "user", "   "),
        _turn("t3", "user", "My manager is Dana."),
    ]
    captured = {}

    def fake_call(prompt: str):
        captured["prompt"] = prompt
        return salience.SalienceSelection(
            selected=[salience.SalienceSelectionItem(turn_index=1, category="relationship")]
        )

    monkeypatch.setattr(salience, "_call_salience_model", fake_call)

    items = salience.select_salient(turns)

    assert "My manager is Dana." in captured["prompt"]
    assert "<tool output 4kb>" not in captured["prompt"]
    assert [i.turn_id for i in items] == ["t3"]


def test_select_salient_empty_when_no_eligible_turns(monkeypatch):
    import landscape.extraction.salience as salience

    called = {"n": 0}

    def fake_call(prompt: str):
        called["n"] += 1
        return salience.SalienceSelection(selected=[])

    monkeypatch.setattr(salience, "_call_salience_model", fake_call)

    items = salience.select_salient([_turn("t1", "tool", "noise")])
    assert items == []
    assert called["n"] == 0


def test_select_salient_ignores_out_of_range_indices(monkeypatch):
    import landscape.extraction.salience as salience

    turns = [_turn("t1", "user", "I live in Berlin.")]

    def fake_call(prompt: str):
        return salience.SalienceSelection(
            selected=[
                salience.SalienceSelectionItem(turn_index=1, category="identity"),
                salience.SalienceSelectionItem(turn_index=9, category="identity"),
            ]
        )

    monkeypatch.setattr(salience, "_call_salience_model", fake_call)
    items = salience.select_salient(turns)
    assert [i.turn_id for i in items] == ["t1"]


@pytest.mark.external
def test_select_salient_real_llm_keeps_facts_drops_chitchat():
    """Sanity check against real Ollama: durable facts kept, greetings dropped."""
    import landscape.extraction.salience as salience

    turns = [
        _turn("g", "user", "hey good morning!"),
        _turn("f", "user", "I lead the Platform team and we use Neo4j."),
    ]
    items = salience.select_salient(turns)
    kept = {item.turn_id for item in items}
    assert "f" in kept
    assert "g" not in kept
