from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "sample_transcript.jsonl"


def test_read_transcript_returns_ordered_eligible_turns():
    from landscape.ingestion.transcript import read_transcript

    turns = read_transcript(FIXTURE)

    assert [(t.role, t.text) for t in turns] == [
        ("user", "My name is Ben and I lead the Landscape project."),
        ("assistant", "Got it, Ben — noted that you lead Landscape."),
    ]
    assert all(t.session_id == "sess-fixture" for t in turns)
    assert turns[0].turn_id == "u1"
    assert turns[1].turn_id == "a1"


def test_read_transcript_session_id_override_wins():
    from landscape.ingestion.transcript import read_transcript

    turns = read_transcript(FIXTURE, session_id="override")
    assert all(t.session_id == "override" for t in turns)


def test_read_transcript_rejects_unknown_client():
    from landscape.ingestion.transcript import read_transcript

    with pytest.raises(ValueError, match="client"):
        read_transcript(FIXTURE, client="cursor")
