"""Unit tests for bench_longmemeval helper functions.

These tests cover the pure-logic helpers that do not need the docker stack.
"""
import os
import sys

import pytest

# The bench script lives in scripts/, not a package. Import its helpers directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.mark.unit
def test_format_docs_for_prompt_joins_page_content():
    from bench_longmemeval import _format_docs_for_prompt
    from langchain_core.documents import Document

    docs = [
        Document(page_content="Alice (Person) [seed]", metadata={"kind": "entity"}),
        Document(page_content="Atlas Corp (Organization) [1 hops via WORKS_FOR]", 
                 metadata={"kind": "entity"}),
        Document(page_content="Alice joined Atlas Corp last year.", metadata={"kind": "chunk"}),
    ]
    result = _format_docs_for_prompt(docs)
    assert "Alice (Person) [seed]" in result
    assert "Atlas Corp (Organization) [1 hops via WORKS_FOR]" in result
    assert "Alice joined Atlas Corp last year." in result


@pytest.mark.unit
def test_format_docs_for_prompt_empty_returns_empty_string():
    from bench_longmemeval import _format_docs_for_prompt

    assert _format_docs_for_prompt([]) == ""


@pytest.mark.unit
def test_parse_judge_response_valid_json():
    from bench_longmemeval import _parse_judge_response

    raw = '{"judgment": "correct", "reason": "matches gold answer"}'
    result = _parse_judge_response(raw)
    assert result["judgment"] == "correct"
    assert result["reason"] == "matches gold answer"


@pytest.mark.unit
def test_parse_judge_response_markdown_wrapped_json():
    from bench_longmemeval import _parse_judge_response

    raw = '```json\n{"judgment": "incorrect", "reason": "wrong answer"}\n```'
    result = _parse_judge_response(raw)
    assert result["judgment"] == "incorrect"


@pytest.mark.unit
def test_parse_judge_response_plain_code_block():
    from bench_longmemeval import _parse_judge_response

    raw = '```\n{"judgment": "abstained", "reason": "not mentioned"}\n```'
    result = _parse_judge_response(raw)
    assert result["judgment"] == "abstained"


@pytest.mark.unit
def test_parse_judge_response_malformed_returns_incorrect():
    from bench_longmemeval import _parse_judge_response

    result = _parse_judge_response("not json at all")
    assert result["judgment"] == "incorrect"
    assert "parse error" in result["reason"]


@pytest.mark.unit
def test_parse_judge_response_missing_judgment_key():
    from bench_longmemeval import _parse_judge_response

    result = _parse_judge_response('{"verdict": "yes"}')
    assert result["judgment"] == "incorrect"
    assert "parse error" in result["reason"]
