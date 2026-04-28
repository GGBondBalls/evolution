"""MockLLMClient + JSON parsing."""

from __future__ import annotations

import json

import pytest

from rm.llm.client import (
    ChatMessage,
    MockLLMClient,
    _normalise_messages,
    _parse_json,
    build_client,
)


def test_mock_default_response():
    c = MockLLMClient(default="hi")
    out = c.chat([{"role": "user", "content": "hello"}])
    assert out.text == "hi"
    assert out.tokens > 0
    assert c.usage.calls == 1


def test_mock_when_rules_match_first():
    c = MockLLMClient(default="default")
    c.when(r"thought", "T-Action: x").when(r"hello", "say hi")
    r1 = c.chat([{"role": "user", "content": "give me a thought"}])
    r2 = c.chat([{"role": "user", "content": "say hello"}])
    assert r1.text.startswith("T-Action")
    assert r2.text == "say hi"


def test_chat_json_repairs_extra_prose():
    c = MockLLMClient(default='Sure, here it is:\n```json\n{"k": 1}\n```')
    out = c.chat_json([{"role": "user", "content": "x"}])
    assert out == {"k": 1}


def test_chat_json_with_chat_message_objects():
    c = MockLLMClient(default='{"a": 2}')
    out = c.chat_json([ChatMessage(role="user", content="x")])
    assert out == {"a": 2}


def test_normalise_message_types():
    msgs = _normalise_messages([
        ChatMessage(role="system", content="s"),
        {"role": "user", "content": "u"},
    ])
    assert msgs == [{"role": "system", "content": "s"},
                    {"role": "user", "content": "u"}]


def test_parse_json_handles_fenced():
    obj = _parse_json('```json\n{"x": 1}\n```')
    assert obj == {"x": 1}


def test_parse_json_handles_trailing_text():
    obj = _parse_json('hi {"x": 1} and then garbage')
    assert obj == {"x": 1}


def test_parse_json_strict_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_json("nope", repair=False)


def test_build_client_mock_path():
    c = build_client({"provider": "mock"})
    assert isinstance(c, MockLLMClient)
