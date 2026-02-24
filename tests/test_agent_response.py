"""Unit tests for AgentResponse and ButtonSpec data types."""

from __future__ import annotations

from gwenn.types import AgentResponse, ButtonSpec


class TestAgentResponse:
    def test_str_returns_text(self):
        resp = AgentResponse(text="hello world")
        assert str(resp) == "hello world"

    def test_buttons_default_none(self):
        resp = AgentResponse(text="hi")
        assert resp.buttons is None

    def test_buttons_attached(self):
        buttons = [[ButtonSpec(label="A"), ButtonSpec(label="B")]]
        resp = AgentResponse(text="choose", buttons=buttons)
        assert resp.buttons is not None
        assert len(resp.buttons) == 1
        assert len(resp.buttons[0]) == 2


class TestButtonSpec:
    def test_value_defaults_to_label(self):
        spec = ButtonSpec(label="Click me")
        assert spec.value == "Click me"

    def test_explicit_value(self):
        spec = ButtonSpec(label="Click me", value="click")
        assert spec.value == "click"
