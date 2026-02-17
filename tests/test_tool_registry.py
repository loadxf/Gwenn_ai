"""
Tests for gwenn.tools.registry.ToolRegistry.
"""

from __future__ import annotations

from gwenn.tools.registry import ToolDefinition, ToolRegistry


def test_get_api_tools_excludes_critical_when_max_risk_is_high():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="safe_tool",
            description="safe",
            input_schema={"type": "object", "properties": {}},
            handler=None,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="critical_tool",
            description="critical",
            input_schema={"type": "object", "properties": {}},
            handler=None,
            risk_level="critical",
        )
    )

    tools = registry.get_api_tools(max_risk="high")
    tool_names = {tool["name"] for tool in tools}

    assert "safe_tool" in tool_names
    assert "critical_tool" not in tool_names


def test_get_api_tools_treats_unknown_risk_as_highest_and_excludes():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="unknown_risk_tool",
            description="unknown risk",
            input_schema={"type": "object", "properties": {}},
            handler=None,
            risk_level="experimental",
        )
    )

    tools = registry.get_api_tools(max_risk="high")
    tool_names = {tool["name"] for tool in tools}

    assert "unknown_risk_tool" not in tool_names

