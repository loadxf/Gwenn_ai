"""
Tests for gwenn.tools.registry.ToolRegistry.
"""

from __future__ import annotations

import pytest

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


def test_invalid_risk_level_coerced_to_critical_and_excluded():
    registry = ToolRegistry()
    tool = ToolDefinition(
        name="unknown_risk_tool",
        description="unknown risk",
        input_schema={"type": "object", "properties": {}},
        handler=None,
        risk_level="experimental",
    )
    # __post_init__ should coerce invalid risk_level to "critical"
    assert tool.risk_level == "critical"

    registry.register(tool)
    tools = registry.get_api_tools(max_risk="high")
    tool_names = {tool["name"] for tool in tools}

    assert "unknown_risk_tool" not in tool_names


def test_register_rejects_name_collisions_unless_override():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="dup_tool",
            description="first",
            input_schema={"type": "object", "properties": {}},
            handler=None,
            risk_level="low",
        )
    )

    with pytest.raises(ValueError):
        registry.register(
            ToolDefinition(
                name="dup_tool",
                description="second",
                input_schema={"type": "object", "properties": {}},
                handler=None,
                risk_level="low",
            )
        )

    registry.register(
        ToolDefinition(
            name="dup_tool",
            description="replacement",
            input_schema={"type": "object", "properties": {}},
            handler=None,
            risk_level="medium",
        ),
        allow_override=True,
    )
    assert registry.get("dup_tool").description == "replacement"
