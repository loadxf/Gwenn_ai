"""Tests for orchestration tool registration and ToolRegistry.get_tools_by_name."""

from __future__ import annotations


from gwenn.tools.builtin import register_builtin_tools
from gwenn.tools.registry import ToolDefinition, ToolRegistry


class TestOrchestrationToolRegistration:
    def test_orchestration_tools_registered(self):
        registry = ToolRegistry()
        register_builtin_tools(registry)

        expected_tools = [
            "spawn_subagent",
            "spawn_swarm",
            "check_subagent",
            "collect_results",
            "cancel_subagent",
        ]
        for name in expected_tools:
            tool = registry.get(name)
            assert tool is not None, f"Tool '{name}' not registered"
            assert tool.is_builtin is True

    def test_orchestration_tool_categories(self):
        registry = ToolRegistry()
        register_builtin_tools(registry)

        for name in [
            "spawn_subagent",
            "spawn_swarm",
            "check_subagent",
            "collect_results",
            "cancel_subagent",
        ]:
            tool = registry.get(name)
            assert tool.category == "orchestration"

    def test_spawn_tools_are_medium_risk(self):
        registry = ToolRegistry()
        register_builtin_tools(registry)

        assert registry.get("spawn_subagent").risk_level == "medium"
        assert registry.get("spawn_swarm").risk_level == "medium"

    def test_check_and_collect_are_low_risk(self):
        registry = ToolRegistry()
        register_builtin_tools(registry)

        assert registry.get("check_subagent").risk_level == "low"
        assert registry.get("collect_results").risk_level == "low"
        assert registry.get("cancel_subagent").risk_level == "low"

    def test_spawn_subagent_schema(self):
        registry = ToolRegistry()
        register_builtin_tools(registry)

        tool = registry.get("spawn_subagent")
        schema = tool.input_schema
        assert "task_description" in schema["properties"]
        assert "tools" in schema["properties"]
        assert "task_description" in schema["required"]


class TestGetToolsByName:
    def test_basic_filtering(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="tool_a",
                description="a",
                risk_level="low",
                input_schema={"type": "object", "properties": {}},
            )
        )
        registry.register(
            ToolDefinition(
                name="tool_b",
                description="b",
                risk_level="medium",
                input_schema={"type": "object", "properties": {}},
            )
        )
        registry.register(
            ToolDefinition(
                name="tool_c",
                description="c",
                risk_level="high",
                input_schema={"type": "object", "properties": {}},
            )
        )

        # Request specific tools
        tools = registry.get_tools_by_name(["tool_a", "tool_b"])
        names = {t["name"] for t in tools}
        assert names == {"tool_a", "tool_b"}

    def test_excludes_above_max_risk(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="low_tool",
                description="low",
                risk_level="low",
                input_schema={"type": "object", "properties": {}},
            )
        )
        registry.register(
            ToolDefinition(
                name="high_tool",
                description="high",
                risk_level="high",
                input_schema={"type": "object", "properties": {}},
            )
        )

        tools = registry.get_tools_by_name(["low_tool", "high_tool"], max_risk="medium")
        names = {t["name"] for t in tools}
        assert "low_tool" in names
        assert "high_tool" not in names

    def test_never_returns_critical(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="critical_tool",
                description="critical",
                risk_level="critical",
                input_schema={"type": "object", "properties": {}},
            )
        )

        tools = registry.get_tools_by_name(["critical_tool"], max_risk="critical")
        assert len(tools) == 0  # CRITICAL never returned

    def test_skips_nonexistent(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="exists",
                description="exists",
                risk_level="low",
                input_schema={"type": "object", "properties": {}},
            )
        )

        tools = registry.get_tools_by_name(["exists", "nope"])
        assert len(tools) == 1
        assert tools[0]["name"] == "exists"

    def test_skips_disabled(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="disabled_tool",
                description="disabled",
                risk_level="low",
                input_schema={"type": "object", "properties": {}},
                enabled=False,
            )
        )

        tools = registry.get_tools_by_name(["disabled_tool"])
        assert len(tools) == 0


class TestGetDefinitionsByName:
    def test_returns_definitions(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="tool_x",
                description="x",
                risk_level="low",
                input_schema={"type": "object", "properties": {}},
            )
        )

        defs = registry.get_definitions_by_name(["tool_x"])
        assert len(defs) == 1
        assert defs[0].name == "tool_x"

    def test_empty_for_nonexistent(self):
        registry = ToolRegistry()
        defs = registry.get_definitions_by_name(["nope"])
        assert defs == []
