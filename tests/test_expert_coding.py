"""End-to-end tests for the expert_coding skill and supporting tool changes.

Covers:
1. Skill file parsing and loading
2. Schema validation for new spawn_subagent/spawn_swarm parameters
3. Handler passthrough for system_prompt, max_iterations, isolation, timeout
4. Skill body rendering with parameter substitution
5. Integration: skill registration as a tool
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gwenn.agent import SentientAgent
from gwenn.orchestration.models import SubagentSpec, SwarmSpec
from gwenn.skills import SkillDefinition, SkillRegistry
from gwenn.skills.loader import parse_skill_file, render_skill_body
from gwenn.tools.builtin import register_builtin_tools
from gwenn.tools.registry import ToolRegistry


SKILL_DIR = Path(__file__).resolve().parent.parent / "gwenn_skills"
SKILL_FILE = SKILL_DIR / "expert_coding.md"


# ---------------------------------------------------------------------------
# 1. Skill file parsing
# ---------------------------------------------------------------------------


class TestSkillFileParsing:
    def test_expert_coding_file_exists(self):
        assert SKILL_FILE.exists(), f"Skill file not found: {SKILL_FILE}"

    def test_parse_expert_coding_skill(self):
        skill = parse_skill_file(SKILL_FILE)
        assert skill is not None
        assert skill.name == "expert_coding"

    def test_metadata_fields(self):
        skill = parse_skill_file(SKILL_FILE)
        assert skill.category == "developer"
        assert skill.version == "1.0"
        assert skill.risk_level == "medium"

    def test_parameters(self):
        skill = parse_skill_file(SKILL_FILE)
        # parse_skill_file normalizes into JSON Schema: {type, properties, required}
        props = skill.parameters["properties"]
        assert "task" in props
        assert "project_path" in props
        assert "experts" in props
        assert "style" in props

    def test_required_parameters(self):
        skill = parse_skill_file(SKILL_FILE)
        required = skill.parameters.get("required", [])
        assert "task" in required
        assert "project_path" in required

    def test_optional_parameters_have_defaults(self):
        skill = parse_skill_file(SKILL_FILE)
        props = skill.parameters["properties"]
        assert props["experts"].get("default") == "auto"
        assert props["style"].get("default") == "thorough"

    def test_style_enum(self):
        skill = parse_skill_file(SKILL_FILE)
        style_param = skill.parameters["properties"]["style"]
        assert set(style_param["enum"]) == {"minimal", "thorough", "rapid"}

    def test_tags_include_key_triggers(self):
        skill = parse_skill_file(SKILL_FILE)
        tags = skill.tags
        for expected in ["code", "implement", "fix", "refactor", "review", "test", "expert"]:
            assert expected in tags, f"Tag '{expected}' missing"

    def test_body_contains_key_sections(self):
        skill = parse_skill_file(SKILL_FILE)
        body = skill.body
        assert "Anti-Vibe-Coding" in body
        assert "Expert Role Catalog" in body
        assert "Phased Workflow" in body
        assert "ARCHITECT" in body
        assert "BACKEND" in body
        assert "REVIEWER" in body
        assert "Final Report" in body

    def test_body_contains_parameter_placeholders(self):
        skill = parse_skill_file(SKILL_FILE)
        body = skill.body
        assert "{task}" in body
        assert "{project_path}" in body
        assert "{experts}" in body
        assert "{style}" in body


# ---------------------------------------------------------------------------
# 2. Schema validation for new spawn_subagent/spawn_swarm parameters
# ---------------------------------------------------------------------------


class TestToolSchemas:
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)

    def test_spawn_subagent_has_system_prompt(self):
        tool = self.registry.get("spawn_subagent")
        props = tool.input_schema["properties"]
        assert "system_prompt" in props
        assert props["system_prompt"]["type"] == "string"

    def test_spawn_subagent_has_max_iterations(self):
        tool = self.registry.get("spawn_subagent")
        props = tool.input_schema["properties"]
        assert "max_iterations" in props
        assert props["max_iterations"]["type"] == "integer"
        assert props["max_iterations"]["default"] == 30

    def test_spawn_subagent_has_isolation(self):
        tool = self.registry.get("spawn_subagent")
        props = tool.input_schema["properties"]
        assert "isolation" in props
        assert set(props["isolation"]["enum"]) == {"in_process", "docker"}

    def test_spawn_subagent_new_params_are_optional(self):
        tool = self.registry.get("spawn_subagent")
        required = tool.input_schema.get("required", [])
        assert "system_prompt" not in required
        assert "max_iterations" not in required
        assert "isolation" not in required

    def test_spawn_swarm_task_has_system_prompt(self):
        tool = self.registry.get("spawn_swarm")
        task_props = tool.input_schema["properties"]["tasks"]["items"]["properties"]
        assert "system_prompt" in task_props
        assert task_props["system_prompt"]["type"] == "string"

    def test_spawn_swarm_task_has_max_iterations(self):
        tool = self.registry.get("spawn_swarm")
        task_props = tool.input_schema["properties"]["tasks"]["items"]["properties"]
        assert "max_iterations" in task_props
        assert task_props["max_iterations"]["type"] == "integer"
        assert task_props["max_iterations"]["default"] == 30

    def test_spawn_swarm_task_has_isolation(self):
        tool = self.registry.get("spawn_swarm")
        task_props = tool.input_schema["properties"]["tasks"]["items"]["properties"]
        assert "isolation" in task_props
        assert set(task_props["isolation"]["enum"]) == {"in_process", "docker"}

    def test_spawn_swarm_task_has_timeout_seconds(self):
        tool = self.registry.get("spawn_swarm")
        task_props = tool.input_schema["properties"]["tasks"]["items"]["properties"]
        assert "timeout_seconds" in task_props
        assert task_props["timeout_seconds"]["type"] == "number"
        assert task_props["timeout_seconds"]["default"] == 120


# ---------------------------------------------------------------------------
# 3. Handler passthrough tests
# ---------------------------------------------------------------------------


def _make_mock_agent():
    """Create a minimal mock agent for handler testing."""
    agent = object.__new__(SentientAgent)
    agent.tool_registry = ToolRegistry()
    register_builtin_tools(agent.tool_registry)

    # Mock orchestrator that captures the spec
    mock_orch = AsyncMock()
    mock_orch.spawn = AsyncMock(return_value="sub-test123")
    mock_orch.spawn_swarm = AsyncMock(return_value="swarm-test123")
    agent.orchestrator = mock_orch

    # Mock config
    agent._config = SimpleNamespace(
        orchestration=SimpleNamespace(default_runtime="in_process"),
    )

    return agent


class TestSpawnSubagentHandler:
    @pytest.mark.asyncio
    async def test_system_prompt_passthrough(self):
        agent = _make_mock_agent()

        # Wire handlers
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_subagent").handler
        result = await handler(
            task_description="test task",
            system_prompt="You are an expert architect.",
        )

        assert "task_id" in result
        spec = agent.orchestrator.spawn.call_args[0][0]
        assert spec.system_prompt == "You are an expert architect."

    @pytest.mark.asyncio
    async def test_empty_system_prompt_becomes_none(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_subagent").handler
        await handler(task_description="test task", system_prompt="")

        spec = agent.orchestrator.spawn.call_args[0][0]
        assert spec.system_prompt is None

    @pytest.mark.asyncio
    async def test_max_iterations_passthrough(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_subagent").handler
        await handler(task_description="test task", max_iterations=25)

        spec = agent.orchestrator.spawn.call_args[0][0]
        assert spec.max_iterations == 25

    @pytest.mark.asyncio
    async def test_max_iterations_capped_at_200(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_subagent").handler
        await handler(task_description="test task", max_iterations=500)

        spec = agent.orchestrator.spawn.call_args[0][0]
        assert spec.max_iterations == 200

    @pytest.mark.asyncio
    async def test_isolation_passthrough_docker(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_subagent").handler
        await handler(task_description="test task", isolation="docker")

        spec = agent.orchestrator.spawn.call_args[0][0]
        assert spec.runtime_tier == "docker"

    @pytest.mark.asyncio
    async def test_isolation_passthrough_in_process(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_subagent").handler
        await handler(task_description="test task", isolation="in_process")

        spec = agent.orchestrator.spawn.call_args[0][0]
        assert spec.runtime_tier == "in_process"

    @pytest.mark.asyncio
    async def test_isolation_empty_falls_back_to_config(self):
        agent = _make_mock_agent()
        agent._config.orchestration.default_runtime = "docker"
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_subagent").handler
        await handler(task_description="test task", isolation="")

        spec = agent.orchestrator.spawn.call_args[0][0]
        assert spec.runtime_tier == "docker"

    @pytest.mark.asyncio
    async def test_isolation_invalid_falls_back_to_config(self):
        agent = _make_mock_agent()
        agent._config.orchestration.default_runtime = "in_process"
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_subagent").handler
        await handler(task_description="test task", isolation="invalid_value")

        spec = agent.orchestrator.spawn.call_args[0][0]
        assert spec.runtime_tier == "in_process"

    @pytest.mark.asyncio
    async def test_all_params_together(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_subagent").handler
        await handler(
            task_description="Implement the login feature",
            system_prompt="You are a Backend Engineer.",
            max_iterations=20,
            isolation="docker",
            tools=["read_file", "write_file"],
            timeout_seconds=300.0,
        )

        spec = agent.orchestrator.spawn.call_args[0][0]
        assert spec.task_description == "Implement the login feature"
        assert spec.system_prompt == "You are a Backend Engineer."
        assert spec.max_iterations == 20
        assert spec.runtime_tier == "docker"
        assert spec.tools == ["read_file", "write_file"]
        assert spec.timeout_seconds == 300.0


class TestSpawnSwarmHandler:
    @pytest.mark.asyncio
    async def test_per_task_system_prompt(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_swarm").handler
        await handler(
            description="Expert implementation",
            tasks=[
                {
                    "task_description": "Backend work",
                    "system_prompt": "You are a Backend Engineer.",
                },
                {
                    "task_description": "Frontend work",
                    "system_prompt": "You are a Frontend Engineer.",
                },
            ],
        )

        swarm_spec = agent.orchestrator.spawn_swarm.call_args[0][0]
        assert len(swarm_spec.agents) == 2
        assert swarm_spec.agents[0].system_prompt == "You are a Backend Engineer."
        assert swarm_spec.agents[1].system_prompt == "You are a Frontend Engineer."

    @pytest.mark.asyncio
    async def test_per_task_max_iterations(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_swarm").handler
        await handler(
            description="Test swarm",
            tasks=[
                {"task_description": "Quick task", "max_iterations": 5},
                {"task_description": "Long task", "max_iterations": 30},
            ],
        )

        swarm_spec = agent.orchestrator.spawn_swarm.call_args[0][0]
        assert swarm_spec.agents[0].max_iterations == 5
        assert swarm_spec.agents[1].max_iterations == 30

    @pytest.mark.asyncio
    async def test_per_task_max_iterations_capped(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_swarm").handler
        await handler(
            description="Test swarm",
            tasks=[{"task_description": "Task", "max_iterations": 999}],
        )

        swarm_spec = agent.orchestrator.spawn_swarm.call_args[0][0]
        assert swarm_spec.agents[0].max_iterations == 200

    @pytest.mark.asyncio
    async def test_per_task_isolation(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_swarm").handler
        await handler(
            description="Mixed isolation",
            tasks=[
                {"task_description": "Trusted", "isolation": "in_process"},
                {"task_description": "Untrusted", "isolation": "docker"},
            ],
        )

        swarm_spec = agent.orchestrator.spawn_swarm.call_args[0][0]
        assert swarm_spec.agents[0].runtime_tier == "in_process"
        assert swarm_spec.agents[1].runtime_tier == "docker"

    @pytest.mark.asyncio
    async def test_per_task_timeout_seconds(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_swarm").handler
        await handler(
            description="Test swarm",
            tasks=[
                {"task_description": "Quick", "timeout_seconds": 60},
                {"task_description": "Slow", "timeout_seconds": 300},
            ],
        )

        swarm_spec = agent.orchestrator.spawn_swarm.call_args[0][0]
        assert swarm_spec.agents[0].timeout_seconds == 60.0
        assert swarm_spec.agents[1].timeout_seconds == 300.0

    @pytest.mark.asyncio
    async def test_per_task_defaults_when_omitted(self):
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_swarm").handler
        await handler(
            description="Minimal swarm",
            tasks=[{"task_description": "Basic task"}],
        )

        swarm_spec = agent.orchestrator.spawn_swarm.call_args[0][0]
        spec = swarm_spec.agents[0]
        assert spec.system_prompt is None
        assert spec.max_iterations == 30
        assert spec.timeout_seconds == 120.0
        assert spec.runtime_tier == "in_process"

    @pytest.mark.asyncio
    async def test_full_expert_swarm(self):
        """Simulate the kind of swarm the expert_coding skill would spawn."""
        agent = _make_mock_agent()
        SentientAgent._wire_orchestration_tool_handlers(agent)

        handler = agent.tool_registry.get("spawn_swarm").handler
        await handler(
            description="Implement user authentication feature",
            tasks=[
                {
                    "task_description": "Implement auth middleware",
                    "system_prompt": "You are a Backend Engineer. Follow anti-vibe-coding rules.",
                    "tools": ["read_file", "write_file", "think_aloud"],
                    "max_iterations": 20,
                    "timeout_seconds": 300,
                    "isolation": "in_process",
                },
                {
                    "task_description": "Implement login form",
                    "system_prompt": "You are a Frontend Engineer. Follow anti-vibe-coding rules.",
                    "tools": ["read_file", "write_file", "think_aloud"],
                    "max_iterations": 20,
                    "timeout_seconds": 300,
                    "isolation": "in_process",
                },
            ],
        )

        swarm_spec = agent.orchestrator.spawn_swarm.call_args[0][0]
        assert len(swarm_spec.agents) == 2

        backend = swarm_spec.agents[0]
        assert backend.system_prompt == "You are a Backend Engineer. Follow anti-vibe-coding rules."
        assert backend.max_iterations == 20
        assert backend.timeout_seconds == 300.0
        assert backend.runtime_tier == "in_process"
        assert backend.tools == ["read_file", "write_file", "think_aloud"]

        frontend = swarm_spec.agents[1]
        assert "Frontend Engineer" in frontend.system_prompt
        assert frontend.max_iterations == 20
        assert frontend.timeout_seconds == 300.0


# ---------------------------------------------------------------------------
# 4. Skill body rendering
# ---------------------------------------------------------------------------


class TestSkillBodyRendering:
    def test_render_with_all_parameters(self):
        skill = parse_skill_file(SKILL_FILE)
        rendered = render_skill_body(skill.body, {
            "task": "Add user authentication",
            "project_path": "/home/bob/myproject",
            "experts": "architect,backend,reviewer",
            "style": "thorough",
        })

        assert "Add user authentication" in rendered
        assert "/home/bob/myproject" in rendered
        assert "architect,backend,reviewer" in rendered
        assert "thorough" in rendered

    def test_render_with_defaults(self):
        skill = parse_skill_file(SKILL_FILE)
        rendered = render_skill_body(skill.body, {
            "task": "Fix login bug",
            "project_path": "/home/bob/myproject",
        })

        # Unsubstituted params remain as {param} — verify task/path are substituted
        assert "Fix login bug" in rendered
        assert "/home/bob/myproject" in rendered

    def test_rendered_body_contains_anti_vibe_directives(self):
        skill = parse_skill_file(SKILL_FILE)
        rendered = render_skill_body(skill.body, {
            "task": "Test task",
            "project_path": "/tmp",
        })

        assert "SMALLEST change" in rendered
        assert "Do NOT refactor adjacent code" in rendered
        assert "Read existing code FIRST" in rendered
        assert "3–5 bullet points" in rendered

    def test_rendered_body_contains_expert_prompts(self):
        skill = parse_skill_file(SKILL_FILE)
        rendered = render_skill_body(skill.body, {
            "task": "Test task",
            "project_path": "/tmp",
        })

        assert "Software Architect" in rendered
        assert "Backend Engineer" in rendered
        assert "Frontend Engineer" in rendered
        assert "Database Engineer" in rendered
        assert "Code Reviewer" in rendered
        assert "Test Engineer" in rendered

    def test_rendered_body_contains_phased_workflow(self):
        skill = parse_skill_file(SKILL_FILE)
        rendered = render_skill_body(skill.body, {
            "task": "Test task",
            "project_path": "/tmp",
            "style": "thorough",
        })

        assert "Phase 1: ANALYZE" in rendered
        assert "Phase 2: PLAN" in rendered
        assert "Phase 3: IMPLEMENT" in rendered
        assert "Phase 4: REVIEW" in rendered
        assert "Phase 5: TEST" in rendered


# ---------------------------------------------------------------------------
# 5. Integration: skill registration as a tool
# ---------------------------------------------------------------------------


def _make_agent(tmp_path):
    agent = object.__new__(SentientAgent)
    agent.tool_registry = ToolRegistry()
    agent.skill_registry = SkillRegistry()
    agent._config = SimpleNamespace(skills_dir=tmp_path)
    agent._last_auto_skill_dev_at = 0.0
    agent._auto_skill_attempts = 0
    agent._auto_skill_created = 0
    return agent


class TestSkillRegistration:
    def test_expert_coding_registers_as_tool(self, tmp_path):
        agent = _make_agent(tmp_path)
        skill = parse_skill_file(SKILL_FILE)

        ok = SentientAgent._register_skill_as_tool(agent, skill)

        assert ok is True
        assert agent.skill_registry.get("expert_coding") is not None

        tool = agent.tool_registry.get("expert_coding")
        assert tool is not None
        assert tool.risk_level == "medium"
        assert tool.category == "skill:developer"
        assert tool.enabled is True

    def test_registered_tool_has_correct_schema(self, tmp_path):
        agent = _make_agent(tmp_path)
        skill = parse_skill_file(SKILL_FILE)
        SentientAgent._register_skill_as_tool(agent, skill)

        tool = agent.tool_registry.get("expert_coding")
        schema = tool.input_schema

        assert schema["type"] == "object"
        assert "task" in schema["properties"]
        assert "project_path" in schema["properties"]
        assert "experts" in schema["properties"]
        assert "style" in schema["properties"]

        # Required params
        assert "task" in schema.get("required", [])
        assert "project_path" in schema.get("required", [])

    def test_registered_tool_handler_returns_rendered_body(self, tmp_path):
        agent = _make_agent(tmp_path)
        skill = parse_skill_file(SKILL_FILE)
        SentientAgent._register_skill_as_tool(agent, skill)

        tool = agent.tool_registry.get("expert_coding")
        result = tool.handler(
            task="Build a REST API",
            project_path="/home/bob/project",
            experts="auto",
            style="thorough",
        )

        assert "[SKILL ACTIVATED: expert_coding v1.0]" in result
        assert "Build a REST API" in result
        assert "/home/bob/project" in result
        assert "Anti-Vibe-Coding" in result
        assert "ARCHITECT" in result

    def test_skill_medium_risk_requires_no_approval(self, tmp_path):
        """Medium risk skills do not require approval (only high/critical do)."""
        agent = _make_agent(tmp_path)
        skill = parse_skill_file(SKILL_FILE)
        SentientAgent._register_skill_as_tool(agent, skill)

        tool = agent.tool_registry.get("expert_coding")
        assert tool.requires_approval is False
