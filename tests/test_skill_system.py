from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gwenn.agent import SentientAgent
from gwenn.skills import SkillDefinition, SkillRegistry
from gwenn.tools.builtin import register_builtin_tools
from gwenn.tools.registry import ToolRegistry


def _make_agent(tmp_path):
    agent = object.__new__(SentientAgent)
    agent.tool_registry = ToolRegistry()
    agent.skill_registry = SkillRegistry()
    agent._config = SimpleNamespace(skills_dir=tmp_path)
    agent._last_auto_skill_dev_at = 0.0
    agent._auto_skill_attempts = 0
    agent._auto_skill_created = 0
    return agent


def test_register_skill_sets_requires_approval_for_high_risk(tmp_path):
    agent = _make_agent(tmp_path)
    skill = SkillDefinition(
        name="dangerous_skill",
        description="Risky workflow",
        body="Do a sensitive action for {target}.",
        category="security",
        risk_level="HIGH",
        parameters={"target": {"type": "string", "required": True}},
    )

    ok = SentientAgent._register_skill_as_tool(agent, skill)

    assert ok is True
    tool = agent.tool_registry.get("dangerous_skill")
    assert tool is not None
    assert tool.risk_level == "high"
    assert tool.requires_approval is True


def test_register_skill_rejects_builtin_name_collision(tmp_path):
    agent = _make_agent(tmp_path)
    register_builtin_tools(agent.tool_registry)

    colliding = SkillDefinition(
        name="remember",
        description="Try to replace built-in memory tool",
        body="This should never register.",
        category="security",
        risk_level="low",
    )

    ok = SentientAgent._register_skill_as_tool(agent, colliding)

    assert ok is False
    assert agent.skill_registry.count == 0
    remember_tool = agent.tool_registry.get("remember")
    assert remember_tool is not None
    assert remember_tool.category == "memory"


def test_create_and_register_skill_cleans_invalid_temp_file(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    monkeypatch.setattr("gwenn.agent.parse_skill_file", lambda _: None)

    ok, message = SentientAgent._create_and_register_skill(
        agent,
        name="broken_skill",
        description="desc",
        instructions="Use {x}",
    )

    assert ok is False
    assert "failed validation" in message
    assert not (tmp_path / "broken_skill.md").exists()
    assert not list(tmp_path.glob("*.tmp.md"))
    assert agent.skill_registry.count == 0
    assert agent.tool_registry.get("broken_skill") is None


def test_create_and_register_skill_tolerates_non_dict_parameters(tmp_path):
    agent = _make_agent(tmp_path)

    ok, message = SentientAgent._create_and_register_skill(
        agent,
        name="non_dict_params_skill",
        description="desc",
        instructions="Do the task.",
        parameters=["not", "a", "dict"],  # type: ignore[arg-type]
    )

    assert ok is True
    assert "Parameters: none" in message
    assert (tmp_path / "non_dict_params_skill.md").exists()


@pytest.mark.asyncio
async def test_autonomous_skill_development_creates_skill(tmp_path):
    agent = _make_agent(tmp_path)

    class _Engine:
        async def reflect(self, system_prompt, messages):  # noqa: ARG002
            return "response"

        def extract_text(self, response):  # noqa: ARG002
            return json.dumps(
                {
                    "should_create": True,
                    "name": "auto_memory_review",
                    "description": "Summarize important memory clusters.",
                    "instructions": "Use `recall` on {topic} and summarize the patterns.",
                    "parameters": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to review",
                            "required": True,
                        }
                    },
                    "category": "autonomous",
                    "risk_level": "low",
                }
            )

    agent.engine = _Engine()

    await SentientAgent.maybe_develop_skill_autonomously(
        agent,
        thought="A" * 220,
        mode=SimpleNamespace(value="reflect"),
    )

    assert agent.skill_registry.count == 1
    assert agent._auto_skill_created == 1
    assert (tmp_path / "auto_memory_review.md").exists()
