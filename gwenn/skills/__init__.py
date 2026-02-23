"""
Skills System — Gwenn's Extensible Capability Library.

Skills are markdown-defined capabilities that extend Gwenn beyond her built-in
Python tools.  Each skill is a .md file with a JSON frontmatter block and a
plain-text instruction body.  When Gwenn invokes a skill, the parameters are
substituted into the instruction body, which Claude then follows using all
available tools.

This follows the Claude Code SDK SKILLS.md pattern: skills are *prompt expansions*
that describe HOW to accomplish a task using existing primitives, rather than
hard-coded Python functions.  The practical advantage is that skills are trivially
user-extensible — no coding required.  A user (or Gwenn herself, via the
skill_builder tool) just writes a .md file and the new capability is live on the
next message.

Skill lifecycle
---------------
1. DISCOVER  — load all *.md files from the skills directory at startup
2. REGISTER  — each skill becomes a ToolDefinition in the ToolRegistry
3. INVOKE    — Claude calls a skill tool; parameters are rendered into the body
4. EXECUTE   — the rendered instruction is returned as the tool result;
               Claude reads it and carries it out with its other tools
5. HOT-LOAD  — skill_builder writes a new .md file and immediately registers it

Skill file format
-----------------
    ---
    {
      "name": "skill_name",
      "description": "Used by Claude to decide when to call this skill.",
      "category": "information",
      "version": "1.0",
      "risk_level": "low",
      "parameters": {
        "param_name": {
          "type": "string",
          "description": "...",
          "required": true
        }
      }
    }
    ---

    Instruction body with {param_name} substitution placeholders.
    Claude follows these instructions, using its tools to complete the task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

VALID_SKILL_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class SkillDefinition:
    """
    A parsed skill ready to be registered as a tool.

    The JSON frontmatter becomes the tool schema exposed to Claude.
    The body is the instruction template rendered at execution time —
    {param_name} placeholders are filled in with the actual call arguments
    before being returned to Claude as the tool result.
    """
    name: str
    description: str
    body: str                               # Instruction template
    category: str = "skill"
    version: str = "1.0"
    risk_level: str = "low"
    parameters: dict[str, Any] = field(default_factory=dict)   # JSON Schema properties
    tags: list[str] = field(default_factory=list)              # Discovery keywords
    source_file: Path | None = None         # Which .md file this came from

    def __post_init__(self) -> None:
        if not self.name or not _SNAKE_CASE_RE.match(self.name):
            logger.warning(
                "skill_definition.invalid_name",
                name=self.name,
                hint="Name must be non-empty snake_case (e.g. 'get_weather')",
            )
        # Normalize risk_level to lowercase, then validate
        self.risk_level = str(self.risk_level or "low").strip().lower()
        if self.risk_level not in VALID_SKILL_RISK_LEVELS:
            logger.warning(
                "skill_definition.invalid_risk_level",
                name=self.name,
                risk_level=self.risk_level,
                coerced_to="low",
            )
            self.risk_level = "low"
        if not isinstance(self.parameters, dict):
            logger.warning(
                "skill_definition.invalid_parameters_type",
                name=self.name,
                provided_type=type(self.parameters).__name__,
                coerced_to="{}",
            )
            self.parameters = {}


class SkillRegistry:
    """
    Registry of all loaded skills.

    Skills live alongside built-in tools in the ToolRegistry — from Gwenn's
    perspective they are identical.  This registry tracks the originating
    SkillDefinitions so they can be listed, inspected, and rebuilt into fresh
    ToolDefinitions after a hot-reload.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition, *, allow_override: bool = False) -> None:
        existing = self._skills.get(skill.name)
        if existing is not None and not allow_override:
            logger.warning(
                "skill_registry.overwrite_blocked",
                name=skill.name,
                hint="Use allow_override=True to replace an existing skill",
            )
            return
        if existing is not None:
            logger.info(
                "skill_registry.overwritten",
                name=skill.name,
                old_version=existing.version,
                new_version=skill.version,
            )
        self._skills[skill.name] = skill
        logger.info("skill_registry.registered", name=skill.name, category=skill.category)

    def unregister(self, name: str) -> bool:
        if name in self._skills:
            del self._skills[name]
            logger.info("skill_registry.unregistered", name=name)
            return True
        return False

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def all_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def generate_catalog(self) -> str:
        """Generate the SKILLS.md catalog content."""
        if not self._skills:
            return (
                "# Gwenn Skills\n\n"
                "No skills loaded yet. Use the `skill_builder` tool to create one,\n"
                "or drop a `.md` file into the `gwenn_skills/` directory and restart.\n"
            )

        lines = [
            "# Gwenn Skills",
            "",
            "Auto-generated by Gwenn on startup. Each entry corresponds to a",
            "skill file in the `gwenn_skills/` directory. Use `skill_builder` to",
            "create new skills — no coding required.",
            "",
            f"**{len(self._skills)} skill{'s' if len(self._skills) != 1 else ''} loaded.**",
            "",
        ]

        # Group by category
        by_category: dict[str, list[SkillDefinition]] = {}
        for skill in sorted(self._skills.values(), key=lambda s: (s.category, s.name)):
            by_category.setdefault(skill.category, []).append(skill)

        for category, skills in sorted(by_category.items()):
            lines.append(f"## {category.replace('_', ' ').title()}")
            lines.append("")
            for skill in skills:
                required = [p for p, v in skill.parameters.items() if v.get("required")]
                optional = [p for p, v in skill.parameters.items() if not v.get("required")]
                param_parts = (
                    [f"`{p}`*" for p in required] + [f"`{p}`" for p in optional]
                )
                lines.append(f"### `{skill.name}`")
                lines.append(skill.description)
                lines.append("")
                lines.append(f"- **Parameters:** {', '.join(param_parts) or 'none'}")
                lines.append(f"- **Risk level:** {skill.risk_level}")
                lines.append(f"- **Version:** {skill.version}")
                if skill.tags:
                    lines.append(f"- **Tags:** {', '.join(skill.tags)}")
                if skill.source_file:
                    lines.append(f"- **File:** `{skill.source_file.name}`")
                lines.append("")

        lines += ["---", "", "*`*` = required parameter*", ""]
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._skills)
