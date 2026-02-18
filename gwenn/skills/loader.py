"""
Skill Loader — Discovers and parses skill definition files.

Skill files are Markdown documents with a JSON frontmatter block:

    ---
    { JSON skill metadata }
    ---

    Instruction body with {param_name} placeholders...

The loader discovers all *.md files in the skills directory (skipping
SKILLS.md which is the auto-generated catalog) and parses them into
SkillDefinition objects ready for registration as tools.

Rendering
---------
When Gwenn invokes a skill tool, the loader's `render_skill_body()` function
substitutes all {param_name} occurrences in the body with the actual values
Claude provided.  The rendered body is returned as the tool result, which
Claude then reads and follows using its other available tools.

This is exactly how Claude Code SDK skills work: the skill expands to a
targeted prompt that describes the task, and the agent carries it out.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog

from gwenn.skills import SkillDefinition

logger = structlog.get_logger(__name__)

# Match --- JSON block --- at the very start of the file
_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(\{.*?\})\s*\n---\s*\n?(.*)", re.DOTALL)


def parse_skill_file(path: Path) -> SkillDefinition | None:
    """
    Parse a skill .md file into a SkillDefinition.

    Returns None (with a warning logged) if the file cannot be parsed.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("skill_loader.read_error", path=str(path), error=str(e))
        return None

    match = _FRONTMATTER_RE.match(raw)
    if not match:
        logger.warning(
            "skill_loader.no_frontmatter",
            path=str(path),
            hint="File must start with --- { JSON } --- frontmatter",
        )
        return None

    frontmatter_text = match.group(1).strip()
    body = match.group(2).strip()

    try:
        meta: dict[str, Any] = json.loads(frontmatter_text)
    except json.JSONDecodeError as e:
        logger.warning("skill_loader.invalid_json", path=str(path), error=str(e))
        return None

    name = meta.get("name")
    description = meta.get("description")
    if not name or not description:
        logger.warning(
            "skill_loader.missing_fields",
            path=str(path),
            missing=[f for f in ("name", "description") if not meta.get(f)],
        )
        return None

    if not body:
        logger.warning("skill_loader.empty_body", path=str(path))
        return None

    return SkillDefinition(
        name=str(name),
        description=str(description),
        body=body,
        category=str(meta.get("category", "skill")),
        version=str(meta.get("version", "1.0")),
        risk_level=str(meta.get("risk_level", "low")),
        parameters=meta.get("parameters", {}),
        tags=list(meta.get("tags", [])),
        source_file=path,
    )


def discover_skills(directory: Path) -> list[SkillDefinition]:
    """
    Discover and parse all skill files in a directory.

    Skips SKILLS.md (the auto-generated catalog) and hidden files.
    Returns parsed SkillDefinitions sorted alphabetically by name.
    """
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("skill_loader.directory_created", path=str(directory))
        return []

    skills: list[SkillDefinition] = []
    for md_file in sorted(directory.glob("*.md")):
        if md_file.name.upper() == "SKILLS.MD":
            continue
        if md_file.name.startswith("."):
            continue

        skill = parse_skill_file(md_file)
        if skill:
            skills.append(skill)
            logger.info("skill_loader.loaded", name=skill.name, file=md_file.name)
        else:
            logger.warning("skill_loader.skipped", file=md_file.name)

    return skills


def render_skill_body(body: str, params: dict[str, Any]) -> str:
    """
    Substitute {param_name} placeholders in a skill body with actual values.

    Unknown placeholders are left unchanged (not an error — the body may
    reference tool names in backticks which look like {tool} patterns).
    """

    def _safe_value(value: Any) -> str:
        # Keep parameter values data-like (escape control chars/newlines) so they
        # cannot inject additional instruction structure into skill prompts.
        if isinstance(value, str):
            encoded = json.dumps(value, ensure_ascii=False)
            return encoded[1:-1]
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            encoded = json.dumps(str(value), ensure_ascii=False)
            return encoded[1:-1]

    result = body
    for key, value in params.items():
        result = result.replace(f"{{{key}}}", _safe_value(value))
    return result


def build_skill_file_content(
    name: str,
    description: str,
    instructions: str,
    parameters: dict[str, Any] | None = None,
    category: str = "skill",
    risk_level: str = "low",
    version: str = "1.0",
    tags: list[str] | None = None,
) -> str:
    """
    Generate the content for a new skill .md file.

    Used by the skill_builder tool to create skills programmatically.
    The output is a valid skill file that can be parsed by parse_skill_file().
    """
    meta: dict[str, Any] = {
        "name": name,
        "description": description,
        "category": category,
        "version": version,
        "risk_level": risk_level,
        "tags": tags or [],
        "parameters": parameters or {},
    }
    frontmatter = json.dumps(meta, indent=2, ensure_ascii=False)
    return f"---\n{frontmatter}\n---\n\n{instructions.strip()}\n"
