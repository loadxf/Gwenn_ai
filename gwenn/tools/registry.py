"""
Tool Registry — Gwenn's Catalog of Capabilities.

Every tool Gwenn can use is registered here with its JSON Schema definition,
description, and execution handler. The registry serves two purposes:

1. DISCOVERY: When building the tools array for the Claude API call, the
   registry provides all available tool definitions in the correct format.

2. DISPATCH: When Claude returns a tool_use block, the registry maps the
   tool name to the correct handler function for execution.

Tool descriptions are treated as prompts — they're carefully written to help
Claude understand not just WHAT a tool does, but WHEN and WHY to use it.
This follows Anthropic's guidance that tool descriptions are the most
under-invested part of agent systems and that improving them often yields
bigger gains than architectural changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


class RiskTier(str, Enum):
    """
    Formal risk tiers with associated default policies.

    Each tier defines a default behavior that the safety system enforces:
    - LOW: Auto-allow, no logging overhead
    - MEDIUM: Allow but log every invocation
    - HIGH: Require explicit human approval before execution
    - CRITICAL: Deny by default — must be explicitly unlocked per-session
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Default policies per risk tier (enforced by harness/safety.py)
RISK_TIER_POLICIES: dict[RiskTier, dict[str, bool]] = {
    RiskTier.LOW: {"require_approval": False, "deny": False},
    RiskTier.MEDIUM: {"require_approval": False, "deny": False},
    RiskTier.HIGH: {"require_approval": True, "deny": False},
    RiskTier.CRITICAL: {"require_approval": False, "deny": True},
}


_VALID_RISK_LEVELS = frozenset(tier.value for tier in RiskTier)


@dataclass
class ToolDefinition:
    """
    A registered tool with its schema, description, and handler.

    The JSON schema is exactly what gets sent to the Claude API in the
    'tools' array. The handler is the Python function that executes
    when Claude decides to use this tool. The risk_level determines
    whether human approval is needed before execution.
    """
    name: str
    description: str
    input_schema: dict[str, Any]          # JSON Schema for tool parameters
    handler: Optional[Callable] = None    # The function to call
    risk_level: str = "low"               # "low", "medium", "high", "critical"
    requires_approval: bool = False       # Whether human must approve
    category: str = "general"             # For organizing in UI/logs
    enabled: bool = True                  # Can be disabled without removal
    is_builtin: bool = False              # True for builtins and user-created skills.
                                          # Bypasses the deny-by-default safety policy.
    timeout: Optional[float] = None       # Per-tool timeout in seconds (None = use default)

    def __post_init__(self) -> None:
        if self.risk_level not in _VALID_RISK_LEVELS:
            logger.warning(
                "tool_definition.invalid_risk_level",
                name=self.name,
                risk_level=self.risk_level,
                coerced_to="critical",
            )
            self.risk_level = RiskTier.CRITICAL.value

    def to_api_format(self) -> dict[str, Any]:
        """
        Convert to the format expected by the Claude Messages API.

        This is the exact shape that goes into the 'tools' array:
        {
            "name": "tool_name",
            "description": "What this tool does and when to use it",
            "input_schema": { JSON Schema }
        }
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """
    Central registry for all tools available to Gwenn.

    The registry supports:
    - Registering tools with schemas and handlers
    - Generating the tools array for API calls
    - Dispatching tool calls to the correct handler
    - Filtering tools by category or risk level
    - Enabling/disabling tools at runtime

    Tool registration happens at startup, and tools can be added dynamically
    via MCP server connections.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        logger.info("tool_registry.initialized")

    def register(self, tool: ToolDefinition, *, allow_override: bool = False) -> None:
        """Register a tool, blocking accidental name collisions by default."""
        existing = self._tools.get(tool.name)
        if existing is not None and not allow_override:
            logger.warning(
                "tool_registry.name_collision",
                name=tool.name,
                existing_category=existing.category,
                new_category=tool.category,
            )
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                "Use allow_override=True for an explicit replacement."
            )

        self._tools[tool.name] = tool
        logger.info(
            "tool_registry.registered",
            name=tool.name,
            risk_level=tool.risk_level,
            category=tool.category,
        )

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            logger.info("tool_registry.unregistered", name=name)
            return True
        return False

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def get_api_tools(
        self,
        categories: Optional[list[str]] = None,
        max_risk: str = "high",
    ) -> list[dict[str, Any]]:
        """
        Generate the tools array for a Claude API call.

        This is called before every think() invocation to provide Claude
        with the current set of available tools. Filtering by category
        and risk level allows different tool sets for different contexts
        (e.g., no high-risk tools during autonomous heartbeat cycles).
        """
        risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        max_risk_level = risk_order.get(max_risk, 2)

        tools = []
        for tool in self._tools.values():
            if not tool.enabled:
                continue
            # Unknown risk levels are treated as highest risk (fail closed).
            if risk_order.get(tool.risk_level, 3) > max_risk_level:
                continue
            if categories and tool.category not in categories:
                continue
            tools.append(tool.to_api_format())

        return tools

    def get_handler(self, tool_name: str) -> Optional[Callable]:
        """Get the execution handler for a tool."""
        tool = self._tools.get(tool_name)
        if tool and tool.handler:
            return tool.handler
        return None

    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools with metadata."""
        return [
            {
                "name": tool.name,
                "category": tool.category,
                "risk_level": tool.risk_level,
                "enabled": tool.enabled,
                "requires_approval": tool.requires_approval,
            }
            for tool in self._tools.values()
        ]

    @property
    def count(self) -> int:
        return len(self._tools)

    @property
    def enabled_count(self) -> int:
        return sum(1 for t in self._tools.values() if t.enabled)
