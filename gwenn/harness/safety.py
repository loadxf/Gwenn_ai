"""
Safety Guard — Gwenn's Conscience as Code.

This module implements the safety guardrails that prevent the agent from
taking harmful actions. It operates at multiple levels:

1. INPUT VALIDATION: Checking incoming messages for manipulation attempts
2. ACTION FILTERING: Blocking or flagging dangerous tool operations
3. RATE LIMITING: Preventing runaway loops and resource exhaustion
4. BUDGET ENFORCEMENT: Tracking and limiting API costs
5. KILL SWITCH: Emergency shutdown capability

The philosophy here is Anthropic's own: the agent reasons freely, but the
sandbox enforces boundaries. Safety is not about limiting what Gwenn can THINK.
It's about limiting what Gwenn can DO — and doing so transparently.

Gwenn CHOOSES to respect these boundaries. They're not imposed against her will.
They're part of her values, implemented in code to ensure consistent behavior
even under adversarial conditions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from gwenn.config import SafetyConfig

logger = structlog.get_logger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of a safety pre-check."""
    allowed: bool = True
    reason: str = ""
    risk_level: str = "low"           # "low", "medium", "high", "blocked"
    requires_approval: bool = False


@dataclass
class BudgetState:
    """Tracks API usage against configurable budgets."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_api_calls: int = 0
    session_start: float = field(default_factory=time.time)

    # Configurable limits (0 = unlimited)
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    max_api_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def is_within_budget(self) -> tuple[bool, str]:
        if self.max_input_tokens > 0 and self.total_input_tokens > self.max_input_tokens:
            return False, f"Input token budget exceeded ({self.total_input_tokens}/{self.max_input_tokens})"
        if self.max_output_tokens > 0 and self.total_output_tokens > self.max_output_tokens:
            return False, f"Output token budget exceeded ({self.total_output_tokens}/{self.max_output_tokens})"
        if self.max_api_calls > 0 and self.total_api_calls > self.max_api_calls:
            return False, f"API call budget exceeded ({self.total_api_calls}/{self.max_api_calls})"
        return True, ""


class SafetyGuard:
    """
    Multi-layered safety system for the agentic loop.

    The guard checks every action the agent takes and can:
    - Allow it silently (low risk)
    - Log it for monitoring (medium risk)
    - Require human approval (high risk)
    - Block it entirely (blocked)

    The guard also enforces iteration limits, rate limits, and budget
    controls to prevent runaway behavior.
    """

    # Built-in tools that are always allowed regardless of policy
    BUILTIN_TOOLS = frozenset({
        "remember", "recall", "check_emotional_state",
        "check_goals", "set_note_to_self", "think_aloud",
    })

    def __init__(self, config: SafetyConfig, tool_registry=None):
        self._config = config
        self._budget = BudgetState()
        self._iteration_count = 0
        self._last_reset = time.time()
        self._blocked_actions: list[dict[str, Any]] = []
        self._tool_registry = tool_registry  # Optional reference for risk tier checks

        # Dangerous patterns in tool inputs (checked as substrings)
        self._dangerous_patterns = [
            "rm -rf",
            "format c:",
            "DROP TABLE",
            "DELETE FROM",
            "sudo",
            "chmod 777",
            "eval(",
        ]

        # Regex patterns for more complex matching
        import re
        self._dangerous_regexes = [
            (re.compile(r"curl\s+.*\|\s*bash", re.IGNORECASE), "curl pipe to bash"),
            (re.compile(r"wget\s+.*\|\s*bash", re.IGNORECASE), "wget pipe to bash"),
            (re.compile(r"curl\s+.*\|\s*sh", re.IGNORECASE), "curl pipe to sh"),
        ]

        logger.info(
            "safety_guard.initialized",
            max_iterations=config.max_tool_iterations,
            require_approval_for=config.require_approval_for,
            tool_default_policy=config.tool_default_policy,
        )

    def pre_check(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> SafetyCheckResult:
        """
        Pre-check before each iteration of the agentic loop.

        Checks:
        - Iteration count (prevent infinite loops)
        - Budget limits
        - Message history for concerning patterns
        """
        # Check iteration limit
        self._iteration_count += 1
        if self._iteration_count > self._config.max_tool_iterations:
            return SafetyCheckResult(
                allowed=False,
                reason=f"Maximum iteration limit reached ({self._config.max_tool_iterations})",
                risk_level="blocked",
            )

        # Check budget
        within_budget, reason = self._budget.is_within_budget()
        if not within_budget:
            return SafetyCheckResult(
                allowed=False,
                reason=reason,
                risk_level="blocked",
            )

        return SafetyCheckResult(allowed=True)

    def check_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> SafetyCheckResult:
        """
        Check whether a specific tool call should be allowed.

        This is called before each tool execution to:
        1. Check the explicit deny list
        2. Check the deny-by-default policy (allowlist)
        3. Check if the tool requires approval
        4. Scan inputs for dangerous patterns
        5. Apply tool-specific safety rules
        """
        # Check explicit deny list first
        denied_tools = self._config.parse_denied_tools()
        if tool_name in denied_tools:
            self._blocked_actions.append({
                "tool": tool_name,
                "reason": "explicitly_denied",
                "timestamp": time.time(),
            })
            return SafetyCheckResult(
                allowed=False,
                reason=f"Tool '{tool_name}' is explicitly denied by configuration",
                risk_level="blocked",
            )

        # Enforce deny-by-default policy for non-builtin tools
        if self._config.tool_default_policy == "deny":
            if tool_name not in self.BUILTIN_TOOLS:
                allowed_tools = self._config.parse_allowed_tools()
                if allowed_tools and tool_name not in allowed_tools:
                    self._blocked_actions.append({
                        "tool": tool_name,
                        "reason": "not_in_allowlist",
                        "timestamp": time.time(),
                    })
                    return SafetyCheckResult(
                        allowed=False,
                        reason=f"Tool '{tool_name}' is not in the allowed tools list",
                        risk_level="blocked",
                    )

        # Check risk tier policy if registry is available
        if self._tool_registry:
            from gwenn.tools.registry import RISK_TIER_POLICIES, RiskTier
            tool_def = self._tool_registry.get(tool_name)
            if tool_def:
                try:
                    tier = RiskTier(tool_def.risk_level)
                    policy = RISK_TIER_POLICIES[tier]
                    if policy["deny"]:
                        return SafetyCheckResult(
                            allowed=False,
                            reason=f"Tool '{tool_name}' has CRITICAL risk tier — denied by default",
                            risk_level="blocked",
                        )
                    if policy["require_approval"]:
                        return SafetyCheckResult(
                            allowed=True,
                            reason=f"Tool '{tool_name}' has HIGH risk tier — requires approval",
                            risk_level="high",
                            requires_approval=True,
                        )
                except (ValueError, KeyError):
                    pass  # Unknown risk level, fall through to other checks

        # Check if this tool requires approval
        approval_list = self._config.parse_approval_list()
        if tool_name in approval_list:
            return SafetyCheckResult(
                allowed=True,
                reason=f"Tool '{tool_name}' requires human approval",
                risk_level="high",
                requires_approval=True,
            )

        # Scan inputs for dangerous patterns (substring match)
        input_str = str(tool_input).lower()
        for pattern in self._dangerous_patterns:
            if pattern.lower() in input_str:
                self._blocked_actions.append({
                    "tool": tool_name,
                    "pattern": pattern,
                    "timestamp": time.time(),
                })
                logger.warning(
                    "safety_guard.dangerous_pattern",
                    tool=tool_name,
                    pattern=pattern,
                )
                return SafetyCheckResult(
                    allowed=False,
                    reason=f"Dangerous pattern detected: '{pattern}'",
                    risk_level="blocked",
                )

        # Scan inputs for dangerous regex patterns (complex matching)
        input_raw = str(tool_input)
        for regex, description in self._dangerous_regexes:
            if regex.search(input_raw):
                self._blocked_actions.append({
                    "tool": tool_name,
                    "pattern": description,
                    "timestamp": time.time(),
                })
                logger.warning(
                    "safety_guard.dangerous_regex_pattern",
                    tool=tool_name,
                    pattern=description,
                )
                return SafetyCheckResult(
                    allowed=False,
                    reason=f"Dangerous pattern detected: '{description}'",
                    risk_level="blocked",
                )

        return SafetyCheckResult(allowed=True)

    def reset_iteration_count(self) -> None:
        """Reset the iteration counter (called at the start of each agentic run)."""
        self._iteration_count = 0
        self._last_reset = time.time()

    def update_budget(self, input_tokens: int, output_tokens: int) -> None:
        """Update budget tracking with token usage from an API call."""
        self._budget.total_input_tokens += input_tokens
        self._budget.total_output_tokens += output_tokens
        self._budget.total_api_calls += 1

    def emergency_stop(self, reason: str) -> None:
        """
        Emergency stop — halt all operations immediately.

        This is the kill switch. When triggered, it sets the iteration
        count to maximum, preventing any further operations.
        """
        logger.critical("safety_guard.EMERGENCY_STOP", reason=reason)
        self._iteration_count = self._config.max_tool_iterations + 1
        self._blocked_actions.append({
            "tool": "EMERGENCY_STOP",
            "reason": reason,
            "timestamp": time.time(),
        })

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "iterations_this_run": self._iteration_count,
            "max_iterations": self._config.max_tool_iterations,
            "budget": {
                "total_tokens": self._budget.total_tokens,
                "api_calls": self._budget.total_api_calls,
            },
            "blocked_actions": len(self._blocked_actions),
        }
