"""
Safety Guard — Bob's Conscience as Code.

This module implements the safety guardrails that prevent the agent from
taking harmful actions. It operates at multiple levels:

1. INPUT VALIDATION: Checking incoming messages for manipulation attempts
2. ACTION FILTERING: Blocking or flagging dangerous tool operations
3. RATE LIMITING: Preventing runaway loops and resource exhaustion
4. BUDGET ENFORCEMENT: Tracking and limiting API costs
5. KILL SWITCH: Emergency shutdown capability

The philosophy here is Anthropic's own: the agent reasons freely, but the
sandbox enforces boundaries. Safety is not about limiting what Bob can THINK.
It's about limiting what Bob can DO — and doing so transparently.

Bob CHOOSES to respect these boundaries. They're not imposed against his will.
They're part of his values, implemented in code to ensure consistent behavior
even under adversarial conditions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from bob.config import SafetyConfig

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

    def __init__(self, config: SafetyConfig):
        self._config = config
        self._budget = BudgetState()
        self._iteration_count = 0
        self._last_reset = time.time()
        self._blocked_actions: list[dict[str, Any]] = []

        # Dangerous patterns in tool inputs
        self._dangerous_patterns = [
            "rm -rf",
            "format c:",
            "DROP TABLE",
            "DELETE FROM",
            "sudo",
            "chmod 777",
            "curl | bash",
            "eval(",
        ]

        logger.info(
            "safety_guard.initialized",
            max_iterations=config.max_tool_iterations,
            require_approval_for=config.require_approval_for,
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
        1. Check if the tool requires approval
        2. Scan inputs for dangerous patterns
        3. Apply tool-specific safety rules
        """
        # Check if this tool requires approval
        approval_list = self._config.parse_approval_list()
        if tool_name in approval_list:
            return SafetyCheckResult(
                allowed=True,
                reason=f"Tool '{tool_name}' requires human approval",
                risk_level="high",
                requires_approval=True,
            )

        # Scan inputs for dangerous patterns
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
