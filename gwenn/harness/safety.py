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

import asyncio
import re
import time
from collections import deque
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
    retry_after_seconds: float = 0.0


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

    # Fallback set of builtin tool names — used when tool_registry is unavailable
    # (e.g. in unit tests that construct SafetyGuard without a registry).
    # The authoritative source at runtime is tool_def.is_builtin, which is set
    # automatically by register_builtin_tools() for every tool it registers.
    # User-created skills also receive is_builtin=True so they are allowed too.
    # Keep this list in sync with gwenn/tools/builtin/__init__.py.
    BUILTIN_TOOLS = frozenset({
        "remember", "recall", "check_emotional_state", "check_goals",
        "set_note_to_self", "think_aloud", "get_datetime", "calculate",
        "fetch_url", "convert_units", "get_calendar", "generate_token",
        "format_json", "encode_decode", "hash_text", "text_stats",
        "get_system_info", "skill_builder", "list_skills",
    })

    def __init__(self, config: SafetyConfig, tool_registry=None):
        self._config = config
        self._budget = BudgetState(
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_api_calls=config.max_api_calls,
        )
        self._iteration_count = 0
        self._last_reset = time.time()
        self._blocked_actions: list[dict[str, Any]] = []
        self._tool_registry = tool_registry  # Optional reference for risk tier checks
        self._emergency_stop_reason: Optional[str] = None
        self._model_calls_last_second: deque[float] = deque()
        self._model_calls_last_minute: deque[float] = deque()

        # Dangerous patterns in tool inputs (checked as substrings, lowercased)
        self._dangerous_patterns = [
            "rm -rf",
            "format c:",
            "drop table",
            "delete from",
            "chmod 777",
            "eval(",
        ]

        # Regex patterns for more complex matching (compiled once at init)
        self._dangerous_regexes = [
            (re.compile(r"\bsudo\b", re.IGNORECASE), "sudo command"),
            (re.compile(r"curl\s+.*\|\s*(?:bash|sh)\b", re.IGNORECASE), "curl pipe to bash"),
            (re.compile(r"wget\s+.*\|\s*(?:bash|sh)\b", re.IGNORECASE), "wget pipe to bash"),
        ]

        logger.info(
            "safety_guard.initialized",
            max_iterations=config.max_tool_iterations,
            require_approval_for=config.require_approval_for,
            tool_default_policy=config.tool_default_policy,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_api_calls=config.max_api_calls,
            max_model_calls_per_second=config.max_model_calls_per_second,
            max_model_calls_per_minute=config.max_model_calls_per_minute,
        )

    def _emergency_stop_check(self) -> Optional[SafetyCheckResult]:
        if self._emergency_stop_reason is None:
            return None
        return SafetyCheckResult(
            allowed=False,
            reason=f"Emergency stop active: {self._emergency_stop_reason}",
            risk_level="blocked",
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
        emergency_block = self._emergency_stop_check()
        if emergency_block:
            return emergency_block

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
        emergency_block = self._emergency_stop_check()
        if emergency_block:
            return emergency_block

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

        # Enforce deny-by-default policy for non-builtin tools.
        # Check is_builtin from the registry first (authoritative); fall back to
        # the static set for situations where the registry isn't wired yet.
        if self._config.tool_default_policy == "deny":
            is_builtin = False
            if self._tool_registry:
                tool_def_check = self._tool_registry.get(tool_name)
                if tool_def_check is not None:
                    is_builtin = tool_def_check.is_builtin
            if not is_builtin:
                is_builtin = tool_name in self.BUILTIN_TOOLS
            if not is_builtin:
                allowed_tools = self._config.parse_allowed_tools()
                if tool_name not in allowed_tools:
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

        return SafetyCheckResult(allowed=True)

    def _prune_rate_windows(self, now: float) -> None:
        while self._model_calls_last_second and now - self._model_calls_last_second[0] >= 1.0:
            self._model_calls_last_second.popleft()
        while self._model_calls_last_minute and now - self._model_calls_last_minute[0] >= 60.0:
            self._model_calls_last_minute.popleft()

    def check_model_call(self) -> SafetyCheckResult:
        """
        Check whether a model API call is currently allowed.

        This gate enforces:
        1. Emergency stop state
        2. Budget headroom (token + call ceilings)
        3. Proactive call-rate limits
        """
        emergency_block = self._emergency_stop_check()
        if emergency_block:
            return emergency_block

        if self._budget.max_input_tokens > 0 and self._budget.total_input_tokens >= self._budget.max_input_tokens:
            return SafetyCheckResult(
                allowed=False,
                reason=(
                    "Input token budget reached "
                    f"({self._budget.total_input_tokens}/{self._budget.max_input_tokens})"
                ),
                risk_level="blocked",
            )
        if self._budget.max_output_tokens > 0 and self._budget.total_output_tokens >= self._budget.max_output_tokens:
            return SafetyCheckResult(
                allowed=False,
                reason=(
                    "Output token budget reached "
                    f"({self._budget.total_output_tokens}/{self._budget.max_output_tokens})"
                ),
                risk_level="blocked",
            )
        if self._budget.max_api_calls > 0 and self._budget.total_api_calls >= self._budget.max_api_calls:
            return SafetyCheckResult(
                allowed=False,
                reason=(
                    "API call budget reached "
                    f"({self._budget.total_api_calls}/{self._budget.max_api_calls})"
                ),
                risk_level="blocked",
            )

        now = time.monotonic()
        self._prune_rate_windows(now)

        retry_after = 0.0
        max_per_second = self._config.max_model_calls_per_second
        max_per_minute = self._config.max_model_calls_per_minute

        if max_per_second > 0 and len(self._model_calls_last_second) >= max_per_second:
            retry_after = max(
                retry_after,
                1.0 - (now - self._model_calls_last_second[0]),
            )
        if max_per_minute > 0 and len(self._model_calls_last_minute) >= max_per_minute:
            retry_after = max(
                retry_after,
                60.0 - (now - self._model_calls_last_minute[0]),
            )

        if retry_after > 0.0:
            return SafetyCheckResult(
                allowed=False,
                reason="Model call rate limit reached",
                risk_level="blocked",
                retry_after_seconds=max(0.01, retry_after),
            )

        self._model_calls_last_second.append(now)
        self._model_calls_last_minute.append(now)
        return SafetyCheckResult(allowed=True)

    async def wait_for_model_call_slot(self) -> None:
        """
        Block until a model call is allowed, or raise on hard safety blocks.
        """
        while True:
            check = self.check_model_call()
            if check.allowed:
                return
            if check.retry_after_seconds > 0:
                await asyncio.sleep(check.retry_after_seconds)
                continue
            raise RuntimeError(f"Safety system intervened: {check.reason}")

    def reset_iteration_count(self) -> None:
        """Reset the iteration counter (called at the start of each agentic run)."""
        self._iteration_count = 0
        self._last_reset = time.time()

    def update_budget(self, input_tokens: int, output_tokens: int) -> None:
        """Update budget tracking with token usage from an API call."""
        self._budget.total_input_tokens += max(0, int(input_tokens))
        self._budget.total_output_tokens += max(0, int(output_tokens))
        self._budget.total_api_calls += 1

    def emergency_stop(self, reason: str) -> None:
        """
        Emergency stop — halt all operations immediately.

        This is the kill switch. When triggered, it sets the iteration
        count to maximum, preventing any further operations.
        """
        logger.critical("safety_guard.EMERGENCY_STOP", reason=reason)
        self._emergency_stop_reason = reason
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
                "max_input_tokens": self._budget.max_input_tokens,
                "max_output_tokens": self._budget.max_output_tokens,
                "max_api_calls": self._budget.max_api_calls,
            },
            "blocked_actions": len(self._blocked_actions),
            "emergency_stop_active": self._emergency_stop_reason is not None,
            "emergency_stop_reason": self._emergency_stop_reason or "",
        }
