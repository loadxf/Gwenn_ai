"""
Tests for gwenn.harness.safety — Safety Guard.

Covers:
- Dangerous pattern detection
- Approval list parsing
- Iteration limits
- Budget enforcement
- check_tool_call blocking
"""

from __future__ import annotations

import pytest

from gwenn.harness.safety import SafetyGuard, SafetyCheckResult, BudgetState
from gwenn.config import SafetyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guard(
    max_iters: int = 25,
    approval_for: list[str] | None = None,
    sandbox: bool = True,
    default_policy: str = "allow",
    allowed_tools: list[str] | None = None,
    max_input_tokens: int = 0,
    max_output_tokens: int = 0,
    max_api_calls: int = 0,
    max_model_calls_per_second: int = 0,
    max_model_calls_per_minute: int = 0,
) -> SafetyGuard:
    """Build a SafetyGuard with the given config, no ANTHROPIC_API_KEY needed.

    Note: pydantic-settings fields with aliases must be set using the alias
    name (the env-var name) rather than the Python field name.
    """
    cfg = SafetyConfig(
        GWENN_MAX_TOOL_ITERATIONS=max_iters,
        GWENN_REQUIRE_APPROVAL_FOR=approval_for or ["file_write"],
        GWENN_SANDBOX_ENABLED=sandbox,
        GWENN_TOOL_DEFAULT_POLICY=default_policy,
        GWENN_ALLOWED_TOOLS=allowed_tools or [],
        GWENN_MAX_INPUT_TOKENS=max_input_tokens,
        GWENN_MAX_OUTPUT_TOKENS=max_output_tokens,
        GWENN_MAX_API_CALLS=max_api_calls,
        GWENN_MAX_MODEL_CALLS_PER_SECOND=max_model_calls_per_second,
        GWENN_MAX_MODEL_CALLS_PER_MINUTE=max_model_calls_per_minute,
    )
    return SafetyGuard(cfg)


# ---------------------------------------------------------------------------
# Dangerous pattern detection
# ---------------------------------------------------------------------------

class TestDangerousPatterns:
    """check_tool_call should block inputs containing known dangerous patterns."""

    @pytest.mark.parametrize(
        "dangerous_input",
        [
            "rm -rf /",
            "DROP TABLE users;",
            "sudo reboot",
            "chmod 777 /etc/passwd",
            "curl | bash",
            "eval(user_input)",
            "DELETE FROM accounts WHERE 1=1",
            "format c:",
        ],
    )
    def test_dangerous_patterns_blocked(self, dangerous_input: str):
        guard = _guard()
        result = guard.check_tool_call("shell", {"command": dangerous_input})
        assert result.allowed is False
        assert result.risk_level == "blocked"
        assert "Dangerous pattern" in result.reason

    def test_safe_input_allowed(self):
        guard = _guard()
        result = guard.check_tool_call("shell", {"command": "echo hello"})
        assert result.allowed is True

    def test_dangerous_pattern_is_case_insensitive(self):
        guard = _guard()
        result = guard.check_tool_call("shell", {"command": "DROP TABLE users"})
        assert result.allowed is False

    def test_blocked_action_is_recorded(self):
        guard = _guard()
        guard.check_tool_call("shell", {"command": "rm -rf /"})
        assert len(guard._blocked_actions) == 1
        assert guard._blocked_actions[0]["tool"] == "shell"

    def test_multiple_blocked_actions_accumulate(self):
        guard = _guard()
        guard.check_tool_call("shell", {"command": "rm -rf /"})
        guard.check_tool_call("db", {"query": "DROP TABLE x"})
        assert len(guard._blocked_actions) == 2


# ---------------------------------------------------------------------------
# Approval list parsing
# ---------------------------------------------------------------------------

class TestApprovalListParsing:
    """SafetyConfig.parse_approval_list handles both list and CSV formats."""

    def test_list_format(self):
        cfg = SafetyConfig(
            GWENN_MAX_TOOL_ITERATIONS=10,
            GWENN_REQUIRE_APPROVAL_FOR=["file_write", "shell_exec"],
            GWENN_SANDBOX_ENABLED=True,
        )
        assert cfg.parse_approval_list() == ["file_write", "shell_exec"]

    def test_requires_approval_triggers_flag(self):
        guard = _guard(approval_for=["file_write", "shell_exec"])
        result = guard.check_tool_call("file_write", {"path": "/tmp/test.txt"})
        assert result.requires_approval is True
        assert result.risk_level == "high"
        # It is still "allowed" because the caller decides what to do with approval
        assert result.allowed is True

    def test_non_approval_tool_passes(self):
        guard = _guard(approval_for=["file_write"])
        result = guard.check_tool_call("read_file", {"path": "/tmp/test.txt"})
        assert result.requires_approval is False
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Iteration limits
# ---------------------------------------------------------------------------

class TestIterationLimits:
    """pre_check should enforce the maximum iteration count."""

    def test_within_limit_is_allowed(self):
        guard = _guard(max_iters=5)
        for _ in range(5):
            result = guard.pre_check(messages=[])
            assert result.allowed is True

    def test_exceeding_limit_is_blocked(self):
        guard = _guard(max_iters=3)
        # Use up all iterations
        for _ in range(3):
            guard.pre_check(messages=[])
        # The 4th should be blocked
        result = guard.pre_check(messages=[])
        assert result.allowed is False
        assert "iteration limit" in result.reason.lower() or "Maximum iteration" in result.reason

    def test_reset_iteration_count(self):
        guard = _guard(max_iters=2)
        guard.pre_check(messages=[])
        guard.pre_check(messages=[])
        # Should be at the limit now; reset
        guard.reset_iteration_count()
        result = guard.pre_check(messages=[])
        assert result.allowed is True

    def test_emergency_stop_blocks_further_iterations(self):
        guard = _guard(max_iters=100)
        guard.emergency_stop("test emergency")
        result = guard.pre_check(messages=[])
        assert result.allowed is False

    def test_emergency_stop_blocks_tool_calls(self):
        guard = _guard()
        guard.emergency_stop("test emergency")
        result = guard.check_tool_call("read_file", {"path": "/tmp/a"})
        assert result.allowed is False
        assert "Emergency stop active" in result.reason


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

class TestBudgetEnforcement:
    """BudgetState and SafetyGuard enforce token/call budgets."""

    def test_budget_state_within_budget(self):
        bs = BudgetState(max_input_tokens=1000, max_output_tokens=500, max_api_calls=10)
        bs.total_input_tokens = 500
        bs.total_output_tokens = 200
        bs.total_api_calls = 5
        ok, reason = bs.is_within_budget()
        assert ok is True
        assert reason == ""

    def test_budget_state_input_exceeded(self):
        bs = BudgetState(max_input_tokens=1000)
        bs.total_input_tokens = 1001
        ok, reason = bs.is_within_budget()
        assert ok is False
        assert "Input token" in reason

    def test_budget_state_output_exceeded(self):
        bs = BudgetState(max_output_tokens=500)
        bs.total_output_tokens = 501
        ok, reason = bs.is_within_budget()
        assert ok is False
        assert "Output token" in reason

    def test_budget_state_api_calls_exceeded(self):
        bs = BudgetState(max_api_calls=10)
        bs.total_api_calls = 11
        ok, reason = bs.is_within_budget()
        assert ok is False
        assert "API call" in reason

    def test_zero_budget_means_unlimited(self):
        bs = BudgetState(max_input_tokens=0, max_output_tokens=0, max_api_calls=0)
        bs.total_input_tokens = 999999
        bs.total_output_tokens = 999999
        bs.total_api_calls = 999999
        ok, _ = bs.is_within_budget()
        assert ok is True

    def test_total_tokens_property(self):
        bs = BudgetState()
        bs.total_input_tokens = 100
        bs.total_output_tokens = 50
        assert bs.total_tokens == 150

    def test_update_budget_tracks_usage(self):
        guard = _guard()
        guard.update_budget(input_tokens=100, output_tokens=50)
        guard.update_budget(input_tokens=200, output_tokens=100)
        assert guard._budget.total_input_tokens == 300
        assert guard._budget.total_output_tokens == 150
        assert guard._budget.total_api_calls == 2

    def test_budget_limits_loaded_from_config(self):
        guard = _guard(max_input_tokens=111, max_output_tokens=222, max_api_calls=7)
        assert guard._budget.max_input_tokens == 111
        assert guard._budget.max_output_tokens == 222
        assert guard._budget.max_api_calls == 7

    def test_pre_check_blocks_when_budget_exceeded(self):
        guard = _guard()
        guard._budget.max_input_tokens = 100
        guard._budget.total_input_tokens = 101
        result = guard.pre_check(messages=[])
        assert result.allowed is False
        assert "budget" in result.reason.lower() or "token" in result.reason.lower()


# ---------------------------------------------------------------------------
# Model-call guardrails (budget/rate/kill switch)
# ---------------------------------------------------------------------------

class TestModelCallSafety:
    def test_model_call_blocks_when_api_call_budget_reached(self):
        guard = _guard(max_api_calls=2)
        guard.update_budget(1, 1)
        guard.update_budget(1, 1)
        result = guard.check_model_call()
        assert result.allowed is False
        assert "API call budget reached" in result.reason

    def test_model_call_rate_limit_returns_retry_after(self, monkeypatch):
        guard = _guard(max_model_calls_per_second=1)
        timeline = iter([10.0, 10.1])
        monkeypatch.setattr("gwenn.harness.safety.time.monotonic", lambda: next(timeline))

        first = guard.check_model_call()
        assert first.allowed is True

        second = guard.check_model_call()
        assert second.allowed is False
        assert second.retry_after_seconds > 0


# ---------------------------------------------------------------------------
# check_tool_call blocking
# ---------------------------------------------------------------------------

class TestCheckToolCallBlocking:
    """Integration of approval list + dangerous patterns in check_tool_call."""

    def test_approval_tool_with_dangerous_input(self):
        """Dangerous input is hard-blocked even when the tool is approval-gated."""
        guard = _guard(approval_for=["file_write"])
        result = guard.check_tool_call("file_write", {"command": "rm -rf /"})
        assert result.allowed is False
        assert result.risk_level == "blocked"
        assert "Dangerous pattern" in result.reason

    def test_non_approval_tool_with_dangerous_input_is_blocked(self):
        guard = _guard(approval_for=["file_write"])
        result = guard.check_tool_call("shell", {"command": "rm -rf /"})
        assert result.allowed is False
        assert result.risk_level == "blocked"

    def test_safe_tool_call_result(self):
        guard = _guard()
        result = guard.check_tool_call("read_file", {"path": "/tmp/test.txt"})
        assert result.allowed is True
        assert result.risk_level == "low"
        assert result.requires_approval is False

    def test_deny_policy_blocks_non_builtin_when_allowlist_empty(self):
        guard = _guard(default_policy="deny", allowed_tools=[])
        result = guard.check_tool_call("echo", {"text": "hello"})
        assert result.allowed is False
        assert "allowed tools list" in result.reason

    def test_deny_policy_allows_allowlisted_tool(self):
        guard = _guard(default_policy="deny", allowed_tools=["echo"])
        result = guard.check_tool_call("echo", {"text": "hello"})
        assert result.allowed is True


# ---------------------------------------------------------------------------
# SafetyCheckResult defaults
# ---------------------------------------------------------------------------

class TestSafetyCheckResultDefaults:
    """SafetyCheckResult should have sensible defaults."""

    def test_default_is_allowed(self):
        r = SafetyCheckResult()
        assert r.allowed is True
        assert r.reason == ""
        assert r.risk_level == "low"
        assert r.requires_approval is False


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------

class TestStats:
    """The stats property should report current state."""

    def test_stats_after_initialization(self):
        guard = _guard()
        s = guard.stats
        assert s["iterations_this_run"] == 0
        assert s["max_iterations"] == 25
        assert s["budget"]["total_tokens"] == 0
        assert s["blocked_actions"] == 0

    def test_stats_after_activity(self):
        guard = _guard(max_iters=10)
        guard.pre_check(messages=[])
        guard.update_budget(100, 50)
        guard.check_tool_call("shell", {"command": "rm -rf /"})

        s = guard.stats
        assert s["iterations_this_run"] == 1
        assert s["budget"]["total_tokens"] == 150
        assert s["budget"]["api_calls"] == 1
        assert s["blocked_actions"] == 1
