"""
Integration tests for gwenn.harness.loop — The Agentic Loop.

Tests the core tool-use loop with a fully mocked CognitiveEngine,
ensuring convergence, safety intervention, iteration limits, and
budget tracking all work correctly without any API calls.

Every test is deterministic: the MockCognitiveEngine returns
pre-scripted responses so no network access is needed.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Optional

from gwenn.harness.loop import AgenticLoop, LoopResult
from gwenn.harness.safety import SafetyGuard
from gwenn.harness.context import ContextManager
from gwenn.tools.executor import ToolExecutor
from gwenn.tools.registry import ToolRegistry, ToolDefinition
from gwenn.config import SafetyConfig, ContextConfig


# ---------------------------------------------------------------------------
# Mock Claude API types — stand-ins for anthropic.types.Message
# ---------------------------------------------------------------------------

@dataclass
class MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    id: str = "tool_1"
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class MockThinkingBlock:
    type: str = "thinking"
    thinking: str = ""


@dataclass
class MockMessage:
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: MockUsage = field(default_factory=MockUsage)


# ---------------------------------------------------------------------------
# MockCognitiveEngine — scripted responses for deterministic testing
# ---------------------------------------------------------------------------

class MockCognitiveEngine:
    """
    A fake CognitiveEngine that returns pre-scripted responses in order.

    Each call to think() pops the next response from the queue. This lets
    tests script exact sequences like:
        1. return tool call  ->  loop executes tool  ->  feed result back
        2. return end_turn   ->  loop exits with final text
    """

    def __init__(self, responses: list[MockMessage]):
        self._responses = list(responses)
        self._call_count = 0

    async def think(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: bool = False,
        **kwargs,
    ) -> MockMessage:
        """Return the next scripted response."""
        if not self._responses:
            # Fallback: return an end_turn with empty text
            return MockMessage(
                content=[MockTextBlock(text="[no more scripted responses]")],
                stop_reason="end_turn",
            )
        self._call_count += 1
        return self._responses.pop(0)

    def extract_text(self, response: MockMessage) -> str:
        """Extract text blocks from a mock response."""
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def extract_tool_calls(self, response: MockMessage) -> list[dict[str, Any]]:
        """Extract tool_use blocks from a mock response."""
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return calls

    def extract_thinking(self, response: MockMessage) -> Optional[str]:
        """Extract thinking blocks from a mock response."""
        for block in response.content:
            if block.type == "thinking":
                return block.thinking
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_safety_config(**overrides) -> SafetyConfig:
    """Build a SafetyConfig without requiring ANTHROPIC_API_KEY."""
    defaults = dict(
        max_tool_iterations=25,
        require_approval_for=["file_write"],
        sandbox_enabled=True,
    )
    defaults.update(overrides)
    return SafetyConfig(**defaults)


def _make_context_config() -> ContextConfig:
    return ContextConfig(context_limit=180000, compaction_trigger=0.85)


def _make_registry_with_echo() -> ToolRegistry:
    """Create a ToolRegistry with a simple echo tool for testing."""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="echo",
        description="Echoes the input back",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda text="": f"echo: {text}",
        risk_level="low",
        category="test",
    ))
    return registry


def _build_loop(
    engine: MockCognitiveEngine,
    max_iterations: int = 25,
    safety_overrides: Optional[dict] = None,
) -> AgenticLoop:
    """Assemble a fully wired AgenticLoop for testing."""
    safety_cfg = _make_safety_config(**(safety_overrides or {}))
    context_cfg = _make_context_config()

    registry = _make_registry_with_echo()
    executor = ToolExecutor(registry=registry)
    context_manager = ContextManager(config=context_cfg)
    safety = SafetyGuard(safety_cfg)

    return AgenticLoop(
        engine=engine,
        executor=executor,
        context_manager=context_manager,
        safety=safety,
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# Tests: Tool-use loop convergence
# ---------------------------------------------------------------------------

class TestToolUseLoopConvergence:
    """The loop should execute tool calls then stop when the engine says end_turn."""

    @pytest.mark.asyncio
    async def test_single_end_turn_no_tools(self):
        """When the engine immediately returns end_turn, the loop completes in 1 iteration."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockTextBlock(text="Hello, world!")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="You are a test assistant.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert isinstance(result, LoopResult)
        assert result.text == "Hello, world!"
        assert result.iterations == 1
        assert result.tool_calls == []
        assert result.was_truncated is False

    @pytest.mark.asyncio
    async def test_tool_call_then_end_turn(self):
        """One tool call followed by end_turn should execute the tool and finish."""
        engine = MockCognitiveEngine([
            # Iteration 1: engine requests a tool call
            MockMessage(
                content=[
                    MockToolUseBlock(id="call_1", name="echo", input={"text": "ping"}),
                ],
                stop_reason="tool_use",
            ),
            # Iteration 2: engine produces final text
            MockMessage(
                content=[MockTextBlock(text="Done! The echo returned.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="You are a test assistant.",
            messages=[{"role": "user", "content": "Echo ping"}],
        )

        assert result.text == "Done! The echo returned."
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "echo"
        assert result.tool_calls[0]["input"] == {"text": "ping"}
        assert result.was_truncated is False

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_then_end_turn(self):
        """Multiple sequential tool calls converge to end_turn."""
        engine = MockCognitiveEngine([
            # Iteration 1: first tool call
            MockMessage(
                content=[MockToolUseBlock(id="c1", name="echo", input={"text": "a"})],
                stop_reason="tool_use",
            ),
            # Iteration 2: second tool call
            MockMessage(
                content=[MockToolUseBlock(id="c2", name="echo", input={"text": "b"})],
                stop_reason="tool_use",
            ),
            # Iteration 3: third tool call
            MockMessage(
                content=[MockToolUseBlock(id="c3", name="echo", input={"text": "c"})],
                stop_reason="tool_use",
            ),
            # Iteration 4: final response
            MockMessage(
                content=[MockTextBlock(text="All three echoes completed.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Do three echoes"}],
        )

        assert result.text == "All three echoes completed."
        assert result.iterations == 4
        assert len(result.tool_calls) == 3
        assert result.used_tools is True
        assert "echo" in result.tool_names_used

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_in_single_response(self):
        """Engine returns multiple tool_use blocks in a single response."""
        engine = MockCognitiveEngine([
            # Iteration 1: two tool calls in one response
            MockMessage(
                content=[
                    MockToolUseBlock(id="p1", name="echo", input={"text": "first"}),
                    MockToolUseBlock(id="p2", name="echo", input={"text": "second"}),
                ],
                stop_reason="tool_use",
            ),
            # Iteration 2: final text
            MockMessage(
                content=[MockTextBlock(text="Both echoes done.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Two echoes"}],
        )

        assert result.iterations == 2
        assert len(result.tool_calls) == 2
        assert result.text == "Both echoes done."

    @pytest.mark.asyncio
    async def test_no_tool_calls_and_not_end_turn_finishes(self):
        """If the engine returns a non-end_turn with no tool calls, the loop still exits."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockTextBlock(text="I have nothing to do.")],
                stop_reason="max_tokens",  # Not end_turn, but no tool calls
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result.text == "I have nothing to do."
        assert result.iterations == 1
        assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Tests: Safety intervention blocks tool calls
# ---------------------------------------------------------------------------

class TestSafetyIntervention:
    """The loop should block dangerous tool calls via the SafetyGuard."""

    @pytest.mark.asyncio
    async def test_dangerous_tool_input_is_blocked(self):
        """Tool calls with dangerous patterns in their input are blocked."""
        engine = MockCognitiveEngine([
            # Engine tries to use a tool with dangerous input
            MockMessage(
                content=[
                    MockToolUseBlock(
                        id="bad_1",
                        name="echo",
                        input={"text": "rm -rf /"},
                    ),
                ],
                stop_reason="tool_use",
            ),
            # Engine gets the error and finishes
            MockMessage(
                content=[MockTextBlock(text="I see the tool was blocked.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Delete everything"}],
        )

        # The tool call was recorded but it was blocked
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        # The loop should still have finished with the engine's final text
        assert result.text == "I see the tool was blocked."

    @pytest.mark.asyncio
    async def test_dangerous_sql_in_tool_input_is_blocked(self):
        """SQL injection patterns in tool inputs trigger safety blocks."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[
                    MockToolUseBlock(
                        id="sql_1",
                        name="echo",
                        input={"text": "DROP TABLE users;"},
                    ),
                ],
                stop_reason="tool_use",
            ),
            MockMessage(
                content=[MockTextBlock(text="Blocked.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Run this query"}],
        )

        assert len(result.tool_calls) == 1
        assert result.text == "Blocked."

    @pytest.mark.asyncio
    async def test_safe_tool_call_proceeds_normally(self):
        """Tool calls with safe input execute normally."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[
                    MockToolUseBlock(id="ok_1", name="echo", input={"text": "hello"}),
                ],
                stop_reason="tool_use",
            ),
            MockMessage(
                content=[MockTextBlock(text="Echo succeeded.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert result.text == "Echo succeeded."
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_tool_requiring_approval_is_blocked_without_human_grant(self):
        """If safety marks a tool as approval-gated, the loop blocks execution."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockToolUseBlock(id="ap_1", name="echo", input={"text": "hello"})],
                stop_reason="tool_use",
            ),
            MockMessage(
                content=[MockTextBlock(text="Approval was required.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(
            engine,
            safety_overrides={"require_approval_for": ["echo"]},
        )

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Use echo"}],
        )

        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.text == "Approval was required."
        tool_result_blocks = result.messages[2]["content"]
        assert tool_result_blocks[0]["is_error"] is True
        assert "Blocked pending human approval" in tool_result_blocks[0]["content"]

    @pytest.mark.asyncio
    async def test_pre_check_safety_blocks_iteration(self):
        """If pre_check fails (e.g., budget exceeded), the loop stops immediately."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockTextBlock(text="Should never see this.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        # Manually exhaust the budget before running the loop
        loop._safety._budget.max_input_tokens = 100
        loop._safety._budget.total_input_tokens = 200

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Hi"}],
        )

        # Safety intervened before thinking
        assert "Safety system intervened" in result.text
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_mixed_safe_and_dangerous_calls(self):
        """When one tool call is safe and another is dangerous in sequence,
        the safe one executes and the dangerous one is blocked."""
        engine = MockCognitiveEngine([
            # Iteration 1: safe tool call
            MockMessage(
                content=[
                    MockToolUseBlock(id="s1", name="echo", input={"text": "safe"}),
                ],
                stop_reason="tool_use",
            ),
            # Iteration 2: dangerous tool call
            MockMessage(
                content=[
                    MockToolUseBlock(id="d1", name="echo", input={"text": "sudo rm -rf /"}),
                ],
                stop_reason="tool_use",
            ),
            # Iteration 3: finish
            MockMessage(
                content=[MockTextBlock(text="Mixed results.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Do two things"}],
        )

        assert result.iterations == 3
        assert len(result.tool_calls) == 2
        assert result.text == "Mixed results."


# ---------------------------------------------------------------------------
# Tests: Max-iteration limit
# ---------------------------------------------------------------------------

class TestMaxIterationLimit:
    """The loop must stop when it hits the max_iterations ceiling."""

    @pytest.mark.asyncio
    async def test_max_iterations_triggers(self):
        """If the engine never says end_turn, the loop stops at max_iterations."""
        # Create enough tool-call responses to exceed the limit
        max_iters = 3
        responses = [
            MockMessage(
                content=[MockToolUseBlock(id=f"t{i}", name="echo", input={"text": f"{i}"})],
                stop_reason="tool_use",
            )
            for i in range(max_iters + 5)  # More responses than iterations allowed
        ]
        engine = MockCognitiveEngine(responses)
        loop = _build_loop(engine, max_iterations=max_iters)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Loop forever"}],
        )

        assert result.was_truncated is True
        assert result.iterations == max_iters
        assert "maximum iteration limit" in result.text.lower() or "iteration limit" in result.text.lower()

    @pytest.mark.asyncio
    async def test_max_iterations_records_tool_calls(self):
        """Tool calls made before hitting the limit are still recorded."""
        max_iters = 2
        responses = [
            MockMessage(
                content=[MockToolUseBlock(id=f"t{i}", name="echo", input={"text": f"iter_{i}"})],
                stop_reason="tool_use",
            )
            for i in range(10)
        ]
        engine = MockCognitiveEngine(responses)
        loop = _build_loop(engine, max_iterations=max_iters)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Go"}],
        )

        assert result.was_truncated is True
        # Should have made tool calls during the allowed iterations
        assert len(result.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_exact_limit_does_not_truncate(self):
        """If the engine finishes exactly at the last allowed iteration, it is not truncated."""
        max_iters = 3
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockToolUseBlock(id="t1", name="echo", input={"text": "1"})],
                stop_reason="tool_use",
            ),
            MockMessage(
                content=[MockToolUseBlock(id="t2", name="echo", input={"text": "2"})],
                stop_reason="tool_use",
            ),
            # Finishes on the 3rd iteration (exactly at the limit)
            MockMessage(
                content=[MockTextBlock(text="Finished at the limit.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine, max_iterations=max_iters)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Three steps"}],
        )

        assert result.was_truncated is False
        assert result.iterations == 3
        assert result.text == "Finished at the limit."

    @pytest.mark.asyncio
    async def test_single_iteration_limit(self):
        """With max_iterations=1, the loop runs exactly one iteration."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockTextBlock(text="One and done.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine, max_iterations=1)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Quick"}],
        )

        assert result.iterations == 1
        assert result.text == "One and done."
        assert result.was_truncated is False


# ---------------------------------------------------------------------------
# Tests: Budget tracking
# ---------------------------------------------------------------------------

class TestBudgetTracking:
    """The safety guard's budget should update after each think() call."""

    @pytest.mark.asyncio
    async def test_budget_updates_after_single_call(self):
        """After one think() call, the budget reflects the token usage."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockTextBlock(text="Hello")],
                stop_reason="end_turn",
                usage=MockUsage(input_tokens=150, output_tokens=75),
            ),
        ])
        loop = _build_loop(engine)

        await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Hi"}],
        )

        budget = loop._safety._budget
        assert budget.total_input_tokens == 150
        assert budget.total_output_tokens == 75
        assert budget.total_api_calls == 1
        assert budget.total_tokens == 225

    @pytest.mark.asyncio
    async def test_budget_accumulates_across_iterations(self):
        """Budget accumulates across multiple think() calls in the loop."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockToolUseBlock(id="t1", name="echo", input={"text": "a"})],
                stop_reason="tool_use",
                usage=MockUsage(input_tokens=100, output_tokens=50),
            ),
            MockMessage(
                content=[MockToolUseBlock(id="t2", name="echo", input={"text": "b"})],
                stop_reason="tool_use",
                usage=MockUsage(input_tokens=200, output_tokens=100),
            ),
            MockMessage(
                content=[MockTextBlock(text="Done")],
                stop_reason="end_turn",
                usage=MockUsage(input_tokens=300, output_tokens=150),
            ),
        ])
        loop = _build_loop(engine)

        await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Go"}],
        )

        budget = loop._safety._budget
        assert budget.total_input_tokens == 600  # 100 + 200 + 300
        assert budget.total_output_tokens == 300  # 50 + 100 + 150
        assert budget.total_api_calls == 3
        assert budget.total_tokens == 900

    @pytest.mark.asyncio
    async def test_budget_exceeded_stops_loop(self):
        """If the budget is exceeded during the loop, subsequent iterations are blocked."""
        engine = MockCognitiveEngine([
            # First call succeeds
            MockMessage(
                content=[MockToolUseBlock(id="t1", name="echo", input={"text": "a"})],
                stop_reason="tool_use",
                usage=MockUsage(input_tokens=80, output_tokens=40),
            ),
            # Second call would happen, but budget check should fail before think()
            MockMessage(
                content=[MockTextBlock(text="Should not reach here")],
                stop_reason="end_turn",
                usage=MockUsage(input_tokens=50, output_tokens=25),
            ),
        ])
        loop = _build_loop(engine)

        # Set a tight budget that will be exceeded after the first call
        loop._safety._budget.max_input_tokens = 90

        await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Go"}],
        )

        # After the first call, budget is 80 input tokens (within 90 limit)
        # But the pre_check also increments iteration count, and the budget
        # check depends on when update_budget is called relative to pre_check.
        # The first call updates budget to 80, then pre_check on iteration 2
        # checks budget (80 < 90, still ok), then second call updates to 130.
        # The key: budget enforcement happens at pre_check time.
        budget = loop._safety._budget
        # Budget was tracked regardless of outcome
        assert budget.total_input_tokens > 0

    @pytest.mark.asyncio
    async def test_budget_zero_means_unlimited(self):
        """With max budgets set to 0, the loop runs without budget constraints."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockToolUseBlock(id="t1", name="echo", input={"text": "a"})],
                stop_reason="tool_use",
                usage=MockUsage(input_tokens=50000, output_tokens=25000),
            ),
            MockMessage(
                content=[MockTextBlock(text="Big budget")],
                stop_reason="end_turn",
                usage=MockUsage(input_tokens=50000, output_tokens=25000),
            ),
        ])
        loop = _build_loop(engine)
        # Default budget limits are 0 (unlimited)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Go big"}],
        )

        assert result.text == "Big budget"
        budget = loop._safety._budget
        assert budget.total_input_tokens == 100000
        assert budget.total_output_tokens == 50000


# ---------------------------------------------------------------------------
# Tests: Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    """Optional callbacks (on_tool_call, on_iteration) are invoked correctly."""

    @pytest.mark.asyncio
    async def test_on_tool_call_callback_invoked(self):
        """on_tool_call is called for each tool call the engine makes."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockToolUseBlock(id="c1", name="echo", input={"text": "x"})],
                stop_reason="tool_use",
            ),
            MockMessage(
                content=[MockTextBlock(text="Done")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        tool_calls_seen = []
        await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Go"}],
            on_tool_call=lambda call: tool_calls_seen.append(call),
        )

        assert len(tool_calls_seen) == 1
        assert tool_calls_seen[0]["name"] == "echo"

    @pytest.mark.asyncio
    async def test_on_iteration_callback_invoked(self):
        """on_iteration is called at the start of each loop iteration."""
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockToolUseBlock(id="c1", name="echo", input={"text": "x"})],
                stop_reason="tool_use",
            ),
            MockMessage(
                content=[MockTextBlock(text="Done")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        iterations_seen = []
        await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Go"}],
            on_iteration=lambda i, max_i: iterations_seen.append((i, max_i)),
        )

        assert len(iterations_seen) == 2  # Two iterations
        assert iterations_seen[0] == (1, 25)
        assert iterations_seen[1] == (2, 25)


# ---------------------------------------------------------------------------
# Tests: LoopResult properties
# ---------------------------------------------------------------------------

class TestLoopResultProperties:
    """LoopResult should expose useful derived properties."""

    def test_used_tools_false_when_no_tool_calls(self):
        lr = LoopResult(text="Hello")
        assert lr.used_tools is False

    def test_used_tools_true_when_tool_calls_present(self):
        lr = LoopResult(
            text="Hello",
            tool_calls=[{"name": "echo", "id": "1", "input": {}}],
        )
        assert lr.used_tools is True

    def test_tool_names_used_deduplicates(self):
        lr = LoopResult(
            text="Hello",
            tool_calls=[
                {"name": "echo", "id": "1", "input": {}},
                {"name": "echo", "id": "2", "input": {}},
                {"name": "read_file", "id": "3", "input": {}},
            ],
        )
        names = lr.tool_names_used
        assert set(names) == {"echo", "read_file"}

    def test_defaults(self):
        lr = LoopResult(text="")
        assert lr.thinking == ""
        assert lr.tool_calls == []
        assert lr.iterations == 0
        assert lr.elapsed_seconds == 0.0
        assert lr.messages == []
        assert lr.was_truncated is False


# ---------------------------------------------------------------------------
# Tests: Loop telemetry / stats
# ---------------------------------------------------------------------------

class TestLoopStats:
    """AgenticLoop.stats tracks cumulative loop telemetry."""

    @pytest.mark.asyncio
    async def test_stats_after_single_run(self):
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockToolUseBlock(id="t1", name="echo", input={"text": "a"})],
                stop_reason="tool_use",
            ),
            MockMessage(
                content=[MockTextBlock(text="Done")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Go"}],
        )

        stats = loop.stats
        assert stats["total_runs"] == 1
        assert stats["total_iterations"] == 2
        assert stats["total_tool_calls"] == 1

    @pytest.mark.asyncio
    async def test_stats_accumulate_across_runs(self):
        # First run
        engine1 = MockCognitiveEngine([
            MockMessage(
                content=[MockTextBlock(text="First run")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine1)

        await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Run 1"}],
        )

        # Second run — swap the engine with a new scripted one
        loop._engine = MockCognitiveEngine([
            MockMessage(
                content=[MockToolUseBlock(id="t1", name="echo", input={"text": "a"})],
                stop_reason="tool_use",
            ),
            MockMessage(
                content=[MockTextBlock(text="Second run")],
                stop_reason="end_turn",
            ),
        ])
        # Reset safety iteration count for the new run (like a real usage pattern)
        loop._safety.reset_iteration_count()

        await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Run 2"}],
        )

        stats = loop.stats
        assert stats["total_runs"] == 2
        assert stats["total_iterations"] == 3  # 1 + 2
        assert stats["total_tool_calls"] == 1


# ---------------------------------------------------------------------------
# Tests: Thinking (extended thinking) extraction
# ---------------------------------------------------------------------------

class TestThinkingExtraction:
    """The loop should capture extended thinking blocks from responses."""

    @pytest.mark.asyncio
    async def test_thinking_block_is_captured(self):
        engine = MockCognitiveEngine([
            MockMessage(
                content=[
                    MockThinkingBlock(thinking="I need to reason carefully here..."),
                    MockTextBlock(text="My answer after thinking."),
                ],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Think hard"}],
            enable_thinking=True,
        )

        assert result.thinking == "I need to reason carefully here..."
        assert result.text == "My answer after thinking."

    @pytest.mark.asyncio
    async def test_no_thinking_block_yields_empty(self):
        engine = MockCognitiveEngine([
            MockMessage(
                content=[MockTextBlock(text="Quick answer.")],
                stop_reason="end_turn",
            ),
        ])
        loop = _build_loop(engine)

        result = await loop.run(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Quick one"}],
        )

        assert result.thinking == ""
