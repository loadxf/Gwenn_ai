"""Tests for gwenn.orchestration.orchestrator — core orchestration logic."""

from __future__ import annotations

import asyncio

import pytest

from gwenn.orchestration.models import SubagentResult, SubagentSpec, SwarmSpec
from gwenn.orchestration.orchestrator import Orchestrator
from gwenn.orchestration.runners import SubagentRunnerBase


class MockConfig:
    """Minimal OrchestrationConfig stand-in for tests."""

    enabled = True
    max_concurrent_subagents = 3
    default_timeout = 30.0
    default_max_iterations = 5
    max_nesting_depth = 3
    subagent_model = ""
    max_total_api_calls = 50
    max_active_swarms = 2
    max_concurrent_api_calls = 2
    default_tools: list[str] = []
    default_runtime = "in_process"
    autonomous_spawn_enabled = True
    autonomous_spawn_cooldown = 0.0  # no cooldown for tests
    autonomous_spawn_max_per_hour = 100


class MockRunner(SubagentRunnerBase):
    """A mock runner that returns a canned result."""

    def __init__(self, result_text: str = "done", delay: float = 0.0):
        self._result_text = result_text
        self._delay = delay

    async def run(self, spec: SubagentSpec) -> SubagentResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        return SubagentResult(
            task_id=spec.task_id,
            status="completed",
            result_text=self._result_text,
            iterations=1,
            elapsed_seconds=self._delay,
        )


class FailingRunner(SubagentRunnerBase):
    async def run(self, spec: SubagentSpec) -> SubagentResult:
        return SubagentResult(
            task_id=spec.task_id,
            status="failed",
            error="intentional failure",
        )


@pytest.fixture
def config():
    return MockConfig()


@pytest.fixture
def orchestrator(config):
    runner = MockRunner()
    return Orchestrator(config=config, runner=runner)


@pytest.mark.asyncio
async def test_spawn_single_task(orchestrator):
    spec = SubagentSpec(task_description="test task")
    task_id = await orchestrator.spawn(spec)

    assert task_id == spec.task_id
    assert orchestrator.active_count >= 0  # may have completed already

    result = await orchestrator.collect_result(task_id)
    assert result is not None
    assert result.status == "completed"
    assert result.result_text == "done"


@pytest.mark.asyncio
async def test_spawn_respects_depth_limit(config):
    runner = MockRunner()
    orch = Orchestrator(config=config, runner=runner)

    spec = SubagentSpec(
        task_description="deep task",
        depth=config.max_nesting_depth,  # at limit
    )

    with pytest.raises(ValueError, match="nesting depth"):
        await orch.spawn(spec)


@pytest.mark.asyncio
async def test_spawn_respects_budget_limit():
    config = MockConfig()
    config.max_total_api_calls = 1

    runner = MockRunner()
    orch = Orchestrator(config=config, runner=runner)

    # First spawn should work
    spec1 = SubagentSpec(task_description="task 1")
    await orch.spawn(spec1)
    await orch.collect_result(spec1.task_id)

    # Budget now exhausted (runner reports 1 iteration)
    spec2 = SubagentSpec(task_description="task 2")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        await orch.spawn(spec2)


@pytest.mark.asyncio
async def test_spawn_disabled():
    config = MockConfig()
    config.enabled = False
    runner = MockRunner()
    orch = Orchestrator(config=config, runner=runner)

    with pytest.raises(RuntimeError, match="disabled"):
        await orch.spawn(SubagentSpec(task_description="test"))


@pytest.mark.asyncio
async def test_spawn_no_runner(config):
    orch = Orchestrator(config=config, runner=None)

    with pytest.raises(RuntimeError, match="runner"):
        await orch.spawn(SubagentSpec(task_description="test"))


@pytest.mark.asyncio
async def test_check_status_running(config):
    runner = MockRunner(delay=0.5)
    orch = Orchestrator(config=config, runner=runner)

    spec = SubagentSpec(task_description="slow task")
    task_id = await orch.spawn(spec)

    # Should be running
    status = await orch.check_status(task_id)
    assert status["status"] == "running"

    # Wait for completion
    await orch.collect_result(task_id)
    status = await orch.check_status(task_id)
    assert status["status"] == "completed"


@pytest.mark.asyncio
async def test_check_status_not_found(orchestrator):
    status = await orchestrator.check_status("nonexistent")
    assert status["status"] == "not_found"


@pytest.mark.asyncio
async def test_cancel_running_task(config):
    runner = MockRunner(delay=10.0)  # will be cancelled before completing
    orch = Orchestrator(config=config, runner=runner)

    spec = SubagentSpec(task_description="long task")
    task_id = await orch.spawn(spec)

    # Cancel it
    cancelled = await orch.cancel(task_id)
    assert cancelled is True

    # Give cancellation time to propagate
    await asyncio.sleep(0.1)

    result = await orch.collect_result(task_id)
    assert result is not None
    assert result.status == "cancelled"


@pytest.mark.asyncio
async def test_spawn_swarm(orchestrator):
    agents = [
        SubagentSpec(task_description="task A"),
        SubagentSpec(task_description="task B"),
    ]
    swarm = SwarmSpec(
        description="parallel test",
        agents=agents,
        aggregation_strategy="concatenate",
    )

    swarm_id = await orchestrator.spawn_swarm(swarm)
    result = await orchestrator.collect_swarm(swarm_id)

    assert result is not None
    assert result.status == "completed"
    assert len(result.individual_results) == 2
    assert "Result 1" in result.aggregated_result
    assert "Result 2" in result.aggregated_result


@pytest.mark.asyncio
async def test_swarm_partial_failure():
    config = MockConfig()

    call_count = 0

    class AlternatingRunner(SubagentRunnerBase):
        async def run(self, spec: SubagentSpec) -> SubagentResult:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return SubagentResult(
                    task_id=spec.task_id,
                    status="failed",
                    error="even task failure",
                )
            return SubagentResult(
                task_id=spec.task_id,
                status="completed",
                result_text=f"result {call_count}",
            )

    orch = Orchestrator(config=config, runner=AlternatingRunner())

    agents = [SubagentSpec(task_description=f"task {i}") for i in range(3)]
    swarm = SwarmSpec(description="mixed", agents=agents)

    swarm_id = await orch.spawn_swarm(swarm)
    result = await orch.collect_swarm(swarm_id)

    assert result is not None
    assert result.status == "partial"


@pytest.mark.asyncio
async def test_collect_completed(config):
    runner = MockRunner()
    orch = Orchestrator(config=config, runner=runner)

    spec = SubagentSpec(task_description="test")
    await orch.spawn(spec)
    await orch.collect_result(spec.task_id)

    completed = orch.collect_completed()
    assert len(completed) == 1
    assert completed[0].task_id == spec.task_id

    # Non-destructive: second call returns the same results
    assert len(orch.collect_completed()) == 1


@pytest.mark.asyncio
async def test_shutdown(config):
    runner = MockRunner(delay=10.0)
    orch = Orchestrator(config=config, runner=runner)

    spec = SubagentSpec(task_description="long task")
    await orch.spawn(spec)

    await orch.shutdown()
    assert orch.active_count == 0


@pytest.mark.asyncio
async def test_aggregate_vote(config):
    class VoteRunner(SubagentRunnerBase):
        async def run(self, spec: SubagentSpec) -> SubagentResult:
            # Most agents say "yes"
            answer = "yes" if "yes" in spec.task_description else "no"
            return SubagentResult(
                task_id=spec.task_id,
                status="completed",
                result_text=answer,
            )

    orch = Orchestrator(config=config, runner=VoteRunner())

    agents = [
        SubagentSpec(task_description="say yes 1"),
        SubagentSpec(task_description="say yes 2"),
        SubagentSpec(task_description="say no"),
    ]
    swarm = SwarmSpec(
        description="vote test",
        agents=agents,
        aggregation_strategy="vote",
    )

    swarm_id = await orch.spawn_swarm(swarm)
    result = await orch.collect_swarm(swarm_id)

    assert result is not None
    assert "yes" in result.aggregated_result


@pytest.mark.asyncio
async def test_autonomous_spawn_rate_limiting(config):
    config.autonomous_spawn_cooldown = 1.0
    config.autonomous_spawn_max_per_hour = 2

    runner = MockRunner()
    orch = Orchestrator(config=config, runner=runner)

    assert orch.can_autonomous_spawn() is True

    orch.record_autonomous_spawn()
    # Should be blocked by cooldown
    assert orch.can_autonomous_spawn() is False

    # After cooldown passes (simulate by manipulating timestamps)
    orch._autonomous_spawn_times[-1] -= 2.0
    assert orch.can_autonomous_spawn() is True

    orch.record_autonomous_spawn()
    orch._autonomous_spawn_times[-1] -= 2.0  # clear cooldown
    # Now at 2 spawns per hour = at limit
    assert orch.can_autonomous_spawn() is False


@pytest.mark.asyncio
async def test_result_truncation(config):
    long_text = "x" * 5000

    class LongRunner(SubagentRunnerBase):
        async def run(self, spec: SubagentSpec) -> SubagentResult:
            return SubagentResult(
                task_id=spec.task_id,
                status="completed",
                result_text=long_text,
            )

    orch = Orchestrator(config=config, runner=LongRunner())
    spec = SubagentSpec(task_description="long output")
    await orch.spawn(spec)

    # Default: truncated
    result = await orch.collect_result(spec.task_id, full=False)
    assert result is not None
    assert len(result.result_text) <= 2100  # 2000 + "[truncated]"

    # Reset for full test
    orch._completed_results[spec.task_id] = SubagentResult(
        task_id=spec.task_id,
        status="completed",
        result_text=long_text,
    )

    # Full: not truncated
    result_full = await orch.collect_result(spec.task_id, full=True)
    assert result_full is not None
    assert len(result_full.result_text) == 5000
