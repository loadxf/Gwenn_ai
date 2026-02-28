"""Tests for orchestrator swarm visualization integration (Phase 6).

Tests that the orchestrator correctly acquires/releases bots from the pool,
emits swarm events and typed inter-agent messages during spawn/complete.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gwenn.events import (
    CompletionMessage,
    EventBus,
    GwennEvent,
    SwarmBotAcquiredEvent,
    SwarmBotReleasedEvent,
    TaskDispatchMessage,
)
from gwenn.orchestration.models import (
    SubagentPersona,
    SubagentResult,
    SubagentSpec,
    SwarmSpec,
)
from gwenn.orchestration.orchestrator import Orchestrator
from gwenn.orchestration.runners import SubagentRunnerBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockConfig:
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
    autonomous_spawn_cooldown = 0.0
    autonomous_spawn_max_per_hour = 100


class MockRunner(SubagentRunnerBase):
    async def run(self, spec: SubagentSpec) -> SubagentResult:
        return SubagentResult(
            task_id=spec.task_id,
            status="completed",
            result_text="done",
            iterations=1,
            elapsed_seconds=0.0,
        )


def _persona(name: str = "Researcher") -> SubagentPersona:
    return SubagentPersona(name=name, role="Research specialist")


# ---------------------------------------------------------------------------
# Orchestrator with event bus
# ---------------------------------------------------------------------------


class TestOrchestratorEventEmission:
    @pytest.mark.asyncio
    async def test_spawn_emits_task_dispatch_message(self) -> None:
        bus = EventBus()
        await bus.start()
        captured: list[GwennEvent] = []
        bus.subscribe("*", lambda e: captured.append(e))

        orch = Orchestrator(
            config=MockConfig(),
            runner=MockRunner(),
            event_bus=bus,
        )

        spec = SubagentSpec(
            task_description="Test task",
            persona=_persona(),
        )
        task_id = await orch.spawn(spec)
        result = await orch.collect_result(task_id)
        # Allow event propagation
        await asyncio.sleep(0.05)

        dispatch_msgs = [e for e in captured if isinstance(e, TaskDispatchMessage)]
        assert len(dispatch_msgs) >= 1
        assert dispatch_msgs[0].recipient_task_id == task_id
        assert dispatch_msgs[0].assigned_persona_name == "Researcher"

        completion_msgs = [e for e in captured if isinstance(e, CompletionMessage)]
        assert len(completion_msgs) >= 1
        assert completion_msgs[0].sender_task_id == task_id

        await bus.stop()

    @pytest.mark.asyncio
    async def test_spawn_without_persona_skips_dispatch(self) -> None:
        bus = EventBus()
        await bus.start()
        captured: list[GwennEvent] = []
        bus.subscribe("*", lambda e: captured.append(e))

        orch = Orchestrator(
            config=MockConfig(),
            runner=MockRunner(),
            event_bus=bus,
        )

        spec = SubagentSpec(task_description="No persona task")
        task_id = await orch.spawn(spec)
        result = await orch.collect_result(task_id)
        await asyncio.sleep(0.05)

        dispatch_msgs = [e for e in captured if isinstance(e, TaskDispatchMessage)]
        assert len(dispatch_msgs) == 0

        await bus.stop()


# ---------------------------------------------------------------------------
# Orchestrator with bot pool
# ---------------------------------------------------------------------------


class TestOrchestratorBotPool:
    @pytest.mark.asyncio
    async def test_spawn_swarm_acquires_bots(self) -> None:
        bus = EventBus()
        await bus.start()
        captured: list[GwennEvent] = []
        bus.subscribe("*", lambda e: captured.append(e))

        mock_pool = AsyncMock()
        mock_slot = MagicMock()
        mock_slot.bot_username = "bot1"
        mock_pool.acquire = AsyncMock(return_value=mock_slot)
        mock_pool.get_slot_for_task = MagicMock(return_value=mock_slot)
        mock_pool.release = AsyncMock()
        mock_pool.release_all = AsyncMock()

        orch = Orchestrator(
            config=MockConfig(),
            runner=MockRunner(),
            event_bus=bus,
            bot_pool=mock_pool,
        )

        swarm = SwarmSpec(
            agents=[
                SubagentSpec(task_description="T1", persona=_persona("Alpha")),
                SubagentSpec(task_description="T2", persona=_persona("Beta")),
            ]
        )

        swarm_id = await orch.spawn_swarm(swarm)
        assert mock_pool.acquire.call_count == 2

        # Collect swarm to trigger release
        await orch.collect_swarm(swarm_id)
        await asyncio.sleep(0.05)

        # Bots should have been released
        assert mock_pool.release.call_count == 2

        # Check events emitted
        acquired_events = [e for e in captured if isinstance(e, SwarmBotAcquiredEvent)]
        released_events = [e for e in captured if isinstance(e, SwarmBotReleasedEvent)]
        assert len(acquired_events) == 2
        assert len(released_events) == 2

        await bus.stop()

    @pytest.mark.asyncio
    async def test_spawn_swarm_skips_agents_without_persona(self) -> None:
        mock_pool = AsyncMock()
        mock_slot = MagicMock()
        mock_slot.bot_username = "bot1"
        mock_pool.acquire = AsyncMock(return_value=mock_slot)
        mock_pool.get_slot_for_task = MagicMock(return_value=None)
        mock_pool.release = AsyncMock()
        mock_pool.release_all = AsyncMock()

        orch = Orchestrator(
            config=MockConfig(),
            runner=MockRunner(),
            bot_pool=mock_pool,
        )

        swarm = SwarmSpec(
            agents=[
                SubagentSpec(task_description="T1", persona=_persona("Alpha")),
                SubagentSpec(task_description="T2"),  # No persona
            ]
        )

        await orch.spawn_swarm(swarm)
        # Only 1 acquire call (the one with persona)
        assert mock_pool.acquire.call_count == 1

    @pytest.mark.asyncio
    async def test_spawn_swarm_without_pool(self) -> None:
        """Swarm works fine without a bot pool (no crash, no events)."""
        orch = Orchestrator(
            config=MockConfig(),
            runner=MockRunner(),
        )

        swarm = SwarmSpec(
            agents=[
                SubagentSpec(task_description="T1", persona=_persona("Alpha")),
            ]
        )

        swarm_id = await orch.spawn_swarm(swarm)
        result = await orch.collect_swarm(swarm_id)
        assert result is not None
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# Orchestrator — send_swarm_message
# ---------------------------------------------------------------------------


class TestSendSwarmMessage:
    @pytest.mark.asyncio
    async def test_send_swarm_message_via_pool(self) -> None:
        bus = EventBus()
        await bus.start()
        captured: list[GwennEvent] = []
        bus.subscribe("*", lambda e: captured.append(e))

        mock_pool = AsyncMock()
        mock_slot = MagicMock()
        mock_slot.bot_username = "bot1"
        mock_pool.get_slot_for_task = MagicMock(return_value=mock_slot)
        mock_pool.send_as = AsyncMock()

        orch = Orchestrator(
            config=MockConfig(),
            runner=MockRunner(),
            event_bus=bus,
            bot_pool=mock_pool,
        )
        # Register origin session for routing
        orch._origin_sessions["task-1"] = "telegram_chat:123"

        ok = await orch.send_swarm_message("swarm-1", "task-1", "Hello world!")
        assert ok is True
        mock_pool.send_as.assert_called_once_with(mock_slot, 123, None, "Hello world!")
        await asyncio.sleep(0.05)

        from gwenn.events import SwarmTurnEvent
        turn_events = [e for e in captured if isinstance(e, SwarmTurnEvent)]
        assert len(turn_events) == 1
        assert turn_events[0].message_preview == "Hello world!"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_send_swarm_message_no_pool(self) -> None:
        orch = Orchestrator(config=MockConfig(), runner=MockRunner())
        ok = await orch.send_swarm_message("swarm-1", "task-1", "msg")
        assert ok is False

    @pytest.mark.asyncio
    async def test_send_swarm_message_no_slot(self) -> None:
        mock_pool = AsyncMock()
        mock_pool.get_slot_for_task = MagicMock(return_value=None)

        orch = Orchestrator(
            config=MockConfig(), runner=MockRunner(), bot_pool=mock_pool
        )
        ok = await orch.send_swarm_message("swarm-1", "task-1", "msg")
        assert ok is False

    @pytest.mark.asyncio
    async def test_send_swarm_message_non_telegram_session(self) -> None:
        mock_pool = AsyncMock()
        mock_slot = MagicMock()
        mock_pool.get_slot_for_task = MagicMock(return_value=mock_slot)

        orch = Orchestrator(
            config=MockConfig(), runner=MockRunner(), bot_pool=mock_pool
        )
        orch._origin_sessions["task-1"] = "discord_channel:456"

        ok = await orch.send_swarm_message("swarm-1", "task-1", "msg")
        assert ok is False


# ---------------------------------------------------------------------------
# Orchestrator — shutdown releases bots
# ---------------------------------------------------------------------------


class TestOrchestratorShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_releases_bots(self) -> None:
        mock_pool = AsyncMock()
        mock_pool.release_all = AsyncMock()

        orch = Orchestrator(
            config=MockConfig(),
            runner=MockRunner(),
            bot_pool=mock_pool,
        )

        await orch.shutdown()
        mock_pool.release_all.assert_called_once()
