"""Tests for typed inter-agent messages and swarm visualization events."""

from __future__ import annotations

import asyncio

import pytest

from gwenn.events import (
    AgentMessage,
    BroadcastMessage,
    CompletionMessage,
    EscalationMessage,
    EventBus,
    GwennEvent,
    RequestHelpMessage,
    StatusUpdateMessage,
    SwarmBotAcquiredEvent,
    SwarmBotReleasedEvent,
    SwarmTurnEvent,
    TaskDispatchMessage,
)


# ---------------------------------------------------------------------------
# Swarm Visualization Events
# ---------------------------------------------------------------------------


class TestSwarmEvents:
    def test_swarm_bot_acquired_event_type(self) -> None:
        event = SwarmBotAcquiredEvent(
            swarm_id="swarm-1", task_id="t1", bot_name="bot1", persona_name="Researcher"
        )
        assert event.event_type == "swarm.bot.acquired"

    def test_swarm_bot_released_event_type(self) -> None:
        event = SwarmBotReleasedEvent(
            swarm_id="swarm-1", task_id="t1", bot_name="bot1"
        )
        assert event.event_type == "swarm.bot.released"

    def test_swarm_turn_event_type(self) -> None:
        event = SwarmTurnEvent(
            swarm_id="swarm-1", task_id="t1", bot_name="bot1", message_preview="Hello!"
        )
        assert event.event_type == "swarm.turn"

    def test_swarm_events_serialize(self) -> None:
        event = SwarmBotAcquiredEvent(
            swarm_id="swarm-1", task_id="t1", bot_name="bot1", persona_name="Alpha"
        )
        data = event.model_dump()
        assert data["swarm_id"] == "swarm-1"
        assert data["persona_name"] == "Alpha"
        assert data["event_type"] == "swarm.bot.acquired"


# ---------------------------------------------------------------------------
# Typed Inter-Agent Messages
# ---------------------------------------------------------------------------


class TestAgentMessageBase:
    def test_agent_message_fields(self) -> None:
        msg = AgentMessage(
            sender_task_id="coordinator",
            recipient_task_id="sub-1",
            swarm_id="swarm-1",
        )
        assert msg.sender_task_id == "coordinator"
        assert msg.recipient_task_id == "sub-1"
        assert msg.swarm_id == "swarm-1"

    def test_agent_message_default_swarm_id(self) -> None:
        msg = AgentMessage(
            sender_task_id="a", recipient_task_id="b"
        )
        assert msg.swarm_id is None


class TestTaskDispatchMessage:
    def test_event_type(self) -> None:
        msg = TaskDispatchMessage(
            sender_task_id="coordinator",
            recipient_task_id="sub-1",
            task_description="Research quantum computing",
        )
        assert msg.event_type == "task.dispatch.message"

    def test_with_persona(self) -> None:
        msg = TaskDispatchMessage(
            sender_task_id="coordinator",
            recipient_task_id="sub-1",
            task_description="Research",
            assigned_persona_name="Researcher",
        )
        assert msg.assigned_persona_name == "Researcher"

    def test_without_persona(self) -> None:
        msg = TaskDispatchMessage(
            sender_task_id="coordinator",
            recipient_task_id="sub-1",
            task_description="Task",
        )
        assert msg.assigned_persona_name is None


class TestStatusUpdateMessage:
    def test_event_type(self) -> None:
        msg = StatusUpdateMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            status="researching",
        )
        assert msg.event_type == "status.update.message"

    def test_with_progress(self) -> None:
        msg = StatusUpdateMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            status="implementing",
            progress_pct=0.75,
            detail="75% complete",
        )
        assert msg.progress_pct == 0.75
        assert msg.detail == "75% complete"


class TestCompletionMessage:
    def test_event_type(self) -> None:
        msg = CompletionMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            result_text="Done!",
        )
        assert msg.event_type == "completion.message"

    def test_with_files(self) -> None:
        msg = CompletionMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            result_text="Modified 2 files",
            files_modified=["a.py", "b.py"],
            success=True,
        )
        assert len(msg.files_modified) == 2
        assert msg.success is True

    def test_failure(self) -> None:
        msg = CompletionMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            result_text="Failed",
            success=False,
        )
        assert msg.success is False


class TestEscalationMessage:
    def test_event_type(self) -> None:
        msg = EscalationMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            reason="blocked",
            detail="Cannot access file",
        )
        assert msg.event_type == "escalation.message"

    def test_with_suggested_action(self) -> None:
        msg = EscalationMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            reason="needs_approval",
            detail="Needs permission",
            suggested_action="Ask user to approve",
        )
        assert msg.suggested_action == "Ask user to approve"


class TestRequestHelpMessage:
    def test_event_type(self) -> None:
        msg = RequestHelpMessage(
            sender_task_id="sub-1",
            recipient_task_id="sub-2",
            request="Help with parsing",
        )
        assert msg.event_type == "request.help.message"

    def test_with_context(self) -> None:
        msg = RequestHelpMessage(
            sender_task_id="sub-1",
            recipient_task_id="sub-2",
            request="Review my output",
            context="I wrote a parser module",
        )
        assert msg.context == "I wrote a parser module"


class TestBroadcastMessage:
    def test_event_type(self) -> None:
        msg = BroadcastMessage(
            sender_task_id="coordinator",
            content="Direction change!",
        )
        assert msg.event_type == "broadcast.message"

    def test_default_recipient(self) -> None:
        msg = BroadcastMessage(
            sender_task_id="coordinator",
            content="Attention all agents",
        )
        assert msg.recipient_task_id == "all"


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestMessageSerialization:
    def test_task_dispatch_round_trip(self) -> None:
        msg = TaskDispatchMessage(
            sender_task_id="coordinator",
            recipient_task_id="sub-1",
            task_description="Test task",
            assigned_persona_name="Researcher",
        )
        data = msg.model_dump()
        restored = TaskDispatchMessage.model_validate(data)
        assert restored.task_description == "Test task"
        assert restored.assigned_persona_name == "Researcher"

    def test_completion_message_round_trip(self) -> None:
        msg = CompletionMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            result_text="All done",
            files_modified=["x.py"],
            success=True,
        )
        data = msg.model_dump()
        restored = CompletionMessage.model_validate(data)
        assert restored.result_text == "All done"
        assert restored.files_modified == ["x.py"]

    def test_escalation_message_round_trip(self) -> None:
        msg = EscalationMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            reason="blocked",
            detail="Stuck on API call",
        )
        data = msg.model_dump()
        restored = EscalationMessage.model_validate(data)
        assert restored.reason == "blocked"


# ---------------------------------------------------------------------------
# Event Bus Integration
# ---------------------------------------------------------------------------


class TestEventBusRouting:
    @pytest.mark.asyncio
    async def test_wildcard_catches_all_agent_messages(self) -> None:
        bus = EventBus()
        await bus.start()
        captured: list[GwennEvent] = []

        bus.subscribe("*", lambda e: captured.append(e))

        msg = TaskDispatchMessage(
            sender_task_id="coordinator",
            recipient_task_id="sub-1",
            task_description="Do stuff",
        )
        await bus.emit_async(msg)

        assert len(captured) == 1
        assert isinstance(captured[0], TaskDispatchMessage)
        await bus.stop()

    @pytest.mark.asyncio
    async def test_pattern_filters_agent_messages(self) -> None:
        bus = EventBus()
        await bus.start()
        completions: list[GwennEvent] = []
        escalations: list[GwennEvent] = []

        bus.subscribe("completion.*", lambda e: completions.append(e))
        bus.subscribe("escalation.*", lambda e: escalations.append(e))

        await bus.emit_async(CompletionMessage(
            sender_task_id="sub-1",
            recipient_task_id="coordinator",
            result_text="Done",
        ))
        await bus.emit_async(EscalationMessage(
            sender_task_id="sub-2",
            recipient_task_id="coordinator",
            reason="blocked",
            detail="Help",
        ))

        assert len(completions) == 1
        assert len(escalations) == 1
        await bus.stop()

    @pytest.mark.asyncio
    async def test_swarm_events_on_bus(self) -> None:
        bus = EventBus()
        await bus.start()
        captured: list[GwennEvent] = []

        bus.subscribe("swarm.*", lambda e: captured.append(e))

        await bus.emit_async(SwarmBotAcquiredEvent(
            swarm_id="s1", task_id="t1", bot_name="b1", persona_name="R"
        ))
        await bus.emit_async(SwarmBotReleasedEvent(
            swarm_id="s1", task_id="t1", bot_name="b1"
        ))
        await bus.emit_async(SwarmTurnEvent(
            swarm_id="s1", task_id="t1", bot_name="b1", message_preview="Hi"
        ))

        assert len(captured) == 3
        await bus.stop()
