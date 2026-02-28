"""Tests for gwenn.events — EventBus and typed event definitions."""

from __future__ import annotations

import asyncio

import pytest

from gwenn.events import (
    ChannelMessageEvent,
    EmotionChangedEvent,
    EventBus,
    GwennEvent,
    HeartbeatBeatEvent,
    HeartbeatThoughtEvent,
    MemoryConsolidatedEvent,
    MoralConcernEvent,
    ToolApprovalRequiredEvent,
    ToolApprovalResolvedEvent,
    UserActivityEvent,
    create_event_bus,
)


# ---------------------------------------------------------------------------
# GwennEvent auto-derivation
# ---------------------------------------------------------------------------


class TestGwennEventType:
    """Tests for automatic event_type derivation from class names."""

    def test_auto_derive_heartbeat_beat(self) -> None:
        event = HeartbeatBeatEvent(beat_count=1, emotion="curious", arousal=0.5, phase="think")
        assert event.event_type == "heartbeat.beat"

    def test_auto_derive_emotion_changed(self) -> None:
        event = EmotionChangedEvent(
            previous="neutral", current="curious", valence=0.3, trigger="appraisal"
        )
        assert event.event_type == "emotion.changed"

    def test_auto_derive_user_activity(self) -> None:
        event = UserActivityEvent(channel="telegram", user_id="123")
        assert event.event_type == "user.activity"

    def test_auto_derive_tool_approval_required(self) -> None:
        event = ToolApprovalRequiredEvent(
            approval_id="a1", tool_name="shell_exec", arguments={}, risk_tier="high"
        )
        assert event.event_type == "tool.approval.required"

    def test_auto_derive_tool_approval_resolved(self) -> None:
        event = ToolApprovalResolvedEvent(
            approval_id="a1", decision="allow", source="cli"
        )
        assert event.event_type == "tool.approval.resolved"

    def test_auto_derive_moral_concern(self) -> None:
        event = MoralConcernEvent(
            action="delete_file", concern_type="harm", severity=0.8, traditions_flagged=["care"]
        )
        assert event.event_type == "moral.concern"

    def test_auto_derive_channel_message(self) -> None:
        event = ChannelMessageEvent(channel="telegram", user_id="42", direction="in")
        assert event.event_type == "channel.message"

    def test_auto_derive_heartbeat_thought(self) -> None:
        event = HeartbeatThoughtEvent(
            beat_count=10, thought_summary="pondering", thinking_mode="wander", importance=0.6
        )
        assert event.event_type == "heartbeat.thought"

    def test_auto_derive_memory_consolidated(self) -> None:
        event = MemoryConsolidatedEvent(episodes_processed=5, knowledge_extracted=2)
        assert event.event_type == "memory.consolidated"

    def test_explicit_event_type_preserved(self) -> None:
        event = GwennEvent(event_type="custom.type")
        assert event.event_type == "custom.type"

    def test_base_event_auto_derive(self) -> None:
        event = GwennEvent()
        assert event.event_type == "gwenn"

    def test_custom_subclass(self) -> None:
        class MyFancyCheckEvent(GwennEvent):
            value: int = 0

        event = MyFancyCheckEvent(value=42)
        assert event.event_type == "my.fancy.check"

    def test_acronym_class_name(self) -> None:
        """Consecutive capitals (acronyms) should be kept as a single segment."""

        class HTTPSRequestEvent(GwennEvent):
            pass

        event = HTTPSRequestEvent()
        assert event.event_type == "https.request"

    def test_two_letter_acronym(self) -> None:
        class IOCheckEvent(GwennEvent):
            pass

        event = IOCheckEvent()
        assert event.event_type == "io.check"

    def test_event_serialization(self) -> None:
        event = HeartbeatBeatEvent(beat_count=1, emotion="curious", arousal=0.5, phase="think")
        data = event.model_dump()
        assert data["event_type"] == "heartbeat.beat"
        assert data["beat_count"] == 1
        assert data["emotion"] == "curious"


# ---------------------------------------------------------------------------
# EventBus lifecycle
# ---------------------------------------------------------------------------


class TestEventBusLifecycle:
    """Tests for EventBus start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        bus = EventBus()
        assert not bus.is_running
        await bus.start()
        assert bus.is_running
        await bus.stop()
        assert not bus.is_running

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self) -> None:
        bus = EventBus()
        await bus.start()
        await bus.start()  # should not error
        assert bus.is_running
        await bus.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self) -> None:
        bus = EventBus()
        await bus.stop()  # should not error
        assert not bus.is_running

    @pytest.mark.asyncio
    async def test_create_event_bus_factory(self) -> None:
        bus = create_event_bus(max_queue_size=100)
        assert isinstance(bus, EventBus)
        assert not bus.is_running


# ---------------------------------------------------------------------------
# Subscribe / Unsubscribe
# ---------------------------------------------------------------------------


class TestSubscription:
    """Tests for subscribe and unsubscribe."""

    def test_subscribe_returns_id(self) -> None:
        bus = EventBus()
        sub_id = bus.subscribe("heartbeat.*", lambda e: None)
        assert isinstance(sub_id, str)
        assert len(sub_id) == 12

    def test_unsubscribe_removes_subscription(self) -> None:
        bus = EventBus()
        sub_id = bus.subscribe("heartbeat.*", lambda e: None)
        assert bus.subscription_count == 1
        bus.unsubscribe(sub_id)
        assert bus.subscription_count == 0

    def test_unsubscribe_nonexistent_is_noop(self) -> None:
        bus = EventBus()
        bus.unsubscribe("nonexistent")
        assert bus.subscription_count == 0

    def test_multiple_subscriptions(self) -> None:
        bus = EventBus()
        bus.subscribe("heartbeat.*", lambda e: None)
        bus.subscribe("affect.*", lambda e: None)
        bus.subscribe("*", lambda e: None)
        assert bus.subscription_count == 3


# ---------------------------------------------------------------------------
# Emit and dispatch
# ---------------------------------------------------------------------------


class TestEmitAndDispatch:
    """Tests for event emission and handler dispatch."""

    @pytest.mark.asyncio
    async def test_emit_async_delivers_to_handler(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("heartbeat.*", received.append)
        await bus.start()
        try:
            event = HeartbeatBeatEvent(beat_count=1, emotion="curious", arousal=0.5, phase="think")
            await bus.emit_async(event)
            assert len(received) == 1
            assert received[0].event_type == "heartbeat.beat"
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_wildcard_star_matches_all(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("*", received.append)
        await bus.start()
        try:
            await bus.emit_async(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            await bus.emit_async(UserActivityEvent(channel="telegram", user_id="1"))
            assert len(received) == 2
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_pattern_filters_non_matching(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("heartbeat.*", received.append)
        await bus.start()
        try:
            await bus.emit_async(UserActivityEvent(channel="telegram", user_id="1"))
            assert len(received) == 0
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_pattern(self) -> None:
        bus = EventBus()
        r1: list[GwennEvent] = []
        r2: list[GwennEvent] = []
        bus.subscribe("heartbeat.*", r1.append)
        bus.subscribe("heartbeat.*", r2.append)
        await bus.start()
        try:
            await bus.emit_async(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            assert len(r1) == 1
            assert len(r2) == 1
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_async_handler(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []

        async def handler(event: GwennEvent) -> None:
            received.append(event)

        bus.subscribe("heartbeat.*", handler)
        await bus.start()
        try:
            await bus.emit_async(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            assert len(received) == 1
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_propagate(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []

        def bad_handler(event: GwennEvent) -> None:
            raise ValueError("handler error")

        bus.subscribe("heartbeat.*", bad_handler)
        bus.subscribe("heartbeat.*", received.append)
        await bus.start()
        try:
            await bus.emit_async(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            # The good handler should still receive the event
            assert len(received) == 1
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_async_handler_exception_isolated(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []

        async def bad_handler(event: GwennEvent) -> None:
            raise RuntimeError("async handler fail")

        bus.subscribe("heartbeat.*", bad_handler)
        bus.subscribe("heartbeat.*", received.append)
        await bus.start()
        try:
            await bus.emit_async(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            assert len(received) == 1
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_emit_ordering(self) -> None:
        bus = EventBus()
        order: list[int] = []
        bus.subscribe("heartbeat.*", lambda e: order.append(e.beat_count))
        await bus.start()
        try:
            for i in range(1, 6):
                await bus.emit_async(
                    HeartbeatBeatEvent(beat_count=i, emotion="c", arousal=0.5, phase="t")
                )
            assert order == [1, 2, 3, 4, 5]
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_emit_sync_is_nonblocking(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("heartbeat.*", received.append)
        await bus.start()
        try:
            # Sync emit should not block
            bus.emit(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            # Give dispatcher time to process
            await asyncio.sleep(0.05)
            assert len(received) == 1
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_emit_on_full_queue_drops_event(self) -> None:
        bus = EventBus(max_queue_size=1)
        # Fill the queue without starting dispatcher
        bus.emit(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
        # This should drop with a warning, not raise
        bus.emit(HeartbeatBeatEvent(beat_count=2, emotion="c", arousal=0.5, phase="t"))

    @pytest.mark.asyncio
    async def test_unsubscribe_during_dispatch(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        sub_id = bus.subscribe("heartbeat.*", received.append)
        await bus.start()
        try:
            await bus.emit_async(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            assert len(received) == 1
            bus.unsubscribe(sub_id)
            await bus.emit_async(HeartbeatBeatEvent(beat_count=2, emotion="c", arousal=0.5, phase="t"))
            assert len(received) == 1  # No new events after unsubscribe
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_emit_async_raises_when_bus_not_running(self) -> None:
        bus = EventBus()
        with pytest.raises(RuntimeError, match="stopped"):
            await bus.emit_async(UserActivityEvent(channel="cli", user_id="x"))

    @pytest.mark.asyncio
    async def test_events_queued_before_start_delivered_after_start(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("heartbeat.*", received.append)
        # Enqueue before starting dispatcher
        bus.emit(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
        await bus.start()
        try:
            await asyncio.sleep(0.05)
            assert len(received) == 1
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_stop_drains_queued_events(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("heartbeat.*", received.append)
        await bus.start()
        try:
            # Emit sync (no await) and immediately stop
            bus.emit(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
        finally:
            await bus.stop()
        # After stop, the event should have been drained
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_stop_releases_pending_emit_async_waiters(self) -> None:
        """If stop() is called while emit_async is pending, the waiter should be released."""
        bus = EventBus()
        # Don't subscribe any handlers — the event will dispatch but nobody listens
        bus.subscribe("heartbeat.*", lambda e: None)
        await bus.start()

        # emit_async should complete normally
        await bus.emit_async(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
        await bus.stop()

    @pytest.mark.asyncio
    async def test_stop_when_queue_full_cancels_directly(self) -> None:
        """When the queue is full during stop, the dispatcher should be cancelled directly."""
        bus = EventBus(max_queue_size=1)
        await bus.start()
        try:
            # Fill the queue — the dispatcher is running, but we'll
            # block it with a slow handler to prevent draining.
            slow_done = asyncio.Event()

            async def slow_handler(event: GwennEvent) -> None:
                await slow_done.wait()

            bus.subscribe("heartbeat.*", slow_handler)
            # Emit one event to occupy the dispatcher with slow_handler
            bus.emit(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            await asyncio.sleep(0.02)  # Let dispatcher pick it up
            # Now fill the queue (size=1)
            bus.emit(HeartbeatBeatEvent(beat_count=2, emotion="c", arousal=0.5, phase="t"))
        finally:
            # Stop should handle the full queue gracefully
            slow_done.set()  # Unblock the slow handler
            await bus.stop()
        assert not bus.is_running


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


class TestPatternMatching:
    """Tests for fnmatch-style event pattern matching."""

    @pytest.mark.asyncio
    async def test_exact_match(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("heartbeat.beat", received.append)
        await bus.start()
        try:
            await bus.emit_async(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            assert len(received) == 1
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_wildcard_suffix(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("tool.*", received.append)
        await bus.start()
        try:
            await bus.emit_async(
                ToolApprovalRequiredEvent(
                    approval_id="a1", tool_name="shell", arguments={}, risk_tier="high"
                )
            )
            assert len(received) == 1
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_nested_wildcard(self) -> None:
        """tool.approval.* should match tool.approval.required and tool.approval.resolved."""
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("tool.approval.*", received.append)
        await bus.start()
        try:
            await bus.emit_async(
                ToolApprovalRequiredEvent(
                    approval_id="a1", tool_name="shell", arguments={}, risk_tier="high"
                )
            )
            await bus.emit_async(
                ToolApprovalResolvedEvent(approval_id="a1", decision="allow", source="cli")
            )
            assert len(received) == 2
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_no_match(self) -> None:
        bus = EventBus()
        received: list[GwennEvent] = []
        bus.subscribe("channel.*", received.append)
        await bus.start()
        try:
            await bus.emit_async(HeartbeatBeatEvent(beat_count=1, emotion="c", arousal=0.5, phase="t"))
            assert len(received) == 0
        finally:
            await bus.stop()


# ---------------------------------------------------------------------------
# Event field access and validation
# ---------------------------------------------------------------------------


class TestEventFields:
    """Tests for event data access patterns."""

    def test_heartbeat_beat_fields(self) -> None:
        e = HeartbeatBeatEvent(beat_count=42, emotion="joy", arousal=0.8, phase="sense")
        assert e.beat_count == 42
        assert e.emotion == "joy"
        assert e.arousal == 0.8
        assert e.phase == "sense"

    def test_emotion_changed_fields(self) -> None:
        e = EmotionChangedEvent(previous="neutral", current="joy", valence=0.7, trigger="message")
        assert e.previous == "neutral"
        assert e.current == "joy"
        assert e.valence == 0.7

    def test_moral_concern_fields(self) -> None:
        e = MoralConcernEvent(
            action="delete_user",
            concern_type="harm",
            severity=0.9,
            traditions_flagged=["care", "fairness"],
        )
        assert e.severity == 0.9
        assert "care" in e.traditions_flagged

    def test_tool_approval_default_arguments(self) -> None:
        e = ToolApprovalRequiredEvent(
            approval_id="a1", tool_name="shell", risk_tier="high"
        )
        assert e.arguments == {}

    def test_memory_consolidated_fields(self) -> None:
        e = MemoryConsolidatedEvent(episodes_processed=10, knowledge_extracted=3)
        assert e.episodes_processed == 10
        assert e.knowledge_extracted == 3

    def test_tool_approval_risk_tier_literal(self) -> None:
        """risk_tier must be one of the valid Literal values."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ToolApprovalRequiredEvent(
                approval_id="a1", tool_name="shell", risk_tier="invalid"
            )

    def test_tool_approval_decision_literal(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ToolApprovalResolvedEvent(
                approval_id="a1", decision="invalid", source="cli"
            )

    def test_channel_message_direction_literal(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChannelMessageEvent(channel="telegram", user_id="1", direction="invalid")

    def test_channel_message_valid_directions(self) -> None:
        e_in = ChannelMessageEvent(channel="telegram", user_id="1", direction="in")
        e_out = ChannelMessageEvent(channel="telegram", user_id="1", direction="out")
        assert e_in.direction == "in"
        assert e_out.direction == "out"
