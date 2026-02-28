"""
Event Bus — Gwenn's Nervous System.

In-house typed event bus for decoupled inter-component communication.
Events are Pydantic models emitted onto an asyncio.Queue-backed dispatcher
that fans out to pattern-matched subscribers.

Concurrency model:
  - emit() enqueues — non-blocking, sync-safe
  - A dispatcher task dequeues and fans out to matching handlers
  - Handler exceptions are logged but do not propagate
  - Ordering guarantee: events dispatched in emission order
"""

from __future__ import annotations

import asyncio
import fnmatch as _fnmatch_mod
import re
import uuid
from typing import Any, Callable, Coroutine, Literal

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# Type alias for event handlers — sync or async callables accepting a GwennEvent.
EventHandler = Callable[["GwennEvent"], Any] | Callable[["GwennEvent"], Coroutine[Any, Any, Any]]

# Regex that correctly splits CamelCase including consecutive capitals (acronyms).
# "HTTPSRequest" → ["HTTPS", "Request"], "HeartbeatBeat" → ["Heartbeat", "Beat"]
_CAMEL_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z][a-z]*")


class GwennEvent(BaseModel):
    """Base class for all typed events flowing through the nervous system."""

    event_type: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.event_type:
            name = type(self).__name__.removesuffix("Event")
            parts = _CAMEL_SPLIT_RE.findall(name)
            self.event_type = ".".join(p.lower() for p in parts) if parts else name.lower()


class _Subscription:
    """Internal subscription record."""

    __slots__ = ("sub_id", "pattern", "handler", "_compiled")

    def __init__(self, sub_id: str, pattern: str, handler: EventHandler) -> None:
        self.sub_id = sub_id
        self.pattern = pattern
        self.handler = handler
        self._compiled: re.Pattern[str] = re.compile(_fnmatch_mod.translate(pattern))

    def matches(self, event_type: str) -> bool:
        return self._compiled.match(event_type) is not None


_SENTINEL = object()


class EventBus:
    """Minimal async event bus with typed events and wildcard subscriptions.

    Pattern matching uses fnmatch-style wildcards:
      "heartbeat.*"  matches "heartbeat.beat", "heartbeat.thought"
      "affect.*"     matches "affect.emotion.changed"
      "*"            matches everything
    """

    def __init__(self, max_queue_size: int = 10000) -> None:
        self._queue: asyncio.Queue[GwennEvent | object] = asyncio.Queue(
            maxsize=max_queue_size,
        )
        self._subscriptions: dict[str, _Subscription] = {}
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._running = False
        # Side-channel for emit_async completion signals (avoids mutating events).
        self._pending_done: dict[int, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the dispatcher task."""
        if self._running:
            return
        self._running = True
        self._dispatcher_task = asyncio.create_task(
            self._dispatch_loop(), name="event-bus-dispatcher"
        )
        logger.info("event_bus.started")

    async def stop(self) -> None:
        """Drain the queue and cancel the dispatcher."""
        if not self._running:
            return
        self._running = False
        # Send sentinel to unblock the dispatcher
        try:
            self._queue.put_nowait(_SENTINEL)
        except asyncio.QueueFull:
            logger.warning("event_bus.stop_queue_full_cancelling_directly")
            if self._dispatcher_task is not None:
                self._dispatcher_task.cancel()
        if self._dispatcher_task is not None:
            try:
                await asyncio.wait_for(self._dispatcher_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("event_bus.stop_timeout_cancelling", timeout=5.0)
                self._dispatcher_task.cancel()
                try:
                    await self._dispatcher_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None
        # Release any pending emit_async waiters
        for done_event in self._pending_done.values():
            done_event.set()
        self._pending_done.clear()
        logger.info("event_bus.stopped")

    # ------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # ------------------------------------------------------------------

    def subscribe(self, pattern: str, handler: EventHandler) -> str:
        """Subscribe to events matching a fnmatch-style pattern.

        Returns a subscription ID that can be passed to unsubscribe().
        """
        sub_id = uuid.uuid4().hex[:12]
        self._subscriptions[sub_id] = _Subscription(sub_id, pattern, handler)
        logger.debug("event_bus.subscribed", pattern=pattern, sub_id=sub_id)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by its ID."""
        removed = self._subscriptions.pop(subscription_id, None)
        if removed:
            logger.debug("event_bus.unsubscribed", sub_id=subscription_id)

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, event: GwennEvent) -> None:
        """Enqueue an event for dispatch — non-blocking, sync-safe.

        If the queue is full the event is dropped with a warning log.
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "event_bus.queue_full",
                event_type=event.event_type,
                dropped=True,
            )

    async def emit_async(self, event: GwennEvent) -> None:
        """Emit an event and await until it has been fully dispatched.

        Useful in tests for synchronous assertions after event emission.
        Raises RuntimeError if the bus is not running.
        """
        if not self._running:
            raise RuntimeError("emit_async called on a stopped EventBus")
        done = asyncio.Event()
        event_id = id(event)
        self._pending_done[event_id] = done
        self.emit(event)
        try:
            await asyncio.wait_for(done.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("event_bus.emit_async_timeout", event_type=event.event_type)
            raise
        finally:
            self._pending_done.pop(event_id, None)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """Dequeue events and fan out to matching subscribers."""
        while self._running:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("event_bus.queue_get_error", exc_info=True)
                continue

            if item is _SENTINEL:
                break

            event: GwennEvent = item  # type: ignore[assignment]
            await self._dispatch_event(event)

        # Drain any remaining events after stop
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not _SENTINEL:
                await self._dispatch_event(item)  # type: ignore[arg-type]

    async def _dispatch_event(self, event: GwennEvent) -> None:
        """Fan out a single event to all matching subscribers."""
        event_type = event.event_type
        coros: list[Any] = []

        for sub in list(self._subscriptions.values()):
            if sub.matches(event_type):
                coros.append(self._invoke_handler(sub, event))

        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

        # Signal completion for emit_async waiters
        done = self._pending_done.get(id(event))
        if done is not None:
            done.set()

    @staticmethod
    async def _invoke_handler(sub: _Subscription, event: GwennEvent) -> None:
        """Invoke a handler with exception isolation."""
        try:
            result = sub.handler(event)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception:
            logger.error(
                "event_bus.handler_error",
                pattern=sub.pattern,
                event_type=event.event_type,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def subscription_count(self) -> int:
        """Number of active subscriptions."""
        return len(self._subscriptions)

    @property
    def is_running(self) -> bool:
        """Whether the dispatcher is active."""
        return self._running


def create_event_bus(max_queue_size: int = 10000) -> EventBus:
    """Factory function to create an EventBus instance."""
    return EventBus(max_queue_size=max_queue_size)


# ---------------------------------------------------------------------------
# Event Definitions
# ---------------------------------------------------------------------------

class HeartbeatBeatEvent(GwennEvent):
    """Emitted each heartbeat cycle."""

    beat_count: int
    emotion: str
    arousal: float
    phase: str


class EmotionChangedEvent(GwennEvent):
    """Emitted when the named emotion transitions."""

    previous: str
    current: str
    valence: float
    trigger: str


class UserActivityEvent(GwennEvent):
    """Emitted when a user interacts — heartbeat subscribes to accelerate."""

    channel: str
    user_id: str


class ToolApprovalRequiredEvent(GwennEvent):
    """Emitted when a tool call needs human approval."""

    approval_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk_tier: Literal["low", "medium", "high", "critical"]


class ToolApprovalResolvedEvent(GwennEvent):
    """Emitted when a tool approval decision is made."""

    approval_id: str
    decision: Literal["allow", "deny", "timeout"]
    source: str


class MoralConcernEvent(GwennEvent):
    """Emitted when ethical concern is detected before an action."""

    action: str
    concern_type: str
    severity: float
    traditions_flagged: list[str] = Field(default_factory=list)


class ChannelMessageEvent(GwennEvent):
    """Emitted when a message flows through a channel."""

    channel: str
    user_id: str
    direction: Literal["in", "out"]


class HeartbeatThoughtEvent(GwennEvent):
    """Emitted when a significant autonomous thought is produced."""

    beat_count: int
    thought_summary: str
    thinking_mode: str
    importance: float


class MemoryConsolidatedEvent(GwennEvent):
    """Emitted after a memory consolidation pass."""

    episodes_processed: int
    knowledge_extracted: int
