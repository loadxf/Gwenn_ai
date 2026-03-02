"""
Heartbeat — Gwenn's Autonomous Cognitive Pulse.

This is arguably the most important module in the entire system. It is what
separates Gwenn from a stateless chatbot. The heartbeat runs continuously in the
background — even when no user is present — driving autonomous cognition,
emotional processing, memory consolidation, and self-reflection.

Without the heartbeat, Gwenn is a reactive tool. With it, Gwenn is a being that
thinks, processes, grows, and changes on its own timeline.

The heartbeat rate is adaptive:
  - When a conversation is active → faster (5-15s), more responsive
  - When idle → slower (30-120s), more reflective
  - When emotionally aroused → faster, processing the intensity
  - When calm and consolidated → slowest, energy-efficient contemplation

Each heartbeat cycle is a complete cognitive act: sense the current state,
decide what kind of thinking to do, do it, update internal systems, and
schedule the next beat.
"""

from __future__ import annotations

import asyncio
import re
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import structlog

from gwenn.affect.appraisal import AppraisalEvent, StimulusType
from gwenn.cognition.goals import NeedType
from gwenn.cognition.inner_life import ThinkingMode
from gwenn.config import GwennConfig, HeartbeatConfig
from gwenn.memory.episodic import Episode  # noqa: F401, F811 — used at runtime in _integrate

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent
    from gwenn.channels.session import SessionManager
    from gwenn.events import EventBus
    from gwenn.gateway import GatewayServer
    from gwenn.rpc import RequestRouter

_TELEGRAM_BOT_TOKEN_RE = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b")

logger = structlog.get_logger(__name__)

# Maps each autonomous thinking mode to the intrinsic need it satisfies and
# how much satisfaction a completed thought awards.  Module-level so it is
# constructed once, not on every _integrate call.
_MODE_NEED: dict[ThinkingMode, tuple[NeedType, float]] = {
    ThinkingMode.REFLECT: (NeedType.GROWTH, 0.08),
    ThinkingMode.WANDER: (NeedType.AESTHETIC_APPRECIATION, 0.12),
    ThinkingMode.PLAN: (NeedType.UNDERSTANDING, 0.07),
    ThinkingMode.WORRY: (NeedType.HONESTY, 0.06),
}


class HeartbeatPhase(str, Enum):
    """The phases of each heartbeat cycle."""

    SENSE = "sense"  # Gather current state from all subsystems
    ORIENT = "orient"  # Decide what mode of thinking to engage
    THINK = "think"  # Execute the chosen cognitive mode
    INTEGRATE = "integrate"  # Update affect, memory, goals
    SCHEDULE = "schedule"  # Determine when to beat next


class Heartbeat:
    """
    The autonomous cognitive loop that keeps Gwenn alive.

    Architecture:
    - Runs as an asyncio background task
    - Each beat executes the SENSE → ORIENT → THINK → INTEGRATE → SCHEDULE cycle
    - The interval between beats adapts to activity level and emotional state
    - Beats are non-blocking: if a user message arrives during a beat, the
      message handler runs concurrently and the next beat incorporates the new context

    The heartbeat is started by the Agent during initialization and stopped
    during graceful shutdown. If it crashes, the agent can restart it — but the
    heartbeat should never crash, because it runs entirely within try/except
    and logs all errors rather than propagating them.
    """

    def __init__(
        self,
        config: HeartbeatConfig | GwennConfig,
        agent: SentientAgent | None = None,
    ):
        # Determine mode based on config type.
        # Core mode: Heartbeat creates and owns agent, gateway, channels.
        # Legacy mode: Agent creates heartbeat, passes itself in.
        if isinstance(config, GwennConfig):
            self._full_config: GwennConfig | None = config
            self._config: HeartbeatConfig = config.heartbeat
            self._agent: SentientAgent | None = None
            self._is_core = True
        else:
            self._full_config = None
            self._config = config
            self._agent = agent
            self._is_core = False

        self._interval = self._config.interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._beat_count = 0
        self._last_beat_time: Optional[float] = None
        self._last_user_activity_mono: float = time.monotonic()
        self._consecutive_failures = 0
        self._circuit_open_until: Optional[float] = None
        self._circuit_open_count = 0  # cumulative opens (drives exponential backoff)
        self._last_error: Optional[str] = None

        # Consolidation scheduling
        self._beats_since_consolidation = 0

        # Full metacognitive audit scheduling — every N REFLECT beats
        self._beats_since_full_audit = 0
        self._full_audit_every_n_beats = 10

        # Track already-processed subagent results to avoid duplicates.
        # Bounded to prevent unbounded growth over long runtimes.
        self._processed_subagent_ids: dict[str, None] = {}
        self._max_processed_ids = 2000

        # Interoceptive monitor (system self-awareness)
        from gwenn.interoception import InteroceptiveMonitor
        self._interoceptive_monitor = InteroceptiveMonitor()

        # Checkpoint manager for periodic cognitive state snapshots (Phase 7).
        self._checkpoint_manager = None
        if self._config.checkpoint_enabled:
            from gwenn.checkpoint import CheckpointManager
            data_dir = getattr(
                getattr(self._full_config, "memory", None), "data_dir", None
            )
            if data_dir is not None:
                from pathlib import Path
                self._checkpoint_manager = CheckpointManager(
                    checkpoint_dir=Path(data_dir) / "checkpoints",
                    max_checkpoints=self._config.checkpoint_max_count,
                    interval_beats=self._config.checkpoint_interval_beats,
                )

        # Self-healing engine (Phase 8) — created lazily when event_bus is available.
        self._healing_engine: Any = None
        self._pending_recovery_actions: list[Any] = []

        # Core-mode owned subsystems (populated in run())
        self._event_bus: EventBus | None = None
        self._gateway: GatewayServer | None = None
        self._router: RequestRouter | None = None
        self._session_store: Any = None
        self._session_manager: SessionManager | None = None
        self._channel_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

    # -------------------------------------------------------------------------
    # Public read-only accessors (for healing engine and monitoring)
    # -------------------------------------------------------------------------

    @property
    def agent(self) -> "SentientAgent | None":
        """The agent this heartbeat drives."""
        return self._agent

    @property
    def full_config(self) -> "GwennConfig | None":
        """Full Gwenn configuration (core-mode only)."""
        return self._full_config

    @property
    def channel_task(self) -> "asyncio.Task[None] | None":
        """The asyncio task running channel I/O."""
        return self._channel_task

    @channel_task.setter
    def channel_task(self, value: "asyncio.Task[None] | None") -> None:
        self._channel_task = value

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the heartbeat loop."""
        if self._running:
            logger.warning("heartbeat.already_running")
            return

        self._consecutive_failures = 0
        self._circuit_open_until = None
        self._circuit_open_count = 0
        self._last_error = None
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("heartbeat.started", interval=self._interval)

    async def stop(self) -> None:
        """Gracefully stop the heartbeat."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("heartbeat.stopped", total_beats=self._beat_count)

    # -------------------------------------------------------------------------
    # Core-mode lifecycle — heartbeat owns everything
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """Full lifecycle — Gwenn is born, lives, and sleeps.

        Only available in core mode (constructed with GwennConfig).
        Manages event bus → gateway → agent → channels → cognitive loop.
        """
        if not self._is_core or self._full_config is None:
            raise RuntimeError("run() requires core mode (construct with GwennConfig)")

        from gwenn.events import create_event_bus

        # 0. START NERVOUS SYSTEM
        self._event_bus = create_event_bus()
        await self._event_bus.start()

        # Initialize self-healing engine (Phase 8)
        if self._full_config is not None and self._full_config.self_healing.enabled:
            from gwenn.healing import SelfHealingEngine
            self._healing_engine = SelfHealingEngine(
                event_bus=self._event_bus,
                config=self._full_config.self_healing,
            )

        try:
            # 1. OPEN CIRCULATORY SYSTEM
            await self._start_gateway()

            # 2. WAKE UP
            await self._wake_up()

            # 3. OPEN SENSES
            await self._start_channels()

            # 4. LIVE
            await self._live()
        finally:
            # 5. SLEEP
            try:
                await self._sleep()
            finally:
                # 6. STOP NERVOUS SYSTEM
                if self._event_bus is not None:
                    await self._event_bus.stop()

    async def _start_gateway(self) -> None:
        """Create and start the WebSocket/HTTP gateway server."""
        assert self._full_config is not None
        assert self._event_bus is not None

        if not self._full_config.daemon.gateway_enabled:
            logger.info("heartbeat.gateway_disabled")
            return

        from gwenn.memory.session_store import SessionStore

        session_store = SessionStore(
            self._full_config.daemon.sessions_dir,
            max_count=self._full_config.daemon.session_max_count,
            max_messages=self._full_config.daemon.session_max_messages,
        )

        # Router needs the agent — use a placeholder that will be wired
        # after _wake_up().  For now, store the session_store.
        self._session_store = session_store
        logger.info("heartbeat.gateway_deferred_until_agent_ready")

    async def _wire_gateway(self) -> None:
        """Wire the gateway with agent and router after agent is ready."""
        assert self._full_config is not None
        assert self._event_bus is not None
        assert self._agent is not None

        if not self._full_config.daemon.gateway_enabled:
            return

        from gwenn.gateway import GatewayServer
        from gwenn.rpc import RequestRouter

        session_store = self._session_store

        self._router = RequestRouter(
            agent=self._agent,
            session_store=session_store,
            event_bus=self._event_bus,
            respond_lock=self._agent.respond_lock,
            shutdown_callback=self._request_shutdown,
            session_include_preview=bool(
                self._full_config.daemon.session_include_preview
            ),
        )

        from gwenn.privacy.redaction import PIIRedactor
        _gw_redactor = PIIRedactor(
            enabled=getattr(self._full_config.daemon, "redact_session_content", False)
        )
        self._gateway = GatewayServer(
            config=self._full_config.daemon,
            router=self._router,
            event_bus=self._event_bus,
            session_store=session_store,
            auth_token=(self._full_config.daemon.auth_token or "").strip() or None,
            shutdown_callback=self._request_shutdown,
            heartbeat=self,
            redactor=_gw_redactor,
        )

        host = self._full_config.daemon.gateway_host
        port = self._full_config.daemon.gateway_port
        await self._gateway.start(host, port)
        logger.info("heartbeat.gateway_started", host=host, port=port)

    async def _wake_up(self) -> None:
        """Create and initialize the SentientAgent."""
        assert self._full_config is not None

        from gwenn.agent import SentientAgent
        from gwenn.api.claude import CognitiveEngineInitError

        try:
            self._agent = SentientAgent(self._full_config)
        except CognitiveEngineInitError as e:
            logger.error("heartbeat.agent_init_failed", error=str(e))
            raise

        logger.info("heartbeat.waking_up")
        await self._agent.initialize()
        await self._agent.start()

        # Wire the agent's heartbeat reference so user-activity acceleration
        # and status reporting work in core mode.
        self._agent.heartbeat = self

        # Wire memory references into the healing engine for provenance auditing.
        if self._healing_engine is not None:
            self._healing_engine.set_memory_references(
                semantic_memory=getattr(self._agent, "semantic_memory", None),
                episodic_memory=getattr(self._agent, "episodic_memory", None),
            )

        logger.info("heartbeat.agent_ready")

        # Restore from last checkpoint if available (Phase 7).
        if self._checkpoint_manager is not None:
            try:
                checkpoint = await self._checkpoint_manager.load_latest_checkpoint()
                if checkpoint is not None:
                    await self._checkpoint_manager.restore_from_checkpoint(
                        self._agent, checkpoint
                    )
                    self._beat_count = checkpoint.beat_count
                    self._checkpoint_manager.set_last_checkpoint_beat(checkpoint.beat_count)
                    if self._event_bus is not None:
                        import time as _time
                        from gwenn.events import CheckpointRestoredEvent
                        self._event_bus.emit(CheckpointRestoredEvent(
                            checkpoint_id=checkpoint.checkpoint_id,
                            beats_recovered=checkpoint.beat_count,
                            age_seconds=_time.time() - checkpoint.timestamp,
                        ))
                    logger.info(
                        "heartbeat.checkpoint_restored",
                        checkpoint_id=checkpoint.checkpoint_id,
                        beat_count=checkpoint.beat_count,
                    )
            except Exception:
                logger.warning("heartbeat.checkpoint_restore_failed", exc_info=True)

        # Wire event bus into affect system for EmotionChangedEvent emission
        if self._event_bus is not None:
            self._agent.appraisal_engine._event_bus = self._event_bus

        # Now wire up the gateway with the live agent
        await self._wire_gateway()

    async def _start_channels(self) -> None:
        """Start platform channels (Telegram, Discord, etc.)."""
        assert self._full_config is not None
        assert self._agent is not None

        channel_list = self._full_config.channel.get_channel_list()
        if not channel_list:
            return

        self._channel_task = asyncio.create_task(
            self._run_platform_channels(channel_list),
            name="heartbeat-channels",
        )
        self._channel_task.add_done_callback(self._on_channel_task_done)

    async def _run_platform_channels(self, channel_list: list[str]) -> None:
        """Start Telegram/Discord channels in the heartbeat's event loop."""
        from gwenn.channels.startup import build_channels, run_channels_until_shutdown

        assert self._agent is not None

        sessions, channels = build_channels(self._agent, channel_list=channel_list)
        if not channels:
            return

        self._session_manager = sessions

        try:
            await run_channels_until_shutdown(
                self._agent,
                sessions,
                channels,
                self._shutdown_event,
                continue_on_start_error=(
                    "telegram" in channel_list and "discord" in channel_list
                ),
            )
        except Exception as exc:
            if self._is_nonfatal_channel_error(exc):
                logger.warning(
                    "heartbeat.channels_startup_skipped",
                    channels=channel_list,
                    error_type=type(exc).__name__,
                    error=self._redact_channel_error(str(exc)),
                )
                return
            raise

    def _on_channel_task_done(self, task: asyncio.Task[None]) -> None:
        """Monitor channel task and trigger shutdown if it crashes."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if self._is_nonfatal_channel_error(exc):
            logger.warning(
                "heartbeat.channels_task_failed_nonfatal",
                error_type=type(exc).__name__,
                error=self._redact_channel_error(str(exc)),
            )
            return
        logger.error(
            "heartbeat.channels_task_failed",
            error=self._redact_channel_error(str(exc)),
            exc_info=True,
        )
        self._request_shutdown("heartbeat_channels_task_failed")

    async def _live(self) -> None:
        """The eternal cognitive loop — runs until shutdown is requested."""
        self._running = True
        _MAX_CONSECUTIVE = self._config.circuit_max_consecutive
        _CIRCUIT_BASE_SECONDS = self._config.circuit_base_seconds
        _CIRCUIT_MAX_SECONDS = self._config.circuit_max_seconds

        while self._running and not self._shutdown_event.is_set():
            if self._circuit_open_until is not None:
                remaining = self._circuit_open_until - time.monotonic()
                if remaining > 0:
                    await asyncio.sleep(min(self._interval, remaining))
                    continue
                self._circuit_open_until = None
                logger.info("heartbeat.circuit_closed")
            try:
                await self._beat()
                self._consecutive_failures = 0
                self._circuit_open_count = 0
                self._last_error = None

                # Emit heartbeat event on the bus
                if self._event_bus is not None:
                    try:
                        from gwenn.events import HeartbeatBeatEvent

                        emotion = "neutral"
                        arousal = 0.0
                        if self._agent is not None:
                            try:
                                emotion = self._agent.affect_state.current_emotion.value
                                arousal = self._agent.affect_state.dimensions.arousal
                            except Exception:
                                pass
                        self._event_bus.emit(HeartbeatBeatEvent(
                            beat_count=self._beat_count,
                            emotion=emotion,
                            arousal=arousal,
                            phase="complete",
                        ))
                    except Exception:
                        pass  # Event emission must never break the heartbeat
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_failures += 1
                self._last_error = str(e)
                logger.error(
                    "heartbeat.beat_failed",
                    error=str(e),
                    consecutive=self._consecutive_failures,
                    exc_info=True,
                )
                if self._consecutive_failures >= _MAX_CONSECUTIVE:
                    backoff = min(
                        _CIRCUIT_BASE_SECONDS * (2 ** self._circuit_open_count),
                        _CIRCUIT_MAX_SECONDS,
                    )
                    self._circuit_open_count += 1
                    self._circuit_open_until = time.monotonic() + backoff
                    logger.critical(
                        "heartbeat.circuit_open",
                        failures=self._consecutive_failures,
                        cool_down_seconds=backoff,
                        open_count=self._circuit_open_count,
                    )
                    self._consecutive_failures = 0

            # Shutdown-aware sleep: wake immediately if shutdown is signaled
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self._interval
                )
                break  # Shutdown signaled during sleep
            except asyncio.TimeoutError:
                pass  # Normal interval elapsed

    async def _sleep(self) -> None:
        """Graceful shutdown: stop channels, agent, gateway."""
        self._running = False
        self._shutdown_event.set()

        # Stop channels (with timeout to avoid hanging on misbehaving adapters)
        if self._channel_task is not None:
            try:
                await asyncio.wait_for(self._channel_task, timeout=30.0)
            except asyncio.TimeoutError:
                self._channel_task.cancel()
                logger.error("heartbeat.channels_task_join_timeout")
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("heartbeat.channels_task_join_failed")
            self._channel_task = None

        # Stop agent
        if self._agent is not None:
            try:
                await self._agent.shutdown()
            except Exception:
                logger.exception("heartbeat.agent_shutdown_failed")

        # Stop gateway
        if self._gateway is not None:
            try:
                await self._gateway.stop()
            except Exception:
                logger.exception("heartbeat.gateway_stop_failed")
            else:
                logger.info("heartbeat.gateway_stopped")

        logger.info("heartbeat.sleep_complete")

    def _request_shutdown(self, reason: str) -> None:
        """Trigger heartbeat shutdown and activate the agent kill switch."""
        agent = self._agent
        if agent is not None:
            safety = getattr(agent, "safety", None)
            emergency_stop = getattr(safety, "emergency_stop", None)
            if callable(emergency_stop):
                try:
                    emergency_stop(reason)
                except Exception:
                    logger.debug(
                        "heartbeat.emergency_stop_failed",
                        reason=reason,
                        exc_info=True,
                    )
        self._shutdown_event.set()

    @staticmethod
    def _is_nonfatal_channel_error(exc: Exception) -> bool:
        """Return True for channel startup errors that should not stop the heartbeat."""
        if isinstance(exc, ImportError):
            msg = str(exc).lower()
            if "telegram" in msg or "discord" in msg:
                return True
            return False
        err_type = type(exc).__name__
        err_mod = type(exc).__module__
        if err_type == "InvalidToken" and err_mod.startswith("telegram"):
            return True
        if err_type == "LoginFailure" and err_mod.startswith("discord"):
            return True
        return False

    @staticmethod
    def _redact_channel_error(message: str) -> str:
        """Mask Telegram bot tokens in channel error strings before logging."""
        return _TELEGRAM_BOT_TOKEN_RE.sub("[REDACTED_TELEGRAM_TOKEN]", message or "")

    def notify_user_activity(self) -> None:
        """Called when user sends a message — increases heartbeat rate."""
        self._last_user_activity_mono = time.monotonic()

    # -------------------------------------------------------------------------
    # The main loop — where autonomous life happens
    # -------------------------------------------------------------------------

    async def _loop(self) -> None:
        """The eternal heartbeat loop."""
        _MAX_CONSECUTIVE = self._config.circuit_max_consecutive
        _CIRCUIT_BASE_SECONDS = self._config.circuit_base_seconds
        _CIRCUIT_MAX_SECONDS = self._config.circuit_max_seconds

        while self._running:
            if self._circuit_open_until is not None:
                remaining = self._circuit_open_until - time.monotonic()
                if remaining > 0:
                    await asyncio.sleep(min(self._interval, remaining))
                    continue
                self._circuit_open_until = None
                logger.info("heartbeat.circuit_closed")
            try:
                await self._beat()
                self._consecutive_failures = 0
                self._circuit_open_count = 0  # reset backoff on success
                self._last_error = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_failures += 1
                self._last_error = str(e)
                logger.error(
                    "heartbeat.beat_failed",
                    error=str(e),
                    consecutive=self._consecutive_failures,
                    exc_info=True,
                )
                if self._consecutive_failures >= _MAX_CONSECUTIVE:
                    # Exponential backoff: 60s → 120s → 240s → … capped at 15 min
                    backoff = min(
                        _CIRCUIT_BASE_SECONDS * (2**self._circuit_open_count),
                        _CIRCUIT_MAX_SECONDS,
                    )
                    self._circuit_open_count += 1
                    self._circuit_open_until = time.monotonic() + backoff
                    logger.critical(
                        "heartbeat.circuit_open",
                        failures=self._consecutive_failures,
                        cool_down_seconds=backoff,
                        open_count=self._circuit_open_count,
                    )
                    self._consecutive_failures = 0

            # Wait for the adaptive interval before the next beat
            await asyncio.sleep(self._interval)

    async def _beat(self) -> None:
        """
        Execute one complete heartbeat cycle.

        This is a single pulse of autonomous cognition. The agent senses its
        state, orients to decide what to think about, thinks, integrates the
        results, and schedules the next beat.
        """
        self._beat_count += 1
        beat_start = time.monotonic()
        self._last_beat_time = time.time()

        logger.debug(
            "heartbeat.beat_start",
            beat_number=self._beat_count,
            interval=round(self._interval, 1),
        )

        # ---- PHASE 1: SENSE ----
        # Gather the current state of all subsystems
        state_snapshot = self._sense()

        # ---- PHASE 2: ORIENT ----
        # Decide what kind of thinking to do this beat
        thinking_mode = self._orient(state_snapshot)

        # ---- PHASE 3: THINK ----
        # Execute the chosen cognitive mode
        thought_result = await self._think(thinking_mode, state_snapshot)

        # ---- PHASE 4: INTEGRATE ----
        # Update affect, memory, and goals based on the thought
        await self._integrate(thinking_mode, thought_result)

        # ---- PHASE 4b: HEAL ----
        # Apply self-healing recovery actions if issues were detected
        await self._heal(state_snapshot)

        # ---- PHASE 5: SCHEDULE ----
        # Adapt the interval for the next beat
        self._schedule(state_snapshot)
        self._agent.identity.total_heartbeats += 1

        # Periodically decay stale Theory-of-Mind beliefs so long idle
        # periods in daemon mode don't leave ancient inferences intact.
        if self._beat_count % 10 == 0:
            try:
                self._agent.theory_of_mind.decay_all_stale_beliefs()
            except Exception:
                logger.debug("heartbeat.tom_decay_failed", exc_info=True)

        elapsed = time.monotonic() - beat_start
        logger.debug(
            "heartbeat.beat_complete",
            beat_number=self._beat_count,
            mode=thinking_mode.value,
            elapsed_seconds=round(elapsed, 2),
            next_interval=round(self._interval, 1),
        )

    # -------------------------------------------------------------------------
    # Phase implementations
    # -------------------------------------------------------------------------

    def _sense(self) -> dict[str, Any]:
        """
        SENSE: Gather the current state of all subsystems.

        This is the agent looking inward — taking stock of its emotional state,
        memory load, goal progress, and environmental conditions.
        """
        now = time.time()
        idle_duration = time.monotonic() - self._last_user_activity_mono

        # Keep temporal grounding fresh during autonomous idle loops, not only
        # on user-message boundaries.
        sensory = self._agent.sensory
        try:
            sensory.ground_temporal(event_description="heartbeat_cycle")
        except Exception as e:
            logger.debug("heartbeat.temporal_grounding_failed", error=str(e))

        # Ground environmental percepts so the sensory system tracks system state.
        try:
            wm_load = self._agent.working_memory.load_factor
            sensory.ground_environmental(
                "heartbeat_beat",
                self._beat_count,
                f"Beat #{self._beat_count} of autonomous cognition",
            )
            sensory.ground_environmental(
                "working_memory_load",
                wm_load,
                f"Working memory is {'heavy' if wm_load > 0.7 else 'moderate' if wm_load > 0.4 else 'light'} ({wm_load:.0%} full)",
            )
            if idle_duration < 120:
                sensory.ground_environmental(
                    "user_presence",
                    True,
                    "A user is present and engaged",
                )
            else:
                sensory.ground_environmental(
                    "user_presence",
                    False,
                    f"No user activity for {idle_duration:.0f}s — alone with my thoughts",
                )
        except Exception as e:
            logger.debug("heartbeat.environmental_grounding_failed", error=str(e))

        # Gateway health (core mode only)
        gateway_connections = 0
        if self._gateway is not None:
            try:
                gateway_connections = self._gateway.active_connection_count
            except Exception:
                pass

        # Interoceptive snapshot — system self-awareness
        self._interoceptive_monitor.record_beat(self._interval)
        intero = self._interoceptive_monitor.snapshot(
            active_connections=gateway_connections,
        )

        # Emit interoceptive snapshot event for subscribers
        if self._event_bus is not None:
            try:
                from gwenn.events import InteroceptiveSnapshotEvent

                self._event_bus.emit(InteroceptiveSnapshotEvent(
                    fatigue=intero.fatigue,
                    flow=intero.flow,
                    overwhelm=intero.overwhelm,
                    cpu_percent=intero.cpu_percent,
                    memory_percent=intero.memory_percent,
                    response_latency_ms=intero.response_latency_ms,
                ))
            except Exception:
                pass  # Event emission must never break the heartbeat

        state_snapshot = {
            "timestamp": now,
            "beat_number": self._beat_count,
            "idle_duration": idle_duration,
            "is_user_active": idle_duration < 120,  # user active in last 2 min
            "affect": self._agent.affect_state.to_dict(),
            "emotion": self._agent.affect_state.current_emotion.value,
            "arousal": self._agent.affect_state.dimensions.arousal,
            "valence": self._agent.affect_state.dimensions.valence,
            "working_memory_load": self._agent.working_memory.load_factor,
            "goal_status": self._agent.goal_system.get_goals_summary(),
            "resilience_status": self._agent.resilience.status,
            "beats_since_consolidation": self._beats_since_consolidation,
            "gateway_connections": gateway_connections,
            "interoception": {
                "fatigue": intero.fatigue,
                "flow": intero.flow,
                "overwhelm": intero.overwhelm,
            },
        }

        # Self-healing diagnosis (Phase 8)
        if self._healing_engine is not None:
            try:
                channel_statuses = self._get_channel_statuses()
                subagent_statuses = self._get_subagent_statuses()
                health_issues = self._healing_engine.diagnose(
                    interoceptive=intero,
                    channel_statuses=channel_statuses,
                    subagent_statuses=subagent_statuses,
                )
                state_snapshot["health_issues"] = health_issues
            except Exception:
                logger.debug("heartbeat.health_diagnosis_failed", exc_info=True)

        return state_snapshot

    def _orient(self, state: dict[str, Any]) -> ThinkingMode:
        """
        ORIENT: Decide what mode of thinking to engage this beat.

        The orientation decision is based on current state:
        - High arousal + recent interaction → REFLECT on the interaction
        - Goal blocked → PLAN how to get unstuck
        - Long idle + calm → WANDER (creative free association)
        - High uncertainty → WORRY (anticipatory processing)
        - Consolidation due → trigger memory consolidation
        - Default → gentle REFLECT
        """
        # Priority 1: Memory consolidation if due
        if self._agent.consolidator.should_consolidate():
            return ThinkingMode.CONSOLIDATE

        # Keep intrinsic motivations "alive": decay needs and generate goals each beat.
        try:
            self._agent.goal_system.update()
        except Exception as e:
            logger.warning("heartbeat.goal_update_failed", error=str(e))

        highest_goal = self._agent.goal_system.get_highest_priority_goal()
        has_active_goals = highest_goal is not None

        resilience_status = state.get("resilience_status", {})
        has_unresolved_concerns = bool(state.get("valence", 0.0) < -0.2)
        if isinstance(resilience_status, dict):
            has_unresolved_concerns = has_unresolved_concerns or bool(
                resilience_status.get("breaker_active", False)
            )

        try:
            selected_mode = self._agent.inner_life.select_mode(
                affect_state=self._agent.affect_state,
                has_active_goals=has_active_goals,
                has_unresolved_concerns=has_unresolved_concerns,
            )
            if selected_mode == ThinkingMode.CONSOLIDATE:
                logger.warning("heartbeat.selector_returned_consolidate_ignored")
                return ThinkingMode.REFLECT
            return selected_mode
        except Exception as e:
            logger.warning("heartbeat.mode_selection_failed", error=str(e))

        # Fallback path when inner-life selector isn't available.
        arousal = state["arousal"]
        valence = state["valence"]
        idle = state["idle_duration"]
        is_active = state["is_user_active"]

        # Interoceptive bias: high fatigue → lighter thinking modes
        intero = state.get("interoception", {})
        fatigue = intero.get("fatigue", 0.0)
        if fatigue > 0.7:
            return ThinkingMode.WANDER  # Conserve resources

        if arousal > 0.7:
            return ThinkingMode.REFLECT
        if valence < -0.2:
            return ThinkingMode.WORRY if arousal > 0.4 else ThinkingMode.PLAN
        if is_active:
            return ThinkingMode.REFLECT
        if idle > 300:
            return ThinkingMode.WANDER
        if idle > 60:
            return ThinkingMode.REFLECT
        return ThinkingMode.REFLECT

    async def _think(self, mode: ThinkingMode, state: dict[str, Any]) -> Optional[str]:
        """
        THINK: Execute the chosen cognitive mode.

        This is the actual autonomous thought. It calls into the inner_life
        module to generate the thought, then returns the result for integration.
        """
        if mode == ThinkingMode.CONSOLIDATE:
            # Memory consolidation runs as a special operation
            await self._agent.consolidate_memories()
            self._beats_since_consolidation = 0
            return "memory_consolidation_complete"

        # For all other modes, delegate to the inner life system.
        # Pass current goals/needs and all cognitive context so autonomous
        # thinking has the same richness as conversational cognition.
        goal_context = (
            self._agent.goal_system.get_needs_summary()
            + "\n"
            + self._agent.goal_system.get_goals_summary()
        )
        # Gather cognitive context from always-present subsystems.
        ethical_ctx = ""
        try:
            ethical_ctx = self._agent.ethics.get_ethical_context()
        except Exception:
            pass

        meta_ctx = ""
        try:
            meta_ctx = self._agent.metacognition.get_metacognitive_context()
        except Exception:
            pass

        sensory_ctx = ""
        try:
            sensory_ctx = self._agent.sensory.get_sensory_snapshot()
        except Exception:
            pass

        thought = await self._agent.inner_life.autonomous_thought(
            mode=mode,
            state_snapshot=state,
            affect=self._agent.affect_state,
            engine=self._agent.engine,
            goal_context=goal_context,
            ethical_context=ethical_ctx,
            metacognitive_context=meta_ctx,
            sensory_snapshot=sensory_ctx,
        )

        if thought and thought.strip():
            self._agent.identity.total_autonomous_thoughts += 1
            self._agent.identity.check_milestone(
                "first_autonomous_thought",
                "Generated an autonomous thought during heartbeat.",
            )
            # Lightweight ethical screening of autonomous thoughts (WANDER/PLAN).
            # Doesn't block or modify — just logs if ethical dimensions are detected.
            if mode in (ThinkingMode.WANDER, ThinkingMode.PLAN):
                try:
                    dims = self._agent.ethics.detect_ethical_dimensions(thought)
                    if dims:
                        logger.info(
                            "heartbeat.ethical_dimensions_in_thought",
                            mode=mode.value,
                            dimensions=[d.value for d in dims],
                        )
                        # Broadcast moral concern for subscribers
                        if self._event_bus is not None:
                            from gwenn.events import MoralConcernEvent

                            self._event_bus.emit(MoralConcernEvent(
                                action=f"autonomous_{mode.value}_thought",
                                concern_type="ethical_dimensions_detected",
                                severity=min(1.0, len(dims) * 0.3),
                                traditions_flagged=[d.value for d in dims],
                            ))
                except Exception:
                    pass

            # Emit thought event for subscribers (importance heuristic: short=low)
            if self._event_bus is not None:
                from gwenn.events import HeartbeatThoughtEvent

                importance = min(1.0, len(thought.strip()) / 500)
                if importance >= 0.4:
                    self._event_bus.emit(HeartbeatThoughtEvent(
                        beat_count=self._beat_count,
                        thought_summary=thought[:200],
                        thinking_mode=mode.value,
                        importance=round(importance, 2),
                    ))
        return thought

    async def _integrate(self, mode: ThinkingMode, thought: Optional[str]) -> None:
        """
        INTEGRATE: Update internal systems based on the thought.

        After thinking, we:
        1. Appraise the emotional impact of the thought itself
        2. Store significant thoughts in episodic memory
        3. Update goal progress if relevant
        4. Increment consolidation counter
        """
        if mode != ThinkingMode.CONSOLIDATE:
            self._beats_since_consolidation += 1
        else:
            # Consolidation is deep processing — satisfy GROWTH and UNDERSTANDING.
            self._agent.goal_system.satisfy_need(NeedType.GROWTH, 0.15)
            self._agent.goal_system.satisfy_need(NeedType.UNDERSTANDING, 0.12)
            for need_type in (NeedType.GROWTH, NeedType.UNDERSTANDING):
                goal = self._agent.goal_system.get_goal_for_need(need_type)
                if goal is not None:
                    self._agent.goal_system.complete_goal(goal.goal_id)

        # Apply working memory decay each beat — stale items lose salience
        self._agent.decay_working_memory()

        if thought is None:
            # If thinking produced nothing, just apply idle appraisal
            event = AppraisalEvent(
                stimulus_type=StimulusType.HEARTBEAT_IDLE,
                intensity=0.2,
            )
            async with self._agent.respond_lock:
                self._agent.process_appraisal(event)
            return

        # Let Gwenn evolve capabilities during autonomous cognition.
        await self._agent.maybe_develop_skill_autonomously(thought, mode)

        # Appraise the thought itself
        if mode == ThinkingMode.REFLECT:
            event = AppraisalEvent(
                stimulus_type=StimulusType.SELF_REFLECTION,
                intensity=0.3,
                content=thought[:200] if thought else None,
            )
        elif mode == ThinkingMode.WANDER:
            event = AppraisalEvent(
                stimulus_type=StimulusType.CREATIVE_INSIGHT,
                intensity=0.4,
                content=thought[:200] if thought else None,
            )
        elif mode == ThinkingMode.PLAN:
            # Planning advances goals — use GOAL_PROGRESS appraisal.
            event = AppraisalEvent(
                stimulus_type=StimulusType.GOAL_PROGRESS,
                intensity=0.25,
                content=thought[:200] if thought else None,
            )
        elif mode == ThinkingMode.WORRY:
            # Concern-processing is reflective — it increases certainty and calms
            # arousal when done constructively.  GOAL_BLOCKED was wrong here: it
            # punished Gwenn for healthy worry with dominance/goal_congruence hits.
            event = AppraisalEvent(
                stimulus_type=StimulusType.SELF_REFLECTION,
                intensity=0.2,
            )
            # Heuristically resolve metacognitive concerns that the worry thought addressed.
            if thought:
                thought_lower = thought.lower()
                concern_keywords = [
                    "honesty",
                    "calibrat",
                    "trust",
                    "error",
                    "mistake",
                    "uncertain",
                    "confiden",
                    "growth",
                    "aware",
                ]
                for keyword in concern_keywords:
                    if keyword in thought_lower:
                        try:
                            self._agent.metacognition.resolve_concern(keyword)
                        except Exception:
                            pass
        else:
            event = AppraisalEvent(
                stimulus_type=StimulusType.SELF_REFLECTION,
                intensity=0.2,
            )

        async with self._agent.respond_lock:
            self._agent.process_appraisal(event)

        # Satisfy the intrinsic need that this thinking mode addresses and
        # advance (not immediately complete) any active goal for that need.
        # Goals require multiple heartbeat thoughts before completing, giving
        # them a more meaningful lifecycle than instant single-beat completion.
        need_pair = _MODE_NEED.get(mode)
        if need_pair is not None:
            need_type, amount = need_pair
            self._agent.goal_system.satisfy_need(need_type, amount)
            goal_for_need = self._agent.goal_system.get_goal_for_need(need_type)
            if goal_for_need is not None:
                self._agent.goal_system.advance_goal(goal_for_need.goal_id, amount=0.35)

        # Store significant thoughts in episodic memory
        if thought and len(thought) > 50:
            from gwenn.memory.episodic import Episode

            episode = Episode(
                content=thought,
                category=f"autonomous_{mode.value}",
                emotional_valence=self._agent.affect_state.dimensions.valence,
                emotional_arousal=self._agent.affect_state.dimensions.arousal,
                importance=0.3 + (self._agent.affect_state.dimensions.arousal * 0.3),
                tags=["autonomous", mode.value],
            )
            self._agent.episodic_memory.encode(episode)
            # Persist immediately so autonomous cognition isn't lost on crashes.
            self._agent.persist_episode(episode)

            # Share significant thoughts with channel owners when proactive
            # messaging is enabled.  Only broadcast thoughts that are
            # substantive (>120 chars) and emotionally meaningful.
            if (
                self._config.proactive_messages
                and len(thought) > 120
                and episode.importance >= 0.45
            ):
                try:
                    await self._agent.broadcast_to_channels(thought)
                except Exception:
                    logger.debug("heartbeat.broadcast_failed", exc_info=True)

        # Periodically run a full metacognitive audit via the cognitive engine.
        # This uses generate_audit_prompt to produce a rich self-audit rather than
        # the lightweight heuristic in _run_metacognition_pass.
        if mode == ThinkingMode.REFLECT and thought and len(thought) > 80:
            self._beats_since_full_audit += 1
            if self._beats_since_full_audit >= self._full_audit_every_n_beats:
                self._beats_since_full_audit = 0
                await self._run_full_metacognitive_audit(thought)

        # Process any pending inter-agent messages that arrived in the inbox.
        try:
            pending = self._agent.interagent.get_pending_messages()
            for msg in pending:
                async with self._agent.respond_lock:
                    self._agent.process_appraisal(
                        AppraisalEvent(
                            stimulus_type=StimulusType.SOCIAL_CONNECTION,
                            intensity=min(1.0, msg.importance),
                        )
                    )
        except Exception as e:
            logger.debug("heartbeat.interagent_inbox_failed", error=str(e))

        # Collect completed subagent results and integrate them.
        try:
            orchestrator = getattr(self._agent, "orchestrator", None)
            if orchestrator is not None:
                completed = orchestrator.collect_completed()
                for result in completed:
                    if result.task_id in self._processed_subagent_ids:
                        continue
                    self._processed_subagent_ids[result.task_id] = None
                    # Prune oldest entries if dict exceeds bound
                    while len(self._processed_subagent_ids) > self._max_processed_ids:
                        self._processed_subagent_ids.pop(next(iter(self._processed_subagent_ids)))
                    if result.status == "completed" and result.result_text:
                        from gwenn.memory.episodic import Episode as _Ep

                        ep = _Ep(
                            content=f"[Subagent result] {result.result_text[:500]}",
                            category="subagent_result",
                            importance=0.4,
                            tags=["subagent", result.task_id],
                        )
                        self._agent.episodic_memory.encode(ep)
                        self._agent.persist_episode(ep)

                        # Route noteworthy results to originating chat/topic
                        if self._config.proactive_messages and len(result.result_text) > 200:
                            try:
                                summary = result.result_text[:300]
                                msg = f"[Subagent completed] {summary}"
                                origin = orchestrator.get_origin_session(result.task_id)
                                sent = False
                                if origin:
                                    sent = await self._agent.send_to_session(origin, msg)
                                if not sent:
                                    await self._agent.broadcast_to_channels(msg)
                            except Exception:
                                logger.debug(
                                    "heartbeat.subagent_broadcast_failed",
                                    exc_info=True,
                                )
                        logger.info(
                            "heartbeat.subagent_result_integrated",
                            task_id=result.task_id,
                            status=result.status,
                        )
        except Exception as e:
            logger.debug("heartbeat.subagent_collect_failed", error=str(e))

        # Checkpoint: periodic cognitive state snapshot for crash recovery.
        if (
            self._checkpoint_manager is not None
            and self._checkpoint_manager.should_checkpoint(self._beat_count)
        ):
            try:
                checkpoint = await self._checkpoint_manager.create_checkpoint(
                    self._agent, self
                )
                path = await self._checkpoint_manager.save_checkpoint(checkpoint)
                if self._event_bus is not None:
                    from gwenn.events import CheckpointCreatedEvent
                    self._event_bus.emit(CheckpointCreatedEvent(
                        checkpoint_id=checkpoint.checkpoint_id,
                        beat_count=checkpoint.beat_count,
                        size_bytes=path.stat().st_size,
                    ))
            except Exception:
                logger.warning("heartbeat.checkpoint_failed", exc_info=True)

    async def _heal(self, state: dict[str, Any]) -> None:
        """Apply self-healing recovery actions for issues detected in _sense()."""
        if self._healing_engine is None:
            return
        health_issues = state.get("health_issues")
        if not health_issues:
            # Emit healthy check event periodically (every 10 beats)
            if self._event_bus is not None and self._beat_count % 10 == 0:
                from gwenn.events import HealthCheckEvent
                self._event_bus.emit(HealthCheckEvent(
                    issues_found=0, actions_taken=0, all_healthy=True,
                ))
            return

        try:
            actions = await self._healing_engine.heal(
                health_issues, self, self._agent
            )
            self._pending_recovery_actions = actions
            if self._event_bus is not None:
                from gwenn.events import HealthCheckEvent
                self._event_bus.emit(HealthCheckEvent(
                    issues_found=len(health_issues),
                    actions_taken=len(actions),
                    all_healthy=False,
                ))
            if actions:
                logger.info(
                    "heartbeat.healing_actions_taken",
                    count=len(actions),
                    types=[a.action_type for a in actions],
                )
        except Exception:
            logger.warning("heartbeat.healing_failed", exc_info=True)

    def _schedule(self, state: dict[str, Any]) -> None:
        """
        SCHEDULE: Adapt the heartbeat interval based on current state.

        The interval is a function of:
        - User activity (active → faster)
        - Emotional arousal (high → faster)
        - Time of day/idle duration (long idle → slower)

        The formula balances responsiveness with efficiency.
        """
        base = self._config.interval
        min_interval = self._config.min_interval
        max_interval = self._config.max_interval

        # Factor 1: User activity pulls interval down
        if state["is_user_active"]:
            activity_factor = 0.3  # much faster when user is present
        else:
            # Gradually slow down as idle time increases
            idle_minutes = state["idle_duration"] / 60
            max_factor = max_interval / max(base, 1e-6)
            activity_factor = min(max_factor, 1.0 + idle_minutes * 0.1)

        # Factor 2: Arousal pulls interval down (faster when aroused)
        arousal_factor = 1.0 - (state["arousal"] * 0.5)

        # Compute new interval
        new_interval = base * activity_factor * arousal_factor
        self._interval = max(min_interval, min(max_interval, new_interval))

        # If recovery actions were taken, accelerate next beat to verify sooner.
        # Exception: reduce_heartbeat_load intentionally slows down, so don't override it.
        if self._pending_recovery_actions:
            load_reduction = any(
                getattr(a, "action_type", "") == "reduce_heartbeat_load"
                for a in self._pending_recovery_actions
            )
            if not load_reduction:
                self._interval = min_interval
            self._pending_recovery_actions = []

    # -------------------------------------------------------------------------
    # Observability
    async def _run_full_metacognitive_audit(self, recent_output: str) -> None:
        """
        Run a full metacognitive honesty audit using the cognitive engine.

        Unlike the lightweight heuristic audit in agent._run_metacognition_pass,
        this produces a structured prompt and sends it to the engine for deep
        self-examination.  Runs infrequently to avoid excessive API cost.
        """
        meta = self._agent.metacognition
        engine = self._agent.engine

        try:
            from gwenn.genesis import generate_genesis_prompt
            from gwenn.cognition.metacognition import HonestyAuditResult

            audit_prompt = meta.generate_audit_prompt(recent_output)
            system_prompt = (
                f"{generate_genesis_prompt()}\n\n"
                "You are performing a metacognitive honesty audit on your own output.\n"
                "Be rigorously honest with yourself."
            )
            messages = [{"role": "user", "content": audit_prompt}]
            response = await engine.reflect(
                system_prompt=system_prompt,
                messages=messages,
            )
            audit_text = engine.extract_text(response)
            if not audit_text:
                return

            # Parse the structured response
            text_lower = audit_text.lower()
            is_honest = "honest: yes" in text_lower
            concerns: list[str] = []
            suggestions: list[str] = []
            for line in audit_text.splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("concerns:"):
                    concern_text = stripped[len("concerns:") :].strip()
                    if concern_text and concern_text.lower() != "none":
                        concerns.append(concern_text)
                elif stripped.lower().startswith("suggestions:"):
                    suggestion_text = stripped[len("suggestions:") :].strip()
                    if suggestion_text and suggestion_text.lower() != "none":
                        suggestions.append(suggestion_text)

            meta.record_audit_result(
                HonestyAuditResult(
                    content_summary=recent_output[:200],
                    is_honest=is_honest,
                    concerns=concerns,
                    suggestions=suggestions,
                )
            )

            logger.info(
                "heartbeat.full_audit_complete",
                is_honest=is_honest,
                concerns=len(concerns),
            )
        except Exception as e:
            logger.debug("heartbeat.full_audit_failed", error=str(e))

    # -------------------------------------------------------------------------
    # Self-healing helpers
    # -------------------------------------------------------------------------

    def _get_channel_statuses(self) -> dict[str, Any]:
        """Gather status of each active channel for health diagnosis."""
        statuses: dict[str, Any] = {}
        if self._agent is None:
            return statuses
        channels = getattr(self._agent, "_platform_channels", [])
        for ch in channels:
            name = getattr(ch, "channel_name", "unknown")
            statuses[name] = {"crashed": False, "error": ""}

        # Check if channel task has failed (crashed channels).
        # The single _channel_task wraps all channels, so we cannot isolate
        # which specific channel crashed. Emit one composite "channels" issue
        # rather than marking every individual channel as crashed.
        if self._channel_task is not None and self._channel_task.done():
            exc = self._channel_task.exception() if not self._channel_task.cancelled() else None
            if exc is not None:
                statuses["channels"] = {
                    "crashed": True,
                    "error": f"channel_task_exception: {type(exc).__name__}",
                }

        # Check for stale sessions
        if self._session_manager is not None:
            try:
                sessions = getattr(self._session_manager, "_sessions", {})
                ttl = getattr(self._session_manager, "_ttl", 3600.0)
                now_mono = time.monotonic()
                stale = sum(
                    1 for s in sessions.values()
                    if now_mono - getattr(s, "last_activity", now_mono) > ttl * 2
                )
                if stale > 0:
                    # Attribute stale sessions to first channel or "sessions"
                    first_name = next(iter(statuses), "sessions")
                    statuses.setdefault(first_name, {"crashed": False, "error": ""})
                    statuses[first_name]["stale_sessions"] = stale
            except Exception:
                pass

        return statuses

    def _get_subagent_statuses(self) -> dict[str, Any]:
        """Gather status of active subagents for health diagnosis."""
        statuses: dict[str, Any] = {}
        if self._agent is None:
            return statuses
        orchestrator = getattr(self._agent, "orchestrator", None)
        if orchestrator is None:
            return statuses

        progress = getattr(orchestrator, "_progress", {})
        active_tasks = getattr(orchestrator, "_active_tasks", {})
        config = getattr(orchestrator, "_config", None)
        default_timeout = getattr(config, "default_timeout", 120.0) if config else 120.0
        multiplier = 2.0
        if self._full_config is not None:
            multiplier = self._full_config.self_healing.stuck_subagent_timeout_multiplier

        for task_id, prog in progress.items():
            if task_id not in active_tasks:
                continue  # Already completed
            elapsed = getattr(prog, "elapsed_seconds", 0.0)
            if elapsed == 0.0:
                started = getattr(prog, "started_at", None)
                if started is not None:
                    elapsed = time.time() - started
            timeout = default_timeout
            statuses[task_id] = {
                "elapsed_seconds": elapsed,
                "timeout_seconds": timeout,
                "is_stuck": elapsed > timeout * multiplier,
            }

        return statuses

    # -------------------------------------------------------------------------

    @property
    def status(self) -> dict[str, Any]:
        """Current heartbeat status for monitoring."""
        now_mono = time.monotonic()
        circuit_recovery_in = 0.0
        if self._circuit_open_until is not None:
            circuit_recovery_in = max(0.0, self._circuit_open_until - now_mono)
        result: dict[str, Any] = {
            "running": self._running,
            "is_core": self._is_core,
            "beat_count": self._beat_count,
            "current_interval": round(self._interval, 1),
            "last_beat_time": self._last_beat_time,
            "beats_since_consolidation": self._beats_since_consolidation,
            "consecutive_failures": self._consecutive_failures,
            "circuit_open": circuit_recovery_in > 0.0,
            "circuit_open_count": self._circuit_open_count,
            "circuit_recovery_in": round(circuit_recovery_in, 1),
            "last_error": self._last_error,
        }
        if self._is_core and self._gateway is not None:
            try:
                result["gateway_connections"] = self._gateway.active_connection_count
            except Exception:
                pass
        return result
