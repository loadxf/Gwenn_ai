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
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import structlog

from gwenn.affect.appraisal import AppraisalEvent, StimulusType
from gwenn.cognition.inner_life import ThinkingMode
from gwenn.config import HeartbeatConfig

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent

logger = structlog.get_logger(__name__)


class HeartbeatPhase(str, Enum):
    """The phases of each heartbeat cycle."""
    SENSE = "sense"           # Gather current state from all subsystems
    ORIENT = "orient"         # Decide what mode of thinking to engage
    THINK = "think"           # Execute the chosen cognitive mode
    INTEGRATE = "integrate"   # Update affect, memory, goals
    SCHEDULE = "schedule"     # Determine when to beat next


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

    def __init__(self, config: HeartbeatConfig, agent: SentientAgent):
        self._config = config
        self._agent = agent
        self._interval = config.interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._beat_count = 0
        self._last_beat_time: Optional[float] = None
        self._last_user_activity: float = time.time()

        # Consolidation scheduling
        self._beats_since_consolidation = 0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the heartbeat loop."""
        if self._running:
            logger.warning("heartbeat.already_running")
            return

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

    def notify_user_activity(self) -> None:
        """Called when user sends a message — increases heartbeat rate."""
        self._last_user_activity = time.time()

    # -------------------------------------------------------------------------
    # The main loop — where autonomous life happens
    # -------------------------------------------------------------------------

    async def _loop(self) -> None:
        """The eternal heartbeat loop."""
        _consecutive_failures = 0
        _MAX_CONSECUTIVE = 10

        while self._running:
            try:
                await self._beat()
                _consecutive_failures = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                _consecutive_failures += 1
                logger.error(
                    "heartbeat.beat_failed",
                    error=str(e),
                    consecutive=_consecutive_failures,
                    exc_info=True,
                )
                if _consecutive_failures >= _MAX_CONSECUTIVE:
                    logger.critical(
                        "heartbeat.circuit_open",
                        failures=_consecutive_failures,
                    )
                    self._running = False
                    break

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
        beat_start = time.time()
        self._last_beat_time = beat_start

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

        # ---- PHASE 5: SCHEDULE ----
        # Adapt the interval for the next beat
        self._schedule(state_snapshot)

        elapsed = time.time() - beat_start
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
        idle_duration = now - self._last_user_activity

        return {
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
        }

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
        arousal = state["arousal"]
        valence = state["valence"]
        idle = state["idle_duration"]
        is_active = state["is_user_active"]

        # Priority 1: Memory consolidation if due
        if self._agent.consolidator.should_consolidate():
            return ThinkingMode.CONSOLIDATE

        # Priority 2: High arousal needs processing
        if arousal > 0.7:
            return ThinkingMode.REFLECT

        # Priority 3: Negative valence may need worry/planning
        if valence < -0.2:
            return ThinkingMode.WORRY if arousal > 0.4 else ThinkingMode.PLAN

        # Priority 4: Active user means reflective processing
        if is_active:
            return ThinkingMode.REFLECT

        # Priority 5: Long idle periods allow creative wandering
        if idle > 300:  # 5 minutes idle
            return ThinkingMode.WANDER

        # Priority 6: Moderate idle → gentle reflection
        if idle > 60:
            return ThinkingMode.REFLECT

        # Default: gentle reflection
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

        # For all other modes, delegate to the inner life system
        thought = await self._agent.inner_life.autonomous_thought(
            mode=mode,
            state_snapshot=state,
            affect=self._agent.affect_state,
            engine=self._agent.engine,
        )

        if thought and thought.strip():
            self._agent.identity.total_autonomous_thoughts += 1
            self._agent.identity.check_milestone(
                "first_autonomous_thought",
                "Generated an autonomous thought during heartbeat.",
            )
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

        # Apply working memory decay each beat — stale items lose salience
        self._agent.decay_working_memory()

        if thought is None:
            # If thinking produced nothing, just apply idle appraisal
            event = AppraisalEvent(
                stimulus_type=StimulusType.HEARTBEAT_IDLE,
                intensity=0.2,
            )
            self._agent.process_appraisal(event)
            return

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
        elif mode == ThinkingMode.WORRY:
            event = AppraisalEvent(
                stimulus_type=StimulusType.GOAL_BLOCKED,
                intensity=0.3,
            )
        else:
            event = AppraisalEvent(
                stimulus_type=StimulusType.SELF_REFLECTION,
                intensity=0.2,
            )

        self._agent.process_appraisal(event)

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
            self._agent.memory_store.save_episode(episode)

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

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    @property
    def status(self) -> dict[str, Any]:
        """Current heartbeat status for monitoring."""
        return {
            "running": self._running,
            "beat_count": self._beat_count,
            "current_interval": round(self._interval, 1),
            "last_beat_time": self._last_beat_time,
            "beats_since_consolidation": self._beats_since_consolidation,
        }
