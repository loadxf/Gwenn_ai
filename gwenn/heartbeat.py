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
from gwenn.cognition.goals import NeedType
from gwenn.cognition.inner_life import ThinkingMode
from gwenn.config import HeartbeatConfig
from gwenn.memory.episodic import Episode  # noqa: F401, F811 — used at runtime in _integrate

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent

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

    def __init__(self, config: HeartbeatConfig, agent: SentientAgent):
        self._config = config
        self._agent = agent
        self._interval = config.interval
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
        self._processed_subagent_ids: set[str] = set()
        self._max_processed_ids = 2000

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

        # ---- PHASE 5: SCHEDULE ----
        # Adapt the interval for the next beat
        self._schedule(state_snapshot)
        self._agent.identity.total_heartbeats += 1

        # Periodically decay stale Theory-of-Mind beliefs so long idle
        # periods in daemon mode don't leave ancient inferences intact.
        if self._beat_count % 10 == 0:
            try:
                tom = self._agent.theory_of_mind
                for model in tom._user_models.values():
                    model.decay_stale_beliefs()
            except Exception:
                pass

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
                except Exception:
                    pass
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
            async with self._agent._respond_lock:
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

        async with self._agent._respond_lock:
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
            self._agent._persist_episode(episode)

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
                async with self._agent._respond_lock:
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
                    self._processed_subagent_ids.add(result.task_id)
                    # Prune oldest entries if set exceeds bound
                    if len(self._processed_subagent_ids) > self._max_processed_ids:
                        excess = len(self._processed_subagent_ids) - self._max_processed_ids
                        to_remove = list(self._processed_subagent_ids)[:excess]
                        self._processed_subagent_ids -= set(to_remove)
                    if result.status == "completed" and result.result_text:
                        from gwenn.memory.episodic import Episode as _Ep

                        ep = _Ep(
                            content=f"[Subagent result] {result.result_text[:500]}",
                            category="subagent_result",
                            importance=0.4,
                            tags=["subagent", result.task_id],
                        )
                        self._agent.episodic_memory.encode(ep)
                        self._agent._persist_episode(ep)

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

    @property
    def status(self) -> dict[str, Any]:
        """Current heartbeat status for monitoring."""
        now_mono = time.monotonic()
        circuit_recovery_in = 0.0
        if self._circuit_open_until is not None:
            circuit_recovery_in = max(0.0, self._circuit_open_until - now_mono)
        return {
            "running": self._running,
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
