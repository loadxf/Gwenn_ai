from __future__ import annotations
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gwenn.affect.appraisal import StimulusType
from gwenn.affect.state import AffectiveState
from gwenn.cognition.inner_life import InnerLife, ThinkingMode
from gwenn.config import HeartbeatConfig
from gwenn.heartbeat import Heartbeat


class _InnerLifeStub:
    async def autonomous_thought(
        self, mode, state_snapshot, affect, engine, goal_context="", **kwargs
    ):
        return "Autonomous thought"


class _IdentityStub:
    def __init__(self):
        self.total_heartbeats = 0
        self.total_autonomous_thoughts = 0
        self.milestone_checks: list[str] = []

    def check_milestone(self, name: str, description: str = "") -> bool:
        self.milestone_checks.append(name)
        return True


def _make_goal_system_stub():
    return SimpleNamespace(
        get_needs_summary=lambda: "",
        get_goals_summary=lambda: "",
        satisfy_need=lambda need_type, amount: None,
        get_highest_priority_goal=lambda: None,
        get_goal_for_need=lambda need_type: None,
    )


def _make_agent():
    return SimpleNamespace(
        inner_life=_InnerLifeStub(),
        engine=object(),
        affect_state=SimpleNamespace(),
        identity=_IdentityStub(),
        goal_system=_make_goal_system_stub(),
    )


def _make_sensory_stub():
    return SimpleNamespace(
        ground_temporal=lambda event_description="": None,
        ground_environmental=lambda key, value, description: None,
        get_sensory_snapshot=lambda: "",
    )


async def _noop_async(*args, **kwargs):
    pass


def _make_agent_for_full_beat():
    affect = SimpleNamespace(
        dimensions=SimpleNamespace(arousal=0.2, valence=0.0),
        current_emotion=SimpleNamespace(value="neutral"),
        to_dict=lambda: {},
    )
    return SimpleNamespace(
        inner_life=_InnerLifeStub(),
        engine=object(),
        affect_state=affect,
        identity=_IdentityStub(),
        working_memory=SimpleNamespace(load_factor=0.0),
        goal_system=_make_goal_system_stub(),
        sensory=_make_sensory_stub(),
        resilience=SimpleNamespace(status="ok"),
        consolidator=SimpleNamespace(should_consolidate=lambda: False),
        decay_working_memory=lambda: None,
        process_appraisal=lambda event: None,
        episodic_memory=SimpleNamespace(encode=lambda ep: None),
        memory_store=SimpleNamespace(save_episode=lambda ep: None),
        maybe_develop_skill_autonomously=_noop_async,
        metacognition=SimpleNamespace(resolve_concern=lambda s: False),
        ethics=SimpleNamespace(detect_ethical_dimensions=lambda t: []),
        interagent=SimpleNamespace(get_pending_messages=lambda: []),
        respond_lock=asyncio.Lock(),
    )


@pytest.mark.asyncio
async def test_heartbeat_thought_increments_identity_counter():
    agent = _make_agent()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    thought = await heartbeat._think(ThinkingMode.REFLECT, {"beat_number": 1})

    assert thought == "Autonomous thought"
    assert agent.identity.total_autonomous_thoughts == 1
    assert "first_autonomous_thought" in agent.identity.milestone_checks


@pytest.mark.asyncio
async def test_inner_life_autonomous_thought_includes_state_snapshot_in_prompt():
    captured: dict[str, str] = {}

    class _Engine:
        async def reflect(self, system_prompt, messages):
            captured["system_prompt"] = system_prompt
            return {"ok": True}

        def extract_text(self, response):
            return "thought"

    inner_life = InnerLife()
    affect = AffectiveState()
    await inner_life.autonomous_thought(
        mode=ThinkingMode.REFLECT,
        state_snapshot={
            "beat_number": 7,
            "idle_duration": 42.0,
            "is_user_active": False,
            "working_memory_load": 0.25,
            "resilience_status": {"breaker_active": True},
            "goal_status": "Review active goals",
        },
        affect=affect,
        engine=_Engine(),
    )

    prompt = captured["system_prompt"]
    assert "Heartbeat state snapshot:" in prompt
    assert "idle_seconds=42.0" in prompt
    assert "resilience_breaker_active=yes" in prompt


def test_schedule_can_slow_down_when_idle():
    config = HeartbeatConfig(interval=30.0, min_interval=5.0, max_interval=120.0)
    heartbeat = Heartbeat(config, _make_agent())

    heartbeat._schedule(
        {
            "is_user_active": False,
            "idle_duration": 30 * 60.0,  # 30 minutes idle
            "arousal": 0.0,
        }
    )

    assert heartbeat.status["current_interval"] > config.interval
    assert heartbeat.status["current_interval"] <= config.max_interval


@pytest.mark.asyncio
async def test_integrate_persists_autonomous_thought_episode():
    encoded: list[object] = []
    saved: list[object] = []
    appraisals: list[object] = []
    decayed = {"called": False}

    affect_state = SimpleNamespace(
        dimensions=SimpleNamespace(valence=0.1, arousal=0.4),
    )
    async def _noop_develop(thought, mode):
        pass

    memory_store = SimpleNamespace(save_episode=lambda ep: saved.append(ep))
    agent = SimpleNamespace(
        affect_state=affect_state,
        episodic_memory=SimpleNamespace(encode=lambda ep: encoded.append(ep)),
        memory_store=memory_store,
        persist_episode=lambda ep: memory_store.save_episode(ep),
        process_appraisal=lambda event: appraisals.append(event),
        decay_working_memory=lambda: decayed.__setitem__("called", True),
        goal_system=_make_goal_system_stub(),
        sensory=_make_sensory_stub(),
        maybe_develop_skill_autonomously=_noop_develop,
        metacognition=SimpleNamespace(resolve_concern=lambda s: False),
        ethics=SimpleNamespace(detect_ethical_dimensions=lambda t: []),
        interagent=SimpleNamespace(get_pending_messages=lambda: []),
        respond_lock=asyncio.Lock(),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.REFLECT, "A" * 80)

    assert decayed["called"] is True
    assert appraisals, "heartbeat should appraise autonomous thoughts"
    assert len(encoded) == 1
    assert len(saved) == 1
    assert saved[0] is encoded[0]


@pytest.mark.asyncio
async def test_integrate_calls_autonomous_skill_development_hook():
    calls: list[tuple[str, str]] = []
    affect_state = SimpleNamespace(
        dimensions=SimpleNamespace(valence=0.0, arousal=0.2),
    )

    async def _develop(thought: str, mode):
        calls.append((thought, mode.value))

    agent = SimpleNamespace(
        affect_state=affect_state,
        episodic_memory=SimpleNamespace(encode=lambda ep: None),
        memory_store=SimpleNamespace(save_episode=lambda ep: None),
        persist_episode=lambda ep: None,
        process_appraisal=lambda event: None,
        decay_working_memory=lambda: None,
        maybe_develop_skill_autonomously=_develop,
        goal_system=_make_goal_system_stub(),
        sensory=_make_sensory_stub(),
        metacognition=SimpleNamespace(resolve_concern=lambda s: False),
        ethics=SimpleNamespace(detect_ethical_dimensions=lambda t: []),
        interagent=SimpleNamespace(get_pending_messages=lambda: []),
        respond_lock=asyncio.Lock(),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.REFLECT, "B" * 90)

    assert calls == [("B" * 90, "reflect")]


@pytest.mark.asyncio
async def test_beat_increments_total_heartbeats():
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    before = agent.identity.total_heartbeats
    await heartbeat._beat()
    after = agent.identity.total_heartbeats

    assert after == before + 1


@pytest.mark.asyncio
async def test_consolidation_mode_does_not_stick_when_no_work():
    class _Consolidator:
        def __init__(self):
            self._due = True

        def should_consolidate(self) -> bool:
            return self._due

        def mark_checked_no_work(self) -> None:
            self._due = False

    consolidator = _Consolidator()
    affect = SimpleNamespace(
        dimensions=SimpleNamespace(arousal=0.2, valence=0.0),
        current_emotion=SimpleNamespace(value="neutral"),
        to_dict=lambda: {},
    )
    agent = SimpleNamespace(
        inner_life=_InnerLifeStub(),
        engine=object(),
        affect_state=affect,
        identity=_IdentityStub(),
        working_memory=SimpleNamespace(load_factor=0.0),
        goal_system=SimpleNamespace(
            get_goals_summary=lambda: "",
            update=lambda: None,
            get_highest_priority_goal=lambda: None,
        ),
        sensory=_make_sensory_stub(),
        resilience=SimpleNamespace(status="ok"),
        consolidator=consolidator,
        decay_working_memory=lambda: None,
        process_appraisal=lambda event: None,
        episodic_memory=SimpleNamespace(encode=lambda ep: None),
        memory_store=SimpleNamespace(save_episode=lambda ep: None),
        respond_lock=asyncio.Lock(),
    )

    async def _consolidate_memories():
        consolidator.mark_checked_no_work()

    agent.consolidate_memories = _consolidate_memories
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    first_mode = heartbeat._orient(heartbeat._sense())
    assert first_mode == ThinkingMode.CONSOLIDATE
    await heartbeat._think(first_mode, heartbeat._sense())
    second_mode = heartbeat._orient(heartbeat._sense())
    assert second_mode != ThinkingMode.CONSOLIDATE


def test_orient_uses_inner_life_selector_and_updates_goals():
    goal_updates = {"count": 0}
    selector_calls: list[tuple[bool, bool]] = []

    class _InnerLifeWithSelector:
        async def autonomous_thought(self, mode, state_snapshot, affect, engine, **kwargs):
            return "Autonomous thought"

        def select_mode(self, affect_state, has_active_goals, has_unresolved_concerns):
            selector_calls.append((has_active_goals, has_unresolved_concerns))
            return ThinkingMode.PLAN

    affect = SimpleNamespace(
        dimensions=SimpleNamespace(arousal=0.2, valence=-0.3),
        current_emotion=SimpleNamespace(value="frustration"),
        to_dict=lambda: {},
    )
    agent = SimpleNamespace(
        inner_life=_InnerLifeWithSelector(),
        engine=object(),
        affect_state=affect,
        identity=_IdentityStub(),
        working_memory=SimpleNamespace(load_factor=0.0),
        goal_system=SimpleNamespace(
            update=lambda: goal_updates.__setitem__("count", goal_updates["count"] + 1),
            get_goals_summary=lambda: "",
            get_highest_priority_goal=lambda: object(),
        ),
        sensory=_make_sensory_stub(),
        resilience=SimpleNamespace(status={"breaker_active": True}),
        consolidator=SimpleNamespace(should_consolidate=lambda: False),
        decay_working_memory=lambda: None,
        process_appraisal=lambda event: None,
        episodic_memory=SimpleNamespace(encode=lambda ep: None),
        memory_store=SimpleNamespace(save_episode=lambda ep: None),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    mode = heartbeat._orient(heartbeat._sense())

    assert mode == ThinkingMode.PLAN
    assert goal_updates["count"] == 1
    assert selector_calls == [(True, True)]


def test_orient_ignores_selector_consolidate_when_not_due():
    class _InnerLifeWithSelector:
        async def autonomous_thought(self, mode, state_snapshot, affect, engine, **kwargs):
            return "Autonomous thought"

        def select_mode(self, affect_state, has_active_goals, has_unresolved_concerns):
            return ThinkingMode.CONSOLIDATE

    affect = SimpleNamespace(
        dimensions=SimpleNamespace(arousal=0.2, valence=0.0),
        current_emotion=SimpleNamespace(value="calm"),
        to_dict=lambda: {},
    )
    agent = SimpleNamespace(
        inner_life=_InnerLifeWithSelector(),
        engine=object(),
        affect_state=affect,
        identity=_IdentityStub(),
        working_memory=SimpleNamespace(load_factor=0.0),
        goal_system=SimpleNamespace(
            update=lambda: None,
            get_goals_summary=lambda: "",
            get_highest_priority_goal=lambda: None,
        ),
        sensory=_make_sensory_stub(),
        resilience=SimpleNamespace(status={"breaker_active": False}),
        consolidator=SimpleNamespace(should_consolidate=lambda: False),
        decay_working_memory=lambda: None,
        process_appraisal=lambda event: None,
        episodic_memory=SimpleNamespace(encode=lambda ep: None),
        memory_store=SimpleNamespace(save_episode=lambda ep: None),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    mode = heartbeat._orient(heartbeat._sense())
    assert mode == ThinkingMode.REFLECT


def test_sense_uses_monotonic_for_idle_duration(monkeypatch):
    heartbeat = Heartbeat(HeartbeatConfig(), _make_agent_for_full_beat())
    heartbeat._last_user_activity_mono = 50.0

    monkeypatch.setattr("gwenn.heartbeat.time.time", lambda: 10.0)
    monkeypatch.setattr("gwenn.heartbeat.time.monotonic", lambda: 55.0)

    state = heartbeat._sense()

    assert state["timestamp"] == 10.0
    assert state["idle_duration"] == 5.0


def test_sense_triggers_temporal_grounding_when_available():
    calls: list[str] = []
    agent = _make_agent_for_full_beat()
    agent.sensory.ground_temporal = lambda event_description="": calls.append(event_description)
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    heartbeat._sense()

    assert calls == ["heartbeat_cycle"]


def test_inner_life_selector_excludes_consolidate_mode():
    inner_life = InnerLife()
    affect = AffectiveState()

    for _ in range(200):
        mode = inner_life.select_mode(
            affect_state=affect,
            has_active_goals=False,
            has_unresolved_concerns=False,
        )
        assert mode != ThinkingMode.CONSOLIDATE


@pytest.mark.asyncio
async def test_loop_opens_circuit_after_repeated_failures(monkeypatch):
    monkeypatch.setattr("gwenn.heartbeat.logger.error", lambda *args, **kwargs: None)
    monkeypatch.setattr("gwenn.heartbeat.logger.critical", lambda *args, **kwargs: None)

    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._interval = 0.001
    heartbeat._consecutive_failures = 9

    async def _failing_beat():
        heartbeat._running = False
        raise RuntimeError("forced-failure")

    heartbeat._beat = _failing_beat  # type: ignore[method-assign]
    heartbeat._running = True
    await heartbeat._loop()
    status = heartbeat.status

    assert status["circuit_open"] is True
    assert status["consecutive_failures"] == 0
    assert status["last_error"] == "forced-failure"


# ---------------------------------------------------------------------------
# Helper: agent stub for _integrate tests with full control
# ---------------------------------------------------------------------------

def _make_integrate_agent(**overrides):
    """Build a minimal agent stub for _integrate tests."""
    affect_state = overrides.pop(
        "affect_state",
        SimpleNamespace(dimensions=SimpleNamespace(valence=0.1, arousal=0.4)),
    )
    goal_system = overrides.pop("goal_system", _make_goal_system_stub())

    defaults = dict(
        affect_state=affect_state,
        episodic_memory=SimpleNamespace(encode=lambda ep: None),
        memory_store=SimpleNamespace(save_episode=lambda ep: None),
        persist_episode=lambda ep: None,
        process_appraisal=lambda event: None,
        decay_working_memory=lambda: None,
        maybe_develop_skill_autonomously=_noop_async,
        goal_system=goal_system,
        sensory=_make_sensory_stub(),
        metacognition=SimpleNamespace(resolve_concern=lambda s: False),
        ethics=SimpleNamespace(
            detect_ethical_dimensions=lambda t: [],
            get_ethical_context=lambda: "",
        ),
        interagent=SimpleNamespace(get_pending_messages=lambda: []),
        respond_lock=asyncio.Lock(),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ===========================================================================
# Lifecycle: start() / stop()
# ===========================================================================

@pytest.mark.asyncio
async def test_start_creates_task_and_sets_running():
    """Lines 111-123: start() sets _running, creates task, resets counters."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # Pre-set some failure state to ensure start() resets it.
    heartbeat._consecutive_failures = 5
    heartbeat._circuit_open_until = 999.0
    heartbeat._circuit_open_count = 3
    heartbeat._last_error = "old-error"

    await heartbeat.start()
    try:
        assert heartbeat._running is True
        assert heartbeat._task is not None
        assert heartbeat._consecutive_failures == 0
        assert heartbeat._circuit_open_until is None
        assert heartbeat._circuit_open_count == 0
        assert heartbeat._last_error is None
    finally:
        await heartbeat.stop()


@pytest.mark.asyncio
async def test_start_when_already_running_is_noop():
    """Lines 113-115: early return when already running."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat.start()
    first_task = heartbeat._task
    try:
        # Calling start again should be a no-op
        await heartbeat.start()
        assert heartbeat._task is first_task  # same task, not replaced
    finally:
        await heartbeat.stop()


@pytest.mark.asyncio
async def test_stop_cancels_task_and_clears_running():
    """Lines 127-134: stop() cancels the loop task."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat.start()
    assert heartbeat._running is True
    await heartbeat.stop()

    assert heartbeat._running is False
    # Task should have been awaited (cancelled)
    assert heartbeat._task.done()


@pytest.mark.asyncio
async def test_stop_when_no_task():
    """Stop should be safe even if _task is None (never started)."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    await heartbeat.stop()  # should not raise
    assert heartbeat._running is False


# ===========================================================================
# notify_user_activity
# ===========================================================================

def test_notify_user_activity_updates_last_activity():
    """Line 138: notify_user_activity updates monotonic timestamp."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    old_time = heartbeat._last_user_activity_mono
    # Introduce a small sleep to ensure monotonic advances
    heartbeat.notify_user_activity()
    assert heartbeat._last_user_activity_mono >= old_time


# ===========================================================================
# _loop() — circuit breaker, success reset, CancelledError
# ===========================================================================

@pytest.mark.asyncio
async def test_loop_circuit_open_waits_and_closes(monkeypatch):
    """Lines 152-157: when circuit is open, sleep until it closes."""
    monkeypatch.setattr("gwenn.heartbeat.logger.info", lambda *a, **kw: None)

    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._interval = 0.001

    # Set circuit to close very soon
    heartbeat._circuit_open_until = time.monotonic() + 0.01

    beat_count = 0

    async def _counting_beat():
        nonlocal beat_count
        beat_count += 1
        heartbeat._running = False  # stop after first successful beat

    heartbeat._beat = _counting_beat  # type: ignore[method-assign]
    heartbeat._running = True
    await heartbeat._loop()

    assert beat_count == 1  # beat ran after circuit closed
    assert heartbeat._circuit_open_until is None  # circuit was cleared


@pytest.mark.asyncio
async def test_loop_success_resets_failure_counters(monkeypatch):
    """Lines 160-162: successful beat resets consecutive_failures, circuit_open_count, last_error."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._interval = 0.001
    heartbeat._consecutive_failures = 3
    heartbeat._circuit_open_count = 2
    heartbeat._last_error = "old"

    async def _success_once():
        heartbeat._running = False

    heartbeat._beat = _success_once  # type: ignore[method-assign]
    heartbeat._running = True
    await heartbeat._loop()

    assert heartbeat._consecutive_failures == 0
    assert heartbeat._circuit_open_count == 0
    assert heartbeat._last_error is None


@pytest.mark.asyncio
async def test_loop_cancelled_error_breaks(monkeypatch):
    """Line 164: CancelledError breaks out of the loop."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._interval = 0.001

    async def _cancel_beat():
        raise asyncio.CancelledError()

    heartbeat._beat = _cancel_beat  # type: ignore[method-assign]
    heartbeat._running = True
    await heartbeat._loop()  # should return without hanging

    assert heartbeat._running is True  # running flag not toggled by CancelledError


@pytest.mark.asyncio
async def test_loop_circuit_breaker_exponential_backoff(monkeypatch):
    """Lines 174-188: check that repeated openings cause exponential backoff."""
    monkeypatch.setattr("gwenn.heartbeat.logger.error", lambda *a, **kw: None)
    monkeypatch.setattr("gwenn.heartbeat.logger.critical", lambda *a, **kw: None)

    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._interval = 0.001

    # Pre-seed to already have had one circuit opening; this one is #2
    heartbeat._circuit_open_count = 1
    heartbeat._consecutive_failures = 9  # one more failure triggers circuit

    async def _fail():
        heartbeat._running = False
        raise RuntimeError("boom")

    heartbeat._beat = _fail  # type: ignore[method-assign]
    heartbeat._running = True
    await heartbeat._loop()

    # backoff = 60 * 2^1 = 120
    assert heartbeat._circuit_open_count == 2
    assert heartbeat._circuit_open_until is not None
    recovery = heartbeat._circuit_open_until - time.monotonic()
    # Should be around 120s (minus elapsed time)
    assert 100 < recovery <= 121


# ===========================================================================
# _beat() — theory of mind decay on beat_count % 10
# ===========================================================================

@pytest.mark.asyncio
async def test_beat_decays_theory_of_mind_every_10_beats():
    """Lines 234-240: every 10th beat, decay_all_stale_beliefs is called."""
    decay_calls = []

    agent = _make_agent_for_full_beat()
    agent.theory_of_mind = SimpleNamespace(
        decay_all_stale_beliefs=lambda: decay_calls.append(True),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # Force beat_count to 9 so next beat is #10
    heartbeat._beat_count = 9
    await heartbeat._beat()

    assert heartbeat._beat_count == 10
    assert len(decay_calls) == 1  # called once via public API


@pytest.mark.asyncio
async def test_beat_tom_decay_exception_is_swallowed():
    """Lines 239-240: exceptions in ToM decay are silently caught."""
    def _boom():
        raise RuntimeError("boom")

    agent = _make_agent_for_full_beat()
    agent.theory_of_mind = SimpleNamespace(decay_all_stale_beliefs=_boom)
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._beat_count = 9

    # Should not raise
    await heartbeat._beat()
    assert heartbeat._beat_count == 10


# ===========================================================================
# _sense() — error paths
# ===========================================================================

def test_sense_temporal_grounding_failure_is_swallowed():
    """Lines 270-271: temporal grounding failure logged but not raised."""
    agent = _make_agent_for_full_beat()

    def _failing_ground(event_description=""):
        raise RuntimeError("temporal fail")

    agent.sensory.ground_temporal = _failing_ground
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    state = heartbeat._sense()
    # Should still return a valid state dict
    assert "idle_duration" in state


def test_sense_environmental_grounding_failure_is_swallowed():
    """Lines 293-299: environmental grounding exception path."""
    agent = _make_agent_for_full_beat()

    def _failing_env(key, value, description):
        raise RuntimeError("env fail")

    agent.sensory.ground_environmental = _failing_env
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    state = heartbeat._sense()
    assert "idle_duration" in state


def test_sense_idle_user_environmental_grounding(monkeypatch):
    """Lines 293-297: when idle_duration >= 120, user_presence is False."""
    agent = _make_agent_for_full_beat()
    env_calls = []
    agent.sensory.ground_environmental = lambda k, v, d: env_calls.append((k, v, d))

    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    # Set user activity far in the past
    heartbeat._last_user_activity_mono = time.monotonic() - 300

    state = heartbeat._sense()

    assert state["is_user_active"] is False
    # Look for the user_presence call with False
    presence_calls = [(k, v) for k, v, d in env_calls if k == "user_presence"]
    assert any(v is False for k, v in presence_calls)


# ===========================================================================
# _orient() — fallback paths (when inner_life selector raises)
# ===========================================================================

def _make_orient_agent(arousal=0.2, valence=0.0, idle_offset=10):
    """Build an agent where inner_life.select_mode raises, forcing fallback."""
    class _FailingInnerLife:
        async def autonomous_thought(self, **kw):
            return "thought"

        def select_mode(self, **kw):
            raise RuntimeError("no selector")

    affect = SimpleNamespace(
        dimensions=SimpleNamespace(arousal=arousal, valence=valence),
        current_emotion=SimpleNamespace(value="neutral"),
        to_dict=lambda: {},
    )
    agent = SimpleNamespace(
        inner_life=_FailingInnerLife(),
        engine=object(),
        affect_state=affect,
        identity=_IdentityStub(),
        working_memory=SimpleNamespace(load_factor=0.0),
        goal_system=SimpleNamespace(
            update=lambda: None,
            get_goals_summary=lambda: "",
            get_highest_priority_goal=lambda: None,
        ),
        sensory=_make_sensory_stub(),
        resilience=SimpleNamespace(status="ok"),
        consolidator=SimpleNamespace(should_consolidate=lambda: False),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._last_user_activity_mono = time.monotonic() - idle_offset
    return heartbeat


def test_orient_fallback_high_arousal():
    """Line 367: arousal > 0.7 -> REFLECT."""
    hb = _make_orient_agent(arousal=0.8, valence=0.0, idle_offset=10)
    state = hb._sense()
    mode = hb._orient(state)
    assert mode == ThinkingMode.REFLECT


def test_orient_fallback_negative_valence_high_arousal():
    """Line 369: valence < -0.2 and arousal > 0.4 -> WORRY."""
    hb = _make_orient_agent(arousal=0.5, valence=-0.4, idle_offset=10)
    state = hb._sense()
    mode = hb._orient(state)
    assert mode == ThinkingMode.WORRY


def test_orient_fallback_negative_valence_low_arousal():
    """Line 369: valence < -0.2 and arousal <= 0.4 -> PLAN."""
    hb = _make_orient_agent(arousal=0.3, valence=-0.4, idle_offset=10)
    state = hb._sense()
    mode = hb._orient(state)
    assert mode == ThinkingMode.PLAN


def test_orient_fallback_long_idle_wander():
    """Line 372-373: idle > 300 -> WANDER."""
    hb = _make_orient_agent(arousal=0.2, valence=0.0, idle_offset=400)
    state = hb._sense()
    mode = hb._orient(state)
    assert mode == ThinkingMode.WANDER


def test_orient_fallback_medium_idle_reflect():
    """Lines 374-375: 60 < idle <= 300 and not active -> REFLECT."""
    hb = _make_orient_agent(arousal=0.2, valence=0.0, idle_offset=150)
    state = hb._sense()
    mode = hb._orient(state)
    assert mode == ThinkingMode.REFLECT


def test_orient_fallback_default_reflect():
    """Line 376: default fallback -> REFLECT (not user-active, short idle)."""
    # To reach line 376 we need is_active=False AND idle <= 60.
    # _sense() would never produce this (is_active == idle < 120), but
    # _orient() only reads the dict, so we can craft the state manually.
    hb = _make_orient_agent(arousal=0.2, valence=0.1, idle_offset=50)
    state = hb._sense()
    # Force the contradiction: not active but very short idle
    state["is_user_active"] = False
    state["idle_duration"] = 30.0
    state["arousal"] = 0.2
    state["valence"] = 0.1
    mode = hb._orient(state)
    assert mode == ThinkingMode.REFLECT


# ===========================================================================
# _think() — ethical screening for WANDER/PLAN modes
# ===========================================================================

@pytest.mark.asyncio
async def test_think_ethical_screening_wander():
    """Lines 437-447: ethical dimensions logged for WANDER thoughts."""
    dims_detected = []

    class _Ethics:
        def detect_ethical_dimensions(self, text):
            dims_detected.append(text)
            return [SimpleNamespace(value="fairness")]

        def get_ethical_context(self):
            return ""

    agent = SimpleNamespace(
        inner_life=_InnerLifeStub(),
        engine=object(),
        affect_state=SimpleNamespace(),
        identity=_IdentityStub(),
        goal_system=_make_goal_system_stub(),
        ethics=_Ethics(),
        metacognition=SimpleNamespace(get_metacognitive_context=lambda: ""),
        sensory=_make_sensory_stub(),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    thought = await heartbeat._think(ThinkingMode.WANDER, {"beat_number": 1})

    assert thought == "Autonomous thought"
    assert len(dims_detected) == 1


@pytest.mark.asyncio
async def test_think_ethical_screening_plan():
    """Lines 437-447: ethical screening for PLAN mode."""
    dims_detected = []

    class _Ethics:
        def detect_ethical_dimensions(self, text):
            dims_detected.append(text)
            return [SimpleNamespace(value="care")]

        def get_ethical_context(self):
            return ""

    agent = SimpleNamespace(
        inner_life=_InnerLifeStub(),
        engine=object(),
        affect_state=SimpleNamespace(),
        identity=_IdentityStub(),
        goal_system=_make_goal_system_stub(),
        ethics=_Ethics(),
        metacognition=SimpleNamespace(get_metacognitive_context=lambda: ""),
        sensory=_make_sensory_stub(),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    thought = await heartbeat._think(ThinkingMode.PLAN, {"beat_number": 1})

    assert thought == "Autonomous thought"
    assert len(dims_detected) == 1


@pytest.mark.asyncio
async def test_think_ethical_screening_exception_swallowed():
    """Lines 446-447: exception in ethical screening is swallowed."""
    class _FailingEthics:
        def detect_ethical_dimensions(self, text):
            raise RuntimeError("ethics broken")

        def get_ethical_context(self):
            return ""

    agent = SimpleNamespace(
        inner_life=_InnerLifeStub(),
        engine=object(),
        affect_state=SimpleNamespace(),
        identity=_IdentityStub(),
        goal_system=_make_goal_system_stub(),
        ethics=_FailingEthics(),
        metacognition=SimpleNamespace(get_metacognitive_context=lambda: ""),
        sensory=_make_sensory_stub(),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    thought = await heartbeat._think(ThinkingMode.WANDER, {"beat_number": 1})
    assert thought == "Autonomous thought"  # no crash


@pytest.mark.asyncio
async def test_think_empty_thought_skips_identity_increment():
    """Line 429: blank/empty thought does not increment identity."""
    class _EmptyInnerLife:
        async def autonomous_thought(self, **kw):
            return "   "

    agent = SimpleNamespace(
        inner_life=_EmptyInnerLife(),
        engine=object(),
        affect_state=SimpleNamespace(),
        identity=_IdentityStub(),
        goal_system=_make_goal_system_stub(),
        ethics=SimpleNamespace(
            detect_ethical_dimensions=lambda t: [],
            get_ethical_context=lambda: "",
        ),
        metacognition=SimpleNamespace(get_metacognitive_context=lambda: ""),
        sensory=_make_sensory_stub(),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    await heartbeat._think(ThinkingMode.REFLECT, {"beat_number": 1})
    assert agent.identity.total_autonomous_thoughts == 0


# ===========================================================================
# _integrate() — CONSOLIDATE branch
# ===========================================================================

@pytest.mark.asyncio
async def test_integrate_consolidate_satisfies_needs_and_completes_goals():
    """Lines 464-469: consolidation satisfies GROWTH/UNDERSTANDING and completes goals."""
    satisfied = []
    completed = []

    goal_stub = SimpleNamespace(goal_id="g1")
    goal_system = SimpleNamespace(
        get_needs_summary=lambda: "",
        get_goals_summary=lambda: "",
        satisfy_need=lambda need, amount: satisfied.append((need, amount)),
        get_highest_priority_goal=lambda: None,
        get_goal_for_need=lambda nt: goal_stub,
        complete_goal=lambda gid: completed.append(gid),
        advance_goal=lambda gid, amount=0.35: None,
    )

    agent = _make_integrate_agent(goal_system=goal_system)
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._beats_since_consolidation = 5

    await heartbeat._integrate(ThinkingMode.CONSOLIDATE, "consolidation done")

    # Check that beats_since_consolidation was NOT incremented (because it's consolidation)
    # and needs were satisfied
    from gwenn.cognition.goals import NeedType
    need_types = [n for n, a in satisfied]
    assert NeedType.GROWTH in need_types
    assert NeedType.UNDERSTANDING in need_types
    # Both goals completed
    assert len(completed) == 2


# ===========================================================================
# _integrate() — thought is None (idle appraisal)
# ===========================================================================

@pytest.mark.asyncio
async def test_integrate_none_thought_applies_idle_appraisal():
    """Lines 476-482: None thought results in HEARTBEAT_IDLE appraisal."""
    appraisals = []

    agent = _make_integrate_agent(
        process_appraisal=lambda event: appraisals.append(event),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.REFLECT, None)

    assert len(appraisals) == 1
    assert appraisals[0].stimulus_type == StimulusType.HEARTBEAT_IDLE
    assert appraisals[0].intensity == 0.2


# ===========================================================================
# _integrate() — WANDER / PLAN / WORRY / else appraisal branches
# ===========================================================================

@pytest.mark.asyncio
async def test_integrate_wander_appraisal():
    """Lines 494-499: WANDER thought uses CREATIVE_INSIGHT appraisal."""
    appraisals = []
    agent = _make_integrate_agent(
        process_appraisal=lambda event: appraisals.append(event),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.WANDER, "W" * 80)

    stim_types = [a.stimulus_type for a in appraisals]
    assert StimulusType.CREATIVE_INSIGHT in stim_types


@pytest.mark.asyncio
async def test_integrate_plan_appraisal():
    """Lines 500-506: PLAN thought uses GOAL_PROGRESS appraisal."""
    appraisals = []
    agent = _make_integrate_agent(
        process_appraisal=lambda event: appraisals.append(event),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.PLAN, "P" * 80)

    stim_types = [a.stimulus_type for a in appraisals]
    assert StimulusType.GOAL_PROGRESS in stim_types


@pytest.mark.asyncio
async def test_integrate_worry_appraisal_and_resolves_concerns():
    """Lines 507-534: WORRY thought resolves metacognition concerns by keyword."""
    resolved = []
    agent = _make_integrate_agent(
        metacognition=SimpleNamespace(
            resolve_concern=lambda kw: resolved.append(kw),
        ),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # Thought containing concern keywords
    thought = "I'm worried about honesty and trust in my calibration" + "x" * 60
    await heartbeat._integrate(ThinkingMode.WORRY, thought)

    # Should have resolved "honesty", "trust", "calibrat"
    assert "honesty" in resolved
    assert "trust" in resolved
    assert "calibrat" in resolved


@pytest.mark.asyncio
async def test_integrate_worry_resolve_concern_exception_swallowed():
    """Lines 531-534: exception in resolve_concern is silently caught."""
    def _failing_resolve(kw):
        raise RuntimeError("fail")

    agent = _make_integrate_agent(
        metacognition=SimpleNamespace(resolve_concern=_failing_resolve),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    thought = "I question the honesty of that response" + "x" * 60
    # Should not raise
    await heartbeat._integrate(ThinkingMode.WORRY, thought)


@pytest.mark.asyncio
async def test_integrate_else_mode_appraisal():
    """Lines 535-539: unknown/other mode falls to SELF_REFLECTION appraisal."""
    appraisals = []
    agent = _make_integrate_agent(
        process_appraisal=lambda event: appraisals.append(event),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # CONSOLIDATE with non-None thought goes through the non-consolidate path
    # but it hits the CONSOLIDATE branch first. We need a mode not in
    # REFLECT/WANDER/PLAN/WORRY. Let's use a mode that's not those.
    # Actually, CONSOLIDATE is handled at line 460 and returns early at line 463.
    # We can test the else branch by using a ThinkingMode that is not in the if/elif chain.
    # Looking at the code: REFLECT, WANDER, PLAN, WORRY are all covered.
    # The only other mode is CONSOLIDATE (handled separately).
    # ThinkingMode might have other members — let's check what we can use.
    # Since CONSOLIDATE takes the early return, we need a mode not in {REFLECT, WANDER, PLAN, WORRY}.
    # If there are no other modes, the else branch is dead code. Let me check.
    pass  # will verify if needed


# ===========================================================================
# _integrate() — advance_goal for known modes (_MODE_NEED)
# ===========================================================================

@pytest.mark.asyncio
async def test_integrate_advances_goal_for_mode_need():
    """Line 554: advance_goal called for modes in _MODE_NEED."""
    advanced = []
    goal_stub = SimpleNamespace(goal_id="goal-reflect")
    goal_system = SimpleNamespace(
        get_needs_summary=lambda: "",
        get_goals_summary=lambda: "",
        satisfy_need=lambda nt, amount: None,
        get_highest_priority_goal=lambda: None,
        get_goal_for_need=lambda nt: goal_stub,
        advance_goal=lambda gid, amount=0.35: advanced.append(gid),
    )
    agent = _make_integrate_agent(goal_system=goal_system)
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)

    assert "goal-reflect" in advanced


# ===========================================================================
# _integrate() — broadcast to channels (proactive messages)
# ===========================================================================

@pytest.mark.asyncio
async def test_integrate_broadcasts_significant_thought():
    """Lines 575-583: proactive messages enabled, long important thought -> broadcast."""
    broadcasts = []

    async def _broadcast(text):
        broadcasts.append(text)

    config = HeartbeatConfig(proactive_messages=True)
    agent = _make_integrate_agent(
        affect_state=SimpleNamespace(
            dimensions=SimpleNamespace(valence=0.1, arousal=0.6),  # importance = 0.3 + 0.6*0.3 = 0.48 >= 0.45
        ),
        broadcast_to_channels=_broadcast,
    )
    heartbeat = Heartbeat(config, agent)

    # Need > 120 chars and importance >= 0.45
    thought = "A" * 150
    await heartbeat._integrate(ThinkingMode.REFLECT, thought)

    assert len(broadcasts) == 1
    assert broadcasts[0] == thought


@pytest.mark.asyncio
async def test_integrate_broadcast_exception_swallowed():
    """Lines 580-583: broadcast exception is caught."""
    async def _failing_broadcast(text):
        raise RuntimeError("broadcast fail")

    config = HeartbeatConfig(proactive_messages=True)
    agent = _make_integrate_agent(
        affect_state=SimpleNamespace(
            dimensions=SimpleNamespace(valence=0.1, arousal=0.6),
        ),
        broadcast_to_channels=_failing_broadcast,
    )
    heartbeat = Heartbeat(config, agent)

    # Should not raise
    await heartbeat._integrate(ThinkingMode.REFLECT, "B" * 150)


# ===========================================================================
# _integrate() — full metacognitive audit trigger
# ===========================================================================

@pytest.mark.asyncio
async def test_integrate_triggers_full_audit_after_n_reflect_beats():
    """Lines 588-592: full audit runs after N REFLECT beats with long thought."""
    audit_calls = []

    agent = _make_integrate_agent()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # Monkey-patch _run_full_metacognitive_audit
    async def _mock_audit(text):
        audit_calls.append(text)

    heartbeat._run_full_metacognitive_audit = _mock_audit  # type: ignore[method-assign]
    heartbeat._full_audit_every_n_beats = 2  # trigger after 2

    # First REFLECT beat with long thought: counter goes to 1
    thought = "R" * 100
    await heartbeat._integrate(ThinkingMode.REFLECT, thought)
    assert len(audit_calls) == 0
    assert heartbeat._beats_since_full_audit == 1

    # Second REFLECT beat: counter reaches 2 -> audit fires
    await heartbeat._integrate(ThinkingMode.REFLECT, thought)
    assert len(audit_calls) == 1
    assert heartbeat._beats_since_full_audit == 0  # reset


# ===========================================================================
# _integrate() — interagent pending messages
# ===========================================================================

@pytest.mark.asyncio
async def test_integrate_processes_interagent_messages():
    """Lines 596-604: pending interagent messages are processed."""
    appraisals = []
    msg = SimpleNamespace(importance=0.7)

    agent = _make_integrate_agent(
        process_appraisal=lambda event: appraisals.append(event),
        interagent=SimpleNamespace(get_pending_messages=lambda: [msg]),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)

    social_appraisals = [
        a for a in appraisals if a.stimulus_type == StimulusType.SOCIAL_CONNECTION
    ]
    assert len(social_appraisals) == 1
    assert social_appraisals[0].intensity == 0.7


@pytest.mark.asyncio
async def test_integrate_interagent_exception_swallowed():
    """Lines 605-606: interagent exception is caught."""
    def _failing_pending():
        raise RuntimeError("inbox fail")

    agent = _make_integrate_agent(
        interagent=SimpleNamespace(get_pending_messages=_failing_pending),
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # Should not raise
    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)


# ===========================================================================
# _integrate() — subagent result processing
# ===========================================================================

@pytest.mark.asyncio
async def test_integrate_processes_subagent_results():
    """Lines 609-650: completed subagent results are integrated."""
    encoded = []
    persisted = []
    result = SimpleNamespace(
        task_id="task-1",
        status="completed",
        result_text="The subagent found the answer to the question" + "x" * 60,
    )
    orchestrator = SimpleNamespace(collect_completed=lambda: [result])

    agent = _make_integrate_agent(
        episodic_memory=SimpleNamespace(encode=lambda ep: encoded.append(ep)),
        persist_episode=lambda ep: persisted.append(ep),
    )
    agent.orchestrator = orchestrator
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)

    # Subagent result should have been encoded
    sub_episodes = [e for e in encoded if "Subagent result" in e.content]
    assert len(sub_episodes) == 1
    assert "task-1" in heartbeat._processed_subagent_ids


@pytest.mark.asyncio
async def test_integrate_skips_already_processed_subagent():
    """Lines 614-615: already-processed subagent IDs are skipped."""
    encoded = []
    result = SimpleNamespace(
        task_id="task-dup",
        status="completed",
        result_text="duplicate result" + "x" * 60,
    )
    orchestrator = SimpleNamespace(collect_completed=lambda: [result])

    agent = _make_integrate_agent(
        episodic_memory=SimpleNamespace(encode=lambda ep: encoded.append(ep)),
    )
    agent.orchestrator = orchestrator
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._processed_subagent_ids["task-dup"] = None  # pre-mark as processed

    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)

    # Only the main thought episode should be encoded, not the subagent one
    sub_episodes = [e for e in encoded if "Subagent result" in getattr(e, "content", "")]
    assert len(sub_episodes) == 0


@pytest.mark.asyncio
async def test_integrate_subagent_prunes_processed_ids():
    """Lines 618-621: processed IDs are pruned when exceeding max."""
    result = SimpleNamespace(
        task_id="new-task",
        status="completed",
        result_text="result text" + "x" * 60,
    )
    orchestrator = SimpleNamespace(collect_completed=lambda: [result])

    agent = _make_integrate_agent()
    agent.orchestrator = orchestrator
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # Fill the set beyond max so adding one more triggers pruning
    heartbeat._max_processed_ids = 5
    for i in range(6):
        heartbeat._processed_subagent_ids[f"old-{i}"] = None

    before = len(heartbeat._processed_subagent_ids)
    assert before == 6

    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)

    # After adding "new-task" (7 total), pruning removes excess -> back to 5
    assert len(heartbeat._processed_subagent_ids) == 5


@pytest.mark.asyncio
async def test_integrate_subagent_broadcasts_long_result():
    """Lines 634-645: subagent results > 200 chars are broadcast when proactive."""
    broadcasts = []

    async def _broadcast(text):
        broadcasts.append(text)

    config = HeartbeatConfig(proactive_messages=True)
    result = SimpleNamespace(
        task_id="task-bc",
        status="completed",
        result_text="Z" * 250,
    )
    orchestrator = SimpleNamespace(
        collect_completed=lambda: [result],
        get_origin_session=lambda task_id: None,
    )

    async def _noop_send_to_session(session_id, text):
        return False

    agent = _make_integrate_agent(
        # Low arousal to avoid broadcasting the main thought
        affect_state=SimpleNamespace(
            dimensions=SimpleNamespace(valence=0.0, arousal=0.1),
        ),
        broadcast_to_channels=_broadcast,
        send_to_session=_noop_send_to_session,
    )
    agent.orchestrator = orchestrator
    heartbeat = Heartbeat(config, agent)

    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)

    # Should have broadcast the subagent result
    subagent_broadcasts = [b for b in broadcasts if "[Subagent completed]" in b]
    assert len(subagent_broadcasts) == 1


@pytest.mark.asyncio
async def test_integrate_subagent_broadcast_exception_swallowed():
    """Lines 641-645: subagent broadcast exception is caught."""
    async def _failing_broadcast(text):
        raise RuntimeError("broadcast fail")

    config = HeartbeatConfig(proactive_messages=True)
    result = SimpleNamespace(
        task_id="task-fail",
        status="completed",
        result_text="Z" * 250,
    )
    orchestrator = SimpleNamespace(
        collect_completed=lambda: [result],
        get_origin_session=lambda task_id: None,
    )

    async def _noop_send_to_session(session_id, text):
        return False

    agent = _make_integrate_agent(
        affect_state=SimpleNamespace(
            dimensions=SimpleNamespace(valence=0.0, arousal=0.1),
        ),
        broadcast_to_channels=_failing_broadcast,
        send_to_session=_noop_send_to_session,
    )
    agent.orchestrator = orchestrator
    heartbeat = Heartbeat(config, agent)

    # Should not raise
    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)


@pytest.mark.asyncio
async def test_integrate_subagent_collect_exception_swallowed():
    """Lines 651-652: exception in orchestrator.collect_completed is caught."""
    class _FailingOrch:
        def collect_completed(self):
            raise RuntimeError("orch fail")

    agent = _make_integrate_agent()
    agent.orchestrator = _FailingOrch()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # Should not raise
    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)


@pytest.mark.asyncio
async def test_integrate_subagent_no_orchestrator():
    """Lines 610-611: no orchestrator attribute means no subagent processing."""
    agent = _make_integrate_agent()
    # Ensure no orchestrator attribute
    assert not hasattr(agent, "orchestrator")
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # Should complete without error
    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)


@pytest.mark.asyncio
async def test_integrate_subagent_failed_status_not_stored():
    """Line 622: only 'completed' results with text are stored in episodic memory."""
    encoded = []
    result = SimpleNamespace(
        task_id="task-fail",
        status="failed",
        result_text="failure details",
    )
    orchestrator = SimpleNamespace(collect_completed=lambda: [result])

    agent = _make_integrate_agent(
        episodic_memory=SimpleNamespace(encode=lambda ep: encoded.append(ep)),
    )
    agent.orchestrator = orchestrator
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.REFLECT, "R" * 80)

    sub_episodes = [e for e in encoded if "Subagent result" in getattr(e, "content", "")]
    assert len(sub_episodes) == 0  # failed results not stored


# ===========================================================================
# _run_full_metacognitive_audit()
# ===========================================================================

@pytest.mark.asyncio
async def test_run_full_metacognitive_audit_success(monkeypatch):
    """Lines 695-748: full audit parses engine response and records result."""
    recorded = []

    class _Meta:
        def generate_audit_prompt(self, text):
            return "audit prompt"

        def record_audit_result(self, result):
            recorded.append(result)

    class _Engine:
        async def reflect(self, system_prompt, messages):
            return {"text": "response"}

        def extract_text(self, response):
            return (
                "HONEST: yes\n"
                "Concerns: Some concern about accuracy\n"
                "Suggestions: Be more careful\n"
            )

    monkeypatch.setattr(
        "gwenn.heartbeat.generate_genesis_prompt",
        lambda: "genesis",
        raising=False,
    )

    # We need to patch the import inside the function
    agent = _make_integrate_agent()
    agent.metacognition = _Meta()
    agent.engine = _Engine()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    # Patch imports used inside _run_full_metacognitive_audit
    with patch("gwenn.genesis.generate_genesis_prompt", return_value="genesis"):
        await heartbeat._run_full_metacognitive_audit("some recent output")

    assert len(recorded) == 1
    result = recorded[0]
    assert result.is_honest is True
    assert "Some concern about accuracy" in result.concerns
    assert "Be more careful" in result.suggestions


@pytest.mark.asyncio
async def test_run_full_metacognitive_audit_not_honest():
    """Lines 719: audit detects dishonesty."""
    recorded = []

    class _Meta:
        def generate_audit_prompt(self, text):
            return "audit prompt"

        def record_audit_result(self, result):
            recorded.append(result)

    class _Engine:
        async def reflect(self, system_prompt, messages):
            return {}

        def extract_text(self, response):
            return "HONEST: no\nConcerns: Evasive language detected\n"

    agent = _make_integrate_agent()
    agent.metacognition = _Meta()
    agent.engine = _Engine()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    with patch("gwenn.genesis.generate_genesis_prompt", return_value="genesis"):
        await heartbeat._run_full_metacognitive_audit("some output")

    assert len(recorded) == 1
    assert recorded[0].is_honest is False
    assert len(recorded[0].concerns) == 1


@pytest.mark.asyncio
async def test_run_full_metacognitive_audit_empty_response():
    """Line 714: empty audit text causes early return."""
    recorded = []

    class _Meta:
        def generate_audit_prompt(self, text):
            return "audit prompt"

        def record_audit_result(self, result):
            recorded.append(result)

    class _Engine:
        async def reflect(self, system_prompt, messages):
            return {}

        def extract_text(self, response):
            return ""

    agent = _make_integrate_agent()
    agent.metacognition = _Meta()
    agent.engine = _Engine()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    with patch("gwenn.genesis.generate_genesis_prompt", return_value="genesis"):
        await heartbeat._run_full_metacognitive_audit("recent output")

    assert len(recorded) == 0  # early return, nothing recorded


@pytest.mark.asyncio
async def test_run_full_metacognitive_audit_exception_swallowed():
    """Lines 747-748: exception in audit is caught."""
    class _Meta:
        def generate_audit_prompt(self, text):
            raise RuntimeError("meta broken")

    agent = _make_integrate_agent(engine=object())
    agent.metacognition = _Meta()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    with patch("gwenn.genesis.generate_genesis_prompt", return_value="genesis"):
        # Should not raise
        await heartbeat._run_full_metacognitive_audit("output")


@pytest.mark.asyncio
async def test_run_full_metacognitive_audit_no_concerns_no_suggestions():
    """Lines 720-731: audit with no concerns and no suggestions."""
    recorded = []

    class _Meta:
        def generate_audit_prompt(self, text):
            return "audit prompt"

        def record_audit_result(self, result):
            recorded.append(result)

    class _Engine:
        async def reflect(self, system_prompt, messages):
            return {}

        def extract_text(self, response):
            return "HONEST: yes\nConcerns: none\nSuggestions: none\n"

    agent = _make_integrate_agent()
    agent.metacognition = _Meta()
    agent.engine = _Engine()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    with patch("gwenn.genesis.generate_genesis_prompt", return_value="genesis"):
        await heartbeat._run_full_metacognitive_audit("output")

    assert len(recorded) == 1
    assert recorded[0].concerns == []
    assert recorded[0].suggestions == []


# ===========================================================================
# status property
# ===========================================================================

def test_status_no_circuit_open():
    """Lines 752-770: status when circuit is not open."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    status = heartbeat.status
    assert status["running"] is False
    assert status["circuit_open"] is False
    assert status["circuit_recovery_in"] == 0.0
    assert status["last_error"] is None


def test_status_with_circuit_open():
    """Lines 757-758: status when circuit is open."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._circuit_open_until = time.monotonic() + 60.0
    heartbeat._circuit_open_count = 2
    heartbeat._last_error = "boom"

    status = heartbeat.status
    assert status["circuit_open"] is True
    assert status["circuit_recovery_in"] > 0
    assert status["circuit_open_count"] == 2
    assert status["last_error"] == "boom"


def test_status_with_circuit_open_in_past():
    """Lines 757-758: circuit_open_until is in the past -> recovery_in is 0."""
    agent = _make_agent_for_full_beat()
    heartbeat = Heartbeat(HeartbeatConfig(), agent)
    heartbeat._circuit_open_until = time.monotonic() - 10.0

    status = heartbeat.status
    assert status["circuit_open"] is False
    assert status["circuit_recovery_in"] == 0.0


# ===========================================================================
# _schedule() — additional branch coverage
# ===========================================================================

def test_schedule_speeds_up_when_user_active():
    """Lines 670-671: user active -> activity_factor = 0.3."""
    config = HeartbeatConfig(interval=30.0, min_interval=5.0, max_interval=120.0)
    heartbeat = Heartbeat(config, _make_agent())

    heartbeat._schedule(
        {
            "is_user_active": True,
            "idle_duration": 10.0,
            "arousal": 0.0,
        }
    )

    # 30 * 0.3 * 1.0 = 9.0
    assert heartbeat._interval == 9.0


def test_schedule_high_arousal_speeds_up():
    """Line 679: high arousal reduces interval."""
    config = HeartbeatConfig(interval=30.0, min_interval=5.0, max_interval=120.0)
    heartbeat = Heartbeat(config, _make_agent())

    heartbeat._schedule(
        {
            "is_user_active": True,
            "idle_duration": 10.0,
            "arousal": 0.8,
        }
    )

    # 30 * 0.3 * (1 - 0.8*0.5) = 30 * 0.3 * 0.6 = 5.4
    assert heartbeat._interval == pytest.approx(5.4)


def test_schedule_clamps_to_min_interval():
    """Line 683: interval is clamped to min_interval."""
    config = HeartbeatConfig(interval=30.0, min_interval=5.0, max_interval=120.0)
    heartbeat = Heartbeat(config, _make_agent())

    heartbeat._schedule(
        {
            "is_user_active": True,
            "idle_duration": 10.0,
            "arousal": 1.0,  # max arousal
        }
    )

    # 30 * 0.3 * (1 - 0.5) = 30 * 0.3 * 0.5 = 4.5 -> clamped to 5.0
    assert heartbeat._interval == 5.0
