from __future__ import annotations
from types import SimpleNamespace

import pytest

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
        resilience=SimpleNamespace(status="ok"),
        consolidator=SimpleNamespace(should_consolidate=lambda: False),
        decay_working_memory=lambda: None,
        process_appraisal=lambda event: None,
        episodic_memory=SimpleNamespace(encode=lambda ep: None),
        memory_store=SimpleNamespace(save_episode=lambda ep: None),
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
    agent = SimpleNamespace(
        affect_state=affect_state,
        episodic_memory=SimpleNamespace(encode=lambda ep: encoded.append(ep)),
        memory_store=SimpleNamespace(save_episode=lambda ep: saved.append(ep)),
        process_appraisal=lambda event: appraisals.append(event),
        decay_working_memory=lambda: decayed.__setitem__("called", True),
        goal_system=_make_goal_system_stub(),
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
        process_appraisal=lambda event: None,
        decay_working_memory=lambda: None,
        maybe_develop_skill_autonomously=_develop,
        goal_system=_make_goal_system_stub(),
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
        goal_system=SimpleNamespace(get_goals_summary=lambda: ""),
        resilience=SimpleNamespace(status="ok"),
        consolidator=consolidator,
        decay_working_memory=lambda: None,
        process_appraisal=lambda event: None,
        episodic_memory=SimpleNamespace(encode=lambda ep: None),
        memory_store=SimpleNamespace(save_episode=lambda ep: None),
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
    agent.sensory = SimpleNamespace(
        ground_temporal=lambda event_description: calls.append(event_description),
    )
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
