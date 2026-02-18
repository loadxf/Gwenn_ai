from __future__ import annotations

from types import SimpleNamespace

import pytest

from gwenn.cognition.inner_life import ThinkingMode
from gwenn.config import HeartbeatConfig
from gwenn.heartbeat import Heartbeat


class _InnerLifeStub:
    async def autonomous_thought(self, mode, state_snapshot, affect, engine):
        return "Autonomous thought"


class _IdentityStub:
    def __init__(self):
        self.total_heartbeats = 0
        self.total_autonomous_thoughts = 0
        self.milestone_checks: list[str] = []

    def check_milestone(self, name: str, description: str = "") -> bool:
        self.milestone_checks.append(name)
        return True


def _make_agent():
    return SimpleNamespace(
        inner_life=_InnerLifeStub(),
        engine=object(),
        affect_state=SimpleNamespace(),
        identity=_IdentityStub(),
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
        goal_system=SimpleNamespace(get_goals_summary=lambda: ""),
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
    )
    heartbeat = Heartbeat(HeartbeatConfig(), agent)

    await heartbeat._integrate(ThinkingMode.REFLECT, "A" * 80)

    assert decayed["called"] is True
    assert appraisals, "heartbeat should appraise autonomous thoughts"
    assert len(encoded) == 1
    assert len(saved) == 1
    assert saved[0] is encoded[0]


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
