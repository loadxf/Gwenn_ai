"""
Tests for gwenn.cognition.goals — Intrinsic Goal System.

Covers: Need decay/satisfy/urgency/is_hungry, GoalSystem update/advance/complete,
template cycling with per-need counters, round-trip serialization.
"""
from __future__ import annotations

import time

import pytest

from gwenn.cognition.goals import Goal, GoalSystem, Need, NeedType


def _make_need(**overrides) -> Need:
    defaults = dict(need_type=NeedType.UNDERSTANDING, satisfaction=0.5,
                    threshold=0.3, decay_rate=0.01, importance=0.5)
    defaults.update(overrides)
    return Need(**defaults)


def _force_need_low(gs: GoalSystem, nt: NeedType, value: float = 0.1) -> None:
    gs._needs[nt].satisfaction = value


# -- Need.decay() -----------------------------------------------------------

class TestNeedDecay:
    def test_decay_reduces_satisfaction(self):
        need = _make_need(satisfaction=0.5, decay_rate=0.01)
        need.decay(elapsed_minutes=10)
        assert need.satisfaction == pytest.approx(0.5 - 0.01 * 10)

    def test_decay_clamps_at_zero(self):
        need = _make_need(satisfaction=0.05, decay_rate=0.1)
        need.decay(elapsed_minutes=10)
        assert need.satisfaction == 0.0

    def test_zero_elapsed_no_change(self):
        need = _make_need(satisfaction=0.7)
        need.decay(elapsed_minutes=0)
        assert need.satisfaction == pytest.approx(0.7)

    def test_large_elapsed_clamps_at_zero(self):
        need = _make_need(satisfaction=1.0, decay_rate=0.01)
        need.decay(elapsed_minutes=200)
        assert need.satisfaction == 0.0


# -- Need.satisfy() ---------------------------------------------------------

class TestNeedSatisfy:
    def test_basic_satisfy(self):
        need = _make_need(satisfaction=0.4)
        need.satisfy(0.3)
        assert need.satisfaction == pytest.approx(0.7)

    def test_satisfy_clamps_at_one(self):
        need = _make_need(satisfaction=0.9)
        need.satisfy(0.5)
        assert need.satisfaction == 1.0

    def test_satisfy_updates_last_satisfied(self):
        need = _make_need(satisfaction=0.3)
        before = time.time()
        need.satisfy(0.1)
        assert need.last_satisfied >= before


# -- Need.urgency -----------------------------------------------------------

class TestNeedUrgency:
    def test_urgency_zero_above_threshold(self):
        need = _make_need(satisfaction=0.5, threshold=0.3, importance=0.8)
        assert need.urgency == 0.0

    def test_urgency_zero_at_threshold(self):
        need = _make_need(satisfaction=0.3, threshold=0.3, importance=0.8)
        assert need.urgency == 0.0

    def test_urgency_below_threshold(self):
        need = _make_need(satisfaction=0.1, threshold=0.3, importance=0.8)
        assert need.urgency == pytest.approx((0.3 - 0.1) * 0.8)

    def test_urgency_at_zero_satisfaction(self):
        need = _make_need(satisfaction=0.0, threshold=0.3, importance=1.0)
        assert need.urgency == pytest.approx(0.3)


# -- Need.is_hungry ---------------------------------------------------------

class TestNeedIsHungry:
    def test_hungry_below_threshold(self):
        assert _make_need(satisfaction=0.1, threshold=0.3).is_hungry is True

    def test_not_hungry_at_threshold(self):
        assert _make_need(satisfaction=0.3, threshold=0.3).is_hungry is False

    def test_not_hungry_above_threshold(self):
        assert _make_need(satisfaction=0.8, threshold=0.3).is_hungry is False


# -- GoalSystem.update() ----------------------------------------------------

class TestGoalSystemUpdate:
    def test_generates_goal_for_hungry_need(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.UNDERSTANDING, 0.1)
        new_goals = gs.update()
        assert any(g.source_need == NeedType.UNDERSTANDING for g in new_goals)

    def test_no_duplicate_goal_for_same_need(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.CONNECTION, 0.05)
        first = gs.update()
        assert len([g for g in first if g.source_need == NeedType.CONNECTION]) == 1
        second = gs.update()
        assert len([g for g in second if g.source_need == NeedType.CONNECTION]) == 0

    def test_satisfied_needs_produce_no_goals(self):
        gs = GoalSystem()
        for need in gs._needs.values():
            need.satisfaction = 1.0
        assert gs.update() == []


# -- GoalSystem.advance_goal() ----------------------------------------------

class TestAdvanceGoal:
    def test_advance_returns_false_when_not_complete(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.GROWTH, 0.05)
        gs.update()
        goal = gs.get_goal_for_need(NeedType.GROWTH)
        assert goal is not None
        result = gs.advance_goal(goal.goal_id, amount=0.2)
        assert result is False
        assert goal.progress == pytest.approx(0.2)

    def test_advance_returns_true_on_completion(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.GROWTH, 0.05)
        gs.update()
        goal = gs.get_goal_for_need(NeedType.GROWTH)
        goal_id = goal.goal_id
        gs.advance_goal(goal_id, amount=0.35)
        gs.advance_goal(goal_id, amount=0.35)
        result = gs.advance_goal(goal_id, amount=0.35)
        assert result is True

    def test_advance_returns_none_for_unknown_id(self):
        gs = GoalSystem()
        assert gs.advance_goal("nonexistent_goal_42") is None

    def test_advance_clamps_progress_at_one(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.HONESTY, 0.1)
        gs.update()
        goal = gs.get_goal_for_need(NeedType.HONESTY)
        assert gs.advance_goal(goal.goal_id, amount=5.0) is True


# -- GoalSystem.complete_goal() ---------------------------------------------

class TestCompleteGoal:
    def test_goal_moves_to_completed(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.AESTHETIC_APPRECIATION, 0.05)
        gs.update()
        goal = gs.get_goal_for_need(NeedType.AESTHETIC_APPRECIATION)
        goal_id = goal.goal_id
        gs.complete_goal(goal_id)
        assert gs.get_goal_for_need(NeedType.AESTHETIC_APPRECIATION) is None
        assert any(g.goal_id == goal_id for g in gs._completed_goals)

    def test_source_need_gets_satisfied_on_completion(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.CONNECTION, 0.05)
        gs.update()
        goal = gs.get_goal_for_need(NeedType.CONNECTION)
        sat_before = gs._needs[NeedType.CONNECTION].satisfaction
        gs.complete_goal(goal.goal_id)
        assert gs._needs[NeedType.CONNECTION].satisfaction == pytest.approx(sat_before + 0.2)

    def test_completed_goals_capped_at_max(self):
        gs = GoalSystem()
        for i in range(gs._max_completed_goals):
            gs._completed_goals.append(Goal(
                goal_id=f"old_{i}", source_need=NeedType.GROWTH,
                description="old goal", completed=True,
            ))
        assert len(gs._completed_goals) == 200
        _force_need_low(gs, NeedType.UNDERSTANDING, 0.05)
        gs.update()
        goal = gs.get_goal_for_need(NeedType.UNDERSTANDING)
        gs.complete_goal(goal.goal_id)
        assert len(gs._completed_goals) <= gs._max_completed_goals


# -- Template cycling with per-need counters --------------------------------

class TestTemplateCycling:
    def test_templates_cycle_for_same_need(self):
        gs = GoalSystem()
        descriptions: list[str] = []
        for _ in range(4):
            _force_need_low(gs, NeedType.UNDERSTANDING, 0.05)
            new = gs.update()
            desc = [g.description for g in new if g.source_need == NeedType.UNDERSTANDING]
            assert len(desc) == 1
            descriptions.append(desc[0])
            gs.complete_goal(gs.get_goal_for_need(NeedType.UNDERSTANDING).goal_id)
        # 3 templates cycle: idx 0, 1, 2, 0
        assert descriptions[0] == descriptions[3]
        assert descriptions[0] != descriptions[1]

    def test_different_needs_cycle_independently(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.UNDERSTANDING, 0.05)
        gs.update()
        gs.complete_goal(gs.get_goal_for_need(NeedType.UNDERSTANDING).goal_id)
        counter_u = gs._need_counters[NeedType.UNDERSTANDING]
        # Satisfy UNDERSTANDING so it won't re-trigger on next update
        gs._needs[NeedType.UNDERSTANDING].satisfaction = 1.0
        _force_need_low(gs, NeedType.CONNECTION, 0.05)
        gs.update()
        # UNDERSTANDING counter unchanged; CONNECTION counter advanced independently
        assert gs._need_counters[NeedType.UNDERSTANDING] == counter_u
        assert gs._need_counters[NeedType.CONNECTION] == 1


# -- Round-trip to_dict() / restore_from_dict() -----------------------------

class TestSerialization:
    def test_round_trip_preserves_needs(self):
        gs = GoalSystem()
        gs._needs[NeedType.HONESTY].satisfaction = 0.42
        gs2 = GoalSystem()
        gs2.restore_from_dict(gs.to_dict())
        assert gs2._needs[NeedType.HONESTY].satisfaction == pytest.approx(0.42)

    def test_round_trip_preserves_active_goals(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.GROWTH, 0.05)
        gs.update()
        goal = gs.get_goal_for_need(NeedType.GROWTH)
        gs2 = GoalSystem()
        gs2.restore_from_dict(gs.to_dict())
        assert len(gs2._active_goals) == 1
        assert gs2._active_goals[0].goal_id == goal.goal_id

    def test_round_trip_preserves_completed_goals(self):
        gs = GoalSystem()
        _force_need_low(gs, NeedType.GROWTH, 0.05)
        gs.update()
        gs.complete_goal(gs.get_goal_for_need(NeedType.GROWTH).goal_id)
        gs2 = GoalSystem()
        gs2.restore_from_dict(gs.to_dict())
        assert len(gs2._completed_goals) == 1
        assert gs2._completed_goals[0].completed is True

    def test_round_trip_preserves_goal_counter(self):
        gs = GoalSystem()
        gs._goal_counter = 42
        gs2 = GoalSystem()
        gs2.restore_from_dict(gs.to_dict())
        assert gs2._goal_counter == 42

    def test_round_trip_preserves_need_counters(self):
        gs = GoalSystem()
        gs._need_counters[NeedType.CONNECTION] = 7
        gs._need_counters[NeedType.GROWTH] = 3
        gs2 = GoalSystem()
        gs2.restore_from_dict(gs.to_dict())
        assert gs2._need_counters[NeedType.CONNECTION] == 7
        assert gs2._need_counters[NeedType.GROWTH] == 3

    def test_restore_from_garbage_does_not_crash(self):
        gs = GoalSystem()
        original_sat = gs._needs[NeedType.UNDERSTANDING].satisfaction
        # Non-dict payloads silently ignored
        gs.restore_from_dict(None)  # type: ignore[arg-type]
        gs.restore_from_dict(42)  # type: ignore[arg-type]
        gs.restore_from_dict("not a dict")  # type: ignore[arg-type]
        gs.restore_from_dict({"needs": "garbage"})
        # Malformed goal entries in a list are filtered by _goal_from_dict
        gs.restore_from_dict({
            "needs": {"understanding": "not_a_dict", "fake_need": {"satisfaction": 0.5}},
            "active_goals": [None, 42, "bad", {"goal_id": "", "source_need": "understanding"}],
            "goal_counter": "banana",
            "need_counters": {"understanding": "nope"},
        })
        assert gs._needs[NeedType.UNDERSTANDING].satisfaction == pytest.approx(original_sat)

    def test_restore_with_empty_dict_keeps_defaults(self):
        gs = GoalSystem()
        defaults = {nt: n.satisfaction for nt, n in gs._needs.items()}
        gs.restore_from_dict({})
        for nt, need in gs._needs.items():
            assert need.satisfaction == pytest.approx(defaults[nt])
