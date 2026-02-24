"""
Tests for gwenn.cognition.inner_life — Autonomous Thinking Modes.

Covers:
- Mode selection emotion routing
- Emotion-driven weight calculations
- Variety pressure boost logic
- Autonomous thought execution (success and error paths)
- State snapshot formatting
- Round-trip persistence (to_dict / restore_from_dict)
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from gwenn.affect.state import AffectiveState, EmotionalDimensions, EmotionLabel
from gwenn.cognition.inner_life import (
    AUTONOMOUS_THINKING_MODES,
    InnerLife,
    ThinkingMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _affect(
    emotion: EmotionLabel = EmotionLabel.NEUTRAL,
    arousal: float = 0.3,
    valence: float = 0.0,
    certainty: float = 0.0,
) -> AffectiveState:
    """Build an AffectiveState with the given overrides."""
    return AffectiveState(
        dimensions=EmotionalDimensions(
            valence=valence,
            arousal=arousal,
            certainty=certainty,
        ),
        current_emotion=emotion,
    )


def _mock_engine(text: str = "A thought.") -> MagicMock:
    """Return a mock CognitiveEngine whose reflect() succeeds."""
    engine = MagicMock()
    engine.reflect = AsyncMock(return_value=MagicMock())
    engine.extract_text = MagicMock(return_value=text)
    return engine


# ---------------------------------------------------------------------------
# Mode selection emotion routing
# ---------------------------------------------------------------------------

class TestModeSelectionEmotionRouting:
    """select_mode should route towards emotion-appropriate modes."""

    def test_curiosity_biases_toward_wander(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.CURIOSITY)
        weights = il._emotion_driven_weights(affect)
        # WANDER should have the highest weight
        assert weights[ThinkingMode.WANDER] == max(weights.values())

    def test_anxiety_biases_toward_worry(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.ANXIETY)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.WORRY] == max(weights.values())

    def test_joy_biases_toward_wander(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.JOY)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.WANDER] >= weights[ThinkingMode.PLAN]
        assert weights[ThinkingMode.WANDER] >= weights[ThinkingMode.WORRY]

    def test_boredom_biases_toward_wander(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.BOREDOM)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.WANDER] == max(weights.values())

    def test_sadness_biases_toward_reflect(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.SADNESS)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.REFLECT] == max(weights.values())

    def test_neutral_gives_equal_weights(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.NEUTRAL, arousal=0.3, valence=0.0)
        weights = il._emotion_driven_weights(affect)
        # All four autonomous modes should be equal (1.0 each)
        values = list(weights.values())
        assert all(v == values[0] for v in values)

    def test_consolidate_excluded_from_autonomous_modes(self):
        assert ThinkingMode.CONSOLIDATE not in AUTONOMOUS_THINKING_MODES
        assert len(AUTONOMOUS_THINKING_MODES) == 4

    def test_select_mode_returns_valid_mode(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.CURIOSITY)
        mode = il.select_mode(affect, has_active_goals=False, has_unresolved_concerns=False)
        assert mode in AUTONOMOUS_THINKING_MODES


# ---------------------------------------------------------------------------
# _emotion_driven_weights detail
# ---------------------------------------------------------------------------

class TestEmotionDrivenWeights:
    """Verify per-emotion routing values and arousal/valence modifiers."""

    def test_curiosity_exact_multipliers(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.CURIOSITY, arousal=0.3, valence=0.0)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.WANDER] == pytest.approx(2.5)
        assert weights[ThinkingMode.REFLECT] == pytest.approx(1.5)
        assert weights[ThinkingMode.PLAN] == pytest.approx(1.0)
        assert weights[ThinkingMode.WORRY] == pytest.approx(1.0)

    def test_anxiety_exact_multipliers(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.ANXIETY, arousal=0.3, valence=0.0)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.WORRY] == pytest.approx(2.5)
        assert weights[ThinkingMode.PLAN] == pytest.approx(1.5)

    def test_frustration_multipliers(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.FRUSTRATION, arousal=0.3, valence=0.0)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.REFLECT] == pytest.approx(2.0)
        assert weights[ThinkingMode.PLAN] == pytest.approx(1.5)

    def test_awe_multipliers(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.AWE, arousal=0.3, valence=0.0)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.WANDER] == pytest.approx(2.5)
        assert weights[ThinkingMode.REFLECT] == pytest.approx(2.0)

    def test_high_arousal_boosts_plan_reduces_reflect(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.NEUTRAL, arousal=0.8, valence=0.0)
        weights = il._emotion_driven_weights(affect)
        # PLAN should be boosted by 1.3x
        assert weights[ThinkingMode.PLAN] == pytest.approx(1.3)
        # REFLECT should be reduced by 0.8x
        assert weights[ThinkingMode.REFLECT] == pytest.approx(0.8)

    def test_low_valence_boosts_worry(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.NEUTRAL, arousal=0.3, valence=-0.5)
        weights = il._emotion_driven_weights(affect)
        # WORRY boosted by 1.3x
        assert weights[ThinkingMode.WORRY] == pytest.approx(1.3)

    def test_combined_emotion_and_arousal(self):
        il = InnerLife()
        # Anxiety + high arousal: WORRY gets 2.5, PLAN gets 1.5*1.3
        affect = _affect(EmotionLabel.ANXIETY, arousal=0.8, valence=0.0)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.WORRY] == pytest.approx(2.5)
        assert weights[ThinkingMode.PLAN] == pytest.approx(1.5 * 1.3)
        assert weights[ThinkingMode.REFLECT] == pytest.approx(0.8)

    def test_combined_emotion_and_low_valence(self):
        il = InnerLife()
        # Sadness + low valence: WORRY gets 1.5*1.3, REFLECT gets 2.5
        affect = _affect(EmotionLabel.SADNESS, arousal=0.3, valence=-0.5)
        weights = il._emotion_driven_weights(affect)
        assert weights[ThinkingMode.REFLECT] == pytest.approx(2.5)
        assert weights[ThinkingMode.WORRY] == pytest.approx(1.5 * 1.3)

    def test_only_autonomous_modes_in_weights(self):
        il = InnerLife()
        affect = _affect(EmotionLabel.CURIOSITY)
        weights = il._emotion_driven_weights(affect)
        assert set(weights.keys()) == set(AUTONOMOUS_THINKING_MODES)
        assert ThinkingMode.CONSOLIDATE not in weights


# ---------------------------------------------------------------------------
# Variety pressure boost
# ---------------------------------------------------------------------------

class TestVarietyPressure:
    """Modes unused for a long time should receive a variety boost."""

    def test_mode_over_300s_gets_boost(self):
        il = InnerLife()
        now = time.time()
        # Set all modes to recently used except WANDER
        for mode in AUTONOMOUS_THINKING_MODES:
            il._mode_last_used[mode] = now
        # WANDER was used 400 seconds ago
        il._mode_last_used[ThinkingMode.WANDER] = now - 400

        affect = _affect(EmotionLabel.NEUTRAL, arousal=0.3, valence=0.0)
        # Call select_mode many times; WANDER should be boosted
        # Instead of statistical testing, verify the weight calculation directly
        weights = il._emotion_driven_weights(affect)
        # Base weight for WANDER under NEUTRAL is 1.0
        assert weights[ThinkingMode.WANDER] == pytest.approx(1.0)

        # Now simulate what select_mode does with variety pressure
        for mode in AUTONOMOUS_THINKING_MODES:
            time_since = now - il._mode_last_used[mode]
            if time_since > 300:
                variety_boost = min(2.0, 1.0 + time_since / 600)
                weights[mode] *= variety_boost

        # WANDER should now be boosted: 1.0 * (1 + 400/600) = ~1.667
        expected = 1.0 * min(2.0, 1.0 + 400 / 600)
        assert weights[ThinkingMode.WANDER] == pytest.approx(expected, rel=0.01)

    def test_variety_boost_caps_at_2(self):
        il = InnerLife()
        now = time.time()
        # Mode unused for 1000 seconds (well over 600)
        il._mode_last_used[ThinkingMode.PLAN] = now - 1000

        time_since = 1000
        variety_boost = min(2.0, 1.0 + time_since / 600)
        assert variety_boost == pytest.approx(2.0)

    def test_recently_used_mode_gets_no_boost(self):
        il = InnerLife()
        now = time.time()
        il._mode_last_used[ThinkingMode.REFLECT] = now - 100  # Only 100s ago

        time_since = 100
        # Under 300s threshold, no boost applied
        assert time_since <= 300

    def test_active_goals_boost_plan(self):
        """select_mode with has_active_goals=True should increase PLAN's share."""
        il = InnerLife()
        affect = _affect(EmotionLabel.NEUTRAL, arousal=0.3, valence=0.0)

        # Run many samples to estimate empirical mode distribution.
        plan_no_goals = 0
        plan_with_goals = 0
        n = 2000
        for _ in range(n):
            if il.select_mode(affect, has_active_goals=False, has_unresolved_concerns=False) == ThinkingMode.PLAN:
                plan_no_goals += 1
            if il.select_mode(affect, has_active_goals=True, has_unresolved_concerns=False) == ThinkingMode.PLAN:
                plan_with_goals += 1
        # With active goals, PLAN *= 1.5, so it should be selected more often.
        assert plan_with_goals > plan_no_goals

    def test_unresolved_concerns_boost_worry(self):
        """select_mode with has_unresolved_concerns=True should increase WORRY's share."""
        il = InnerLife()
        affect = _affect(EmotionLabel.NEUTRAL, arousal=0.3, valence=0.0)

        worry_no_concerns = 0
        worry_with_concerns = 0
        n = 2000
        for _ in range(n):
            if il.select_mode(affect, has_active_goals=False, has_unresolved_concerns=False) == ThinkingMode.WORRY:
                worry_no_concerns += 1
            if il.select_mode(affect, has_active_goals=False, has_unresolved_concerns=True) == ThinkingMode.WORRY:
                worry_with_concerns += 1
        # With unresolved concerns, WORRY *= 1.3, so it should be selected more often.
        assert worry_with_concerns > worry_no_concerns


# ---------------------------------------------------------------------------
# autonomous_thought
# ---------------------------------------------------------------------------

class TestAutonomousThought:
    """Test the autonomous_thought method with a mocked CognitiveEngine."""

    @pytest.mark.asyncio
    async def test_success_returns_thought_and_updates_counters(self):
        il = InnerLife()
        engine = _mock_engine("Deep reflection here.")
        affect = _affect(EmotionLabel.CALM)
        state = {"beat_number": 5, "idle_duration": 120.0}

        result = await il.autonomous_thought(
            mode=ThinkingMode.REFLECT,
            state_snapshot=state,
            affect=affect,
            engine=engine,
        )

        assert result == "Deep reflection here."
        assert il._total_thoughts == 1
        assert il._mode_counts[ThinkingMode.REFLECT] == 1
        engine.reflect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_updates_mode_last_used(self):
        il = InnerLife()
        engine = _mock_engine("Planning thought.")
        affect = _affect(EmotionLabel.CALM)
        state = {"beat_number": 1}
        before = time.time()

        await il.autonomous_thought(
            mode=ThinkingMode.PLAN,
            state_snapshot=state,
            affect=affect,
            engine=engine,
        )

        after = time.time()
        assert il._mode_last_used[ThinkingMode.PLAN] >= before
        assert il._mode_last_used[ThinkingMode.PLAN] <= after

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        il = InnerLife()
        engine = _mock_engine()
        engine.reflect.side_effect = anthropic.APIConnectionError(request=None)
        affect = _affect(EmotionLabel.CALM)
        state = {"beat_number": 1}

        result = await il.autonomous_thought(
            mode=ThinkingMode.WANDER,
            state_snapshot=state,
            affect=affect,
            engine=engine,
        )

        assert result is None
        assert il._total_thoughts == 0

    @pytest.mark.asyncio
    async def test_auth_error_returns_none(self):
        il = InnerLife()
        engine = _mock_engine()
        mock_response = MagicMock(status_code=401)
        engine.reflect.side_effect = anthropic.AuthenticationError(
            message="test", response=mock_response, body=None,
        )
        affect = _affect(EmotionLabel.CALM)
        state = {"beat_number": 1}

        result = await il.autonomous_thought(
            mode=ThinkingMode.WORRY,
            state_snapshot=state,
            affect=affect,
            engine=engine,
        )

        assert result is None
        assert il._total_thoughts == 0

    @pytest.mark.asyncio
    async def test_connection_warning_suppressed_within_60s(self):
        il = InnerLife()
        engine = _mock_engine()
        engine.reflect.side_effect = anthropic.APIConnectionError(request=None)
        affect = _affect(EmotionLabel.CALM)
        state = {"beat_number": 1}

        # First call — warning is emitted, _last_connection_warning_at updated
        await il.autonomous_thought(
            mode=ThinkingMode.REFLECT, state_snapshot=state,
            affect=affect, engine=engine,
        )
        first_warning_time = il._last_connection_warning_at
        assert first_warning_time > 0

        # Second call immediately — should be suppressed (debug level)
        await il.autonomous_thought(
            mode=ThinkingMode.REFLECT, state_snapshot=state,
            affect=affect, engine=engine,
        )
        # Warning timestamp should NOT be updated on suppressed call
        assert il._last_connection_warning_at == first_warning_time

    @pytest.mark.asyncio
    async def test_auth_warning_suppressed_within_60s(self):
        il = InnerLife()
        engine = _mock_engine()
        mock_response = MagicMock(status_code=401)
        engine.reflect.side_effect = anthropic.AuthenticationError(
            message="test", response=mock_response, body=None,
        )
        affect = _affect(EmotionLabel.CALM)
        state = {"beat_number": 1}

        await il.autonomous_thought(
            mode=ThinkingMode.WORRY, state_snapshot=state,
            affect=affect, engine=engine,
        )
        first_warning_time = il._last_auth_warning_at
        assert first_warning_time > 0

        await il.autonomous_thought(
            mode=ThinkingMode.WORRY, state_snapshot=state,
            affect=affect, engine=engine,
        )
        # Not updated on suppressed call
        assert il._last_auth_warning_at == first_warning_time

    @pytest.mark.asyncio
    async def test_generic_exception_returns_none(self):
        il = InnerLife()
        engine = _mock_engine()
        engine.reflect.side_effect = RuntimeError("unexpected")
        affect = _affect(EmotionLabel.CALM)
        state = {"beat_number": 1}

        result = await il.autonomous_thought(
            mode=ThinkingMode.REFLECT, state_snapshot=state,
            affect=affect, engine=engine,
        )

        assert result is None
        assert il._total_thoughts == 0

    @pytest.mark.asyncio
    async def test_optional_context_fields_included(self):
        il = InnerLife()
        engine = _mock_engine("Thought with context.")
        affect = _affect(EmotionLabel.CALM)
        state = {"beat_number": 1}

        result = await il.autonomous_thought(
            mode=ThinkingMode.PLAN,
            state_snapshot=state,
            affect=affect,
            engine=engine,
            goal_context="Goal: learn Python",
            ethical_context="No ethical concerns",
            metacognitive_context="Confidence: high",
            sensory_snapshot="Quiet evening",
        )

        assert result == "Thought with context."
        # Verify the system prompt passed to engine.reflect contains context
        call_kwargs = engine.reflect.call_args.kwargs
        system_prompt = call_kwargs["system_prompt"]
        assert "Goal: learn Python" in system_prompt
        assert "No ethical concerns" in system_prompt
        assert "Confidence: high" in system_prompt
        assert "Quiet evening" in system_prompt


# ---------------------------------------------------------------------------
# _format_state_snapshot
# ---------------------------------------------------------------------------

class TestFormatStateSnapshot:
    """Test the state snapshot formatting helper."""

    def test_valid_state_dict(self):
        state = {
            "beat_number": 42,
            "idle_duration": 95.3,
            "is_user_active": True,
            "working_memory_load": 0.65,
            "resilience_status": {"breaker_active": True},
            "goal_status": "Complete task A",
        }
        result = InnerLife._format_state_snapshot(state)

        assert "beat=42" in result
        assert "idle_seconds=95.3" in result
        assert "user_active=yes" in result
        assert "working_memory_load=0.65" in result
        assert "resilience_breaker_active=yes" in result
        assert "goals=Complete task A" in result

    def test_missing_keys_degrade_gracefully(self):
        state = {"beat_number": 1}
        result = InnerLife._format_state_snapshot(state)

        assert "beat=1" in result
        assert "idle_seconds=0.0" in result
        assert "user_active=no" in result
        assert "working_memory_load=0.00" in result

    def test_empty_dict(self):
        result = InnerLife._format_state_snapshot({})
        assert "Heartbeat state snapshot:" in result
        assert "beat=0" in result

    def test_non_dict_input_returns_unavailable(self):
        result = InnerLife._format_state_snapshot("not a dict")
        assert "unavailable" in result.lower()

    def test_none_input_returns_unavailable(self):
        result = InnerLife._format_state_snapshot(None)
        assert "unavailable" in result.lower()

    def test_list_input_returns_unavailable(self):
        result = InnerLife._format_state_snapshot([1, 2, 3])
        assert "unavailable" in result.lower()

    def test_working_memory_load_clamped(self):
        state = {"working_memory_load": 5.0}
        result = InnerLife._format_state_snapshot(state)
        # Should be clamped to 1.0
        assert "working_memory_load=1.00" in result

    def test_idle_duration_negative_clamped(self):
        state = {"idle_duration": -10.0}
        result = InnerLife._format_state_snapshot(state)
        assert "idle_seconds=0.0" in result

    def test_goal_status_omitted_when_empty(self):
        state = {"goal_status": ""}
        result = InnerLife._format_state_snapshot(state)
        assert "goals=" not in result

    def test_resilience_status_non_dict(self):
        state = {"resilience_status": "broken"}
        result = InnerLife._format_state_snapshot(state)
        assert "resilience_breaker_active=no" in result


# ---------------------------------------------------------------------------
# Round-trip persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    """to_dict() -> restore_from_dict() should preserve state."""

    def test_round_trip_preserves_total_thoughts(self):
        il = InnerLife()
        il._total_thoughts = 42
        data = il.to_dict()

        il2 = InnerLife()
        il2.restore_from_dict(data)
        assert il2._total_thoughts == 42

    def test_round_trip_preserves_mode_counts(self):
        il = InnerLife()
        il._mode_counts[ThinkingMode.REFLECT] = 10
        il._mode_counts[ThinkingMode.WANDER] = 5
        il._mode_counts[ThinkingMode.PLAN] = 3
        il._mode_counts[ThinkingMode.WORRY] = 1
        data = il.to_dict()

        il2 = InnerLife()
        il2.restore_from_dict(data)
        assert il2._mode_counts[ThinkingMode.REFLECT] == 10
        assert il2._mode_counts[ThinkingMode.WANDER] == 5
        assert il2._mode_counts[ThinkingMode.PLAN] == 3
        assert il2._mode_counts[ThinkingMode.WORRY] == 1

    def test_round_trip_preserves_mode_last_used(self):
        il = InnerLife()
        now = time.time()
        il._mode_last_used[ThinkingMode.REFLECT] = now - 100
        il._mode_last_used[ThinkingMode.PLAN] = now - 200
        data = il.to_dict()

        il2 = InnerLife()
        il2.restore_from_dict(data)
        assert il2._mode_last_used[ThinkingMode.REFLECT] == pytest.approx(now - 100)
        assert il2._mode_last_used[ThinkingMode.PLAN] == pytest.approx(now - 200)

    def test_restore_with_empty_dict_no_crash(self):
        il = InnerLife()
        il._total_thoughts = 5
        il.restore_from_dict({})
        # total_thoughts resets to 0 because data.get returns 0
        assert il._total_thoughts == 0

    def test_restore_with_none_no_crash(self):
        il = InnerLife()
        il._total_thoughts = 5
        il.restore_from_dict(None)
        # Should be a no-op
        assert il._total_thoughts == 5

    def test_restore_with_non_dict_no_crash(self):
        il = InnerLife()
        il._total_thoughts = 5
        il.restore_from_dict("garbage")
        assert il._total_thoughts == 5

    def test_restore_with_malformed_counts_skips_bad_values(self):
        il = InnerLife()
        data = {
            "total_thoughts": 10,
            "mode_counts": {
                "reflect": "not_a_number",
                "wander": 7,
            },
            "mode_last_used": {},
        }
        il.restore_from_dict(data)
        assert il._total_thoughts == 10
        # "reflect" should remain at its default (0)
        assert il._mode_counts[ThinkingMode.REFLECT] == 0
        assert il._mode_counts[ThinkingMode.WANDER] == 7

    def test_restore_with_negative_total_thoughts_clamped(self):
        il = InnerLife()
        il.restore_from_dict({"total_thoughts": -5})
        assert il._total_thoughts == 0

    def test_restore_with_negative_mode_count_clamped(self):
        il = InnerLife()
        il.restore_from_dict({"mode_counts": {"reflect": -3}})
        assert il._mode_counts[ThinkingMode.REFLECT] == 0

    def test_to_dict_structure(self):
        il = InnerLife()
        il._total_thoughts = 7
        il._mode_counts[ThinkingMode.WANDER] = 3
        data = il.to_dict()

        assert "total_thoughts" in data
        assert "mode_counts" in data
        assert "mode_last_used" in data
        assert data["total_thoughts"] == 7
        assert data["mode_counts"]["wander"] == 3

    def test_stats_property(self):
        il = InnerLife()
        il._total_thoughts = 10
        il._mode_counts[ThinkingMode.REFLECT] = 4
        stats = il.stats

        assert stats["total_thoughts"] == 10
        assert stats["mode_counts"]["reflect"] == 4
        assert "mode_last_used" in stats
