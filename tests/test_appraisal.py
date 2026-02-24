"""
Tests for gwenn.affect.appraisal — Emotional Appraisal Engine.

Covers:
- Each StimulusType produces expected dimensional changes
- Intensity scaling (0.0 vs 0.5 vs 1.0)
- Momentum factor (emotional inertia)
- Baseline pull (drift toward equilibrium)
- Clamping (dimensions stay within valid ranges)
- Emotion reclassification after appraisal
- Edge cases: unknown stimulus type, zero intensity, time elapsed tracking
"""

from __future__ import annotations

import time

import pytest

from gwenn.affect.appraisal import AppraisalEngine, AppraisalEvent, StimulusType
from gwenn.affect.state import AffectiveState, EmotionalDimensions, EmotionLabel
from gwenn.config import AffectConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    momentum_decay: float = 0.0,
    baseline_pull: float = 0.0,
    arousal_ceiling: float = 0.92,
    distress_threshold: float = 0.80,
    distress_duration_limit: float = 300.0,
) -> AffectConfig:
    """Create an AffectConfig with configurable momentum and baseline pull.

    By default momentum_decay=0 and baseline_pull=0, which makes tests
    deterministic: dimensional changes are applied directly without blending
    or drift.
    """
    return AffectConfig(
        arousal_ceiling=arousal_ceiling,
        distress_threshold=distress_threshold,
        distress_duration_limit=distress_duration_limit,
        momentum_decay=momentum_decay,
        baseline_pull=baseline_pull,
    )


def _zero_state() -> AffectiveState:
    """An AffectiveState with all dimensions at zero and zero baseline."""
    return AffectiveState(
        dimensions=EmotionalDimensions(
            valence=0.0, arousal=0.0, dominance=0.0,
            certainty=0.0, goal_congruence=0.0,
        ),
        baseline=EmotionalDimensions(
            valence=0.0, arousal=0.0, dominance=0.0,
            certainty=0.0, goal_congruence=0.0,
        ),
    )


def _default_state() -> AffectiveState:
    """An AffectiveState with default baseline (the one AffectiveState uses)."""
    return AffectiveState(
        dimensions=EmotionalDimensions(
            valence=0.0, arousal=0.3, dominance=0.0,
            certainty=0.0, goal_congruence=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# StimulusType dimensional changes
# ---------------------------------------------------------------------------

class TestStimulusTypeDimensionalChanges:
    """Each StimulusType should shift the correct dimensions in the expected direction."""

    def _appraise_from_zero(self, stimulus_type: StimulusType, intensity: float = 1.0):
        """Appraise a stimulus from a zeroed-out state with no momentum/baseline pull."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        event = AppraisalEvent(stimulus_type=stimulus_type, intensity=intensity)
        state = _zero_state()
        return engine.appraise(event, state)

    def test_user_message_increases_valence_arousal_goal_congruence(self):
        result = self._appraise_from_zero(StimulusType.USER_MESSAGE)
        assert result.dimensions.valence > 0.0
        assert result.dimensions.arousal > 0.0
        assert result.dimensions.goal_congruence > 0.0

    def test_tool_success_increases_valence_dominance_certainty_goal(self):
        result = self._appraise_from_zero(StimulusType.TOOL_SUCCESS)
        assert result.dimensions.valence > 0.0
        assert result.dimensions.dominance > 0.0
        assert result.dimensions.certainty > 0.0
        assert result.dimensions.goal_congruence > 0.0

    def test_tool_failure_decreases_valence_dominance_certainty_goal(self):
        result = self._appraise_from_zero(StimulusType.TOOL_FAILURE)
        assert result.dimensions.valence < 0.0
        assert result.dimensions.dominance < 0.0
        assert result.dimensions.certainty < 0.0
        assert result.dimensions.goal_congruence < 0.0

    def test_goal_progress_increases_valence_arousal_dominance_goal(self):
        result = self._appraise_from_zero(StimulusType.GOAL_PROGRESS)
        assert result.dimensions.valence > 0.0
        assert result.dimensions.arousal > 0.0
        assert result.dimensions.dominance > 0.0
        assert result.dimensions.goal_congruence > 0.0

    def test_goal_blocked_decreases_valence_dominance_goal_increases_arousal(self):
        result = self._appraise_from_zero(StimulusType.GOAL_BLOCKED)
        assert result.dimensions.valence < 0.0
        assert result.dimensions.arousal > 0.0  # Frustration is activating
        assert result.dimensions.dominance < 0.0
        assert result.dimensions.goal_congruence < 0.0

    def test_novel_information_increases_valence_arousal_decreases_certainty(self):
        result = self._appraise_from_zero(StimulusType.NOVEL_INFORMATION)
        assert result.dimensions.valence > 0.0
        assert result.dimensions.arousal > 0.0
        assert result.dimensions.certainty < 0.0  # Novelty reduces certainty
        assert result.dimensions.goal_congruence > 0.0

    def test_creative_insight_strongly_increases_valence_arousal_goal(self):
        result = self._appraise_from_zero(StimulusType.CREATIVE_INSIGHT)
        assert result.dimensions.valence > 0.0
        assert result.dimensions.arousal > 0.0
        assert result.dimensions.dominance > 0.0
        assert result.dimensions.goal_congruence > 0.0
        # Creative insight should be the strongest positive valence change
        user_msg = self._appraise_from_zero(StimulusType.USER_MESSAGE)
        assert result.dimensions.valence > user_msg.dimensions.valence

    def test_social_connection_increases_valence_goal_congruence(self):
        result = self._appraise_from_zero(StimulusType.SOCIAL_CONNECTION)
        assert result.dimensions.valence > 0.0
        assert result.dimensions.arousal > 0.0
        assert result.dimensions.dominance > 0.0
        assert result.dimensions.goal_congruence > 0.0

    def test_social_disconnect_decreases_valence_dominance_goal(self):
        result = self._appraise_from_zero(StimulusType.SOCIAL_DISCONNECT)
        assert result.dimensions.valence < 0.0
        assert result.dimensions.arousal > 0.0
        assert result.dimensions.dominance < 0.0
        assert result.dimensions.goal_congruence < 0.0

    def test_time_passing_decreases_arousal_and_slightly_valence(self):
        # Start with some arousal so the decrease is visible (arousal is clamped to [0, 1])
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.0, arousal=0.5, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            baseline=EmotionalDimensions(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
        )
        event = AppraisalEvent(stimulus_type=StimulusType.TIME_PASSING, intensity=1.0)
        result = engine.appraise(event, state)
        assert result.dimensions.arousal < 0.5  # Arousal decreased
        assert result.dimensions.valence < 0.0  # Slight negative drift

    def test_self_reflection_increases_certainty_decreases_arousal(self):
        result = self._appraise_from_zero(StimulusType.SELF_REFLECTION)
        assert result.dimensions.certainty > 0.0
        # Arousal decreases, but starts at 0.0 so clamp keeps it at 0.0
        assert result.dimensions.arousal == pytest.approx(0.0)

    def test_error_occurred_decreases_valence_dominance_certainty_increases_arousal(self):
        result = self._appraise_from_zero(StimulusType.ERROR_OCCURRED)
        assert result.dimensions.valence < 0.0
        assert result.dimensions.arousal > 0.0
        assert result.dimensions.dominance < 0.0
        assert result.dimensions.certainty < 0.0

    def test_heartbeat_idle_slightly_decreases_arousal_increases_valence(self):
        # Start with some arousal so decrease is visible
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.0, arousal=0.5, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            baseline=EmotionalDimensions(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
        )
        event = AppraisalEvent(stimulus_type=StimulusType.HEARTBEAT_IDLE, intensity=1.0)
        result = engine.appraise(event, state)
        assert result.dimensions.arousal < 0.5
        assert result.dimensions.valence > 0.0


# ---------------------------------------------------------------------------
# Exact dimensional values for specific stimulus types
# ---------------------------------------------------------------------------

class TestExactDimensionalValues:
    """With no momentum/baseline, dimensional changes should exactly match appraisal rules * intensity."""

    def test_user_message_exact_values_at_full_intensity(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=1.0)
        result = engine.appraise(event, state)
        assert result.dimensions.valence == pytest.approx(0.15)
        assert result.dimensions.arousal == pytest.approx(0.10)
        assert result.dimensions.dominance == pytest.approx(0.0)
        assert result.dimensions.certainty == pytest.approx(0.0)
        assert result.dimensions.goal_congruence == pytest.approx(0.10)

    def test_user_message_negative_valence_hint_can_drive_negative_valence(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(
            stimulus_type=StimulusType.USER_MESSAGE,
            intensity=1.0,
            metadata={"valence_hint": -1.0},
        )
        result = engine.appraise(event, state)
        assert result.dimensions.valence < 0.0

    def test_user_message_positive_valence_hint_strengthens_positive_valence(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        neutral_event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=1.0)
        hinted_event = AppraisalEvent(
            stimulus_type=StimulusType.USER_MESSAGE,
            intensity=1.0,
            metadata={"valence_hint": 1.0},
        )
        neutral_result = engine.appraise(neutral_event, state)
        hinted_result = engine.appraise(hinted_event, state)
        assert hinted_result.dimensions.valence > neutral_result.dimensions.valence

    def test_tool_success_exact_values(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.TOOL_SUCCESS, intensity=1.0)
        result = engine.appraise(event, state)
        assert result.dimensions.valence == pytest.approx(0.20)
        assert result.dimensions.dominance == pytest.approx(0.15)
        assert result.dimensions.certainty == pytest.approx(0.10)
        assert result.dimensions.goal_congruence == pytest.approx(0.20)

    def test_tool_failure_exact_values(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.TOOL_FAILURE, intensity=1.0)
        result = engine.appraise(event, state)
        assert result.dimensions.valence == pytest.approx(-0.15)
        assert result.dimensions.dominance == pytest.approx(-0.10)
        assert result.dimensions.certainty == pytest.approx(-0.15)
        assert result.dimensions.goal_congruence == pytest.approx(-0.10)

    def test_goal_progress_exact_values(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.GOAL_PROGRESS, intensity=1.0)
        result = engine.appraise(event, state)
        assert result.dimensions.valence == pytest.approx(0.25)
        assert result.dimensions.arousal == pytest.approx(0.10)
        assert result.dimensions.dominance == pytest.approx(0.10)
        assert result.dimensions.goal_congruence == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Intensity scaling
# ---------------------------------------------------------------------------

class TestIntensityScaling:
    """Dimensional changes should scale linearly with event intensity."""

    def test_zero_intensity_produces_no_change(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.CREATIVE_INSIGHT, intensity=0.0)
        result = engine.appraise(event, state)
        # With zero intensity, all dimensions stay at zero
        assert result.dimensions.valence == pytest.approx(0.0)
        assert result.dimensions.arousal == pytest.approx(0.0)
        assert result.dimensions.dominance == pytest.approx(0.0)
        assert result.dimensions.certainty == pytest.approx(0.0)
        assert result.dimensions.goal_congruence == pytest.approx(0.0)

    def test_half_intensity_produces_half_change(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)

        # Full intensity
        full_event = AppraisalEvent(stimulus_type=StimulusType.GOAL_PROGRESS, intensity=1.0)
        full_result = engine.appraise(full_event, _zero_state())

        # Half intensity (new engine to reset time)
        engine2 = AppraisalEngine(config)
        half_event = AppraisalEvent(stimulus_type=StimulusType.GOAL_PROGRESS, intensity=0.5)
        half_result = engine2.appraise(half_event, _zero_state())

        assert half_result.dimensions.valence == pytest.approx(full_result.dimensions.valence / 2.0)
        assert half_result.dimensions.arousal == pytest.approx(full_result.dimensions.arousal / 2.0)
        assert half_result.dimensions.dominance == pytest.approx(full_result.dimensions.dominance / 2.0)
        assert half_result.dimensions.goal_congruence == pytest.approx(
            full_result.dimensions.goal_congruence / 2.0
        )

    def test_intensity_scales_linearly(self):
        """Check multiple intensity levels scale proportionally."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
        valences = []
        for intensity in intensities:
            engine = AppraisalEngine(config)
            event = AppraisalEvent(stimulus_type=StimulusType.TOOL_SUCCESS, intensity=intensity)
            result = engine.appraise(event, _zero_state())
            valences.append(result.dimensions.valence)

        # Each step should be the same delta
        for i in range(1, len(valences)):
            expected = 0.20 * intensities[i]  # TOOL_SUCCESS valence base is 0.20
            assert valences[i] == pytest.approx(expected)

    def test_full_intensity_applies_full_rule_values(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        event = AppraisalEvent(stimulus_type=StimulusType.ERROR_OCCURRED, intensity=1.0)
        result = engine.appraise(event, _zero_state())
        # Base rules for ERROR_OCCURRED: valence=-0.15, arousal=0.15, dominance=-0.15, certainty=-0.20
        assert result.dimensions.valence == pytest.approx(-0.15)
        assert result.dimensions.arousal == pytest.approx(0.15)
        assert result.dimensions.dominance == pytest.approx(-0.15)
        assert result.dimensions.certainty == pytest.approx(-0.20)


# ---------------------------------------------------------------------------
# AppraisalEvent intensity clamping
# ---------------------------------------------------------------------------

class TestAppraisalEventIntensityClamping:
    """AppraisalEvent.__post_init__ should clamp intensity to [0.0, 1.0]."""

    def test_negative_intensity_clamped_to_zero(self):
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=-0.5)
        assert event.intensity == 0.0

    def test_intensity_above_one_clamped_to_one(self):
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=2.5)
        assert event.intensity == 1.0

    def test_intensity_in_range_unchanged(self):
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=0.7)
        assert event.intensity == pytest.approx(0.7)

    def test_intensity_boundary_zero(self):
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=0.0)
        assert event.intensity == 0.0

    def test_intensity_boundary_one(self):
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=1.0)
        assert event.intensity == 1.0

    def test_default_intensity_is_half(self):
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE)
        assert event.intensity == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Momentum factor (emotional inertia)
# ---------------------------------------------------------------------------

class TestMomentumFactor:
    """Momentum decay controls how much the previous emotional state carries forward."""

    def test_zero_momentum_means_new_state_fully_applied(self):
        """With momentum_decay=0, the blend weight is (1-0)=1, so new dims fully replace old."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.GOAL_PROGRESS, intensity=1.0)
        result = engine.appraise(event, state)
        # With momentum_decay=0, weight = 1 - 0 = 1, meaning fully new dimensions
        assert result.dimensions.valence == pytest.approx(0.25)
        assert result.dimensions.goal_congruence == pytest.approx(0.30)

    def test_high_momentum_preserves_old_state(self):
        """With momentum_decay close to 1, old state dominates."""
        config = _make_config(momentum_decay=0.9, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        # Start with a specific state
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.5, arousal=0.5, dominance=0.5,
                certainty=0.5, goal_congruence=0.5,
            ),
            baseline=EmotionalDimensions(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
        )
        event = AppraisalEvent(stimulus_type=StimulusType.TOOL_FAILURE, intensity=1.0)
        result = engine.appraise(event, state)
        # With momentum_decay=0.9, blend weight=(1-0.9)=0.1, mostly old state preserved
        # New raw dimensions: valence=0.5+(-0.15)=0.35, dominance=0.5+(-0.10)=0.40, etc.
        # Blended: old * 0.9 + new * 0.1
        # Valence: 0.5 * 0.9 + 0.35 * 0.1 = 0.45 + 0.035 = 0.485
        assert result.dimensions.valence == pytest.approx(0.485)

    def test_momentum_half_blends_equally(self):
        """With momentum_decay=0.5, old and new are blended 50/50."""
        config = _make_config(momentum_decay=0.5, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=1.0)
        result = engine.appraise(event, state)
        # New raw dims: valence=0.15, arousal=0.10, goal_congruence=0.10
        # Blend: old(zero) * 0.5 + new * 0.5
        # Valence: 0.0 * 0.5 + 0.15 * 0.5 = 0.075
        assert result.dimensions.valence == pytest.approx(0.075)
        assert result.dimensions.arousal == pytest.approx(0.05)
        assert result.dimensions.goal_congruence == pytest.approx(0.05)

    def test_default_momentum_decay_is_085(self):
        """The default AffectConfig has momentum_decay=0.85."""
        config = AffectConfig(
            arousal_ceiling=0.92,
            distress_threshold=0.80,
            distress_duration_limit=300.0,
        )
        assert config.momentum_decay == pytest.approx(0.85)

    def test_momentum_makes_change_gradual(self):
        """Higher momentum means smaller per-appraisal change from starting state."""
        config_low = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        config_high = _make_config(momentum_decay=0.8, baseline_pull=0.0)
        engine_low = AppraisalEngine(config_low)
        engine_high = AppraisalEngine(config_high)

        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.CREATIVE_INSIGHT, intensity=1.0)

        result_low = engine_low.appraise(event, state)
        result_high = engine_high.appraise(event, state)

        # Low momentum should produce bigger change from zero
        assert abs(result_low.dimensions.valence) > abs(result_high.dimensions.valence)


# ---------------------------------------------------------------------------
# Baseline pull (drift toward equilibrium)
# ---------------------------------------------------------------------------

class TestBaselinePull:
    """baseline_pull controls gentle drift toward the agent's baseline emotional state."""

    def test_zero_baseline_pull_means_no_drift(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        # State with non-zero baseline but dimensions far from it
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=-0.5, arousal=0.8, dominance=-0.5,
                certainty=-0.5, goal_congruence=-0.5,
            ),
            baseline=EmotionalDimensions(
                valence=0.2, arousal=0.3, dominance=0.1,
                certainty=0.0, goal_congruence=0.1,
            ),
        )
        event = AppraisalEvent(stimulus_type=StimulusType.HEARTBEAT_IDLE, intensity=0.0)
        result = engine.appraise(event, state)
        # Zero intensity + zero baseline pull = no change from current dims
        # (momentum_decay=0 means blend weight=1, new dims = current + 0 = current)
        assert result.dimensions.valence == pytest.approx(-0.5)

    def test_nonzero_baseline_pull_moves_toward_baseline(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.1)
        engine = AppraisalEngine(config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.8, arousal=0.5, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            baseline=EmotionalDimensions(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
        )
        event = AppraisalEvent(stimulus_type=StimulusType.HEARTBEAT_IDLE, intensity=0.0)
        result = engine.appraise(event, state)
        # After momentum step: dims = (0.8, 0.5, ...) since momentum=0 and intensity=0
        # After baseline pull (0.1 toward zero baseline): valence = 0.8 * 0.9 + 0.0 * 0.1 = 0.72
        assert result.dimensions.valence == pytest.approx(0.72)

    def test_full_baseline_pull_snaps_to_baseline(self):
        """With baseline_pull=1.0, state should become the baseline."""
        config = _make_config(momentum_decay=0.0, baseline_pull=1.0)
        engine = AppraisalEngine(config)
        baseline = EmotionalDimensions(
            valence=0.2, arousal=0.3, dominance=0.1,
            certainty=0.0, goal_congruence=0.1,
        )
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.9, arousal=0.9, dominance=0.9,
                certainty=0.9, goal_congruence=0.9,
            ),
            baseline=baseline,
        )
        event = AppraisalEvent(stimulus_type=StimulusType.HEARTBEAT_IDLE, intensity=0.0)
        result = engine.appraise(event, state)
        assert result.dimensions.valence == pytest.approx(baseline.valence)
        assert result.dimensions.arousal == pytest.approx(baseline.arousal)
        assert result.dimensions.dominance == pytest.approx(baseline.dominance)
        assert result.dimensions.certainty == pytest.approx(baseline.certainty)
        assert result.dimensions.goal_congruence == pytest.approx(baseline.goal_congruence)

    def test_baseline_pull_default_is_005(self):
        """The default baseline_pull is 0.05."""
        config = AffectConfig(
            arousal_ceiling=0.92,
            distress_threshold=0.80,
            distress_duration_limit=300.0,
        )
        assert config.baseline_pull == pytest.approx(0.05)

    def test_repeated_baseline_pull_converges_toward_baseline(self):
        """Multiple appraisals with idle events should gradually approach baseline."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.2)
        engine = AppraisalEngine(config)
        baseline = EmotionalDimensions(
            valence=0.0, arousal=0.0, dominance=0.0,
            certainty=0.0, goal_congruence=0.0,
        )
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=1.0, arousal=1.0, dominance=1.0,
                certainty=1.0, goal_congruence=1.0,
            ),
            baseline=baseline,
        )
        event = AppraisalEvent(stimulus_type=StimulusType.HEARTBEAT_IDLE, intensity=0.0)

        prev_valence = state.dimensions.valence
        for _ in range(10):
            state = engine.appraise(event, state)
            assert state.dimensions.valence < prev_valence
            prev_valence = state.dimensions.valence

        # After 10 rounds of 20% pull toward zero, should be much closer to zero
        assert state.dimensions.valence < 0.2


# ---------------------------------------------------------------------------
# Clamping (dimensions stay in valid ranges)
# ---------------------------------------------------------------------------

class TestClamping:
    """After appraisal, all dimensions must remain within their valid ranges."""

    def test_repeated_positive_stimuli_clamp_valence_at_one(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.CREATIVE_INSIGHT, intensity=1.0)
        # Apply many positive stimuli
        for _ in range(20):
            state = engine.appraise(event, state)
        assert state.dimensions.valence <= 1.0
        assert state.dimensions.arousal <= 1.0
        assert state.dimensions.dominance <= 1.0
        assert state.dimensions.goal_congruence <= 1.0

    def test_repeated_negative_stimuli_clamp_valence_at_minus_one(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.GOAL_BLOCKED, intensity=1.0)
        for _ in range(30):
            state = engine.appraise(event, state)
        assert state.dimensions.valence >= -1.0
        assert state.dimensions.dominance >= -1.0
        assert state.dimensions.goal_congruence >= -1.0

    def test_arousal_never_goes_below_zero(self):
        """Arousal is [0, 1], not [-1, 1]."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        # TIME_PASSING decreases arousal; starting from 0 it should stay at 0
        event = AppraisalEvent(stimulus_type=StimulusType.TIME_PASSING, intensity=1.0)
        for _ in range(10):
            state = engine.appraise(event, state)
        assert state.dimensions.arousal >= 0.0

    def test_all_dimensions_within_bounds_after_mixed_stimuli(self):
        """Random sequence of stimuli should never push dimensions out of bounds."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        stimuli = list(StimulusType)
        for stim in stimuli * 5:
            event = AppraisalEvent(stimulus_type=stim, intensity=1.0)
            state = engine.appraise(event, state)
            assert -1.0 <= state.dimensions.valence <= 1.0
            assert 0.0 <= state.dimensions.arousal <= 1.0
            assert -1.0 <= state.dimensions.dominance <= 1.0
            assert -1.0 <= state.dimensions.certainty <= 1.0
            assert -1.0 <= state.dimensions.goal_congruence <= 1.0


# ---------------------------------------------------------------------------
# Emotion reclassification after appraisal
# ---------------------------------------------------------------------------

class TestEmotionReclassification:
    """The appraisal engine should reclassify the named emotion based on new dimensions."""

    def test_appraisal_updates_emotion_label(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        # Start neutral
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.0, arousal=0.4, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            baseline=EmotionalDimensions(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            current_emotion=EmotionLabel.NEUTRAL,
        )
        # Push toward high positive valence + medium arousal -> JOY
        event = AppraisalEvent(stimulus_type=StimulusType.CREATIVE_INSIGHT, intensity=1.0)
        result = engine.appraise(event, state)
        # After: valence=0.30, arousal=0.60 => high arousal, positive valence
        # With arousal=0.60 and valence=0.30 => not > 0.6 arousal, so medium arousal
        # classify: 0.3 <= arousal <= 0.6, valence=0.30 < 0.4 but > 0.1 -> CURIOSITY or SATISFACTION
        # The exact label depends on the final dimension values
        assert result.current_emotion != EmotionLabel.NEUTRAL or result.previous_emotion is not None

    def test_previous_emotion_tracked_on_change(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.0, arousal=0.4, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            baseline=EmotionalDimensions(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            current_emotion=EmotionLabel.NEUTRAL,
        )
        # Apply strong positive to shift to a non-neutral emotion
        event = AppraisalEvent(stimulus_type=StimulusType.GOAL_PROGRESS, intensity=1.0)
        result = engine.appraise(event, state)
        if result.current_emotion != EmotionLabel.NEUTRAL:
            assert result.previous_emotion == EmotionLabel.NEUTRAL

    def test_duration_resets_on_emotion_change(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.5, arousal=0.5, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            baseline=EmotionalDimensions(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            current_emotion=EmotionLabel.JOY,
            duration_in_current_state=60.0,
        )
        # Apply negative stimuli to shift from JOY to something else
        event = AppraisalEvent(stimulus_type=StimulusType.GOAL_BLOCKED, intensity=1.0)
        result = engine.appraise(event, state)
        if result.current_emotion != EmotionLabel.JOY:
            assert result.duration_in_current_state == 0.0

    def test_same_emotion_preserves_duration(self):
        """If emotion stays the same, duration accumulates."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        # Start in NEUTRAL (medium arousal, near-zero valence)
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.0, arousal=0.4, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            baseline=EmotionalDimensions(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            current_emotion=EmotionLabel.NEUTRAL,
            duration_in_current_state=10.0,
        )
        # Very mild stimulus unlikely to change emotion
        event = AppraisalEvent(stimulus_type=StimulusType.HEARTBEAT_IDLE, intensity=0.1)
        result = engine.appraise(event, state)
        # Duration should have increased (not reset), since emotion likely stays NEUTRAL
        if result.current_emotion == EmotionLabel.NEUTRAL:
            assert result.duration_in_current_state > 10.0

    def test_strong_negative_chain_reaches_frustration_or_anxiety(self):
        """Repeated negative events should push to a clearly negative emotion."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        state.current_emotion = EmotionLabel.NEUTRAL
        event = AppraisalEvent(stimulus_type=StimulusType.GOAL_BLOCKED, intensity=1.0)
        for _ in range(5):
            state = engine.appraise(event, state)
        # Should be a negative emotion
        negative_emotions = {
            EmotionLabel.FRUSTRATION, EmotionLabel.ANXIETY,
            EmotionLabel.SADNESS, EmotionLabel.CONFUSION,
        }
        assert state.current_emotion in negative_emotions

    def test_strong_positive_chain_reaches_positive_emotion(self):
        """Repeated positive events should push to a clearly positive emotion."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        state.current_emotion = EmotionLabel.NEUTRAL
        event = AppraisalEvent(stimulus_type=StimulusType.CREATIVE_INSIGHT, intensity=1.0)
        for _ in range(5):
            state = engine.appraise(event, state)
        positive_emotions = {
            EmotionLabel.JOY, EmotionLabel.EXCITEMENT,
            EmotionLabel.SATISFACTION, EmotionLabel.CURIOSITY,
            EmotionLabel.CALM, EmotionLabel.AFFECTION,
            EmotionLabel.AWE, EmotionLabel.SURPRISE,
        }
        assert state.current_emotion in positive_emotions


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: unknown stimulus type handling, zero intensity, metadata, etc."""

    def test_unknown_stimulus_type_produces_no_dimensional_change(self):
        """If an unrecognized stimulus type is used (via _appraisal_rules.get default),
        the state should only change due to momentum/baseline, not stimulus deltas."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        # Remove a known rule and test fallback behavior
        # We cannot add a truly unknown enum member, but we can test the .get({}) fallback
        # by temporarily clearing the rules
        engine._appraisal_rules.clear()
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=1.0)
        result = engine.appraise(event, state)
        # No rules -> no changes -> all zeros
        assert result.dimensions.valence == pytest.approx(0.0)
        assert result.dimensions.arousal == pytest.approx(0.0)
        assert result.dimensions.dominance == pytest.approx(0.0)
        assert result.dimensions.certainty == pytest.approx(0.0)
        assert result.dimensions.goal_congruence == pytest.approx(0.0)

    def test_zero_intensity_from_any_stimulus_produces_no_change(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        state = _zero_state()
        for stim_type in StimulusType:
            engine_inner = AppraisalEngine(config)
            event = AppraisalEvent(stimulus_type=stim_type, intensity=0.0)
            result = engine_inner.appraise(event, state)
            assert result.dimensions.valence == pytest.approx(0.0), f"Failed for {stim_type}"
            assert result.dimensions.arousal == pytest.approx(0.0), f"Failed for {stim_type}"
            assert result.dimensions.dominance == pytest.approx(0.0), f"Failed for {stim_type}"
            assert result.dimensions.certainty == pytest.approx(0.0), f"Failed for {stim_type}"
            assert result.dimensions.goal_congruence == pytest.approx(0.0), f"Failed for {stim_type}"

    def test_appraisal_event_metadata_passthrough(self):
        event = AppraisalEvent(
            stimulus_type=StimulusType.USER_MESSAGE,
            intensity=0.5,
            content="Hello there",
            metadata={"user_id": "alice", "channel": "discord"},
        )
        assert event.content == "Hello there"
        assert event.metadata["user_id"] == "alice"
        assert event.metadata["channel"] == "discord"

    def test_appraisal_event_default_metadata_is_empty_dict(self):
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE)
        assert event.metadata == {}

    def test_appraisal_event_default_content_is_none(self):
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE)
        assert event.content is None

    def test_time_elapsed_tracking(self):
        """The appraisal engine should track time between appraisals."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        state.duration_in_current_state = 0.0
        event = AppraisalEvent(stimulus_type=StimulusType.HEARTBEAT_IDLE, intensity=0.0)

        result1 = engine.appraise(event, state)
        # duration_in_current_state should have increased by the elapsed time
        assert result1.duration_in_current_state >= 0.0

        result2 = engine.appraise(event, result1)
        # Should continue to accumulate
        assert result2.duration_in_current_state >= result1.duration_in_current_state

    def test_timestamp_is_updated(self):
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        before = time.time()
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=0.5)
        result = engine.appraise(event, state)
        after = time.time()
        assert before <= result.timestamp <= after

    def test_momentum_field_is_nonnegative(self):
        """The momentum field on the returned state should be abs(distance)."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.TOOL_FAILURE, intensity=1.0)
        result = engine.appraise(event, state)
        assert result.momentum >= 0.0

    def test_baseline_is_preserved(self):
        """Appraisal should not modify the baseline of the returned state."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        baseline = EmotionalDimensions(
            valence=0.2, arousal=0.3, dominance=0.1,
            certainty=0.0, goal_congruence=0.1,
        )
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            baseline=baseline,
        )
        event = AppraisalEvent(stimulus_type=StimulusType.CREATIVE_INSIGHT, intensity=1.0)
        result = engine.appraise(event, state)
        assert result.baseline.valence == pytest.approx(baseline.valence)
        assert result.baseline.arousal == pytest.approx(baseline.arousal)
        assert result.baseline.dominance == pytest.approx(baseline.dominance)
        assert result.baseline.certainty == pytest.approx(baseline.certainty)
        assert result.baseline.goal_congruence == pytest.approx(baseline.goal_congruence)

    def test_appraise_returns_new_state_object(self):
        """appraise() should return a new AffectiveState, not mutate the input."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=0.5)
        result = engine.appraise(event, state)
        assert result is not state

    def test_last_appraisal_time_updates(self):
        """The engine's internal _last_appraisal_time should advance after appraise()."""
        config = _make_config(momentum_decay=0.0, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        initial_time = engine._last_appraisal_time
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=0.5)
        engine.appraise(event, state)
        assert engine._last_appraisal_time >= initial_time


# ---------------------------------------------------------------------------
# All StimulusType enum members have appraisal rules
# ---------------------------------------------------------------------------

class TestAppraisalRuleCoverage:
    """Every StimulusType should have a corresponding entry in the appraisal rules."""

    def test_all_stimulus_types_have_rules(self):
        config = _make_config()
        engine = AppraisalEngine(config)
        for stim_type in StimulusType:
            assert stim_type in engine._appraisal_rules, (
                f"{stim_type} has no appraisal rule"
            )

    def test_all_rules_reference_valid_dimension_names(self):
        config = _make_config()
        engine = AppraisalEngine(config)
        valid_dims = {"valence", "arousal", "dominance", "certainty", "goal_congruence"}
        for stim_type, rules in engine._appraisal_rules.items():
            for dim_name in rules:
                assert dim_name in valid_dims, (
                    f"{stim_type} references invalid dimension '{dim_name}'"
                )


# ---------------------------------------------------------------------------
# Integration: momentum + baseline_pull together
# ---------------------------------------------------------------------------

class TestMomentumAndBaselinePullIntegration:
    """Test the interplay between momentum and baseline pull."""

    def test_with_both_momentum_and_baseline_pull(self):
        """With realistic settings, changes are damped by momentum and pulled toward baseline."""
        config = _make_config(momentum_decay=0.85, baseline_pull=0.05)
        engine = AppraisalEngine(config)
        baseline = EmotionalDimensions(
            valence=0.2, arousal=0.3, dominance=0.1,
            certainty=0.0, goal_congruence=0.1,
        )
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=0.0, arousal=0.3, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
            ),
            baseline=baseline,
            current_emotion=EmotionLabel.NEUTRAL,
        )
        event = AppraisalEvent(stimulus_type=StimulusType.USER_MESSAGE, intensity=0.5)
        result = engine.appraise(event, state)
        # Should have moved slightly positive due to user_message
        # But damped by momentum and pulled slightly toward baseline
        assert result.dimensions.valence > 0.0

    def test_idle_system_converges_to_baseline(self):
        """Repeated idle heartbeats should drive the system toward baseline."""
        config = _make_config(momentum_decay=0.5, baseline_pull=0.1)
        engine = AppraisalEngine(config)
        baseline = EmotionalDimensions(
            valence=0.2, arousal=0.3, dominance=0.1,
            certainty=0.0, goal_congruence=0.1,
        )
        state = AffectiveState(
            dimensions=EmotionalDimensions(
                valence=-0.8, arousal=0.9, dominance=-0.5,
                certainty=-0.5, goal_congruence=-0.5,
            ),
            baseline=baseline,
            current_emotion=EmotionLabel.ANXIETY,
        )
        event = AppraisalEvent(stimulus_type=StimulusType.HEARTBEAT_IDLE, intensity=0.5)
        for _ in range(50):
            state = engine.appraise(event, state)

        # After many idle heartbeats, should be much closer to baseline
        assert abs(state.dimensions.valence - baseline.valence) < 0.3
        assert abs(state.dimensions.arousal - baseline.arousal) < 0.3

    def test_strong_stimulus_overcomes_momentum(self):
        """Even with high momentum, a very strong repeated stimulus should shift the state."""
        config = _make_config(momentum_decay=0.85, baseline_pull=0.0)
        engine = AppraisalEngine(config)
        state = _zero_state()
        event = AppraisalEvent(stimulus_type=StimulusType.CREATIVE_INSIGHT, intensity=1.0)
        for _ in range(20):
            state = engine.appraise(event, state)
        # Despite 85% momentum, repeated strong stimuli should have moved valence significantly
        assert state.dimensions.valence > 0.1


# ---------------------------------------------------------------------------
# StimulusType enum values
# ---------------------------------------------------------------------------

class TestStimulusTypeEnum:
    """StimulusType enum should have expected members and string values."""

    def test_all_expected_members_exist(self):
        expected = {
            "USER_MESSAGE", "TOOL_SUCCESS", "TOOL_FAILURE",
            "GOAL_PROGRESS", "GOAL_BLOCKED", "NOVEL_INFORMATION",
            "CREATIVE_INSIGHT", "SOCIAL_CONNECTION", "SOCIAL_DISCONNECT",
            "TIME_PASSING", "SELF_REFLECTION", "ERROR_OCCURRED",
            "HEARTBEAT_IDLE",
        }
        actual = {e.name for e in StimulusType}
        assert expected == actual

    def test_stimulus_type_is_str_enum(self):
        assert isinstance(StimulusType.USER_MESSAGE, str)
        assert StimulusType.USER_MESSAGE == "user_message"
        assert StimulusType.TOOL_FAILURE == "tool_failure"
