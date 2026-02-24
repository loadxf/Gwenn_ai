"""
Tests for gwenn.affect — Affective State and Resilience Circuit.

Covers:
- EmotionalDimensions clamping
- Emotion classification from dimensions
- Resilience circuit breaker (arousal ceiling)
- Distress timeout
- Habituation factor
"""

from __future__ import annotations

import time

import pytest

from gwenn.affect.state import AffectiveState, EmotionalDimensions, EmotionLabel
from gwenn.affect.resilience import ResilienceCircuit


# ---------------------------------------------------------------------------
# EmotionalDimensions clamping
# ---------------------------------------------------------------------------

class TestEmotionalDimensionsClamping:
    """clamp() should keep all dimensions within valid ranges."""

    def test_clamp_valence_above_max(self):
        d = EmotionalDimensions(valence=2.0)
        clamped = d.clamp()
        assert clamped.valence == 1.0

    def test_clamp_valence_below_min(self):
        d = EmotionalDimensions(valence=-2.0)
        clamped = d.clamp()
        assert clamped.valence == -1.0

    def test_clamp_arousal_above_max(self):
        d = EmotionalDimensions(arousal=1.5)
        clamped = d.clamp()
        assert clamped.arousal == 1.0

    def test_clamp_arousal_below_min(self):
        d = EmotionalDimensions(arousal=-0.5)
        clamped = d.clamp()
        assert clamped.arousal == 0.0

    def test_clamp_dominance(self):
        d = EmotionalDimensions(dominance=3.0)
        assert d.clamp().dominance == 1.0
        d2 = EmotionalDimensions(dominance=-3.0)
        assert d2.clamp().dominance == -1.0

    def test_clamp_certainty(self):
        d = EmotionalDimensions(certainty=5.0)
        assert d.clamp().certainty == 1.0
        d2 = EmotionalDimensions(certainty=-5.0)
        assert d2.clamp().certainty == -1.0

    def test_clamp_goal_congruence(self):
        d = EmotionalDimensions(goal_congruence=2.0)
        assert d.clamp().goal_congruence == 1.0
        d2 = EmotionalDimensions(goal_congruence=-2.0)
        assert d2.clamp().goal_congruence == -1.0

    def test_clamp_preserves_valid_values(self):
        d = EmotionalDimensions(
            valence=0.5, arousal=0.3, dominance=-0.2, certainty=0.1, goal_congruence=0.0
        )
        clamped = d.clamp()
        assert clamped.valence == pytest.approx(0.5)
        assert clamped.arousal == pytest.approx(0.3)
        assert clamped.dominance == pytest.approx(-0.2)
        assert clamped.certainty == pytest.approx(0.1)
        assert clamped.goal_congruence == pytest.approx(0.0)

    def test_clamp_all_extreme_values(self):
        d = EmotionalDimensions(
            valence=10.0, arousal=-5.0, dominance=99.0, certainty=-99.0, goal_congruence=50.0
        )
        clamped = d.clamp()
        assert clamped.valence == 1.0
        assert clamped.arousal == 0.0
        assert clamped.dominance == 1.0
        assert clamped.certainty == -1.0
        assert clamped.goal_congruence == 1.0

    def test_blend_produces_clamped_result(self):
        d1 = EmotionalDimensions(valence=0.8, arousal=0.9)
        d2 = EmotionalDimensions(valence=0.6, arousal=0.7)
        blended = d1.blend(d2, weight=0.5)
        assert -1.0 <= blended.valence <= 1.0
        assert 0.0 <= blended.arousal <= 1.0

    def test_distance_from(self):
        d1 = EmotionalDimensions(valence=0.0, arousal=0.0, dominance=0.0, certainty=0.0, goal_congruence=0.0)
        d2 = EmotionalDimensions(valence=1.0, arousal=0.0, dominance=0.0, certainty=0.0, goal_congruence=0.0)
        assert d1.distance_from(d2) == pytest.approx(1.0)

    def test_distance_from_identical_is_zero(self):
        d = EmotionalDimensions(valence=0.5, arousal=0.3)
        assert d.distance_from(d) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Emotion classification from dimensions
# ---------------------------------------------------------------------------

class TestEmotionClassification:
    """classify_emotion maps dimensional values to named emotions."""

    def test_high_arousal_positive_valence_with_certainty_is_excitement(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.8, certainty=0.5)
        )
        assert state.classify_emotion() == EmotionLabel.EXCITEMENT

    def test_high_arousal_positive_valence_without_certainty_is_surprise(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.8, certainty=-0.5)
        )
        assert state.classify_emotion() == EmotionLabel.SURPRISE

    def test_high_arousal_negative_valence_low_certainty_is_anxiety(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.5, arousal=0.8, certainty=-0.5)
        )
        assert state.classify_emotion() == EmotionLabel.ANXIETY

    def test_high_arousal_negative_valence_high_certainty_is_frustration(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.5, arousal=0.8, certainty=0.5)
        )
        assert state.classify_emotion() == EmotionLabel.FRUSTRATION

    def test_medium_arousal_high_valence_is_joy(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.5, certainty=0.0)
        )
        assert state.classify_emotion() == EmotionLabel.JOY

    def test_medium_arousal_moderate_positive_low_certainty_is_curiosity(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.2, arousal=0.4, certainty=0.0)
        )
        assert state.classify_emotion() == EmotionLabel.CURIOSITY

    def test_medium_arousal_moderate_positive_high_certainty_is_satisfaction(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.2, arousal=0.4, certainty=0.5)
        )
        assert state.classify_emotion() == EmotionLabel.SATISFACTION

    def test_medium_arousal_negative_is_sadness(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.5, arousal=0.4, certainty=0.0)
        )
        assert state.classify_emotion() == EmotionLabel.SADNESS

    def test_medium_arousal_slightly_negative_low_certainty_is_confusion(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.2, arousal=0.4, certainty=-0.5)
        )
        assert state.classify_emotion() == EmotionLabel.CONFUSION

    def test_low_arousal_positive_high_certainty_is_calm(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.2, certainty=0.5)
        )
        assert state.classify_emotion() == EmotionLabel.CALM

    def test_low_arousal_positive_low_certainty_is_affection(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.2, certainty=-0.5)
        )
        assert state.classify_emotion() == EmotionLabel.AFFECTION

    def test_low_arousal_negative_is_sadness(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.5, arousal=0.1, certainty=0.0)
        )
        assert state.classify_emotion() == EmotionLabel.SADNESS

    def test_very_low_arousal_neutral_is_boredom(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.0, arousal=0.1, certainty=0.0)
        )
        assert state.classify_emotion() == EmotionLabel.BOREDOM

    def test_neutral_medium_arousal_is_neutral(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.0, arousal=0.4, certainty=0.0)
        )
        assert state.classify_emotion() == EmotionLabel.NEUTRAL

    def test_update_classification_tracks_transition(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.5),
            current_emotion=EmotionLabel.NEUTRAL,
        )
        state.update_classification()
        assert state.current_emotion == EmotionLabel.JOY
        assert state.previous_emotion == EmotionLabel.NEUTRAL

    def test_update_classification_resets_duration(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.5),
            current_emotion=EmotionLabel.NEUTRAL,
            duration_in_current_state=120.0,
        )
        state.update_classification()
        # Since the classification changed, duration should be reset
        assert state.duration_in_current_state == 0.0


# ---------------------------------------------------------------------------
# Resilience circuit breaker — arousal ceiling
# ---------------------------------------------------------------------------

class TestArousalCeiling:
    """The resilience circuit should clamp arousal at the configured ceiling."""

    def test_arousal_above_ceiling_is_clamped(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.99)
        )
        result = circuit.check(state)
        assert result.dimensions.arousal == pytest.approx(affect_config.arousal_ceiling)

    def test_arousal_below_ceiling_is_untouched(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.5)
        )
        result = circuit.check(state)
        assert result.dimensions.arousal == pytest.approx(0.5)

    def test_arousal_ceiling_also_dampens_valence(self, affect_config):
        """When arousal hits the ceiling, valence is slightly reduced (multiplied by 0.95)."""
        circuit = ResilienceCircuit(affect_config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.8, arousal=0.99)
        )
        result = circuit.check(state)
        assert result.dimensions.valence == pytest.approx(0.8 * 0.95)

    def test_arousal_at_exactly_ceiling_is_not_modified(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=affect_config.arousal_ceiling)
        )
        result = circuit.check(state)
        # At the ceiling exactly, no modification (not strictly greater)
        assert result.dimensions.arousal == pytest.approx(affect_config.arousal_ceiling)
        assert result.dimensions.valence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Distress timeout
# ---------------------------------------------------------------------------

class TestDistressTimeout:
    """Sustained negative state should trigger forced recovery after duration_limit."""

    def test_distress_detection(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.9, arousal=0.5)
        )
        circuit.check(state)
        assert circuit.is_distressed is True

    def test_no_distress_on_positive_state(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.5)
        )
        circuit.check(state)
        assert circuit.is_distressed is False

    def test_distress_resolves_when_valence_improves(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        # First: enter distress
        distressed = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.9, arousal=0.5)
        )
        circuit.check(distressed)
        assert circuit.is_distressed is True

        # Then: recover
        recovered = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.3, arousal=0.3)
        )
        circuit.check(recovered)
        assert circuit.is_distressed is False

    def test_forced_recovery_after_duration_limit(self, affect_config):
        circuit = ResilienceCircuit(affect_config)

        # Enter distress
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.9, arousal=0.5)
        )
        circuit.check(state)

        # Simulate time passing beyond the duration limit by backdating distress_start
        circuit._distress_start = time.time() - (affect_config.distress_duration_limit + 10)

        # Check again — forced recovery should pull valence toward baseline
        result = circuit.check(state)
        # Valence should have moved toward the baseline (which is 0.2 by default)
        assert result.dimensions.valence > -0.9

    def test_distress_duration_property(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        assert circuit.distress_duration == 0.0

        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.9, arousal=0.5)
        )
        circuit.check(state)
        # Should be a very small positive number now
        assert circuit.distress_duration >= 0.0

    def test_high_arousal_and_moderate_negative_also_triggers_distress(self, affect_config):
        """Distress also triggers when arousal > 0.7 and valence < -0.3."""
        circuit = ResilienceCircuit(affect_config)
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=-0.4, arousal=0.8)
        )
        circuit.check(state)
        assert circuit.is_distressed is True

    def test_status_property(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        s = circuit.status
        assert "breaker_active" in s
        assert "distress_duration" in s
        assert "arousal_ceiling" in s
        assert s["arousal_ceiling"] == affect_config.arousal_ceiling


# ---------------------------------------------------------------------------
# Habituation factor
# ---------------------------------------------------------------------------

class TestHabituation:
    """Repeated identical stimuli should have diminishing emotional impact."""

    def test_first_stimulus_has_full_impact(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        factor = circuit.get_habituation_factor("error_message")
        assert factor == pytest.approx(1.0)

    def test_repeated_stimulus_diminishes(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        factors = []
        for _ in range(5):
            factors.append(circuit.get_habituation_factor("error_message"))
        # Each subsequent factor should be smaller
        for i in range(1, len(factors)):
            assert factors[i] < factors[i - 1]

    def test_habituation_never_goes_below_minimum(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        # Hit the same stimulus many times
        factor = None
        for _ in range(100):
            factor = circuit.get_habituation_factor("flood")
        assert factor >= 0.1  # minimum 10% impact

    def test_different_stimuli_are_independent(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        # Habituate to "error_a"
        for _ in range(5):
            circuit.get_habituation_factor("error_a")

        # "error_b" should still have full impact
        factor_b = circuit.get_habituation_factor("error_b")
        assert factor_b == pytest.approx(1.0)

    def test_habituation_decay_rate(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        # First call: full impact
        f1 = circuit.get_habituation_factor("test_stim")
        assert f1 == pytest.approx(1.0)
        # Second call: should be decay_rate^1 = 0.7
        f2 = circuit.get_habituation_factor("test_stim")
        assert f2 == pytest.approx(0.7)
        # Third call: should be decay_rate^2 = 0.49
        f3 = circuit.get_habituation_factor("test_stim")
        assert f3 == pytest.approx(0.49)

    def test_habituation_resets_after_window(self, affect_config):
        circuit = ResilienceCircuit(affect_config)
        # Habituate
        circuit.get_habituation_factor("reset_test")
        circuit.get_habituation_factor("reset_test")

        # Simulate time passing beyond the habituation window (120s)
        circuit._habituation["reset_test"] = (time.time() - 200, 5)

        # Should reset to full impact
        factor = circuit.get_habituation_factor("reset_test")
        assert factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AffectiveState serialization and prompt
# ---------------------------------------------------------------------------

class TestAffectiveStateSerialization:
    """AffectiveState.to_dict and to_prompt_fragment should produce valid output."""

    def test_to_dict_contains_all_fields(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.4),
            current_emotion=EmotionLabel.JOY,
        )
        d = state.to_dict()
        assert d["valence"] == pytest.approx(0.5)
        assert d["arousal"] == pytest.approx(0.4)
        assert d["emotion"] == "joy"

    def test_to_prompt_fragment_is_string(self):
        state = AffectiveState(
            dimensions=EmotionalDimensions(valence=0.5, arousal=0.8),
            current_emotion=EmotionLabel.EXCITEMENT,
        )
        fragment = state.to_prompt_fragment()
        assert isinstance(fragment, str)
        assert "excitement" in fragment.lower()

    def test_prompt_fragment_intensity_levels(self):
        # Low arousal -> "mildly"
        low = AffectiveState(dimensions=EmotionalDimensions(arousal=0.2))
        assert "mildly" in low.to_prompt_fragment()

        # Medium arousal -> "quite"
        mid = AffectiveState(dimensions=EmotionalDimensions(arousal=0.5))
        assert "quite" in mid.to_prompt_fragment()

        # High arousal -> "intensely"
        high = AffectiveState(dimensions=EmotionalDimensions(arousal=0.8))
        assert "intensely" in high.to_prompt_fragment()
