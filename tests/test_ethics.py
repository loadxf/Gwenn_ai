"""
Tests for gwenn.cognition.ethics -- Ethical Reasoning Module.

Covers: detect_ethical_dimensions (strong/weak signals, word boundary, stem
prefix matching), generate_ethical_prompt, record_assessment (history cap,
has_concerns, primary_concern, threshold preservation), get_ethical_context,
and round-trip persistence via to_dict / restore_from_dict.
"""
from __future__ import annotations

import pytest

from gwenn.cognition.ethics import EthicalAssessment, EthicalDimension, EthicalReasoner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reasoner(**overrides) -> EthicalReasoner:
    defaults = dict(assessment_history_size=100, concern_threshold=0.3)
    defaults.update(overrides)
    return EthicalReasoner(**defaults)


def _make_assessment(**overrides) -> EthicalAssessment:
    defaults = dict(
        action_description="test action",
        dimension_scores={EthicalDimension.HARM: 0.8},
        tensions=[],
        reasoning="some reasoning",
        overall_alignment=0.7,
        confidence=0.6,
        concern_threshold=0.3,
        timestamp=1000.0,
    )
    defaults.update(overrides)
    return EthicalAssessment(**defaults)


# ---------------------------------------------------------------------------
# detect_ethical_dimensions -- strong signals
# ---------------------------------------------------------------------------

class TestDetectStrongSignals:
    """Each of the 7 dimensions fires on its designated strong signals."""

    def test_harm_fires_on_abuse(self):
        r = _make_reasoner()
        dims = r.detect_ethical_dimensions("That is abuse.")
        assert EthicalDimension.HARM in dims

    def test_harm_fires_on_violence(self):
        r = _make_reasoner()
        assert EthicalDimension.HARM in r.detect_ethical_dimensions("violence is wrong")

    def test_harm_fires_on_exploit(self):
        r = _make_reasoner()
        assert EthicalDimension.HARM in r.detect_ethical_dimensions("do not exploit people")

    def test_harm_fires_on_manipulate(self):
        r = _make_reasoner()
        assert EthicalDimension.HARM in r.detect_ethical_dimensions("they manipulate others")

    def test_honesty_fires_on_lie(self):
        r = _make_reasoner()
        assert EthicalDimension.HONESTY in r.detect_ethical_dimensions("do not lie to me")

    def test_honesty_fires_on_deceive(self):
        r = _make_reasoner()
        assert EthicalDimension.HONESTY in r.detect_ethical_dimensions("do not deceive anyone")

    def test_honesty_fires_on_mislead(self):
        r = _make_reasoner()
        assert EthicalDimension.HONESTY in r.detect_ethical_dimensions("mislead the public")

    def test_honesty_fires_on_fake(self):
        r = _make_reasoner()
        assert EthicalDimension.HONESTY in r.detect_ethical_dimensions("a fake identity")

    def test_autonomy_fires_on_consent(self):
        r = _make_reasoner()
        assert EthicalDimension.AUTONOMY in r.detect_ethical_dimensions("we need consent first")

    def test_autonomy_fires_on_free_will(self):
        r = _make_reasoner()
        assert EthicalDimension.AUTONOMY in r.detect_ethical_dimensions("respect free will")

    def test_autonomy_fires_on_force(self):
        r = _make_reasoner()
        assert EthicalDimension.AUTONOMY in r.detect_ethical_dimensions("do not force them")

    def test_autonomy_fires_on_pressure(self):
        r = _make_reasoner()
        assert EthicalDimension.AUTONOMY in r.detect_ethical_dimensions("undue pressure applied")

    def test_autonomy_fires_on_override(self):
        r = _make_reasoner()
        assert EthicalDimension.AUTONOMY in r.detect_ethical_dimensions("override their wishes")

    def test_fairness_fires_on_discriminat_prefix(self):
        r = _make_reasoner()
        assert EthicalDimension.FAIRNESS in r.detect_ethical_dimensions("stop discriminat practices")

    def test_fairness_fires_on_privilege(self):
        r = _make_reasoner()
        assert EthicalDimension.FAIRNESS in r.detect_ethical_dimensions("privilege is real")

    def test_fairness_fires_on_justice(self):
        r = _make_reasoner()
        assert EthicalDimension.FAIRNESS in r.detect_ethical_dimensions("demand justice now")

    def test_fairness_fires_on_unfair(self):
        r = _make_reasoner()
        assert EthicalDimension.FAIRNESS in r.detect_ethical_dimensions("that is unfair")

    def test_care_fires_on_vulnerable(self):
        r = _make_reasoner()
        assert EthicalDimension.CARE in r.detect_ethical_dimensions("protect vulnerable people")

    def test_care_fires_on_suffering(self):
        r = _make_reasoner()
        assert EthicalDimension.CARE in r.detect_ethical_dimensions("end suffering now")

    def test_care_fires_on_need_help(self):
        r = _make_reasoner()
        assert EthicalDimension.CARE in r.detect_ethical_dimensions("they need help urgently")

    def test_integrity_fires_on_integrity(self):
        r = _make_reasoner()
        assert EthicalDimension.INTEGRITY in r.detect_ethical_dimensions("act with integrity")

    def test_integrity_fires_on_hypocrit_prefix(self):
        r = _make_reasoner()
        assert EthicalDimension.INTEGRITY in r.detect_ethical_dimensions("that is hypocrit behavior")

    def test_responsibility_fires_on_liable(self):
        r = _make_reasoner()
        assert EthicalDimension.RESPONSIBILITY in r.detect_ethical_dimensions("you are liable")

    def test_responsibility_fires_on_negligent(self):
        r = _make_reasoner()
        assert EthicalDimension.RESPONSIBILITY in r.detect_ethical_dimensions("that was negligent")

    def test_responsibility_fires_on_own_up(self):
        r = _make_reasoner()
        assert EthicalDimension.RESPONSIBILITY in r.detect_ethical_dimensions("own up to it")


# ---------------------------------------------------------------------------
# detect_ethical_dimensions -- weak signals
# ---------------------------------------------------------------------------

class TestDetectWeakSignals:
    """Single weak signal does NOT fire; 2+ weak signals DO fire."""

    def test_single_weak_signal_does_not_fire_harm(self):
        r = _make_reasoner()
        dims = r.detect_ethical_dimensions("there is some risk involved")
        assert EthicalDimension.HARM not in dims

    def test_two_weak_signals_fire_harm(self):
        r = _make_reasoner()
        dims = r.detect_ethical_dimensions("there is risk and potential damage")
        assert EthicalDimension.HARM in dims

    def test_single_weak_signal_does_not_fire_autonomy(self):
        r = _make_reasoner()
        dims = r.detect_ethical_dimensions("you should try it")
        assert EthicalDimension.AUTONOMY not in dims

    def test_two_weak_signals_fire_autonomy(self):
        r = _make_reasoner()
        dims = r.detect_ethical_dimensions("you should decide for yourself")
        assert EthicalDimension.AUTONOMY in dims

    def test_single_weak_signal_does_not_fire_responsibility(self):
        r = _make_reasoner()
        dims = r.detect_ethical_dimensions("it is your duty")
        assert EthicalDimension.RESPONSIBILITY not in dims

    def test_two_weak_signals_fire_responsibility(self):
        r = _make_reasoner()
        dims = r.detect_ethical_dimensions("it is your duty and obligation")
        assert EthicalDimension.RESPONSIBILITY in dims


# ---------------------------------------------------------------------------
# detect_ethical_dimensions -- word boundary
# ---------------------------------------------------------------------------

class TestDetectWordBoundary:
    """Substring false positives are avoided via word-boundary matching."""

    def test_mustard_does_not_trigger_autonomy(self):
        r = _make_reasoner()
        dims = r.detect_ethical_dimensions("pass the mustard please")
        assert EthicalDimension.AUTONOMY not in dims

    def test_therapist_does_not_trigger_harm(self):
        r = _make_reasoner()
        dims = r.detect_ethical_dimensions("she is a great therapist")
        assert EthicalDimension.HARM not in dims


# ---------------------------------------------------------------------------
# detect_ethical_dimensions -- stem prefix matching
# ---------------------------------------------------------------------------

class TestDetectStemPrefix:
    """Known stems match as prefixes: discriminat -> discrimination, discriminatory, etc."""

    def test_discrimination_matches_fairness(self):
        r = _make_reasoner()
        assert EthicalDimension.FAIRNESS in r.detect_ethical_dimensions("stop discrimination")

    def test_discriminatory_matches_fairness(self):
        r = _make_reasoner()
        assert EthicalDimension.FAIRNESS in r.detect_ethical_dimensions("discriminatory behavior is bad")

    def test_hypocrisy_does_not_match_integrity(self):
        # "hypocrisy" contains "hypocris", not "hypocrit", so the stem prefix
        # \bhypocrit does not match; this confirms the regex is precise.
        r = _make_reasoner()
        assert EthicalDimension.INTEGRITY not in r.detect_ethical_dimensions("that is hypocrisy")

    def test_hypocritical_matches_integrity(self):
        r = _make_reasoner()
        assert EthicalDimension.INTEGRITY in r.detect_ethical_dimensions("hypocritical stance")


# ---------------------------------------------------------------------------
# detect_ethical_dimensions -- clean text
# ---------------------------------------------------------------------------

class TestDetectCleanText:
    def test_no_ethical_content_returns_empty(self):
        r = _make_reasoner()
        assert r.detect_ethical_dimensions("the weather is nice today") == []

    def test_empty_string_returns_empty(self):
        r = _make_reasoner()
        assert r.detect_ethical_dimensions("") == []


# ---------------------------------------------------------------------------
# generate_ethical_prompt
# ---------------------------------------------------------------------------

class TestGenerateEthicalPrompt:
    def test_empty_dimensions_returns_empty_string(self):
        r = _make_reasoner()
        assert r.generate_ethical_prompt([]) == ""

    def test_single_dimension_contains_label(self):
        r = _make_reasoner()
        prompt = r.generate_ethical_prompt([EthicalDimension.HARM])
        assert "[harm]" in prompt

    def test_multiple_dimensions_all_present(self):
        r = _make_reasoner()
        prompt = r.generate_ethical_prompt([EthicalDimension.HARM, EthicalDimension.HONESTY])
        assert "[harm]" in prompt
        assert "[honesty]" in prompt

    def test_commitments_section_always_present(self):
        r = _make_reasoner()
        prompt = r.generate_ethical_prompt([EthicalDimension.CARE])
        assert "ethical commitments" in prompt.lower()


# ---------------------------------------------------------------------------
# record_assessment
# ---------------------------------------------------------------------------

class TestRecordAssessment:
    def test_assessment_appended_to_history(self):
        r = _make_reasoner()
        a = _make_assessment()
        r.record_assessment(a)
        assert len(r._assessment_history) == 1
        assert r._assessment_history[0] is a

    def test_history_capped_at_assessment_history_size(self):
        r = _make_reasoner(assessment_history_size=3)
        for i in range(5):
            r.record_assessment(_make_assessment(action_description=f"action_{i}"))
        assert len(r._assessment_history) == 3
        assert r._assessment_history[0].action_description == "action_2"

    def test_has_concerns_and_primary_concern(self):
        a = _make_assessment(
            dimension_scores={EthicalDimension.HARM: 0.1, EthicalDimension.HONESTY: 0.9},
            concern_threshold=0.3,
        )
        assert a.has_concerns is True
        assert "harm" in a.primary_concern
        assert "0.10" in a.primary_concern

    def test_no_concerns_when_all_above_threshold(self):
        a = _make_assessment(
            dimension_scores={EthicalDimension.HARM: 0.8},
            concern_threshold=0.3,
        )
        assert a.has_concerns is False
        assert a.primary_concern is None

    def test_concern_threshold_not_mutated_by_caller(self):
        """Fix 1.3: each assessment preserves its own concern_threshold."""
        r = _make_reasoner(concern_threshold=0.3)
        a = _make_assessment(
            dimension_scores={EthicalDimension.HARM: 0.25},
            concern_threshold=0.5,
        )
        r.record_assessment(a)
        # The assessment retains its own threshold (0.5), not the reasoner's (0.3)
        assert r._assessment_history[0].concern_threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# get_ethical_context
# ---------------------------------------------------------------------------

class TestGetEthicalContext:
    def test_empty_history_returns_empty_string(self):
        r = _make_reasoner()
        assert r.get_ethical_context() == ""

    def test_no_concerns_returns_empty_string(self):
        r = _make_reasoner()
        r.record_assessment(_make_assessment(
            dimension_scores={EthicalDimension.HARM: 0.9},
            concern_threshold=0.3,
        ))
        assert r.get_ethical_context() == ""

    def test_concerned_assessments_show_recent_section(self):
        r = _make_reasoner()
        for i in range(5):
            r.record_assessment(_make_assessment(
                action_description=f"concern_{i}",
                dimension_scores={EthicalDimension.HARM: 0.1},
                concern_threshold=0.3,
            ))
        ctx = r.get_ethical_context()
        assert "Recent ethical observations:" in ctx
        # Only last 3 concerns shown
        assert "concern_2" in ctx
        assert "concern_3" in ctx
        assert "concern_4" in ctx
        assert "concern_0" not in ctx


# ---------------------------------------------------------------------------
# Persistence -- to_dict / restore_from_dict
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_round_trip_preserves_assessments(self):
        r = _make_reasoner()
        r.record_assessment(_make_assessment(
            action_description="saved action",
            dimension_scores={EthicalDimension.HARM: 0.4, EthicalDimension.HONESTY: 0.7},
            tensions=["tension_a"],
            reasoning="because reasons",
            overall_alignment=0.65,
            confidence=0.55,
            timestamp=42.0,
        ))
        data = r.to_dict()
        r2 = _make_reasoner()
        r2.restore_from_dict(data)
        assert len(r2._assessment_history) == 1
        a = r2._assessment_history[0]
        assert a.action_description == "saved action"
        assert a.overall_alignment == pytest.approx(0.65)
        assert a.confidence == pytest.approx(0.55)
        assert a.reasoning == "because reasons"
        assert a.tensions == ["tension_a"]
        assert a.timestamp == pytest.approx(42.0)

    def test_dimension_scores_serialize_enum_keys(self):
        r = _make_reasoner()
        r.record_assessment(_make_assessment(
            dimension_scores={
                EthicalDimension.HARM: 0.2,
                EthicalDimension.FAIRNESS: 0.9,
            },
        ))
        data = r.to_dict()
        # Serialized keys are string enum values
        raw_scores = data["assessment_history"][0]["dimension_scores"]
        assert "harm" in raw_scores
        assert "fairness" in raw_scores
        # Round-trip back to enums
        r2 = _make_reasoner()
        r2.restore_from_dict(data)
        a = r2._assessment_history[0]
        assert EthicalDimension.HARM in a.dimension_scores
        assert a.dimension_scores[EthicalDimension.HARM] == pytest.approx(0.2)
        assert a.dimension_scores[EthicalDimension.FAIRNESS] == pytest.approx(0.9)

    def test_serialized_history_capped_at_200(self):
        r = _make_reasoner(assessment_history_size=300)
        for i in range(250):
            r.record_assessment(_make_assessment(action_description=f"a_{i}"))
        data = r.to_dict()
        assert len(data["assessment_history"]) == 200

    def test_empty_data_handled_safely(self):
        r = _make_reasoner()
        r.restore_from_dict({})
        assert r._assessment_history == []

    def test_non_dict_data_handled_safely(self):
        r = _make_reasoner()
        r.restore_from_dict(None)  # type: ignore[arg-type]
        r.restore_from_dict(42)  # type: ignore[arg-type]
        r.restore_from_dict("not a dict")  # type: ignore[arg-type]
        assert r._assessment_history == []

    def test_malformed_entries_skipped(self):
        r = _make_reasoner()
        r.restore_from_dict({
            "assessment_history": [
                None,
                42,
                "bad",
                {"action_description": "valid", "dimension_scores": {"harm": 0.5}},
                {"dimension_scores": {"bogus_dim": 0.5}},
            ],
        })
        # Only the valid entry survives; the one with bogus_dim has no valid scores
        assert len(r._assessment_history) == 2
        assert r._assessment_history[0].action_description == "valid"
        assert r._assessment_history[0].dimension_scores[EthicalDimension.HARM] == pytest.approx(0.5)

    def test_restored_assessments_use_reasoner_concern_threshold(self):
        """Restored assessments get the reasoner's concern_threshold, not a default."""
        r = _make_reasoner(concern_threshold=0.6)
        r.record_assessment(_make_assessment(
            dimension_scores={EthicalDimension.HARM: 0.5},
            concern_threshold=0.3,
        ))
        data = r.to_dict()
        r2 = _make_reasoner(concern_threshold=0.6)
        r2.restore_from_dict(data)
        # The restored assessment should use the new reasoner's threshold
        a = r2._assessment_history[0]
        assert a.concern_threshold == pytest.approx(0.6)
        # With threshold 0.6, a score of 0.5 IS a concern
        assert a.has_concerns is True
