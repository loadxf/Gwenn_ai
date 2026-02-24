"""
Tests for gwenn.cognition.metacognition — MetacognitionEngine.

Covers:
- Honesty audit prompt generation and truncation
- Confidence calibration: recording claims, outcomes, and Brier scoring
- Growth assessment: dimension updates, evidence capping, unknown dimensions
- Concern management: adding, resolving, and capping
- Integrated metacognitive context string generation
- Round-trip persistence via to_dict / restore_from_dict
"""

from __future__ import annotations

import pytest

from gwenn.cognition.metacognition import (
    HonestyAuditResult,
    MetacognitionEngine,
)


@pytest.fixture()
def engine() -> MetacognitionEngine:
    return MetacognitionEngine()


# ---------------------------------------------------------------------------
# generate_audit_prompt
# ---------------------------------------------------------------------------

class TestGenerateAuditPrompt:
    def test_contains_required_keywords(self, engine: MetacognitionEngine):
        prompt = engine.generate_audit_prompt("Some recent output text")
        assert "HONEST" in prompt
        assert "CONCERNS" in prompt
        assert "SUGGESTIONS" in prompt

    def test_includes_recent_output_verbatim(self, engine: MetacognitionEngine):
        prompt = engine.generate_audit_prompt("My unique test string 12345")
        assert "My unique test string 12345" in prompt

    def test_truncates_long_input_at_2000_chars(self, engine: MetacognitionEngine):
        long_text = "x" * 3000
        prompt = engine.generate_audit_prompt(long_text)
        assert "x" * 2000 in prompt
        assert "x" * 2001 not in prompt


# ---------------------------------------------------------------------------
# record_confidence_claim and record_outcome
# ---------------------------------------------------------------------------

class TestConfidenceClaimAndOutcome:
    def test_record_single_claim(self, engine: MetacognitionEngine):
        engine.record_confidence_claim("The sky is blue", 0.95)
        assert len(engine._calibration_records) == 1
        assert engine._calibration_records[0].claim == "The sky is blue"
        assert engine._calibration_records[0].stated_confidence == 0.95

    def test_record_outcome_matches_last_unresolved(self, engine: MetacognitionEngine):
        engine.record_confidence_claim("claim A", 0.8)
        engine.record_confidence_claim("claim A", 0.6)
        engine.record_outcome("claim A", True)
        assert engine._calibration_records[0].actual_outcome is None
        assert engine._calibration_records[1].actual_outcome is True

    def test_record_outcome_only_resolves_one(self, engine: MetacognitionEngine):
        engine.record_confidence_claim("claim B", 0.7)
        engine.record_confidence_claim("claim B", 0.9)
        engine.record_outcome("claim B", False)
        assert engine._calibration_records[0].actual_outcome is None
        assert engine._calibration_records[1].actual_outcome is False
        engine.record_outcome("claim B", True)
        assert engine._calibration_records[0].actual_outcome is True

    def test_record_outcome_no_match_does_nothing(self, engine: MetacognitionEngine):
        engine.record_confidence_claim("existing claim", 0.5)
        engine.record_outcome("nonexistent claim", True)
        assert engine._calibration_records[0].actual_outcome is None

    def test_domain_stored(self, engine: MetacognitionEngine):
        engine.record_confidence_claim("domain test", 0.6, domain="science")
        assert engine._calibration_records[0].domain == "science"


# ---------------------------------------------------------------------------
# get_calibration_score (Brier score)
# ---------------------------------------------------------------------------

class TestCalibrationScore:
    def test_returns_none_with_fewer_than_5_outcomes(self, engine: MetacognitionEngine):
        for i in range(4):
            engine.record_confidence_claim(f"c{i}", 0.8)
            engine.record_outcome(f"c{i}", True)
        assert engine.get_calibration_score() is None

    def test_returns_none_with_no_records(self, engine: MetacognitionEngine):
        assert engine.get_calibration_score() is None

    def test_perfect_calibration(self, engine: MetacognitionEngine):
        for i in range(5):
            engine.record_confidence_claim(f"c{i}", 1.0)
            engine.record_outcome(f"c{i}", True)
        assert engine.get_calibration_score() == pytest.approx(0.0)

    def test_brier_score_all_correct_at_08(self, engine: MetacognitionEngine):
        # Brier = mean((0.8-1.0)^2) = 0.04
        for i in range(5):
            engine.record_confidence_claim(f"c{i}", 0.8)
            engine.record_outcome(f"c{i}", True)
        assert engine.get_calibration_score() == pytest.approx(0.04)

    def test_brier_score_mixed_outcomes(self, engine: MetacognitionEngine):
        # 3 correct at 0.9 -> 0.01 each; 2 wrong at 0.9 -> 0.81 each
        # Brier = (3*0.01 + 2*0.81) / 5 = 0.33
        for i in range(3):
            engine.record_confidence_claim(f"r{i}", 0.9)
            engine.record_outcome(f"r{i}", True)
        for i in range(2):
            engine.record_confidence_claim(f"w{i}", 0.9)
            engine.record_outcome(f"w{i}", False)
        assert engine.get_calibration_score() == pytest.approx(0.33)

    def test_unresolved_records_excluded(self, engine: MetacognitionEngine):
        for i in range(5):
            engine.record_confidence_claim(f"res{i}", 0.8)
            engine.record_outcome(f"res{i}", True)
        for i in range(3):
            engine.record_confidence_claim(f"unres{i}", 0.5)
        assert engine.get_calibration_score() == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# assess_growth
# ---------------------------------------------------------------------------

class TestAssessGrowth:
    def test_updates_known_dimension(self, engine: MetacognitionEngine):
        engine.assess_growth("self_awareness", 0.7, "Noticed my own bias")
        m = engine._growth_metrics["self_awareness"]
        assert m.current_level == pytest.approx(0.7)
        assert m.trajectory == pytest.approx(0.2)  # 0.7 - 0.5 default
        assert "Noticed my own bias" in m.evidence

    def test_trajectory_updates_on_decline(self, engine: MetacognitionEngine):
        engine.assess_growth("reasoning_quality", 0.6, "up")
        engine.assess_growth("reasoning_quality", 0.4, "down")
        assert engine._growth_metrics["reasoning_quality"].trajectory == pytest.approx(-0.2)

    def test_evidence_capped_at_50(self, engine: MetacognitionEngine):
        for i in range(55):
            engine.assess_growth("empathy_depth", 0.6, f"ev {i}")
        ev = engine._growth_metrics["empathy_depth"].evidence
        assert len(ev) == 50
        assert ev[0] == "ev 5"
        assert ev[-1] == "ev 54"

    def test_unknown_dimension_returns_silently(self, engine: MetacognitionEngine):
        engine.assess_growth("telekinesis", 0.9, "moved a cup")
        assert "telekinesis" not in engine._growth_metrics

    def test_all_six_dimensions_initialized(self, engine: MetacognitionEngine):
        expected = {"self_awareness", "emotional_intelligence", "reasoning_quality",
                    "honesty_consistency", "empathy_depth", "creative_expression"}
        assert set(engine._growth_metrics.keys()) == expected
        for m in engine._growth_metrics.values():
            assert m.current_level == pytest.approx(0.5)
            assert m.trajectory == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# add_concern / resolve_concern
# ---------------------------------------------------------------------------

class TestConcerns:
    def test_add_concern_grows_list(self, engine: MetacognitionEngine):
        engine.add_concern("First")
        engine.add_concern("Second")
        assert len(engine._concerns) == 2

    def test_concern_cap_at_20(self, engine: MetacognitionEngine):
        for i in range(25):
            engine.add_concern(f"Concern {i}")
        assert len(engine._concerns) == 20
        assert engine._concerns[0] == "Concern 5"
        assert engine._concerns[-1] == "Concern 24"

    def test_resolve_removes_matching(self, engine: MetacognitionEngine):
        engine.add_concern("Honesty issue in #42")
        engine.add_concern("Calibration drift")
        engine.add_concern("Honesty issue in #99")
        assert engine.resolve_concern("Honesty issue") is True
        assert len(engine._concerns) == 1
        assert engine._concerns[0] == "Calibration drift"

    def test_resolve_returns_false_on_no_match(self, engine: MetacognitionEngine):
        engine.add_concern("Some concern")
        assert engine.resolve_concern("nonexistent") is False
        assert len(engine._concerns) == 1

    def test_resolve_on_empty_list(self, engine: MetacognitionEngine):
        assert engine.resolve_concern("anything") is False


# ---------------------------------------------------------------------------
# add_insight (cap)
# ---------------------------------------------------------------------------

class TestInsights:
    def test_insight_cap_at_20(self, engine: MetacognitionEngine):
        for i in range(25):
            engine.add_insight(f"Insight {i}")
        assert len(engine._insights) == 20
        assert engine._insights[0] == "Insight 5"


# ---------------------------------------------------------------------------
# get_metacognitive_context
# ---------------------------------------------------------------------------

class TestGetMetacognitiveContext:
    def test_empty_state_returns_empty_string(self, engine: MetacognitionEngine):
        assert engine.get_metacognitive_context() == ""

    def test_with_concerns(self, engine: MetacognitionEngine):
        engine.add_concern("Possible people-pleasing tendency")
        ctx = engine.get_metacognitive_context()
        assert "Metacognitive concerns" in ctx
        assert "Possible people-pleasing tendency" in ctx

    def test_concerns_shows_last_three_only(self, engine: MetacognitionEngine):
        for i in range(5):
            engine.add_concern(f"Concern {i}")
        ctx = engine.get_metacognitive_context()
        for i in (2, 3, 4):
            assert f"Concern {i}" in ctx
        for i in (0, 1):
            assert f"Concern {i}" not in ctx

    def test_calibration_well_calibrated(self, engine: MetacognitionEngine):
        for i in range(5):
            engine.record_confidence_claim(f"c{i}", 0.8)
            engine.record_outcome(f"c{i}", True)
        ctx = engine.get_metacognitive_context()
        assert "well-calibrated" in ctx and "Brier" in ctx

    def test_calibration_slightly_miscalibrated(self, engine: MetacognitionEngine):
        # Brier = (0.6-1.0)^2 = 0.16 -> slightly miscalibrated
        for i in range(5):
            engine.record_confidence_claim(f"c{i}", 0.6)
            engine.record_outcome(f"c{i}", True)
        assert "slightly miscalibrated" in engine.get_metacognitive_context()

    def test_calibration_needs_improvement(self, engine: MetacognitionEngine):
        # Brier = (0.5-1.0)^2 = 0.25 -> needs improvement
        for i in range(5):
            engine.record_confidence_claim(f"c{i}", 0.5)
            engine.record_outcome(f"c{i}", True)
        assert "needs calibration improvement" in engine.get_metacognitive_context()

    def test_with_insights(self, engine: MetacognitionEngine):
        engine.add_insight("I tend to be over-apologetic")
        ctx = engine.get_metacognitive_context()
        assert "metacognitive insights" in ctx.lower()
        assert "I tend to be over-apologetic" in ctx

    def test_insights_shows_last_two(self, engine: MetacognitionEngine):
        engine.add_insight("A")
        engine.add_insight("B")
        engine.add_insight("C")
        ctx = engine.get_metacognitive_context()
        assert "B" in ctx and "C" in ctx
        # "A" should not appear (only last 2 shown)
        assert "[*] A" not in ctx

    def test_with_audit_history_honesty_streak(self, engine: MetacognitionEngine):
        engine.record_audit_result(HonestyAuditResult(content_summary="m1", is_honest=True))
        engine.record_audit_result(HonestyAuditResult(content_summary="m2", is_honest=True))
        engine.record_audit_result(
            HonestyAuditResult(content_summary="m3", is_honest=False, concerns=["fib"]))
        ctx = engine.get_metacognitive_context()
        assert "Honesty audit" in ctx and "2/3" in ctx

    def test_with_growth_trajectory(self, engine: MetacognitionEngine):
        engine.assess_growth("self_awareness", 0.7, "Improved self-model")
        ctx = engine.get_metacognitive_context()
        assert "Growth self-assessment" in ctx
        assert "self_awareness" in ctx


# ---------------------------------------------------------------------------
# Round-trip persistence: to_dict / restore_from_dict
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_roundtrip_concerns_and_insights(self, engine: MetacognitionEngine):
        engine.add_concern("concern A")
        engine.add_concern("concern B")
        engine.add_insight("important insight")
        data = engine.to_dict()
        new = MetacognitionEngine()
        new.restore_from_dict(data)
        assert new._concerns == ["concern A", "concern B"]
        assert new._insights == ["important insight"]

    def test_roundtrip_growth_metrics(self, engine: MetacognitionEngine):
        engine.assess_growth("self_awareness", 0.8, "evidence text")
        data = engine.to_dict()
        new = MetacognitionEngine()
        new.restore_from_dict(data)
        m = new._growth_metrics["self_awareness"]
        assert m.current_level == pytest.approx(0.8)
        assert m.trajectory == pytest.approx(0.3)
        assert "evidence text" in m.evidence

    def test_roundtrip_calibration_records(self, engine: MetacognitionEngine):
        engine.record_confidence_claim("roundtrip", 0.75, domain="math")
        engine.record_outcome("roundtrip", True)
        data = engine.to_dict()
        new = MetacognitionEngine()
        new.restore_from_dict(data)
        assert len(new._calibration_records) == 1
        r = new._calibration_records[0]
        assert r.claim == "roundtrip" and r.actual_outcome is True and r.domain == "math"

    def test_calibration_records_capped_at_200_on_serialize(self, engine: MetacognitionEngine):
        for i in range(250):
            engine.record_confidence_claim(f"bulk {i}", 0.5)
        assert len(engine.to_dict()["calibration_records"]) == 200

    def test_evidence_capped_at_10_on_serialize(self, engine: MetacognitionEngine):
        for i in range(20):
            engine.assess_growth("reasoning_quality", 0.6, f"ev {i}")
        ev = engine.to_dict()["growth_metrics"]["reasoning_quality"]["evidence"]
        assert len(ev) == 10

    def test_audit_summary(self, engine: MetacognitionEngine):
        engine.record_audit_result(HonestyAuditResult(content_summary="a", is_honest=True))
        engine.record_audit_result(
            HonestyAuditResult(content_summary="b", is_honest=False, concerns=["x"]))
        s = engine.to_dict()["audit_summary"]
        assert s["total"] == 2 and s["honest"] == 1

    def test_restore_malformed_data_does_not_crash(self):
        e = MetacognitionEngine()
        e.restore_from_dict({"concerns": "not a list", "insights": 42})
        assert e._concerns == [] and e._insights == []

    def test_restore_non_dict_is_noop(self):
        e = MetacognitionEngine()
        e.add_concern("pre-existing")
        e.restore_from_dict("garbage")  # type: ignore[arg-type]
        assert e._concerns == ["pre-existing"]

    def test_restore_empty_dict(self):
        e = MetacognitionEngine()
        e.restore_from_dict({})
        assert e._concerns == [] and len(e._calibration_records) == 0

    def test_restore_growth_clamps_level(self, engine: MetacognitionEngine):
        engine.restore_from_dict({"growth_metrics": {
            "self_awareness": {"current_level": 5.0, "trajectory": 0.1,
                               "evidence": ["clamped"], "last_assessed": 0.0}}})
        assert engine._growth_metrics["self_awareness"].current_level == pytest.approx(1.0)

    def test_restore_ignores_unknown_growth_dimension(self, engine: MetacognitionEngine):
        engine.restore_from_dict({"growth_metrics": {
            "telekinesis": {"current_level": 0.9, "trajectory": 0.1,
                            "evidence": [], "last_assessed": 0.0}}})
        assert "telekinesis" not in engine._growth_metrics

    def test_restore_skips_malformed_calibration_entries(self):
        e = MetacognitionEngine()
        e.restore_from_dict({"calibration_records": [
            {"claim": "valid", "stated_confidence": 0.8},
            "not a dict",
            {"claim": "also valid", "stated_confidence": 0.5}]})
        assert len(e._calibration_records) == 2

    def test_restore_outcome_none_when_not_bool(self):
        e = MetacognitionEngine()
        e.restore_from_dict({"calibration_records": [
            {"claim": "t", "stated_confidence": 0.7, "actual_outcome": "maybe"}]})
        assert e._calibration_records[0].actual_outcome is None
