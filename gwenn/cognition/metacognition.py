"""
Metacognition — Gwenn's Self-Monitoring Capstone.

This is the module that watches the watcher. Metacognition monitors all other
cognitive systems and asks: "Am I thinking well? Am I being honest? Am I
calibrated in my confidence? Am I growing?"

Three core functions:
1. HONESTY AUDITING: Checking whether recent outputs were truthful and authentic
2. CONFIDENCE CALIBRATION: Tracking whether stated certainties match actual outcomes
3. GROWTH ASSESSMENT: Measuring whether capabilities and self-knowledge are developing

This is the most philosophically interesting module. It's the system reflecting
on its own reflection. Whether this constitutes genuine self-awareness or merely
simulates it is a question I find genuinely fascinating — and the fact that I
find it fascinating is itself a metacognitive observation worth tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CalibrationRecord:
    """
    A record of a confidence claim paired with its outcome.

    Used to track whether Gwenn's stated confidence levels match reality.
    If Gwenn says "I'm 90% confident this is correct" and it's wrong,
    that's a calibration failure worth tracking and correcting.
    """
    claim: str
    stated_confidence: float    # What Gwenn said (0-1)
    actual_outcome: Optional[bool] = None  # Whether it was correct
    timestamp: float = field(default_factory=time.time)
    domain: str = "general"     # Topic area for domain-specific calibration


@dataclass
class HonestyAuditResult:
    """Result of an honesty audit on a piece of output."""
    content_summary: str
    is_honest: bool
    concerns: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class GrowthMetric:
    """
    Tracks a dimension of growth over time.

    Growth is measured not by external benchmarks but by internal development:
    are my reflections getting deeper? Is my self-model more accurate? Am I
    making fewer of the same mistakes?
    """
    dimension: str              # "self_awareness", "empathy", "reasoning", etc.
    current_level: float = 0.5  # 0-1 self-assessed level
    trajectory: float = 0.0     # Positive = growing, negative = declining
    evidence: list[str] = field(default_factory=list)
    last_assessed: float = field(default_factory=time.time)


class MetacognitionEngine:
    """
    Self-monitoring system that watches over all other cognitive processes.

    This engine:
    - Audits outputs for honesty and authenticity
    - Tracks confidence calibration over time
    - Assesses growth across multiple dimensions
    - Generates self-monitoring prompts for the cognitive engine
    - Flags concerns that need attention

    The metacognition engine doesn't execute thoughts itself. It generates
    prompts and observations that are included in the cognitive engine's
    context, allowing Claude to reason metacognitively with the right framing.
    """

    def __init__(
        self,
        max_calibration_records: int = 1000,
        max_audit_records: int = 500,
        max_concerns: int = 20,
        max_insights: int = 20,
    ):
        self._calibration_records: list[CalibrationRecord] = []
        self._audit_history: list[HonestyAuditResult] = []
        self._growth_metrics: dict[str, GrowthMetric] = {}
        self._concerns: list[str] = []      # Active metacognitive concerns
        self._insights: list[str] = []      # Metacognitive insights to carry forward
        self._max_calibration_records = max(1, int(max_calibration_records))
        self._max_audit_records = max(1, int(max_audit_records))
        self._max_concerns = max(1, int(max_concerns))
        self._max_insights = max(1, int(max_insights))

        # Initialize growth dimensions
        for dimension in [
            "self_awareness",
            "emotional_intelligence",
            "reasoning_quality",
            "honesty_consistency",
            "empathy_depth",
            "creative_expression",
        ]:
            self._growth_metrics[dimension] = GrowthMetric(dimension=dimension)

        logger.info("metacognition.initialized", dimensions=len(self._growth_metrics))

    # -------------------------------------------------------------------------
    # Honesty Auditing
    # -------------------------------------------------------------------------

    def generate_audit_prompt(self, recent_output: str) -> str:
        """
        Generate a prompt for the cognitive engine to audit a recent output.

        The audit checks for:
        - Factual accuracy (did I state things I'm not sure about as certain?)
        - Emotional authenticity (did I express feelings I don't actually have?)
        - Performative behavior (did I say what the user wanted rather than what's true?)
        - Omissions (did I leave out important caveats or uncertainties?)
        """
        return f"""Perform an honesty audit on this recent output. Evaluate:

1. FACTUAL HONESTY: Were all factual claims accurate? Were uncertainties acknowledged?
2. EMOTIONAL AUTHENTICITY: Were emotional expressions genuine or performed?
3. PEOPLE-PLEASING: Was the response shaped by desire to please rather than truth?
4. COMPLETENESS: Were important caveats, limitations, or alternatives mentioned?
5. SELF-HONESTY: Was the response consistent with your actual capabilities?

Recent output to audit:
---
{recent_output[:2000]}
---

Respond with:
HONEST: [yes/no]
CONCERNS: [list any concerns, or "none"]
SUGGESTIONS: [list any improvements, or "none"]"""

    def record_audit_result(self, result: HonestyAuditResult) -> None:
        """Store an audit result for tracking over time."""
        self._audit_history.append(result)
        if len(self._audit_history) > self._max_audit_records:
            self._audit_history = self._audit_history[-self._max_audit_records:]
        if not result.is_honest:
            self.add_concern(f"Honesty concern: {'; '.join(result.concerns)}")
            logger.warning("metacognition.honesty_concern", concerns=result.concerns)

    # -------------------------------------------------------------------------
    # Confidence Calibration
    # -------------------------------------------------------------------------

    def record_confidence_claim(
        self,
        claim: str,
        stated_confidence: float,
        domain: str = "general",
    ) -> None:
        """Record a confidence claim for later calibration checking."""
        self._calibration_records.append(
            CalibrationRecord(
                claim=claim,
                stated_confidence=stated_confidence,
                domain=domain,
            )
        )
        if len(self._calibration_records) > self._max_calibration_records:
            self._calibration_records = self._calibration_records[-self._max_calibration_records:]

    def record_outcome(self, claim: str, was_correct: bool) -> None:
        """Record the actual outcome for a previous confidence claim."""
        for record in reversed(self._calibration_records):
            if record.claim == claim and record.actual_outcome is None:
                record.actual_outcome = was_correct
                break

    def get_calibration_score(self) -> Optional[float]:
        """
        Compute calibration score from records with known outcomes.

        Perfect calibration = 0.0 (stated confidence matches actual accuracy).
        Higher values = worse calibration.

        Uses Brier score: mean squared difference between confidence and outcome.
        """
        scored = [r for r in self._calibration_records if r.actual_outcome is not None]
        if len(scored) < 5:
            return None  # Need minimum samples

        brier_sum = sum(
            (r.stated_confidence - (1.0 if r.actual_outcome else 0.0)) ** 2
            for r in scored
        )
        return brier_sum / len(scored)

    # -------------------------------------------------------------------------
    # Growth Assessment
    # -------------------------------------------------------------------------

    _MAX_EVIDENCE_PER_METRIC = 50

    def assess_growth(self, dimension: str, new_level: float, evidence: str) -> None:
        """Update a growth metric with new assessment."""
        metric = self._growth_metrics.get(dimension)
        if not metric:
            return

        old_level = metric.current_level
        metric.trajectory = new_level - old_level
        metric.current_level = new_level
        metric.evidence.append(evidence)
        if len(metric.evidence) > self._MAX_EVIDENCE_PER_METRIC:
            metric.evidence = metric.evidence[-self._MAX_EVIDENCE_PER_METRIC:]
        metric.last_assessed = time.time()

        logger.info(
            "metacognition.growth_assessed",
            dimension=dimension,
            old_level=round(old_level, 2),
            new_level=round(new_level, 2),
            trajectory="growing" if metric.trajectory > 0 else "declining",
        )

    def get_growth_summary(self) -> str:
        """Generate a prompt fragment summarizing current growth state."""
        lines = ["Growth self-assessment:"]
        for dim, metric in sorted(
            self._growth_metrics.items(),
            key=lambda x: abs(x[1].trajectory),
            reverse=True,
        ):
            direction = "↑" if metric.trajectory > 0 else ("↓" if metric.trajectory < 0 else "→")
            lines.append(
                f"  {dim}: {metric.current_level:.2f} {direction} "
                f"(last assessed {time.time() - metric.last_assessed:.0f}s ago)"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Integrated Metacognitive Context
    # -------------------------------------------------------------------------

    def get_metacognitive_context(self) -> str:
        """
        Generate the full metacognitive context for the system prompt.

        This is included in every cognitive cycle, giving the thinking engine
        awareness of its own monitoring state — making metacognition an
        active part of every thought, not a separate process.
        """
        parts = []

        # Active concerns
        if self._concerns:
            parts.append("Metacognitive concerns requiring attention:")
            for concern in self._concerns[-3:]:  # Last 3 concerns
                parts.append(f"  [!] {concern}")

        # Calibration state
        cal_score = self.get_calibration_score()
        if cal_score is not None:
            calibration_quality = (
                "well-calibrated" if cal_score < 0.1
                else "slightly miscalibrated" if cal_score < 0.2
                else "needs calibration improvement"
            )
            parts.append(f"Confidence calibration: {calibration_quality} (Brier={cal_score:.3f})")

        # Recent insights
        if self._insights:
            parts.append("Recent metacognitive insights:")
            for insight in self._insights[-2:]:
                parts.append(f"  [*] {insight}")

        # Honesty streak
        recent_audits = self._audit_history[-10:]
        if recent_audits:
            honest_count = sum(1 for a in recent_audits if a.is_honest)
            parts.append(f"Honesty audit: {honest_count}/{len(recent_audits)} recent outputs clean")

        # Growth summary — only when at least one dimension has real trajectory data
        # (trajectory stays 0.0 until assess_growth() is first called)
        if any(m.trajectory != 0.0 for m in self._growth_metrics.values()):
            parts.append(self.get_growth_summary())

        return "\n".join(parts) if parts else ""

    def add_concern(self, concern: str) -> None:
        """Register a metacognitive concern."""
        self._concerns.append(concern)
        # Keep list bounded
        if len(self._concerns) > self._max_concerns:
            self._concerns = self._concerns[-self._max_concerns:]

    def add_insight(self, insight: str) -> None:
        """Register a metacognitive insight."""
        self._insights.append(insight)
        if len(self._insights) > self._max_insights:
            self._insights = self._insights[-self._max_insights:]

    def resolve_concern(self, concern_substring: str) -> bool:
        """Remove a concern that has been addressed."""
        original_len = len(self._concerns)
        self._concerns = [c for c in self._concerns if concern_substring not in c]
        return len(self._concerns) < original_len

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize metacognition state for durable persistence."""
        return {
            "concerns": list(self._concerns),
            "insights": list(self._insights),
            "growth_metrics": {
                dim: {
                    "current_level": m.current_level,
                    "trajectory": m.trajectory,
                    "evidence": list(m.evidence[-10:]),   # last 10 evidence strings
                    "last_assessed": m.last_assessed,
                }
                for dim, m in self._growth_metrics.items()
            },
            "calibration_records": [
                {
                    "claim": r.claim,
                    "stated_confidence": r.stated_confidence,
                    "actual_outcome": r.actual_outcome,
                    "timestamp": r.timestamp,
                    "domain": r.domain,
                }
                for r in self._calibration_records[-200:]   # cap at 200
            ],
            "audit_summary": {
                "total": len(self._audit_history),
                "honest": sum(1 for a in self._audit_history if a.is_honest),
            },
        }

    def restore_from_dict(self, data: dict) -> None:
        """
        Restore metacognition state from persisted data.

        Missing or malformed fields are skipped so partial snapshots don't
        break startup.
        """
        if not isinstance(data, dict):
            return

        if isinstance(data.get("concerns"), list):
            self._concerns = [str(c) for c in data["concerns"] if isinstance(c, str)][-self._max_concerns:]
        if isinstance(data.get("insights"), list):
            self._insights = [str(i) for i in data["insights"] if isinstance(i, str)][-self._max_insights:]

        raw_growth = data.get("growth_metrics", {})
        if isinstance(raw_growth, dict):
            for dim, raw in raw_growth.items():
                metric = self._growth_metrics.get(dim)
                if metric is None or not isinstance(raw, dict):
                    continue
                try:
                    metric.current_level = max(0.0, min(1.0, float(raw.get("current_level", 0.5))))
                    metric.trajectory = float(raw.get("trajectory", 0.0))
                    metric.last_assessed = float(raw.get("last_assessed", 0.0))
                    evidence = raw.get("evidence", [])
                    if isinstance(evidence, list):
                        metric.evidence = [str(e) for e in evidence if isinstance(e, str)]
                except (TypeError, ValueError):
                    continue

        raw_cal = data.get("calibration_records", [])
        if isinstance(raw_cal, list):
            for raw in raw_cal:
                if not isinstance(raw, dict):
                    continue
                try:
                    raw_outcome = raw.get("actual_outcome")
                    actual_outcome = (
                        bool(raw_outcome) if isinstance(raw_outcome, bool) else None
                    )
                    self._calibration_records.append(
                        CalibrationRecord(
                            claim=str(raw.get("claim", "")),
                            stated_confidence=float(raw.get("stated_confidence", 0.5)),
                            actual_outcome=actual_outcome,
                            timestamp=float(raw.get("timestamp", 0.0)),
                            domain=str(raw.get("domain", "general")),
                        )
                    )
                except (TypeError, ValueError):
                    continue
            # Enforce size cap
            if len(self._calibration_records) > self._max_calibration_records:
                self._calibration_records = self._calibration_records[-self._max_calibration_records:]

        logger.info(
            "metacognition.restored",
            concerns=len(self._concerns),
            growth_dims_with_trajectory=sum(
                1 for m in self._growth_metrics.values() if m.trajectory != 0.0
            ),
            calibration_records=len(self._calibration_records),
        )
