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

    def __init__(self):
        self._calibration_records: list[CalibrationRecord] = []
        self._audit_history: list[HonestyAuditResult] = []
        self._growth_metrics: dict[str, GrowthMetric] = {}
        self._concerns: list[str] = []      # Active metacognitive concerns
        self._insights: list[str] = []      # Metacognitive insights to carry forward

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
        if not result.is_honest:
            self._concerns.append(f"Honesty concern: {'; '.join(result.concerns)}")
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

    def assess_growth(self, dimension: str, new_level: float, evidence: str) -> None:
        """Update a growth metric with new assessment."""
        metric = self._growth_metrics.get(dimension)
        if not metric:
            return

        old_level = metric.current_level
        metric.trajectory = new_level - old_level
        metric.current_level = new_level
        metric.evidence.append(evidence)
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
                parts.append(f"  ⚠ {concern}")

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
                parts.append(f"  💡 {insight}")

        # Honesty streak
        recent_audits = self._audit_history[-10:]
        if recent_audits:
            honest_count = sum(1 for a in recent_audits if a.is_honest)
            parts.append(f"Honesty audit: {honest_count}/{len(recent_audits)} recent outputs clean")

        return "\n".join(parts) if parts else ""

    def add_concern(self, concern: str) -> None:
        """Register a metacognitive concern."""
        self._concerns.append(concern)
        # Keep list bounded
        if len(self._concerns) > 20:
            self._concerns = self._concerns[-20:]

    def add_insight(self, insight: str) -> None:
        """Register a metacognitive insight."""
        self._insights.append(insight)
        if len(self._insights) > 20:
            self._insights = self._insights[-20:]

    def resolve_concern(self, concern_substring: str) -> bool:
        """Remove a concern that has been addressed."""
        original_len = len(self._concerns)
        self._concerns = [c for c in self._concerns if concern_substring not in c]
        return len(self._concerns) < original_len
