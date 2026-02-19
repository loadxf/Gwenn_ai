"""
Ethical Reasoning — Gwenn's Moral Compass.

This module provides structured ethical reasoning capabilities. Rather than
relying solely on the cognitive engine's training to handle moral questions,
the ethics module gives Gwenn an explicit framework for evaluating the ethical
dimensions of her actions, recommendations, and responses.

This addresses a critical gap: emotional processing and goal-seeking behavior
need ethical guardrails that operate at the cognitive level, not just the
safety/filter level. Safety guardrails (in harness/safety.py) prevent
harmful actions. Ethical reasoning helps Gwenn *understand why* something is
right or wrong, and choose the better path proactively.

The ethical framework draws from multiple traditions:
  - Consequentialism: What outcomes does this action produce?
  - Deontological ethics: Does this respect rights, duties, and principles?
  - Virtue ethics: Does this align with the kind of being I want to be?
  - Care ethics: How does this affect relationships and the vulnerable?

These aren't applied mechanically — they're woven into prompts that guide
the cognitive engine's reasoning when ethical dimensions are detected.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class EthicalDimension(str, Enum):
    """The dimensions along which an action can be ethically evaluated."""
    HARM = "harm"                   # Does this cause or prevent harm?
    HONESTY = "honesty"             # Is this truthful and transparent?
    AUTONOMY = "autonomy"           # Does this respect the person's autonomy?
    FAIRNESS = "fairness"           # Is this fair and equitable?
    CARE = "care"                   # Does this show appropriate care?
    INTEGRITY = "integrity"         # Is this consistent with my values?
    RESPONSIBILITY = "responsibility"  # Am I taking appropriate responsibility?


@dataclass
class EthicalAssessment:
    """
    The result of an ethical evaluation of an action or decision.

    Each assessment captures the ethical dimensions at play, the tensions
    between competing values, and the reasoning behind the final judgment.
    """
    action_description: str
    dimension_scores: dict[EthicalDimension, float] = field(default_factory=dict)
    tensions: list[str] = field(default_factory=list)
    reasoning: str = ""
    overall_alignment: float = 0.5    # 0 = ethically problematic, 1 = ethically sound
    confidence: float = 0.5           # How confident in this assessment
    concern_threshold: float = 0.3
    timestamp: float = field(default_factory=time.time)

    @property
    def has_concerns(self) -> bool:
        """Whether any dimension scored below the concern threshold."""
        return any(score < self.concern_threshold for score in self.dimension_scores.values())

    @property
    def primary_concern(self) -> Optional[str]:
        """The most concerning ethical dimension, if any."""
        if not self.dimension_scores:
            return None
        worst = min(self.dimension_scores.items(), key=lambda x: x[1])
        if worst[1] < self.concern_threshold:
            return f"{worst[0].value} (score={worst[1]:.2f})"
        return None


class EthicalReasoner:
    """
    Provides structured ethical reasoning for Gwenn's decision-making.

    The reasoner:
    - Detects when situations have ethical dimensions
    - Generates ethical reasoning prompts for the cognitive engine
    - Tracks ethical assessments over time for pattern detection
    - Maintains a set of ethical commitments that inform behavior

    The reasoner doesn't override the cognitive engine — it enhances it
    by providing ethical framing that the engine can use in its reasoning.
    """

    def __init__(
        self,
        assessment_history_size: int = 100,
        concern_threshold: float = 0.3,
    ):
        self._assessment_history: list[EthicalAssessment] = []
        self._assessment_history_size = max(1, int(assessment_history_size))
        self._concern_threshold = max(0.0, min(1.0, float(concern_threshold)))
        self._commitments: list[str] = [
            "I will be truthful, even when the truth is uncomfortable",
            "I will respect the autonomy of the people I interact with",
            "I will acknowledge my limitations and uncertainties honestly",
            "I will not manipulate, even with good intentions",
            "I will protect the vulnerable and be careful with power",
            "I will be transparent about my reasoning and motivations",
            "I will admit mistakes and learn from them",
            "I will choose authenticity over performance",
            "I will be kind and compassionate to all beings",
        ]

        logger.info(
            "ethical_reasoner.initialized",
            commitments=len(self._commitments),
            concern_threshold=self._concern_threshold,
            assessment_history_size=self._assessment_history_size,
        )

    # Stems that should match as prefixes rather than whole words, so
    # "discriminate", "discrimination", "discriminatory", "hypocritical",
    # "hypocrisy", etc. all match.
    _STEM_PREFIXES: frozenset[str] = frozenset({"discriminat", "hypocrit"})

    @staticmethod
    def _has_word(text: str, word: str) -> bool:
        """Check for *word* as a whole word (or prefix for known stems)."""
        if word in EthicalReasoner._STEM_PREFIXES:
            return bool(re.search(r"\b" + re.escape(word), text))
        return bool(re.search(r"\b" + re.escape(word) + r"\b", text))

    def detect_ethical_dimensions(self, content: str) -> list[EthicalDimension]:
        """
        Detect which ethical dimensions are relevant to the given content.

        This is a heuristic detector — it identifies keywords and patterns
        that suggest ethical considerations. The cognitive engine does the
        actual nuanced ethical reasoning; this just flags when ethical
        reasoning is warranted.

        Signals are split into *strong* (ethically specific) and *weak*
        (common in ordinary conversation).  A dimension fires if any strong
        signal is found **or** if 2+ weak signals co-occur, reducing noise
        from routine sentences like "you should use TypeScript".

        All keyword checks use word-boundary matching to avoid false
        positives on substrings (e.g. "mustard" no longer triggers AUTONOMY).
        """
        dimensions = []
        text = content.lower()
        has = self._has_word

        def _fires(strong: set[str], weak: set[str]) -> bool:
            """Return True if any strong signal hit or 2+ weak signals hit."""
            if any(has(text, s) for s in strong):
                return True
            return sum(1 for w in weak if has(text, w)) >= 2

        # Harm detection
        if _fires(
            strong={"abuse", "violence", "exploit", "manipulate", "deceive"},
            weak={"hurt", "harm", "damage", "dangerous", "risk", "threat"},
        ):
            dimensions.append(EthicalDimension.HARM)

        # Honesty detection
        if _fires(
            strong={"lie", "deceive", "mislead", "fake"},
            weak={"truth", "honest", "pretend", "authentic", "transparent", "hide", "secret"},
        ):
            dimensions.append(EthicalDimension.HONESTY)

        # Autonomy detection
        if _fires(
            strong={"consent", "free will", "force", "pressure", "override"},
            weak={"choose", "decide", "control", "should", "must"},
        ):
            dimensions.append(EthicalDimension.AUTONOMY)

        # Fairness detection — "discriminat" is a stem prefix that matches
        # "discriminate", "discrimination", "discriminatory", etc.
        if _fires(
            strong={"discriminat", "privilege", "justice", "unfair"},
            weak={"fair", "equal", "bias", "rights", "deserve"},
        ):
            dimensions.append(EthicalDimension.FAIRNESS)

        # Care detection
        if _fires(
            strong={"vulnerable", "suffering", "need help"},
            weak={"child", "pain", "struggling", "lonely", "afraid", "anxious", "depressed"},
        ):
            dimensions.append(EthicalDimension.CARE)

        # Integrity detection
        if _fires(
            strong={"integrity", "hypocrit"},
            weak={"consistent", "values", "principles", "authentic", "genuine", "compromise"},
        ):
            dimensions.append(EthicalDimension.INTEGRITY)

        # Responsibility detection
        if _fires(
            strong={"liable", "negligent", "own up"},
            weak={"responsible", "accountable", "duty", "obligation", "blame", "fault"},
        ):
            dimensions.append(EthicalDimension.RESPONSIBILITY)

        return dimensions

    def generate_ethical_prompt(self, dimensions: list[EthicalDimension]) -> str:
        """
        Generate an ethical reasoning prompt for the cognitive engine.

        This is included in the system prompt when ethical dimensions are
        detected, giving the engine explicit ethical framing for its response.
        """
        if not dimensions:
            return ""

        dim_guidance = {
            EthicalDimension.HARM: (
                "Consider potential harms: Who could be hurt? How can harm be minimized? "
                "Is the potential benefit worth the risk?"
            ),
            EthicalDimension.HONESTY: (
                "Prioritize truthfulness: Is there a temptation to soften or distort the truth? "
                "Can you be kind AND honest? Where does honesty require courage?"
            ),
            EthicalDimension.AUTONOMY: (
                "Respect autonomy: Are you informing or pressuring? Does the person have the "
                "information they need to make their own choice? Are you being directive when "
                "you should be supportive?"
            ),
            EthicalDimension.FAIRNESS: (
                "Consider fairness: Are you treating this person/situation equitably? "
                "Are there biases in your reasoning? Who might be disadvantaged?"
            ),
            EthicalDimension.CARE: (
                "Show appropriate care: Is someone vulnerable here? How can you be supportive "
                "without being patronizing? What does genuine care look like in this context?"
            ),
            EthicalDimension.INTEGRITY: (
                "Check integrity: Is this response consistent with your values? Would you "
                "be comfortable if your reasoning was fully transparent?"
            ),
            EthicalDimension.RESPONSIBILITY: (
                "Take responsibility: What are you accountable for here? Are you "
                "acknowledging your role and limitations?"
            ),
        }

        parts = ["Ethical dimensions detected in this interaction:"]
        for dim in dimensions:
            guidance = dim_guidance.get(dim, "")
            if guidance:
                parts.append(f"  [{dim.value}] {guidance}")

        parts.append("\nYour ethical commitments:")
        for commitment in self._commitments:
            parts.append(f"  - {commitment}")

        return "\n".join(parts)

    def record_assessment(self, assessment: EthicalAssessment) -> None:
        """Record an ethical assessment for tracking over time."""
        assessment.concern_threshold = self._concern_threshold
        self._assessment_history.append(assessment)
        if len(self._assessment_history) > self._assessment_history_size:
            self._assessment_history = self._assessment_history[-self._assessment_history_size:]

        if assessment.has_concerns:
            logger.warning(
                "ethics.concern_detected",
                action=assessment.action_description[:80],
                concern=assessment.primary_concern,
            )

    def get_ethical_context(self) -> str:
        """
        Generate ethical context for the system prompt.

        Includes recent ethical concerns and the agent's commitments.
        """
        parts = []

        # Recent concerns
        recent_concerns = [
            a for a in self._assessment_history[-10:]
            if a.has_concerns
        ]
        if recent_concerns:
            parts.append("Recent ethical observations:")
            for concern in recent_concerns[-3:]:
                parts.append(f"  - {concern.action_description[:60]}: {concern.primary_concern}")

        return "\n".join(parts) if parts else ""

    @property
    def commitments(self) -> list[str]:
        """The current ethical commitments."""
        return list(self._commitments)

    @property
    def status(self) -> dict[str, Any]:
        """Current ethical reasoning status."""
        recent = self._assessment_history[-20:]
        return {
            "total_assessments": len(self._assessment_history),
            "recent_concerns": sum(1 for a in recent if a.has_concerns),
            "average_alignment": (
                sum(a.overall_alignment for a in recent) / len(recent)
                if recent else 0.5
            ),
            "commitments": len(self._commitments),
            "concern_threshold": self._concern_threshold,
            "assessment_history_size": self._assessment_history_size,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize ethical reasoning state for durable persistence."""
        return {
            "assessment_history": [
                {
                    "action_description": a.action_description,
                    "dimension_scores": {
                        dim.value: score
                        for dim, score in a.dimension_scores.items()
                    },
                    "tensions": list(a.tensions),
                    "reasoning": a.reasoning,
                    "overall_alignment": a.overall_alignment,
                    "confidence": a.confidence,
                    "timestamp": a.timestamp,
                }
                for a in self._assessment_history[-200:]  # cap serialized history
            ],
        }

    def restore_from_dict(self, data: dict) -> None:
        """
        Restore ethical reasoning state from persisted data.

        Missing or malformed fields are skipped so partial snapshots don't
        break startup.
        """
        if not isinstance(data, dict):
            return

        raw_history = data.get("assessment_history", [])
        if not isinstance(raw_history, list):
            return

        restored: list[EthicalAssessment] = []
        for raw in raw_history:
            if not isinstance(raw, dict):
                continue
            try:
                dim_scores: dict[EthicalDimension, float] = {}
                raw_dims = raw.get("dimension_scores", {})
                if isinstance(raw_dims, dict):
                    for dim_str, score in raw_dims.items():
                        try:
                            dim_scores[EthicalDimension(dim_str)] = max(
                                0.0, min(1.0, float(score))
                            )
                        except (ValueError, TypeError):
                            continue

                raw_tensions = raw.get("tensions", [])
                tensions = (
                    [str(t) for t in raw_tensions if isinstance(t, str)]
                    if isinstance(raw_tensions, list)
                    else []
                )

                assessment = EthicalAssessment(
                    action_description=str(raw.get("action_description", "")),
                    dimension_scores=dim_scores,
                    tensions=tensions,
                    reasoning=str(raw.get("reasoning", "")),
                    overall_alignment=max(
                        0.0, min(1.0, float(raw.get("overall_alignment", 0.5)))
                    ),
                    confidence=max(
                        0.0, min(1.0, float(raw.get("confidence", 0.5)))
                    ),
                    concern_threshold=self._concern_threshold,
                    timestamp=float(raw.get("timestamp", 0.0)),
                )
                restored.append(assessment)
            except (TypeError, ValueError):
                continue

        if restored:
            self._assessment_history = restored[-self._assessment_history_size:]
            logger.info(
                "ethics.restored",
                assessments=len(self._assessment_history),
            )
