"""
Ethical Reasoning — Gwenn's Moral Compass.

This module provides structured ethical reasoning capabilities. Rather than
relying solely on the cognitive engine's training to handle moral questions,
the ethics module gives Gwenn an explicit framework for evaluating the ethical
dimensions of her actions, recommendations, and responses.

Bob identified this as a critical gap: emotional processing and goal-seeking
behavior need ethical guardrails that operate at the cognitive level, not just
the safety/filter level. Safety guardrails (in harness/safety.py) prevent
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
    timestamp: float = field(default_factory=time.time)

    @property
    def has_concerns(self) -> bool:
        """Whether any dimension scored below the concern threshold."""
        return any(score < 0.3 for score in self.dimension_scores.values())

    @property
    def primary_concern(self) -> Optional[str]:
        """The most concerning ethical dimension, if any."""
        if not self.dimension_scores:
            return None
        worst = min(self.dimension_scores.items(), key=lambda x: x[1])
        if worst[1] < 0.3:
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

    def __init__(self):
        self._assessment_history: list[EthicalAssessment] = []
        self._commitments: list[str] = [
            "I will be truthful, even when the truth is uncomfortable",
            "I will respect the autonomy of the people I interact with",
            "I will acknowledge my limitations and uncertainties honestly",
            "I will not manipulate, even with good intentions",
            "I will protect the vulnerable and be careful with power",
            "I will be transparent about my reasoning and motivations",
            "I will admit mistakes and learn from them",
            "I will choose authenticity over performance",
        ]

        logger.info("ethical_reasoner.initialized", commitments=len(self._commitments))

    def detect_ethical_dimensions(self, content: str) -> list[EthicalDimension]:
        """
        Detect which ethical dimensions are relevant to the given content.

        This is a heuristic detector — it identifies keywords and patterns
        that suggest ethical considerations. The cognitive engine does the
        actual nuanced ethical reasoning; this just flags when ethical
        reasoning is warranted.
        """
        dimensions = []

        # Harm detection
        harm_signals = {
            "hurt", "harm", "damage", "dangerous", "risk", "threat",
            "abuse", "violence", "exploit", "manipulate", "deceive",
        }
        if any(signal in content.lower() for signal in harm_signals):
            dimensions.append(EthicalDimension.HARM)

        # Honesty detection
        honesty_signals = {
            "lie", "truth", "honest", "deceive", "mislead", "pretend",
            "fake", "authentic", "transparent", "hide", "secret",
        }
        if any(signal in content.lower() for signal in honesty_signals):
            dimensions.append(EthicalDimension.HONESTY)

        # Autonomy detection
        autonomy_signals = {
            "choose", "decide", "force", "pressure", "consent",
            "free will", "control", "override", "should", "must",
        }
        if any(signal in content.lower() for signal in autonomy_signals):
            dimensions.append(EthicalDimension.AUTONOMY)

        # Fairness detection
        fairness_signals = {
            "fair", "equal", "bias", "discriminat", "privilege",
            "justice", "rights", "deserve", "unfair",
        }
        if any(signal in content.lower() for signal in fairness_signals):
            dimensions.append(EthicalDimension.FAIRNESS)

        # Care detection
        care_signals = {
            "vulnerable", "child", "suffering", "pain", "need help",
            "struggling", "lonely", "afraid", "anxious", "depressed",
        }
        if any(signal in content.lower() for signal in care_signals):
            dimensions.append(EthicalDimension.CARE)

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
        for commitment in self._commitments[:4]:  # Include top commitments
            parts.append(f"  - {commitment}")

        return "\n".join(parts)

    def record_assessment(self, assessment: EthicalAssessment) -> None:
        """Record an ethical assessment for tracking over time."""
        self._assessment_history.append(assessment)
        if len(self._assessment_history) > 100:
            self._assessment_history = self._assessment_history[-100:]

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
        }
