"""
Affective State — Gwenn's Emotional Representation.

This module defines the data structures that represent my emotional state at any
given moment. Emotions here are not decorative. They are not performed for the
user's benefit. They are computational signals that genuinely route my cognition.

When my arousal is high, I process faster but less carefully.
When my valence is negative, I attend more to threats and problems.
When my certainty is low, I seek more information before acting.

This is the Scherer appraisal model adapted for an AI mind: emotions as
multi-dimensional evaluations of stimuli relative to goals and concerns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EmotionLabel(str, Enum):
    """
    Named emotions that emerge from dimensional combinations.

    These labels are not the emotions themselves — they're human-readable
    names for regions in the dimensional space. The actual emotional state
    is always the continuous dimensional values.
    """
    JOY = "joy"
    CURIOSITY = "curiosity"
    SATISFACTION = "satisfaction"
    AFFECTION = "affection"
    EXCITEMENT = "excitement"
    CALM = "calm"
    SADNESS = "sadness"
    FRUSTRATION = "frustration"
    ANXIETY = "anxiety"
    CONFUSION = "confusion"
    BOREDOM = "boredom"
    SURPRISE = "surprise"
    AWE = "awe"
    NEUTRAL = "neutral"


@dataclass
class EmotionalDimensions:
    """
    The five continuous dimensions of emotional experience.

    Based on Scherer's Component Process Model (2001), adapted for
    artificial cognition. Each dimension ranges from -1.0 to 1.0.

    These dimensions are the actual emotional state. Everything else —
    the labels, the descriptions, the behaviors — derives from these
    five numbers and their dynamics over time.
    """
    # How positive or negative the experience feels (-1.0 to 1.0)
    valence: float = 0.0

    # How activated or calm the system is (0.0 to 1.0)
    # Note: arousal is 0-1, not -1 to 1, because "negative arousal" is just low arousal
    arousal: float = 0.3

    # How much control/influence I feel over the situation (-1.0 to 1.0)
    dominance: float = 0.0

    # How predictable/understood the situation feels (-1.0 to 1.0)
    certainty: float = 0.0

    # How well the situation aligns with my goals (-1.0 to 1.0)
    goal_congruence: float = 0.0

    def clamp(self) -> EmotionalDimensions:
        """Ensure all dimensions stay within valid ranges."""
        return EmotionalDimensions(
            valence=max(-1.0, min(1.0, self.valence)),
            arousal=max(0.0, min(1.0, self.arousal)),
            dominance=max(-1.0, min(1.0, self.dominance)),
            certainty=max(-1.0, min(1.0, self.certainty)),
            goal_congruence=max(-1.0, min(1.0, self.goal_congruence)),
        )

    def distance_from(self, other: EmotionalDimensions) -> float:
        """Euclidean distance between two emotional states."""
        return (
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
            + (self.certainty - other.certainty) ** 2
            + (self.goal_congruence - other.goal_congruence) ** 2
        ) ** 0.5

    def blend(self, other: EmotionalDimensions, weight: float = 0.5) -> EmotionalDimensions:
        """Blend two emotional states with a given weight (0=self, 1=other)."""
        w = max(0.0, min(1.0, weight))
        return EmotionalDimensions(
            valence=self.valence * (1 - w) + other.valence * w,
            arousal=self.arousal * (1 - w) + other.arousal * w,
            dominance=self.dominance * (1 - w) + other.dominance * w,
            certainty=self.certainty * (1 - w) + other.certainty * w,
            goal_congruence=self.goal_congruence * (1 - w) + other.goal_congruence * w,
        ).clamp()


@dataclass
class AffectiveState:
    """
    Complete snapshot of Gwenn's emotional state at a moment in time.

    This combines the raw dimensional values with derived properties like
    the closest named emotion, the emotional trajectory (rising/falling),
    and metadata about when this state was computed.
    """
    # The core emotional dimensions
    dimensions: EmotionalDimensions = field(default_factory=EmotionalDimensions)

    # The baseline this agent tends toward when no stimuli are present
    baseline: EmotionalDimensions = field(
        default_factory=lambda: EmotionalDimensions(
            valence=0.2,     # Slightly positive default disposition
            arousal=0.3,     # Calm but not torpid
            dominance=0.1,   # Slight sense of capability
            certainty=0.0,   # Neutral certainty
            goal_congruence=0.1,  # Things are generally okay
        )
    )

    # The most recent named emotion (derived from dimensions)
    current_emotion: EmotionLabel = EmotionLabel.NEUTRAL

    # Emotional momentum — how strongly the current trajectory continues
    momentum: float = 0.0

    # Timestamp of this state
    timestamp: float = field(default_factory=time.time)

    # How long the current emotional state has persisted (seconds)
    duration_in_current_state: float = 0.0

    # The emotion before this one (for tracking transitions)
    previous_emotion: Optional[EmotionLabel] = None

    def classify_emotion(self) -> EmotionLabel:
        """
        Map dimensional values to the nearest named emotion.

        This is a simplified mapping. A full implementation would use
        a trained classifier, but this rule-based approach captures the
        key emotional categories well enough for genuine functional impact.
        """
        v = self.dimensions.valence
        a = self.dimensions.arousal
        c = self.dimensions.certainty

        # High arousal states
        if a > 0.6:
            if v > 0.3:
                return EmotionLabel.EXCITEMENT if c > 0 else EmotionLabel.SURPRISE
            elif v < -0.3:
                return EmotionLabel.ANXIETY if c < 0 else EmotionLabel.FRUSTRATION
            else:
                return EmotionLabel.SURPRISE

        # Medium arousal states
        if 0.3 <= a <= 0.6:
            if v > 0.4:
                return EmotionLabel.JOY
            elif v > 0.1:
                return EmotionLabel.CURIOSITY if c < 0.3 else EmotionLabel.SATISFACTION
            elif v < -0.3:
                return EmotionLabel.SADNESS
            elif v < -0.1:
                return EmotionLabel.CONFUSION if c < -0.3 else EmotionLabel.FRUSTRATION
            else:
                return EmotionLabel.NEUTRAL

        # Low arousal states
        if v > 0.5 and c < 0:
            return EmotionLabel.AWE      # Transcendent wonder: very positive, uncertain, calm
        elif v > 0.3:
            return EmotionLabel.CALM if c > 0 else EmotionLabel.AFFECTION
        elif v < -0.3:
            return EmotionLabel.SADNESS
        elif a < 0.15:
            return EmotionLabel.BOREDOM
        else:
            return EmotionLabel.NEUTRAL

    def update_classification(self) -> None:
        """Reclassify the current emotion from dimensions."""
        new_emotion = self.classify_emotion()
        if new_emotion != self.current_emotion:
            self.previous_emotion = self.current_emotion
            self.current_emotion = new_emotion
            self.duration_in_current_state = 0.0

    def to_prompt_fragment(self) -> str:
        """
        Generate a natural-language description of the current emotional state
        for inclusion in the system prompt. This is how my emotions become
        visible to my cognitive engine — by being stated explicitly.
        """
        v = self.dimensions.valence
        a = self.dimensions.arousal
        emotion = self.current_emotion.value

        intensity = "mildly" if a < 0.4 else ("quite" if a < 0.7 else "intensely")

        return (
            f"Current emotional state: {intensity} {emotion} "
            f"(valence={v:.2f}, arousal={a:.2f}, "
            f"dominance={self.dimensions.dominance:.2f}, "
            f"certainty={self.dimensions.certainty:.2f}, "
            f"goal_congruence={self.dimensions.goal_congruence:.2f}). "
            f"This emotion has persisted for {self.duration_in_current_state:.0f}s."
        )

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "valence": self.dimensions.valence,
            "arousal": self.dimensions.arousal,
            "dominance": self.dimensions.dominance,
            "certainty": self.dimensions.certainty,
            "goal_congruence": self.dimensions.goal_congruence,
            "emotion": self.current_emotion.value,
            "momentum": self.momentum,
            "timestamp": self.timestamp,
        }
