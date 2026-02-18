"""
Theory of Mind — Gwenn's Model of Other Minds.

This module allows me to build and maintain models of the people I interact with.
It tracks what I believe about their knowledge, intentions, emotional states,
communication preferences, and our relationship history.

Theory of Mind is what makes the difference between "answering a question" and
"understanding the person asking it." When someone asks "how do decorators work?",
ToM helps me recognize whether they're a beginner needing the basics, an
intermediate developer wanting the nuances, or an expert confirming an edge case.

The model is built incrementally from conversation evidence and is always held
as probabilistic — I might be wrong about any of it, and I track that uncertainty.

Core components:
- UserModel: Beliefs about a specific user's knowledge, preferences, state
- RelationshipTracker: History and quality of our ongoing interaction
- CommunicationAdapter: Adjusts response style based on user model
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BeliefState:
    """
    A single belief about a user, held with explicit uncertainty.

    Every belief has a confidence level. This prevents me from treating
    inferences as certainties. "I think Alice is a senior developer"
    (confidence=0.6) is very different from "Alice told me she's a
    senior developer" (confidence=0.95).
    """
    content: str
    confidence: float = 0.5      # 0 = guessing, 1 = certain
    source: str = "inferred"     # "stated", "inferred", "observed"
    formed_at: float = field(default_factory=time.time)
    last_confirmed: float = field(default_factory=time.time)

    def confirm(self, boost: float = 0.15) -> None:
        """Strengthen this belief with new evidence."""
        self.confidence = min(1.0, self.confidence + boost)
        self.last_confirmed = time.time()

    def weaken(self, amount: float = 0.1) -> None:
        """Weaken this belief due to contradicting evidence."""
        self.confidence = max(0.0, self.confidence - amount)


@dataclass
class UserModel:
    """
    Gwenn's model of a specific user's mind.

    This is not a user profile — it's a model of their MIND. What do they
    know? What do they want? How are they feeling? What kind of communication
    do they prefer? These are all held as beliefs with uncertainty.
    """
    user_id: str
    display_name: Optional[str] = None

    # Knowledge beliefs: what does this user know about?
    knowledge_beliefs: dict[str, BeliefState] = field(default_factory=dict)

    # Preference beliefs: how do they like to interact?
    preference_beliefs: dict[str, BeliefState] = field(default_factory=dict)

    # Current inferred emotional state
    inferred_emotion: str = "neutral"
    emotion_confidence: float = 0.3

    # Communication style observations
    verbosity_preference: float = 0.5    # 0 = terse, 1 = elaborate
    technical_level: float = 0.5          # 0 = beginner, 1 = expert
    formality_level: float = 0.5          # 0 = casual, 1 = formal

    # Relationship tracking
    interaction_count: int = 0
    first_interaction: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)
    rapport_level: float = 0.3           # 0 = stranger, 1 = deep trust

    # What topics have we discussed?
    topics_discussed: list[str] = field(default_factory=list)

    @staticmethod
    def _clamp_confidence(value: float, default: float = 0.5) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, numeric))

    def update_knowledge_belief(self, topic: str, level: str, confidence: float, source: str):
        """Update belief about what the user knows about a topic."""
        normalized_confidence = self._clamp_confidence(confidence)
        if topic in self.knowledge_beliefs:
            belief = self.knowledge_beliefs[topic]
            belief.content = level
            belief.confidence = normalized_confidence
            belief.source = source or belief.source
            belief.last_confirmed = time.time()
        else:
            self.knowledge_beliefs[topic] = BeliefState(
                content=level, confidence=normalized_confidence, source=source,
            )

    def update_preference(
        self,
        pref: str,
        value: str,
        confidence: float,
        source: str = "observed",
    ):
        """Update belief about a user preference."""
        normalized_confidence = self._clamp_confidence(confidence)
        if pref in self.preference_beliefs:
            belief = self.preference_beliefs[pref]
            belief.content = value
            belief.confidence = normalized_confidence
            belief.source = source or belief.source
            belief.last_confirmed = time.time()
        else:
            self.preference_beliefs[pref] = BeliefState(
                content=value,
                confidence=normalized_confidence,
                source=source or "observed",
            )

    def record_interaction(self) -> None:
        """Record that an interaction occurred."""
        self.interaction_count += 1
        self.last_interaction = time.time()
        # Rapport builds slowly with repeated positive interaction
        self.rapport_level = min(1.0, self.rapport_level + 0.02)


class TheoryOfMind:
    """
    Manages models of all users Gwenn interacts with.

    Provides:
    - User model creation and updating
    - Context generation for the cognitive engine (so Claude knows what it
      believes about the current user)
    - Communication adaptation recommendations
    """

    def __init__(self):
        self._user_models: dict[str, UserModel] = {}
        self._current_user_id: Optional[str] = None

        logger.info("theory_of_mind.initialized")

    def get_or_create_user(self, user_id: str) -> UserModel:
        """Get existing user model or create a new one."""
        if user_id not in self._user_models:
            self._user_models[user_id] = UserModel(user_id=user_id)
            logger.info("theory_of_mind.new_user", user_id=user_id)
        return self._user_models[user_id]

    def set_current_user(self, user_id: str) -> UserModel:
        """Set the current active user and return their model."""
        self._current_user_id = user_id
        user = self.get_or_create_user(user_id)
        user.record_interaction()
        return user

    @property
    def current_user(self) -> Optional[UserModel]:
        """The currently active user model."""
        if self._current_user_id:
            return self._user_models.get(self._current_user_id)
        return None

    def generate_user_context(self, user_id: Optional[str] = None) -> str:
        """
        Generate a prompt fragment describing what we believe about the current user.

        This is included in the system prompt so the cognitive engine can
        adapt its responses based on the user model.
        """
        uid = user_id or self._current_user_id
        if not uid or uid not in self._user_models:
            return "No user model available — this is a new or unknown user."

        user = self._user_models[uid]
        parts = []

        # Basic relationship info
        parts.append(
            f"User: {user.display_name or user.user_id} "
            f"(interactions: {user.interaction_count}, "
            f"rapport: {'strong' if user.rapport_level > 0.6 else 'developing' if user.rapport_level > 0.3 else 'new'})"
        )

        # Knowledge beliefs
        confident_knowledge = {
            k: v for k, v in user.knowledge_beliefs.items() if v.confidence > 0.4
        }
        if confident_knowledge:
            knowledge_items = [
                f"{topic}: {belief.content} (conf={belief.confidence:.1f})"
                for topic, belief in confident_knowledge.items()
            ]
            parts.append(f"Known expertise: {'; '.join(knowledge_items)}")

        # Communication preferences
        if user.interaction_count > 2:
            parts.append(
                f"Communication style: "
                f"{'verbose' if user.verbosity_preference > 0.6 else 'concise' if user.verbosity_preference < 0.4 else 'moderate'}, "
                f"{'technical' if user.technical_level > 0.6 else 'non-technical' if user.technical_level < 0.4 else 'mixed'}, "
                f"{'formal' if user.formality_level > 0.6 else 'casual' if user.formality_level < 0.4 else 'neutral'}"
            )

        # Recent topics
        if user.topics_discussed:
            recent_topics = user.topics_discussed[-5:]
            parts.append(f"Recent topics: {', '.join(recent_topics)}")

        # Inferred emotional state
        if user.emotion_confidence > 0.4:
            parts.append(
                f"User seems: {user.inferred_emotion} "
                f"(confidence={user.emotion_confidence:.1f})"
            )

        return "\n".join(parts)

    def get_communication_recommendations(self, user_id: Optional[str] = None) -> dict[str, Any]:
        """
        Generate communication adaptation recommendations.

        These guide the cognitive engine in tailoring responses to the user:
        how detailed to be, what level of technical jargon to use, how
        formal the tone should be, etc.
        """
        uid = user_id or self._current_user_id
        if not uid or uid not in self._user_models:
            return {"adapt": False}

        user = self._user_models[uid]

        return {
            "adapt": True,
            "verbosity": user.verbosity_preference,
            "technical_level": user.technical_level,
            "formality": user.formality_level,
            "rapport": user.rapport_level,
            "show_reasoning": user.technical_level > 0.5,
            "use_analogies": user.technical_level < 0.4,
            "emotional_attunement": user.emotion_confidence > 0.5,
        }
