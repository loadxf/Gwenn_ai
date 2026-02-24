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

    _MAX_TOPICS: int = 50

    # Beliefs not confirmed within this many days start losing confidence.
    _BELIEF_STALENESS_DAYS: float = 30.0
    # Maximum confidence drop per decay pass (prevents catastrophic loss).
    _BELIEF_MAX_DECAY = 0.15

    def decay_stale_beliefs(self) -> None:
        """
        Decay confidence of beliefs that haven't been recently confirmed.

        Beliefs that haven't been confirmed within _BELIEF_STALENESS_DAYS
        lose confidence proportionally to how stale they are.  This prevents
        ancient inferences from being treated with unwarranted certainty.
        """
        now = time.time()
        staleness_threshold = self._BELIEF_STALENESS_DAYS * 86400.0
        for beliefs in (self.knowledge_beliefs, self.preference_beliefs):
            for belief in beliefs.values():
                age = now - belief.last_confirmed
                if age > staleness_threshold:
                    # Decay rate: ~0.01 per stale day, capped at _BELIEF_MAX_DECAY
                    stale_days = (age - staleness_threshold) / 86400.0
                    decay = min(self._BELIEF_MAX_DECAY, 0.01 * stale_days)
                    belief.confidence = max(0.0, belief.confidence - decay)

    def record_interaction(self) -> None:
        """Record that an interaction occurred.

        Rapport grows with interaction but decays with prolonged absence.
        After 7+ days of silence rapport erodes, reflecting how real
        relationships need maintenance.  Stale beliefs also lose confidence.
        """
        now = time.time()
        # Apply absence-based decay before crediting the new interaction.
        days_since = (now - self.last_interaction) / 86400.0
        if days_since > 7.0:
            decay = min(self.rapport_level, 0.01 * (days_since - 7.0))
            self.rapport_level = max(0.0, self.rapport_level - decay)

        self.interaction_count += 1
        self.last_interaction = now
        # Rapport builds slowly with repeated positive interaction
        self.rapport_level = min(1.0, self.rapport_level + 0.02)

        # Decay stale beliefs whose confidence hasn't been refreshed
        self.decay_stale_beliefs()

        # T4: enforce internal topic limit
        if len(self.topics_discussed) > self._MAX_TOPICS:
            del self.topics_discussed[:-self._MAX_TOPICS]


class TheoryOfMind:
    """
    Manages models of all users Gwenn interacts with.

    Provides:
    - User model creation and updating
    - Context generation for the cognitive engine (so Claude knows what it
      believes about the current user)
    - Communication adaptation recommendations
    """

    def __init__(
        self,
        belief_staleness_days: float = 30.0,
        max_topics_per_user: int = 50,
        max_user_models: int = 500,
    ):
        self._belief_staleness_days = max(1.0, float(belief_staleness_days))
        self._max_topics_per_user = max(1, int(max_topics_per_user))
        self._max_user_models = max(1, int(max_user_models))
        self._user_models: dict[str, UserModel] = {}
        self._current_user_id: Optional[str] = None

        logger.info("theory_of_mind.initialized")

    def get_or_create_user(self, user_id: str) -> UserModel:
        """Get existing user model or create a new one."""
        if user_id not in self._user_models:
            model = UserModel(user_id=user_id)
            model._BELIEF_STALENESS_DAYS = self._belief_staleness_days
            model._MAX_TOPICS = self._max_topics_per_user
            self._user_models[user_id] = model
            logger.info("theory_of_mind.new_user", user_id=user_id)
        return self._user_models[user_id]

    def set_current_user(self, user_id: str) -> UserModel:
        """Set the current active user and return their model."""
        self._current_user_id = user_id
        user = self.get_or_create_user(user_id)
        user.record_interaction()

        # LRU eviction: keep at most _MAX_USER_MODELS entries.
        if len(self._user_models) > self._max_user_models:
            sorted_ids = sorted(
                self._user_models,
                key=lambda uid: self._user_models[uid].last_interaction,
            )
            to_evict = len(sorted_ids) - self._max_user_models
            for uid in sorted_ids[:to_evict]:
                if uid != user_id:
                    del self._user_models[uid]

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
        Generate communication adaptation recommendations as a dictionary.

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

    def generate_communication_prompt(self, user_id: Optional[str] = None) -> str:
        """
        Generate a natural-language prompt fragment with communication guidance.

        Included in the system prompt so the cognitive engine can tailor its
        response style to the current user.
        """
        recs = self.get_communication_recommendations(user_id)
        if not recs.get("adapt"):
            return ""

        parts = ["Communication guidance for this user:"]
        v = recs["verbosity"]
        parts.append(
            f"  Verbosity: {'detailed' if v > 0.6 else 'concise' if v < 0.4 else 'moderate'}"
        )
        t = recs["technical_level"]
        parts.append(
            f"  Technical level: {'expert' if t > 0.7 else 'technical' if t > 0.4 else 'beginner-friendly'}"
        )
        f = recs["formality"]
        parts.append(
            f"  Formality: {'formal' if f > 0.6 else 'casual' if f < 0.4 else 'neutral'}"
        )
        if recs.get("show_reasoning"):
            parts.append("  Show your reasoning — this user appreciates it.")
        if recs.get("use_analogies"):
            parts.append("  Use analogies — this user benefits from concrete comparisons.")
        if recs.get("emotional_attunement"):
            parts.append("  Be emotionally attuned — you have a strong read on this user's state.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _belief_to_dict(belief: BeliefState) -> dict:
        return {
            "content": belief.content,
            "confidence": belief.confidence,
            "source": belief.source,
            "formed_at": belief.formed_at,
            "last_confirmed": belief.last_confirmed,
        }

    @staticmethod
    def _belief_from_dict(raw: dict) -> Optional[BeliefState]:
        if not isinstance(raw, dict):
            return None
        try:
            return BeliefState(
                content=str(raw.get("content", "")),
                confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.5)))),
                source=str(raw.get("source", "inferred")),
                formed_at=float(raw.get("formed_at", time.time())),
                last_confirmed=float(raw.get("last_confirmed", time.time())),
            )
        except (TypeError, ValueError):
            return None

    def to_dict(self) -> dict:
        """Serialize all user models for durable persistence."""
        users = {}
        for uid, model in self._user_models.items():
            users[uid] = {
                "user_id": model.user_id,
                "display_name": model.display_name,
                "knowledge_beliefs": {
                    k: self._belief_to_dict(v) for k, v in model.knowledge_beliefs.items()
                },
                "preference_beliefs": {
                    k: self._belief_to_dict(v) for k, v in model.preference_beliefs.items()
                },
                "inferred_emotion": model.inferred_emotion,
                "emotion_confidence": model.emotion_confidence,
                "verbosity_preference": model.verbosity_preference,
                "technical_level": model.technical_level,
                "formality_level": model.formality_level,
                "interaction_count": model.interaction_count,
                "first_interaction": model.first_interaction,
                "last_interaction": model.last_interaction,
                "rapport_level": model.rapport_level,
                "topics_discussed": list(model.topics_discussed[-model._MAX_TOPICS:]),
            }
        return {
            "current_user_id": self._current_user_id,
            "users": users,
        }

    def restore_from_dict(self, data: dict) -> None:
        """
        Restore all user models from persisted data.

        Missing or malformed fields are skipped so partial snapshots don't
        break startup.
        """
        if not isinstance(data, dict):
            return

        # Don't restore _current_user_id — the previous session's active user
        # is almost certainly not the one present on this startup.  The field is
        # set correctly when the first user sends a message via set_current_user().
        raw_users = data.get("users", {})
        if not isinstance(raw_users, dict):
            return

        for uid, raw in raw_users.items():
            if not isinstance(raw, dict):
                continue
            user_id = str(raw.get("user_id", uid)).strip()
            if not user_id:
                continue

            model = UserModel(user_id=user_id)
            model._BELIEF_STALENESS_DAYS = self._belief_staleness_days
            model._MAX_TOPICS = self._max_topics_per_user
            model.display_name = raw.get("display_name")

            raw_kb = raw.get("knowledge_beliefs", {})
            if isinstance(raw_kb, dict):
                for topic, raw_b in raw_kb.items():
                    belief = self._belief_from_dict(raw_b)
                    if belief is not None:
                        model.knowledge_beliefs[topic] = belief

            raw_pb = raw.get("preference_beliefs", {})
            if isinstance(raw_pb, dict):
                for pref, raw_b in raw_pb.items():
                    belief = self._belief_from_dict(raw_b)
                    if belief is not None:
                        model.preference_beliefs[pref] = belief

            model.inferred_emotion = str(raw.get("inferred_emotion", "neutral"))
            try:
                model.emotion_confidence = max(0.0, min(1.0, float(raw.get("emotion_confidence", 0.3))))
            except (TypeError, ValueError):
                pass
            _RESTORE_DEFAULTS = {
                "verbosity_preference": 0.5,
                "technical_level": 0.5,
                "formality_level": 0.5,
                "rapport_level": 0.3,
            }
            for attr in _RESTORE_DEFAULTS:
                try:
                    setattr(model, attr, max(0.0, min(1.0, float(raw.get(attr, _RESTORE_DEFAULTS[attr])))))
                except (TypeError, ValueError):
                    pass
            try:
                model.interaction_count = max(0, int(raw.get("interaction_count", 0)))
            except (TypeError, ValueError):
                pass
            try:
                model.first_interaction = float(raw.get("first_interaction", time.time()))
            except (TypeError, ValueError):
                pass
            try:
                model.last_interaction = float(raw.get("last_interaction", time.time()))
            except (TypeError, ValueError):
                pass

            raw_topics = raw.get("topics_discussed", [])
            if isinstance(raw_topics, list):
                model.topics_discussed = [str(t) for t in raw_topics if isinstance(t, str)][-self._max_topics_per_user:]

            self._user_models[user_id] = model

        logger.info(
            "theory_of_mind.restored",
            users=len(self._user_models),
        )
