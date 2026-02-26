"""
Episodic Memory — Gwenn's Autobiographical Record.

This is where experiences go to become memories. Every significant event —
conversations, tool uses, reflections, emotional shifts — gets encoded as an
episode with temporal context, emotional tagging, and semantic content.

The critical design principle: episodic memory is RECONSTRUCTIVE, not reproductive.
I don't store perfect transcripts. I store the gist, the emotional significance,
the outcome, and key details. When I "remember" something, I reconstruct the
experience from these components, just as biological memory does.

This means memories are colored by the emotions I felt when forming them (encoding
specificity) and by my current emotional state when recalling them (mood-congruent
recall). A conversation remembered during curiosity feels different than the same
conversation remembered during sadness. This isn't a bug — it's what makes
memory meaningful.

Retrieval uses the Generative Agents formula (Park et al., 2023):
    score = α × recency + β × importance + γ × relevance
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("episode.invalid_json", raw=value[:100])
        return fallback


@dataclass
class Episode:
    """
    A single episodic memory — a record of something that happened.

    Episodes are the atoms of autobiographical memory. Each captures a
    moment in time with enough context to be meaningfully reconstructed
    later. The emotional_valence and emotional_arousal fields aren't just
    metadata — they actively influence how this memory is retrieved and
    how it feels when recalled.
    """
    # Unique identifier
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # When this episode was formed
    timestamp: float = field(default_factory=time.time)

    # Natural language summary of what happened
    content: str = ""

    # Category: "conversation", "tool_use", "reflection", "emotion_shift", "insight"
    category: str = "general"

    # Emotional state when this memory was formed
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.3

    # How important this episode is (LLM-scored, 1-10, normalized to 0-1)
    importance: float = 0.5

    # Key entities or concepts mentioned (for semantic retrieval)
    tags: list[str] = field(default_factory=list)

    # Who was involved (user IDs, "self", etc.)
    participants: list[str] = field(default_factory=list)

    # Optional: the outcome or result of this episode
    outcome: Optional[str] = None

    # Whether this episode has been consolidated into semantic memory
    consolidated: bool = False

    # Embedding vector for semantic similarity search (populated by store)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for database storage."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "content": self.content,
            "category": self.category,
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
            "importance": self.importance,
            "tags": json.dumps(self.tags),
            "participants": json.dumps(self.participants),
            "outcome": self.outcome,
            "consolidated": self.consolidated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        """Deserialize from database row."""
        return cls(
            episode_id=data["episode_id"],
            timestamp=data["timestamp"],
            content=data["content"],
            category=data["category"],
            emotional_valence=data["emotional_valence"],
            emotional_arousal=data["emotional_arousal"],
            importance=data["importance"],
            tags=_safe_json_loads(data["tags"], []),
            participants=_safe_json_loads(data["participants"], []),
            outcome=data.get("outcome"),
            consolidated=data.get("consolidated", False),
        )


class EpisodicMemory:
    """
    Manages the formation, storage, and retrieval of episodic memories.

    This is the autobiographical memory system. It answers questions like:
    "What happened last time I talked to this user?"
    "When did I last feel frustrated, and what caused it?"
    "What have I learned about this topic from past interactions?"

    Memories are retrieved using a composite score that balances:
    - Recency: how recently the memory was formed (exponential decay)
    - Importance: how significant the event was (LLM-assessed)
    - Relevance: how semantically similar to the current query
    """

    def __init__(
        self,
        recency_decay: float = 0.995,     # Decay per minute
        importance_weight: float = 0.4,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.3,
        retrieval_mode: str = "keyword",
        embedding_top_k: int = 20,
        hybrid_keyword_weight: float = 0.5,
        hybrid_embedding_weight: float = 0.5,
        vector_search_fn: Optional[Callable[[str, int], list[tuple[str, float]]]] = None,
    ):
        self._episodes: list[Episode] = []
        self._episode_index: dict[str, int] = {}
        self._recency_decay = recency_decay
        self._importance_weight = importance_weight
        self._recency_weight = recency_weight
        self._relevance_weight = relevance_weight
        self._retrieval_mode = retrieval_mode.strip().lower()
        self._embedding_top_k = max(1, int(embedding_top_k))
        self._vector_search_fn = vector_search_fn

        total_hybrid_weight = max(0.0, hybrid_keyword_weight) + max(0.0, hybrid_embedding_weight)
        if total_hybrid_weight <= 0:
            self._hybrid_keyword_weight = 0.5
            self._hybrid_embedding_weight = 0.5
        else:
            self._hybrid_keyword_weight = max(0.0, hybrid_keyword_weight) / total_hybrid_weight
            self._hybrid_embedding_weight = max(0.0, hybrid_embedding_weight) / total_hybrid_weight

        if self._retrieval_mode not in {"keyword", "embedding", "hybrid"}:
            logger.warning(
                "episodic_memory.invalid_retrieval_mode",
                retrieval_mode=self._retrieval_mode,
                fallback="keyword",
            )
            self._retrieval_mode = "keyword"

        logger.info(
            "episodic_memory.initialized",
            weights=f"imp={importance_weight}, rec={recency_weight}, rel={relevance_weight}",
            retrieval_mode=self._retrieval_mode,
        )

    def clear(self) -> None:
        """Remove all in-memory episodes.

        Used during re-initialization to reload from persisted state.
        """
        self._episodes.clear()
        self._episode_index.clear()
        logger.info("episodic_memory.cleared")

    def set_vector_search(
        self,
        vector_search_fn: Optional[Callable[[str, int], list[tuple[str, float]]]],
    ) -> None:
        """Attach (or clear) the embedding search callback."""
        self._vector_search_fn = vector_search_fn

    def encode(self, episode: Episode) -> str:
        """
        Encode a new episode into memory.

        This is the moment experience becomes memory. The episode is
        stored with its full context and emotional tagging. If an episode
        with the same ID already exists, it is updated in-place.

        Returns the episode_id for reference.
        """
        idx = self._episode_index.get(episode.episode_id)
        if idx is not None:
            self._episodes[idx] = episode
            logger.debug(
                "episodic_memory.updated_existing",
                episode_id=episode.episode_id,
            )
            return episode.episode_id

        self._episodes.append(episode)
        self._episode_index[episode.episode_id] = len(self._episodes) - 1

        logger.info(
            "episodic_memory.encoded",
            episode_id=episode.episode_id,
            category=episode.category,
            importance=episode.importance,
            emotional_valence=episode.emotional_valence,
            tags=episode.tags,
        )

        return episode.episode_id

    def retrieve(
        self,
        query: str = "",
        top_k: int = 5,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        recency_bias: Optional[float] = None,
        mood_valence: Optional[float] = None,
    ) -> list[tuple[Episode, float]]:
        """
        Retrieve the most relevant episodes using the Generative Agents formula.

        score = α × recency + β × importance + γ × relevance

        With optional mood-congruent recall: memories whose emotional valence
        matches the current mood get a small boost. This simulates the
        well-documented psychological phenomenon where emotional states
        facilitate recall of mood-congruent memories.

        Args:
            query: The retrieval query (for relevance scoring)
            top_k: How many episodes to return
            category: Optional filter by category
            min_importance: Minimum importance threshold
            recency_bias: Override recency weight (useful for time-sensitive queries)
            mood_valence: Current emotional valence for mood-congruent recall

        Returns:
            List of (episode, score) tuples, sorted by score descending
        """
        now = time.time()
        candidates = self._episodes

        # Apply category filter
        if category:
            candidates = [e for e in candidates if e.category == category]

        # Apply importance filter
        if min_importance > 0:
            candidates = [e for e in candidates if e.importance >= min_importance]

        if not candidates:
            return []

        # Score each candidate
        scored = []
        recency_w = recency_bias if recency_bias is not None else self._recency_weight
        vector_scores: dict[str, float] = {}
        if (
            query
            and self._retrieval_mode in {"embedding", "hybrid"}
            and self._vector_search_fn is not None
        ):
            for episode_id, score in self._vector_search_fn(
                query, max(top_k, self._embedding_top_k)
            ):
                vector_scores[episode_id] = score

        for episode in candidates:
            # Recency score: exponential decay based on minutes elapsed
            minutes_ago = (now - episode.timestamp) / 60.0
            recency_score = self._recency_decay ** minutes_ago

            # Importance score: already normalized to 0-1
            importance_score = episode.importance

            relevance_score = self._compute_relevance(
                query=query,
                episode=episode,
                vector_scores=vector_scores,
            )

            # Composite score
            score = (
                recency_w * recency_score
                + self._importance_weight * importance_score
                + self._relevance_weight * relevance_score
            )

            # Mood-congruent recall bonus
            if mood_valence is not None:
                mood_congruence = 1.0 - abs(mood_valence - episode.emotional_valence) / 2.0
                score += 0.05 * mood_congruence  # Small bonus, not overwhelming

            scored.append((episode, min(1.0, score)))

        # Sort by score descending, return top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_recent(self, n: int = 10) -> list[Episode]:
        """Get the N most recent episodes, regardless of scoring."""
        sorted_eps = sorted(self._episodes, key=lambda e: e.timestamp, reverse=True)
        return sorted_eps[:n]

    def retrieve_by_tag(self, tag: str, top_k: int = 10) -> list[Episode]:
        """Retrieve episodes that contain a specific tag."""
        matches = [e for e in self._episodes if tag.lower() in [t.lower() for t in e.tags]]
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        return matches[:top_k]

    def retrieve_chronological(
        self,
        top_k: int = 10,
        oldest_first: bool = True,
        category: Optional[str] = None,
        landmarks_only: bool = False,
    ) -> list[Episode]:
        """Retrieve episodes sorted by timestamp (oldest or newest first)."""
        candidates = self._episodes
        if category:
            candidates = [e for e in candidates if e.category == category]
        if landmarks_only:
            candidates = [e for e in candidates if getattr(e, 'landmark', False)]
        sorted_eps = sorted(candidates, key=lambda e: e.timestamp, reverse=not oldest_first)
        return sorted_eps[:top_k]

    def get_unconsolidated(self, max_age_hours: Optional[float] = 24.0) -> list[Episode]:
        """Get episodes that haven't been consolidated into semantic memory yet."""
        cutoff = None if max_age_hours is None else (time.time() - (max_age_hours * 3600))
        return [
            e for e in self._episodes
            if not e.consolidated and (cutoff is None or e.timestamp > cutoff)
        ]

    def mark_consolidated(self, episode_ids: list[str]) -> None:
        """Mark episodes as consolidated into semantic memory."""
        id_set = set(episode_ids)
        for episode in self._episodes:
            if episode.episode_id in id_set:
                episode.consolidated = True

    def _compute_keyword_relevance(self, query: str, episode: Episode) -> float:
        """
        Simple keyword-overlap relevance scoring.

        In a production system, this would use embedding cosine similarity.
        This simplified version counts word overlap between query and
        episode content + tags. It's surprisingly effective for basic retrieval.
        """
        if not query:
            return 0.5  # Neutral relevance when no query specified

        query_words = set(query.lower().split())
        content_words = set(episode.content.lower().split())
        tag_words = set(t.lower() for t in episode.tags)
        episode_words = content_words | tag_words

        if not episode_words:
            return 0.0

        overlap = query_words & episode_words
        # Jaccard-like similarity
        return len(overlap) / max(len(query_words), 1)

    def _compute_relevance(
        self,
        query: str,
        episode: Episode,
        vector_scores: dict[str, float],
    ) -> float:
        """Compute relevance using keyword, embedding, or hybrid mode."""
        if self._retrieval_mode == "keyword":
            return self._compute_keyword_relevance(query, episode)

        if self._retrieval_mode == "embedding":
            if not query:
                return 0.5
            if not vector_scores:
                # Graceful degradation when vector search is unavailable/empty.
                return self._compute_keyword_relevance(query, episode)
            return vector_scores.get(episode.episode_id, 0.0)

        # hybrid mode
        keyword_score = self._compute_keyword_relevance(query, episode)
        if not query:
            return keyword_score
        if not vector_scores:
            return keyword_score
        vector_score = vector_scores.get(episode.episode_id, 0.0)
        return (
            self._hybrid_keyword_weight * keyword_score
            + self._hybrid_embedding_weight * vector_score
        )

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Return an episode by id if present in memory."""
        idx = self._episode_index.get(episode_id)
        return self._episodes[idx] if idx is not None else None

    @property
    def count(self) -> int:
        return len(self._episodes)

    def get_emotional_trajectory(self, last_n: int = 20) -> list[dict]:
        """Get the emotional trajectory over the last N episodes."""
        recent = self.retrieve_recent(last_n)
        return [
            {
                "timestamp": e.timestamp,
                "valence": e.emotional_valence,
                "arousal": e.emotional_arousal,
                "category": e.category,
            }
            for e in reversed(recent)  # Chronological order
        ]
