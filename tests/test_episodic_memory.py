"""
Tests for gwenn.memory.episodic — Episodic Memory.

Covers:
- Retrieve scoring (recency, importance, relevance weighting)
- Mood-congruent retrieval bias
- Encoding and basic retrieval
- Category filtering
- Unconsolidated episode retrieval
"""

from __future__ import annotations

import time

import pytest

from gwenn.memory.episodic import Episode, EpisodicMemory


# ---------------------------------------------------------------------------
# Encoding and basic retrieval
# ---------------------------------------------------------------------------

class TestEncodingAndRetrieval:
    """Basic encode / retrieve round-trip behaviour."""

    def test_encode_returns_episode_id(self, episodic_memory):
        ep = Episode(episode_id="ep-1", content="hello world")
        result = episodic_memory.encode(ep)
        assert result == "ep-1"

    def test_count_increases_after_encode(self, episodic_memory):
        assert episodic_memory.count == 0
        episodic_memory.encode(Episode(content="first"))
        assert episodic_memory.count == 1
        episodic_memory.encode(Episode(content="second"))
        assert episodic_memory.count == 2

    def test_retrieve_returns_encoded_episodes(self, episodic_memory):
        ep = Episode(episode_id="ep-1", content="test content", importance=0.5)
        episodic_memory.encode(ep)
        results = episodic_memory.retrieve(top_k=5)
        assert len(results) == 1
        returned_ep, _ = results[0]
        assert returned_ep.episode_id == "ep-1"

    def test_retrieve_recent_ordering(self, populated_episodic_memory):
        recent = populated_episodic_memory.retrieve_recent(n=3)
        assert len(recent) == 3
        # Should be in reverse-chronological order
        for i in range(len(recent) - 1):
            assert recent[i].timestamp >= recent[i + 1].timestamp

    def test_retrieve_by_tag(self, populated_episodic_memory):
        results = populated_episodic_memory.retrieve_by_tag("python")
        assert len(results) >= 1
        assert all("python" in [t.lower() for t in ep.tags] for ep in results)

    def test_retrieve_by_tag_case_insensitive(self, populated_episodic_memory):
        results = populated_episodic_memory.retrieve_by_tag("Python")
        assert len(results) >= 1

    def test_retrieve_empty_memory_returns_empty(self, episodic_memory):
        results = episodic_memory.retrieve(query="anything", top_k=5)
        assert results == []

    def test_episode_to_dict_and_from_dict_round_trip(self):
        ep = Episode(
            episode_id="ep-rt",
            timestamp=1000.0,
            content="round trip",
            category="conversation",
            emotional_valence=0.3,
            emotional_arousal=0.5,
            importance=0.7,
            tags=["a", "b"],
            participants=["alice"],
            outcome="good",
            consolidated=False,
        )
        d = ep.to_dict()
        restored = Episode.from_dict(d)
        assert restored.episode_id == ep.episode_id
        assert restored.content == ep.content
        assert restored.tags == ep.tags
        assert restored.participants == ep.participants
        assert restored.outcome == ep.outcome


# ---------------------------------------------------------------------------
# Retrieve scoring (recency, importance, relevance weighting)
# ---------------------------------------------------------------------------

class TestRetrieveScoring:
    """Ensure the composite score formula is applied correctly."""

    def test_higher_importance_scores_higher(self):
        """An episode with higher importance should score higher, all else equal."""
        now = time.time()
        mem = EpisodicMemory(
            importance_weight=1.0, recency_weight=0.0, relevance_weight=0.0
        )
        low = Episode(episode_id="low", content="low importance", importance=0.2, timestamp=now)
        high = Episode(episode_id="high", content="high importance", importance=0.9, timestamp=now)
        mem.encode(low)
        mem.encode(high)

        results = mem.retrieve(top_k=2)
        assert results[0][0].episode_id == "high"
        assert results[1][0].episode_id == "low"

    def test_more_recent_scores_higher(self):
        """A more recent episode should score higher when recency is the only weight."""
        now = time.time()
        mem = EpisodicMemory(
            importance_weight=0.0, recency_weight=1.0, relevance_weight=0.0
        )
        old = Episode(episode_id="old", content="old episode", importance=0.5, timestamp=now - 3600)
        new = Episode(episode_id="new", content="new episode", importance=0.5, timestamp=now)
        mem.encode(old)
        mem.encode(new)

        results = mem.retrieve(top_k=2)
        assert results[0][0].episode_id == "new"

    def test_relevance_boosts_matching_query(self):
        """Episodes whose content matches the query should rank higher."""
        now = time.time()
        mem = EpisodicMemory(
            importance_weight=0.0, recency_weight=0.0, relevance_weight=1.0
        )
        match = Episode(
            episode_id="match", content="python decorators are useful",
            importance=0.5, timestamp=now,
        )
        no_match = Episode(
            episode_id="nomatch", content="the weather is nice today",
            importance=0.5, timestamp=now,
        )
        mem.encode(match)
        mem.encode(no_match)

        results = mem.retrieve(query="python decorators", top_k=2)
        assert results[0][0].episode_id == "match"

    def test_recency_bias_override(self, populated_episodic_memory):
        """Passing recency_bias should override the default recency weight."""
        results_biased = populated_episodic_memory.retrieve(query="", top_k=5, recency_bias=0.99)
        # With a very high recency bias, the most recent episode should dominate
        most_recent_id = results_biased[0][0].episode_id
        # The most recent episode in the sample set is ep-conv-1 (1 min ago)
        assert most_recent_id == "ep-conv-1"

    def test_min_importance_filter(self, populated_episodic_memory):
        """Episodes below min_importance should be excluded."""
        results = populated_episodic_memory.retrieve(min_importance=0.75, top_k=10)
        for ep, _ in results:
            assert ep.importance >= 0.75


# ---------------------------------------------------------------------------
# Mood-congruent retrieval bias
# ---------------------------------------------------------------------------

class TestMoodCongruentRetrieval:
    """Mood-congruent recall should slightly boost episodes matching mood valence."""

    def test_positive_mood_boosts_positive_episodes(self):
        """With positive mood_valence, positive-valence episodes should get a boost."""
        now = time.time()
        mem = EpisodicMemory(
            importance_weight=0.0, recency_weight=0.0, relevance_weight=0.0
        )
        positive = Episode(
            episode_id="pos", content="happy moment",
            emotional_valence=0.8, importance=0.5, timestamp=now,
        )
        negative = Episode(
            episode_id="neg", content="sad moment",
            emotional_valence=-0.8, importance=0.5, timestamp=now,
        )
        mem.encode(positive)
        mem.encode(negative)

        # With all weights at 0, only mood congruence differs
        results = mem.retrieve(query="", top_k=2, mood_valence=0.8)
        assert results[0][0].episode_id == "pos"

    def test_negative_mood_boosts_negative_episodes(self):
        """With negative mood_valence, negative-valence episodes should get a boost."""
        now = time.time()
        mem = EpisodicMemory(
            importance_weight=0.0, recency_weight=0.0, relevance_weight=0.0
        )
        positive = Episode(
            episode_id="pos", content="happy moment",
            emotional_valence=0.8, importance=0.5, timestamp=now,
        )
        negative = Episode(
            episode_id="neg", content="sad moment",
            emotional_valence=-0.8, importance=0.5, timestamp=now,
        )
        mem.encode(positive)
        mem.encode(negative)

        results = mem.retrieve(query="", top_k=2, mood_valence=-0.8)
        assert results[0][0].episode_id == "neg"

    def test_no_mood_valence_means_no_bias(self):
        """When mood_valence is None, no mood bonus is applied."""
        now = time.time()
        mem = EpisodicMemory(
            importance_weight=0.0, recency_weight=0.0, relevance_weight=0.0
        )
        ep1 = Episode(
            episode_id="a", content="a", emotional_valence=0.9,
            importance=0.5, timestamp=now,
        )
        ep2 = Episode(
            episode_id="b", content="b", emotional_valence=-0.9,
            importance=0.5, timestamp=now,
        )
        mem.encode(ep1)
        mem.encode(ep2)

        results = mem.retrieve(query="", top_k=2, mood_valence=None)
        # With all weights zero and no mood, both should score the same (0.0)
        scores = [score for _, score in results]
        assert scores[0] == pytest.approx(scores[1], abs=1e-9)


# ---------------------------------------------------------------------------
# Category filtering
# ---------------------------------------------------------------------------

class TestCategoryFiltering:
    """Category filter on retrieve should restrict results."""

    def test_filter_conversation(self, populated_episodic_memory):
        results = populated_episodic_memory.retrieve(category="conversation", top_k=10)
        assert len(results) >= 1
        for ep, _ in results:
            assert ep.category == "conversation"

    def test_filter_tool_use(self, populated_episodic_memory):
        results = populated_episodic_memory.retrieve(category="tool_use", top_k=10)
        assert len(results) >= 1
        for ep, _ in results:
            assert ep.category == "tool_use"

    def test_filter_nonexistent_category_returns_empty(self, populated_episodic_memory):
        results = populated_episodic_memory.retrieve(category="nonexistent", top_k=10)
        assert results == []


# ---------------------------------------------------------------------------
# Unconsolidated episode retrieval
# ---------------------------------------------------------------------------

class TestUnconsolidatedEpisodes:
    """get_unconsolidated and mark_consolidated behaviour."""

    def test_new_episodes_are_unconsolidated(self, populated_episodic_memory):
        unconsolidated = populated_episodic_memory.get_unconsolidated(max_age_hours=24.0)
        assert len(unconsolidated) == 5  # all sample episodes are within 24h

    def test_mark_consolidated_removes_from_unconsolidated(self, populated_episodic_memory):
        populated_episodic_memory.mark_consolidated(["ep-conv-1", "ep-tool-1"])
        unconsolidated = populated_episodic_memory.get_unconsolidated(max_age_hours=24.0)
        ids = [e.episode_id for e in unconsolidated]
        assert "ep-conv-1" not in ids
        assert "ep-tool-1" not in ids
        assert len(unconsolidated) == 3

    def test_get_unconsolidated_respects_max_age(self, episodic_memory):
        old_ep = Episode(
            episode_id="old",
            content="very old",
            timestamp=time.time() - 48 * 3600,  # 48 hours ago
        )
        new_ep = Episode(
            episode_id="new",
            content="recent",
            timestamp=time.time() - 60,  # 1 minute ago
        )
        episodic_memory.encode(old_ep)
        episodic_memory.encode(new_ep)

        unconsolidated = episodic_memory.get_unconsolidated(max_age_hours=24.0)
        ids = [e.episode_id for e in unconsolidated]
        assert "new" in ids
        assert "old" not in ids


# ---------------------------------------------------------------------------
# Keyword relevance helper
# ---------------------------------------------------------------------------

class TestKeywordRelevance:
    """The simplified keyword-overlap relevance function."""

    def test_empty_query_returns_neutral(self, episodic_memory):
        ep = Episode(content="some content")
        score = episodic_memory._compute_keyword_relevance("", ep)
        assert score == 0.5

    def test_full_overlap_returns_one(self, episodic_memory):
        ep = Episode(content="python decorators")
        score = episodic_memory._compute_keyword_relevance("python decorators", ep)
        assert score == pytest.approx(1.0)

    def test_partial_overlap(self, episodic_memory):
        ep = Episode(content="python decorators are great")
        score = episodic_memory._compute_keyword_relevance("python java", ep)
        # 1 out of 2 query words match
        assert score == pytest.approx(0.5)

    def test_no_overlap_returns_zero(self, episodic_memory):
        ep = Episode(content="python decorators")
        score = episodic_memory._compute_keyword_relevance("java spring", ep)
        assert score == pytest.approx(0.0)

    def test_tags_contribute_to_relevance(self, episodic_memory):
        ep = Episode(content="something", tags=["python", "decorators"])
        score = episodic_memory._compute_keyword_relevance("python", ep)
        assert score > 0.0


# ---------------------------------------------------------------------------
# Emotional trajectory
# ---------------------------------------------------------------------------

class TestEmotionalTrajectory:
    """get_emotional_trajectory returns chronological emotion data."""

    def test_trajectory_length(self, populated_episodic_memory):
        trajectory = populated_episodic_memory.get_emotional_trajectory(last_n=3)
        assert len(trajectory) == 3

    def test_trajectory_chronological_order(self, populated_episodic_memory):
        trajectory = populated_episodic_memory.get_emotional_trajectory(last_n=5)
        for i in range(len(trajectory) - 1):
            assert trajectory[i]["timestamp"] <= trajectory[i + 1]["timestamp"]
