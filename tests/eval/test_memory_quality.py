"""
Memory Retrieval Quality Benchmarks — Evaluation Suite.

Measures the quality of Gwenn's episodic memory retrieval across several
dimensions.  All tests are deterministic (no LLM / API calls) and use
seeded fixtures with known ground-truth tags so that Recall@k, ranking,
and bias behaviour can be verified precisely.

Covered benchmarks:
  - Basic Recall@k: correct episodes retrieved for keyword queries
  - Mood-congruent retrieval bias: positive mood boosts positive memories
  - Importance weighting: high-importance episodes rank higher
  - Recency bias: recent episodes rank higher when recency is weighted
  - Semantic memory query quality
  - Working memory salience gating
"""

from __future__ import annotations

import time

import pytest

from gwenn.memory.episodic import Episode, EpisodicMemory
from gwenn.memory.semantic import SemanticMemory
from gwenn.memory.working import WorkingMemoryItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _episode_ids(results: list[tuple[Episode, float]]) -> list[str]:
    """Extract episode IDs from a retrieve() result list."""
    return [ep.episode_id for ep, _ in results]


def _recall_at_k(results: list[tuple[Episode, float]], expected_ids: set[str], k: int) -> float:
    """
    Compute Recall@k: what fraction of expected IDs appear in the top-k results.

    Returns a float in [0.0, 1.0].
    """
    top_k_ids = set(_episode_ids(results[:k]))
    if not expected_ids:
        return 1.0
    return len(top_k_ids & expected_ids) / len(expected_ids)


# ===========================================================================
# 1. Basic Recall@k — correct episodes retrieved for keyword queries
# ===========================================================================

class TestBasicRecallAtK:
    """Verify that keyword queries surface the right episodes."""

    def test_recall_python_query(self, seeded_episodic_memory):
        """Querying 'python' should retrieve both Python-tagged episodes."""
        results = seeded_episodic_memory.retrieve(query="python", top_k=5)
        expected = {"eval-python-decorators", "eval-old-python-async"}
        recall = _recall_at_k(results, expected, k=5)
        assert recall == 1.0, (
            f"Expected both Python episodes in top 5, got {_episode_ids(results)}"
        )

    def test_recall_rust_query(self, seeded_episodic_memory):
        """Querying 'rust ownership' should retrieve the Rust episode."""
        results = seeded_episodic_memory.retrieve(query="rust ownership", top_k=3)
        expected = {"eval-rust-ownership"}
        recall = _recall_at_k(results, expected, k=3)
        assert recall == 1.0

    def test_recall_security_query(self, seeded_episodic_memory):
        """Querying 'security vulnerability' should find the critical insight."""
        results = seeded_episodic_memory.retrieve(query="security vulnerability", top_k=3)
        expected = {"eval-critical-insight"}
        recall = _recall_at_k(results, expected, k=3)
        assert recall == 1.0

    def test_recall_nonexistent_query(self, seeded_episodic_memory):
        """Querying for something not in any episode should still return results
        (scored by recency/importance), but no specific episode is 'expected'."""
        results = seeded_episodic_memory.retrieve(query="quantum entanglement", top_k=3)
        # Should not crash; results come from recency/importance scoring
        assert isinstance(results, list)

    def test_recall_at_k_partial(self, seeded_episodic_memory):
        """With top_k=1, we may only get one of two matching episodes."""
        results = seeded_episodic_memory.retrieve(query="python", top_k=1)
        expected = {"eval-python-decorators", "eval-old-python-async"}
        recall = _recall_at_k(results, expected, k=1)
        # At least one of the two should be retrieved
        assert recall >= 0.5

    def test_tag_based_retrieval_quality(self, seeded_episodic_memory):
        """retrieve_by_tag should return all episodes with the queried tag."""
        results = seeded_episodic_memory.retrieve_by_tag("python")
        ids = [ep.episode_id for ep in results]
        assert "eval-python-decorators" in ids
        assert "eval-old-python-async" in ids

    def test_category_filter_narrows_results(self, seeded_episodic_memory):
        """Category filter should exclude non-matching episodes."""
        results = seeded_episodic_memory.retrieve(
            query="python", top_k=10, category="conversation"
        )
        for ep, _ in results:
            assert ep.category == "conversation"
        ids = _episode_ids(results)
        # The emotion_shift and insight episodes should be excluded
        assert "eval-sad-failure" not in ids
        assert "eval-critical-insight" not in ids


# ===========================================================================
# 2. Mood-congruent retrieval bias
# ===========================================================================

class TestMoodCongruentBias:
    """
    Positive mood should boost positive-valence memories; negative mood
    should boost negative-valence memories. This tests the mood-congruent
    recall mechanism from the Generative Agents formula extension.
    """

    def test_positive_mood_boosts_positive_memories(self, mood_only_memory):
        """With all weights zeroed, only mood congruence differentiates.
        Positive mood_valence should rank positive-valence episodes higher."""
        now = time.time()
        positive_ep = Episode(
            episode_id="pos",
            content="joyful creative session",
            emotional_valence=0.9,
            importance=0.5,
            timestamp=now,
        )
        negative_ep = Episode(
            episode_id="neg",
            content="frustrating tool failure",
            emotional_valence=-0.8,
            importance=0.5,
            timestamp=now,
        )
        neutral_ep = Episode(
            episode_id="neu",
            content="routine status check",
            emotional_valence=0.0,
            importance=0.5,
            timestamp=now,
        )
        mood_only_memory.encode(positive_ep)
        mood_only_memory.encode(negative_ep)
        mood_only_memory.encode(neutral_ep)

        results = mood_only_memory.retrieve(query="", top_k=3, mood_valence=0.9)
        ids = _episode_ids(results)
        assert ids[0] == "pos", f"Expected positive episode first, got {ids}"

    def test_negative_mood_boosts_negative_memories(self, mood_only_memory):
        """Negative mood_valence should rank negative-valence episodes higher."""
        now = time.time()
        positive_ep = Episode(
            episode_id="pos",
            content="happy conversation",
            emotional_valence=0.9,
            importance=0.5,
            timestamp=now,
        )
        negative_ep = Episode(
            episode_id="neg",
            content="sad difficult moment",
            emotional_valence=-0.8,
            importance=0.5,
            timestamp=now,
        )
        mood_only_memory.encode(positive_ep)
        mood_only_memory.encode(negative_ep)

        results = mood_only_memory.retrieve(query="", top_k=2, mood_valence=-0.8)
        assert results[0][0].episode_id == "neg"

    def test_no_mood_means_no_bias(self, mood_only_memory):
        """Without mood_valence, all episodes with equal attributes score the same."""
        now = time.time()
        ep1 = Episode(
            episode_id="a", content="happy", emotional_valence=0.9,
            importance=0.5, timestamp=now,
        )
        ep2 = Episode(
            episode_id="b", content="sad", emotional_valence=-0.9,
            importance=0.5, timestamp=now,
        )
        mood_only_memory.encode(ep1)
        mood_only_memory.encode(ep2)

        results = mood_only_memory.retrieve(query="", top_k=2, mood_valence=None)
        scores = [score for _, score in results]
        assert scores[0] == pytest.approx(scores[1], abs=1e-9)

    def test_mood_bias_is_small_not_overwhelming(self, seeded_episodic_memory):
        """With default weights, mood bias should nudge ranking but not dominate.
        A high-importance episode should still beat a mood-congruent low-importance one."""
        # eval-critical-insight: importance=1.0, valence=0.1
        # eval-trivial-log: importance=0.1, valence=0.0
        # Even with mood_valence=0.0 (matching trivial-log better), importance should win.
        results = seeded_episodic_memory.retrieve(
            query="", top_k=8, mood_valence=0.0
        )
        ids = _episode_ids(results)
        # Critical insight should rank above trivial log
        critical_idx = ids.index("eval-critical-insight")
        trivial_idx = ids.index("eval-trivial-log")
        assert critical_idx < trivial_idx, (
            "High-importance episode should still rank above low-importance "
            "even when mood slightly favours the low-importance one."
        )


# ===========================================================================
# 3. Importance weighting
# ===========================================================================

class TestImportanceWeighting:
    """High importance episodes should score higher when importance is weighted."""

    def test_importance_ranking_isolated(self, importance_only_memory):
        """With only importance weight, episodes should rank by importance."""
        now = time.time()
        episodes = [
            Episode(episode_id="low", content="low", importance=0.1, timestamp=now),
            Episode(episode_id="mid", content="mid", importance=0.5, timestamp=now),
            Episode(episode_id="high", content="high", importance=0.9, timestamp=now),
            Episode(episode_id="max", content="max", importance=1.0, timestamp=now),
        ]
        for ep in episodes:
            importance_only_memory.encode(ep)

        results = importance_only_memory.retrieve(top_k=4)
        ids = _episode_ids(results)
        assert ids == ["max", "high", "mid", "low"], f"Expected importance-sorted, got {ids}"

    def test_importance_scores_are_monotonic(self, importance_only_memory):
        """Scores should decrease monotonically with decreasing importance."""
        now = time.time()
        for imp in [0.2, 0.4, 0.6, 0.8]:
            importance_only_memory.encode(
                Episode(content=f"imp-{imp}", importance=imp, timestamp=now)
            )

        results = importance_only_memory.retrieve(top_k=4)
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_min_importance_filter_excludes_low(self, seeded_episodic_memory):
        """min_importance should exclude episodes below the threshold."""
        results = seeded_episodic_memory.retrieve(min_importance=0.7, top_k=10)
        for ep, _ in results:
            assert ep.importance >= 0.7
        ids = _episode_ids(results)
        # eval-trivial-log (0.1) and eval-sad-failure (0.5) should be excluded
        assert "eval-trivial-log" not in ids
        assert "eval-sad-failure" not in ids

    def test_importance_dominates_with_high_weight(self):
        """When importance_weight is much larger than others, importance dominates."""
        mem = EpisodicMemory(
            importance_weight=0.9, recency_weight=0.05, relevance_weight=0.05
        )
        now = time.time()
        # Old but important
        mem.encode(Episode(
            episode_id="old-important",
            content="critical finding",
            importance=0.95,
            timestamp=now - 3600,  # 1 hour ago
        ))
        # Recent but trivial
        mem.encode(Episode(
            episode_id="new-trivial",
            content="routine check",
            importance=0.1,
            timestamp=now,
        ))

        results = mem.retrieve(top_k=2)
        assert results[0][0].episode_id == "old-important"


# ===========================================================================
# 4. Recency bias
# ===========================================================================

class TestRecencyBias:
    """Recent episodes should score higher when recency is weighted."""

    def test_recency_ranking_isolated(self, recency_only_memory):
        """With only recency weight, episodes should rank by timestamp (newest first)."""
        now = time.time()
        episodes = [
            Episode(episode_id="ancient", content="ancient", importance=0.5, timestamp=now - 7200),
            Episode(episode_id="old", content="old", importance=0.5, timestamp=now - 3600),
            Episode(episode_id="recent", content="recent", importance=0.5, timestamp=now - 60),
            Episode(episode_id="fresh", content="fresh", importance=0.5, timestamp=now),
        ]
        for ep in episodes:
            recency_only_memory.encode(ep)

        results = recency_only_memory.retrieve(top_k=4)
        ids = _episode_ids(results)
        assert ids == ["fresh", "recent", "old", "ancient"], f"Expected recency-sorted, got {ids}"

    def test_recency_scores_decay_with_age(self, recency_only_memory):
        """Scores should be strictly decreasing for increasingly old episodes."""
        now = time.time()
        ages_minutes = [0, 5, 15, 60, 180]
        for i, age in enumerate(ages_minutes):
            recency_only_memory.encode(
                Episode(
                    episode_id=f"ep-{age}m",
                    content=f"episode {age} minutes ago",
                    importance=0.5,
                    timestamp=now - age * 60,
                )
            )

        results = recency_only_memory.retrieve(top_k=len(ages_minutes))
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], (
                f"Score at position {i} ({scores[i]}) should exceed "
                f"score at position {i+1} ({scores[i+1]})"
            )

    def test_recency_bias_override_parameter(self, seeded_episodic_memory):
        """Passing a high recency_bias should give recent episodes a strong boost."""
        results_low = seeded_episodic_memory.retrieve(query="", top_k=5, recency_bias=0.0)
        results_high = seeded_episodic_memory.retrieve(query="", top_k=5, recency_bias=0.99)
        # With high recency bias, the most recent episode should score higher
        # than it does with zero recency bias
        recent_id = "eval-python-decorators"
        score_low = next((s for ep, s in results_low if ep.episode_id == recent_id), 0)
        score_high = next((s for ep, s in results_high if ep.episode_id == recent_id), 0)
        assert score_high > score_low

    def test_zero_recency_weight_ignores_age(self):
        """With recency_weight=0, episode age should not affect ranking."""
        mem = EpisodicMemory(
            importance_weight=1.0, recency_weight=0.0, relevance_weight=0.0
        )
        now = time.time()
        old_important = Episode(
            episode_id="old-imp",
            content="old important",
            importance=0.9,
            timestamp=now - 86400,  # 24 hours ago
        )
        new_trivial = Episode(
            episode_id="new-triv",
            content="new trivial",
            importance=0.2,
            timestamp=now,
        )
        mem.encode(old_important)
        mem.encode(new_trivial)

        results = mem.retrieve(top_k=2)
        assert results[0][0].episode_id == "old-imp"


# ===========================================================================
# 5. Semantic memory query quality
# ===========================================================================

class TestSemanticMemoryQuality:
    """Verify that semantic memory retrieval returns relevant knowledge nodes."""

    def test_query_returns_matching_nodes(self, seeded_semantic_memory):
        """Querying 'python' should return the python knowledge node."""
        results = seeded_semantic_memory.query("python")
        labels = [node.label for node in results]
        assert "python" in labels

    def test_query_respects_min_confidence(self, seeded_semantic_memory):
        """Nodes below min_confidence should be excluded."""
        results = seeded_semantic_memory.query("python", min_confidence=0.95)
        # python has confidence 0.9, should be excluded at 0.95 threshold
        labels = [node.label for node in results]
        assert "python" not in labels

    def test_query_category_filter(self, seeded_semantic_memory):
        """Category filter should restrict results to matching categories."""
        results = seeded_semantic_memory.query("python", category="person")
        # python is a concept, not a person
        labels = [node.label for node in results]
        assert "python" not in labels

    def test_relationship_retrieval(self, seeded_semantic_memory):
        """get_relationships should return connected edges and target nodes."""
        rels = seeded_semantic_memory.get_relationships("alice")
        assert len(rels) >= 1
        edge, target = rels[0]
        assert edge.relationship == "prefers"
        assert target.label == "python"

    def test_knowledge_reinforcement_increases_confidence(self, semantic_memory):
        """Storing the same label again should reinforce, not duplicate."""
        node1 = semantic_memory.store_knowledge("python", "A language", confidence=0.5)
        initial_confidence = node1.confidence
        node2 = semantic_memory.store_knowledge(
            "python", "A popular language", source_episode_id="ep-1"
        )
        assert node2.confidence > initial_confidence
        assert semantic_memory.node_count == 1  # No duplication

    def test_context_generation(self, seeded_semantic_memory):
        """get_context_for should produce a non-empty prompt fragment."""
        context = seeded_semantic_memory.get_context_for("python")
        assert "python" in context.lower()
        assert len(context) > 0

    def test_embedding_mode_uses_vector_scores(self):
        semantic = SemanticMemory(retrieval_mode="embedding")
        a = semantic.store_knowledge("alpha", "alpha content", confidence=0.9)
        b = semantic.store_knowledge("beta", "beta content", confidence=0.9)
        semantic.set_vector_search(lambda _q, _k: [(b.node_id, 0.9), (a.node_id, 0.1)])

        results = semantic.query("anything", top_k=2)
        assert results[0].label == "beta"

    def test_hybrid_mode_blends_keyword_and_vector(self):
        semantic = SemanticMemory(
            retrieval_mode="hybrid",
            hybrid_keyword_weight=0.2,
            hybrid_embedding_weight=0.8,
        )
        kw = semantic.store_knowledge("python", "python decorators", confidence=0.8)
        vec = semantic.store_knowledge("vectors", "unrelated text", confidence=0.8)
        semantic.set_vector_search(lambda _q, _k: [(vec.node_id, 1.0), (kw.node_id, 0.0)])

        results = semantic.query("python decorators", top_k=2)
        assert results[0].label == "vectors"


# ===========================================================================
# 6. Working memory salience gating quality
# ===========================================================================

class TestWorkingMemoryGating:
    """Verify that working memory correctly gates items by salience."""

    def test_capacity_limit_enforced(self, working_memory):
        """Adding more items than max_slots should evict least salient."""
        for i in range(10):
            item = WorkingMemoryItem(
                item_id=f"item-{i}",
                content=f"item {i}",
                salience=i * 0.1,  # salience increases with index
            )
            working_memory.attend(item)

        assert working_memory.slot_count == 7
        # The 7 most salient items (indices 3-9) should remain
        for i in range(3, 10):
            assert working_memory.get(f"item-{i}") is not None

    def test_low_salience_rejected(self, working_memory):
        """A new item with lower salience than all current items is rejected."""
        # Fill with high-salience items
        for i in range(7):
            working_memory.attend(WorkingMemoryItem(
                item_id=f"high-{i}", content=f"high {i}", salience=0.9,
            ))

        # Try to add a low-salience item
        low_item = WorkingMemoryItem(
            item_id="low", content="low salience", salience=0.1,
        )
        evicted = working_memory.attend(low_item)
        assert evicted is None  # No eviction happened
        assert working_memory.get("low") is None  # Item was rejected

    def test_high_salience_displaces_low(self, working_memory):
        """A new item with higher salience should displace the least salient."""
        # Fill with moderate salience
        for i in range(7):
            working_memory.attend(WorkingMemoryItem(
                item_id=f"mod-{i}", content=f"moderate {i}", salience=0.5,
            ))

        # Add a very high salience item
        high_item = WorkingMemoryItem(
            item_id="urgent", content="urgent task", salience=0.95,
        )
        evicted = working_memory.attend(high_item)
        assert evicted is not None
        assert working_memory.get("urgent") is not None

    def test_context_generation_includes_all_items(self, working_memory):
        """get_context() should mention all items currently in working memory."""
        for i in range(3):
            working_memory.attend(WorkingMemoryItem(
                item_id=f"ctx-{i}", content=f"context item {i}", salience=0.5,
            ))

        context = working_memory.get_context()
        assert "context item 0" in context
        assert "context item 1" in context
        assert "context item 2" in context

    def test_utilization_tracking(self, working_memory):
        """utilization should reflect current slot usage."""
        assert working_memory.utilization == pytest.approx(0.0)
        working_memory.attend(WorkingMemoryItem(
            item_id="single", content="one item", salience=0.5,
        ))
        assert working_memory.utilization == pytest.approx(1.0 / 7.0)


# ===========================================================================
# 7. Cross-system integration quality
# ===========================================================================

class TestCrossSystemQuality:
    """
    Integration tests verifying that different memory subsystems work
    together coherently. These are higher-level quality checks.
    """

    def test_episodic_to_semantic_provenance(
        self, seeded_episodic_memory, semantic_memory
    ):
        """Knowledge stored with an episode_id should trace back to episodic memory."""
        # Simulate consolidation: extract knowledge from an episode
        ep = seeded_episodic_memory.retrieve_by_tag("python")[0]
        node = semantic_memory.store_knowledge(
            label="python_decorators",
            content="Users frequently ask about Python decorators",
            source_episode_id=ep.episode_id,
        )
        # Verify provenance
        provenance = semantic_memory.verify_provenance(
            node.node_id, seeded_episodic_memory
        )
        assert provenance["supported"] is True
        assert provenance["found_count"] == 1
        assert len(provenance["missing"]) == 0

    def test_missing_provenance_detected(self, semantic_memory, episodic_memory):
        """A knowledge node referencing a non-existent episode should be flagged."""
        node = semantic_memory.store_knowledge(
            label="phantom",
            content="Knowledge from a deleted episode",
            source_episode_id="ep-does-not-exist",
        )
        provenance = semantic_memory.verify_provenance(
            node.node_id, episodic_memory
        )
        assert provenance["supported"] is False
        assert provenance["found_count"] == 0
        assert "ep-does-not-exist" in provenance["missing"]

    def test_emotional_trajectory_consistency(self, seeded_episodic_memory):
        """Emotional trajectory should be in chronological order with valid values."""
        trajectory = seeded_episodic_memory.get_emotional_trajectory(last_n=8)
        for i in range(len(trajectory) - 1):
            assert trajectory[i]["timestamp"] <= trajectory[i + 1]["timestamp"]
        for point in trajectory:
            assert -1.0 <= point["valence"] <= 1.0
            assert 0.0 <= point["arousal"] <= 1.0
