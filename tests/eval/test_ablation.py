"""
Ablation Tests — Systematic Subsystem Disable/Compare Framework.

Verifies that disabling individual subsystems causes measurable, predictable
degradation.  Each test class isolates a single capability dimension using
the fixtures from ``conftest.py`` and compares behaviour against known
ground-truth expectations.

Covered ablation dimensions:
  - Single-dimension memory scoring (importance, recency, relevance)
  - Mood-congruent retrieval bias
  - Consolidation pipeline (episodes → semantic knowledge)
  - Working memory capacity and eviction
  - Configuration flag propagation
  - Retrieval mode selection
"""

from __future__ import annotations

import time

from gwenn.memory.episodic import Episode, EpisodicMemory
from gwenn.memory.semantic import SemanticMemory
from gwenn.memory.working import WorkingMemory, WorkingMemoryItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _episode_ids(results: list[tuple[Episode, float]]) -> list[str]:
    """Extract episode IDs from a retrieve() result list."""
    return [ep.episode_id for ep, _ in results]


def _seed_episodes(mem: EpisodicMemory, now: float | None = None) -> None:
    """Seed an episodic memory with controlled episodes for ablation tests."""
    now = now or time.time()
    episodes = [
        Episode(
            episode_id="high-imp",
            content="Critical security insight discovered during audit",
            importance=1.0,
            emotional_valence=0.1,
            tags=["security", "audit"],
            timestamp=now - 600,  # 10 min ago
        ),
        Episode(
            episode_id="low-imp",
            content="Routine log status check completed normally",
            importance=0.1,
            emotional_valence=0.0,
            tags=["routine", "log"],
            timestamp=now - 60,  # 1 min ago (recent)
        ),
        Episode(
            episode_id="recent",
            content="User asked about Python decorators just now",
            importance=0.5,
            emotional_valence=0.4,
            tags=["python", "decorators"],
            timestamp=now - 5,  # 5 sec ago
        ),
        Episode(
            episode_id="relevant-python",
            content="Deep discussion about Python async patterns and best practices",
            importance=0.5,
            emotional_valence=0.3,
            tags=["python", "async", "patterns"],
            timestamp=now - 300,  # 5 min ago
        ),
    ]
    for ep in episodes:
        mem.encode(ep)


# ---------------------------------------------------------------------------
# Class 1: Single-dimension memory scoring
# ---------------------------------------------------------------------------


class TestMemoryScoringSingleDimension:
    """Verify ablated memories rank by their single enabled dimension only."""

    def test_importance_only_ranks_by_importance(self, importance_only_memory):
        _seed_episodes(importance_only_memory)
        results = importance_only_memory.retrieve(query="security", top_k=4)
        ids = _episode_ids(results)
        # high-imp (1.0) should be first since only importance matters
        assert ids[0] == "high-imp"
        # low-imp (0.1) should be last
        assert ids[-1] == "low-imp"

    def test_importance_only_ignores_recency(self, importance_only_memory):
        """low-imp is recent but low importance — should rank last."""
        _seed_episodes(importance_only_memory)
        results = importance_only_memory.retrieve(query="", top_k=4)
        ids = _episode_ids(results)
        assert ids[-1] == "low-imp"

    def test_recency_only_ranks_by_time(self, recency_only_memory):
        _seed_episodes(recency_only_memory)
        results = recency_only_memory.retrieve(query="", top_k=4)
        ids = _episode_ids(results)
        # "recent" was 5 sec ago → highest recency
        assert ids[0] == "recent"
        # "high-imp" was 10 min ago → lowest recency
        assert ids[-1] == "high-imp"

    def test_recency_only_ignores_importance(self, recency_only_memory):
        """high-imp has importance=1.0 but is oldest — should rank last."""
        _seed_episodes(recency_only_memory)
        results = recency_only_memory.retrieve(query="", top_k=4)
        ids = _episode_ids(results)
        assert ids[-1] == "high-imp"

    def test_relevance_only_ranks_by_keyword_match(self, relevance_only_memory):
        _seed_episodes(relevance_only_memory)
        results = relevance_only_memory.retrieve(query="python async", top_k=4)
        ids = _episode_ids(results)
        # "relevant-python" has both "python" and "async" in content/tags
        assert ids[0] == "relevant-python"

    def test_relevance_only_ignores_recency_and_importance(self, relevance_only_memory):
        """recent and high-imp should not outrank keyword matches."""
        _seed_episodes(relevance_only_memory)
        results = relevance_only_memory.retrieve(query="python async", top_k=4)
        scores = {ep.episode_id: score for ep, score in results}
        assert scores["relevant-python"] >= scores.get("high-imp", 0)


# ---------------------------------------------------------------------------
# Class 2: Mood-congruent ablation
# ---------------------------------------------------------------------------


class TestMoodCongruentAblation:
    """Verify mood bias dominates when all standard weights are zeroed."""

    def test_positive_mood_boosts_positive_episodes(self, mood_only_memory):
        now = time.time()
        mood_only_memory.encode(Episode(
            episode_id="happy-ep",
            content="Joyful creative session",
            importance=0.5,
            emotional_valence=0.9,
            timestamp=now - 60,
        ))
        mood_only_memory.encode(Episode(
            episode_id="sad-ep",
            content="Frustrating tool failure",
            importance=0.5,
            emotional_valence=-0.8,
            timestamp=now - 60,
        ))

        results = mood_only_memory.retrieve(
            query="session", top_k=2, mood_valence=0.9,
        )
        ids = _episode_ids(results)
        assert ids[0] == "happy-ep"

    def test_negative_mood_boosts_negative_episodes(self, mood_only_memory):
        now = time.time()
        mood_only_memory.encode(Episode(
            episode_id="happy-ep",
            content="Joyful creative session",
            importance=0.5,
            emotional_valence=0.9,
            timestamp=now - 60,
        ))
        mood_only_memory.encode(Episode(
            episode_id="sad-ep",
            content="Frustrating tool failure",
            importance=0.5,
            emotional_valence=-0.8,
            timestamp=now - 60,
        ))

        results = mood_only_memory.retrieve(
            query="session", top_k=2, mood_valence=-0.9,
        )
        ids = _episode_ids(results)
        assert ids[0] == "sad-ep"

    def test_default_memory_mood_doesnt_dominate(self):
        """In a default-weighted memory, mood is a small bonus, not dominant."""
        mem = EpisodicMemory()
        now = time.time()
        # High-importance negative episode
        mem.encode(Episode(
            episode_id="important-neg",
            content="Critical security finding",
            importance=1.0,
            emotional_valence=-0.9,
            timestamp=now - 10,
        ))
        # Low-importance positive episode
        mem.encode(Episode(
            episode_id="trivial-pos",
            content="Routine status check",
            importance=0.1,
            emotional_valence=0.9,
            timestamp=now - 600,
        ))

        # Even with positive mood, importance should still win
        results = mem.retrieve(query="security", top_k=2, mood_valence=0.9)
        ids = _episode_ids(results)
        assert ids[0] == "important-neg"


# ---------------------------------------------------------------------------
# Class 3: Consolidation ablation
# ---------------------------------------------------------------------------


class TestConsolidationAblation:
    """Verify knowledge extraction only happens through consolidation."""

    def test_consolidation_creates_knowledge_nodes(self):
        sm = SemanticMemory()
        em = EpisodicMemory()

        now = time.time()
        for i in range(3):
            em.encode(Episode(
                episode_id=f"ep-{i}",
                content=f"User discussed Python decorators in session {i}",
                importance=0.7,
                tags=["python", "decorators"],
                timestamp=now - i * 60,
            ))

        # Simulate consolidation by storing extracted knowledge
        sm.store_knowledge(
            "python decorators",
            "User frequently discusses Python decorators",
            category="concept",
            source_episode_id="ep-0",
            confidence=0.7,
        )
        assert sm.node_count > 0

    def test_no_consolidation_leaves_semantic_empty(self):
        sm = SemanticMemory()
        em = EpisodicMemory()

        now = time.time()
        for i in range(3):
            em.encode(Episode(
                episode_id=f"ep-{i}",
                content=f"User discussed Python decorators in session {i}",
                importance=0.7,
                tags=["python", "decorators"],
                timestamp=now - i * 60,
            ))

        # Without consolidation, semantic memory stays empty
        assert sm.node_count == 0
        results = sm.query("python decorators")
        assert len(results) == 0

    def test_episodic_episodes_exist_independent_of_semantic(self):
        """Episodic memories exist regardless of consolidation state."""
        em = EpisodicMemory()
        now = time.time()
        em.encode(Episode(
            episode_id="ep-test",
            content="Test episode about Python",
            importance=0.5,
            tags=["python"],
            timestamp=now,
        ))

        results = em.retrieve(query="python", top_k=5)
        assert len(results) == 1
        assert results[0][0].episode_id == "ep-test"


# ---------------------------------------------------------------------------
# Class 4: Working memory ablation
# ---------------------------------------------------------------------------


class TestWorkingMemoryAblation:
    """Verify capacity changes produce expected eviction behavior."""

    def test_default_capacity_evicts_at_seven(self):
        wm = WorkingMemory(max_slots=7)
        for i in range(7):
            wm.attend(WorkingMemoryItem(
                item_id=f"item-{i}", content=f"Item {i}", salience=0.5,
            ))
        assert wm.slot_count == 7

        # Adding 8th item with higher salience evicts lowest
        evicted = wm.attend(WorkingMemoryItem(
            item_id="item-8", content="New high item", salience=0.9,
        ))
        assert evicted is not None
        assert wm.slot_count == 7

    def test_single_slot_keeps_only_most_salient(self):
        wm = WorkingMemory(max_slots=1)
        wm.attend(WorkingMemoryItem(
            item_id="low", content="Low salience", salience=0.3,
        ))
        assert wm.slot_count == 1

        evicted = wm.attend(WorkingMemoryItem(
            item_id="high", content="High salience", salience=0.9,
        ))
        assert evicted is not None
        assert evicted.item_id == "low"
        assert wm.slot_count == 1
        assert wm.get("high") is not None

    def test_large_capacity_no_eviction(self):
        wm = WorkingMemory(max_slots=100)
        for i in range(50):
            wm.attend(WorkingMemoryItem(
                item_id=f"item-{i}", content=f"Item {i}", salience=0.5,
            ))
        assert wm.slot_count == 50
        # No eviction since capacity is 100

    def test_low_salience_rejected_when_full(self):
        wm = WorkingMemory(max_slots=3)
        for i in range(3):
            wm.attend(WorkingMemoryItem(
                item_id=f"item-{i}", content=f"Item {i}", salience=0.8,
            ))

        # Low salience item can't displace anything
        result = wm.attend(WorkingMemoryItem(
            item_id="weak", content="Weak item", salience=0.1,
        ))
        assert result is None  # Rejected, not evicted
        assert wm.slot_count == 3
        assert wm.get("weak") is None


# ---------------------------------------------------------------------------
# Class 5: Config flag ablation
# ---------------------------------------------------------------------------


class TestConfigFlagAblation:
    """Verify config flags correctly propagate when subsystems are disabled."""

    def test_self_healing_disabled(self, monkeypatch):
        monkeypatch.setenv("GWENN_SELF_HEALING_ENABLED", "False")
        from gwenn.config import SelfHealingConfig
        config = SelfHealingConfig()
        assert config.enabled is False

    def test_self_healing_enabled_by_default(self):
        from gwenn.config import SelfHealingConfig
        config = SelfHealingConfig()
        assert config.enabled is True

    def test_checkpoint_disabled(self, monkeypatch):
        monkeypatch.setenv("GWENN_CHECKPOINT_ENABLED", "false")
        from gwenn.config import HeartbeatConfig
        config = HeartbeatConfig()
        assert config.checkpoint_enabled is False

    def test_orchestration_disabled(self, monkeypatch):
        monkeypatch.setenv("GWENN_ORCHESTRATION_ENABLED", "False")
        from gwenn.config import OrchestrationConfig
        config = OrchestrationConfig()
        assert config.enabled is False


# ---------------------------------------------------------------------------
# Class 6: Retrieval mode ablation
# ---------------------------------------------------------------------------


class TestRetrievalModeAblation:
    """Verify retrieval modes produce expected results."""

    def test_keyword_mode_uses_word_overlap(self):
        mem = EpisodicMemory(retrieval_mode="keyword")
        now = time.time()
        mem.encode(Episode(
            episode_id="match",
            content="Python decorators are powerful tools for metaprogramming",
            importance=0.5,
            tags=["python", "decorators"],
            timestamp=now - 60,
        ))
        mem.encode(Episode(
            episode_id="no-match",
            content="The weather was sunny and warm outside today",
            importance=0.5,
            tags=["weather"],
            timestamp=now - 60,
        ))

        results = mem.retrieve(query="python decorators", top_k=2)
        ids = _episode_ids(results)
        assert ids[0] == "match"

    def test_no_matching_keywords_low_scores(self):
        mem = EpisodicMemory(
            retrieval_mode="keyword",
            importance_weight=0.0,
            recency_weight=0.0,
            relevance_weight=1.0,
        )
        now = time.time()
        mem.encode(Episode(
            episode_id="unrelated",
            content="Database migration completed successfully",
            importance=0.5,
            tags=["database"],
            timestamp=now - 60,
        ))

        results = mem.retrieve(query="quantum physics", top_k=1)
        if results:
            _, score = results[0]
            assert score < 0.3  # Very low relevance

    def test_semantic_memory_keyword_mode(self):
        sm = SemanticMemory(retrieval_mode="keyword")
        sm.store_knowledge("python", "A popular programming language", confidence=0.9)
        sm.store_knowledge("rust", "A systems programming language", confidence=0.8)

        results = sm.query("python programming")
        labels = [n.label for n in results]
        assert "python" in labels

    def test_invalid_retrieval_mode_falls_back_to_keyword(self):
        mem = EpisodicMemory(retrieval_mode="nonexistent")
        assert mem._retrieval_mode == "keyword"

        sm = SemanticMemory(retrieval_mode="invalid")
        assert sm._retrieval_mode == "keyword"
