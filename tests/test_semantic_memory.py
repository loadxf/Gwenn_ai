"""
Tests for gwenn.memory.semantic — Semantic Memory.

Covers:
- add_relationship() deduplication (P1-1 validation)
- store_knowledge() immutable node protection
- get_relationships() direction parameter (P2-1 validation)
- verify_provenance() without _episodes fallback (P1-2 validation)
- node_id full UUID (P2-2 validation)
- Basic query behaviour
"""

from __future__ import annotations

import pytest

from gwenn.memory.episodic import Episode, EpisodicMemory
from gwenn.memory.semantic import SemanticMemory


# ---------------------------------------------------------------------------
# add_relationship deduplication (P1-1)
# ---------------------------------------------------------------------------

class TestRelationshipDeduplication:
    """Repeated add_relationship calls should not create duplicate edges."""

    def test_duplicate_relationship_not_added(self, semantic_memory):
        semantic_memory.store_knowledge(label="Alice", content="A user", category="person")
        semantic_memory.store_knowledge(label="Python", content="A language", category="concept")

        edge1 = semantic_memory.add_relationship("Alice", "Python", "likes")
        edge2 = semantic_memory.add_relationship("Alice", "Python", "likes")

        assert semantic_memory.edge_count == 1
        assert edge1 is not None
        assert edge2 is not None
        # Second call should return the existing edge
        assert edge2.edge_id == edge1.edge_id

    def test_different_relationships_not_deduplicated(self, semantic_memory):
        semantic_memory.store_knowledge(label="Alice", content="A user")
        semantic_memory.store_knowledge(label="Python", content="A language")

        semantic_memory.add_relationship("Alice", "Python", "likes")
        semantic_memory.add_relationship("Alice", "Python", "studies")

        assert semantic_memory.edge_count == 2

    def test_reinforced_edge_updates_strength(self, semantic_memory):
        semantic_memory.store_knowledge(label="A", content="a")
        semantic_memory.store_knowledge(label="B", content="b")

        semantic_memory.add_relationship("A", "B", "related_to", strength=0.3)
        edge = semantic_memory.add_relationship("A", "B", "related_to", strength=0.8)

        assert edge.strength == pytest.approx(0.8)
        assert semantic_memory.edge_count == 1


# ---------------------------------------------------------------------------
# store_knowledge immutable node protection
# ---------------------------------------------------------------------------

class TestImmutableNodeProtection:
    """Immutable nodes should not be overwritten by store_knowledge."""

    def test_immutable_node_not_overwritten(self, semantic_memory):
        node = semantic_memory.store_knowledge(
            label="genesis:identity",
            content="I am Gwenn",
            category="self",
            confidence=1.0,
        )
        node.metadata["immutable"] = True

        # Try to overwrite
        result = semantic_memory.store_knowledge(
            label="genesis:identity",
            content="I am someone else",
            category="self",
            confidence=0.9,
        )

        assert result.content == "I am Gwenn"
        assert result.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_relationships direction parameter (P2-1)
# ---------------------------------------------------------------------------

class TestGetRelationshipsDirection:
    """get_relationships should support outgoing, incoming, and both."""

    def _setup_triangle(self, sm: SemanticMemory):
        sm.store_knowledge(label="A", content="a")
        sm.store_knowledge(label="B", content="b")
        sm.store_knowledge(label="C", content="c")
        sm.add_relationship("A", "B", "knows")
        sm.add_relationship("C", "A", "follows")

    def test_outgoing_default(self, semantic_memory):
        self._setup_triangle(semantic_memory)
        rels = semantic_memory.get_relationships("A")
        assert len(rels) == 1
        edge, node = rels[0]
        assert edge.relationship == "knows"
        assert node.label == "B"

    def test_incoming(self, semantic_memory):
        self._setup_triangle(semantic_memory)
        rels = semantic_memory.get_relationships("A", direction="incoming")
        assert len(rels) == 1
        edge, node = rels[0]
        assert edge.relationship == "follows"
        assert node.label == "C"

    def test_both_directions(self, semantic_memory):
        self._setup_triangle(semantic_memory)
        rels = semantic_memory.get_relationships("A", direction="both")
        assert len(rels) == 2


# ---------------------------------------------------------------------------
# verify_provenance without _episodes fallback (P1-2)
# ---------------------------------------------------------------------------

class TestVerifyProvenance:
    """verify_provenance should work via get_episode only."""

    def test_provenance_with_episodic_memory(self, semantic_memory):
        em = EpisodicMemory()
        ep = Episode(episode_id="ep-1", content="Alice likes Python decorators")
        em.encode(ep)

        node = semantic_memory.store_knowledge(
            label="Alice preference",
            content="Alice likes Python decorators",
            source_episode_id="ep-1",
        )

        result = semantic_memory.verify_provenance(node.node_id, em)
        assert result["supported"] is True
        assert result["found_count"] == 1
        assert result["missing"] == []

    def test_provenance_missing_episode(self, semantic_memory):
        em = EpisodicMemory()
        node = semantic_memory.store_knowledge(
            label="orphan fact",
            content="something",
            source_episode_id="ep-gone",
        )

        result = semantic_memory.verify_provenance(node.node_id, em)
        assert result["supported"] is False
        assert result["missing"] == ["ep-gone"]

    def test_provenance_without_get_episode_method(self, semantic_memory):
        """An object without get_episode should treat all episodes as missing."""
        node = semantic_memory.store_knowledge(
            label="no method",
            content="test",
            source_episode_id="ep-x",
        )

        class FakeMemory:
            pass

        result = semantic_memory.verify_provenance(node.node_id, FakeMemory())
        assert result["supported"] is False
        assert result["missing"] == ["ep-x"]


# ---------------------------------------------------------------------------
# node_id full UUID (P2-2)
# ---------------------------------------------------------------------------

class TestNodeIdFullUuid:
    """node_id should be a full UUID, not truncated."""

    def test_node_id_length(self, semantic_memory):
        node = semantic_memory.store_knowledge(label="test node", content="test")
        # Full UUID is 36 chars (32 hex + 4 dashes)
        assert len(node.node_id) == 36
        assert node.node_id.count("-") == 4


# ---------------------------------------------------------------------------
# Basic query behaviour
# ---------------------------------------------------------------------------

class TestBasicQuery:
    """Basic query, get_context_for, and node_count / edge_count."""

    def test_query_by_keyword(self, semantic_memory):
        semantic_memory.store_knowledge(
            label="Python language",
            content="Python is a popular programming language",
            category="concept",
            confidence=0.9,
        )
        semantic_memory.store_knowledge(
            label="Weather today",
            content="The weather is sunny and warm",
            category="fact",
            confidence=0.7,
        )

        results = semantic_memory.query("Python programming", top_k=2)
        assert len(results) >= 1
        assert results[0].label == "Python language"

    def test_query_min_confidence_filter(self, semantic_memory):
        semantic_memory.store_knowledge(
            label="low conf", content="tentative fact", confidence=0.1,
        )
        semantic_memory.store_knowledge(
            label="high conf", content="certain fact", confidence=0.9,
        )

        results = semantic_memory.query("fact", min_confidence=0.5)
        labels = [n.label for n in results]
        assert "high conf" in labels
        assert "low conf" not in labels

    def test_node_and_edge_counts(self, semantic_memory):
        assert semantic_memory.node_count == 0
        assert semantic_memory.edge_count == 0

        semantic_memory.store_knowledge(label="A", content="a")
        semantic_memory.store_knowledge(label="B", content="b")
        semantic_memory.add_relationship("A", "B", "related_to")

        assert semantic_memory.node_count == 2
        assert semantic_memory.edge_count == 1
