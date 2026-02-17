"""
Tests for gwenn.memory.consolidation — Consolidation Engine.

Covers:
- Parsing FACT lines into semantic nodes
- Parsing RELATIONSHIP lines into edges
- Parsing SELF and PATTERN lines
- Robustness to malformed lines
- Marking episodes as consolidated
"""

from __future__ import annotations

import pytest

from gwenn.memory.consolidation import ConsolidationEngine
from gwenn.memory.episodic import Episode, EpisodicMemory
from gwenn.memory.semantic import SemanticMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_engine_with_episodes() -> tuple[ConsolidationEngine, EpisodicMemory, SemanticMemory]:
    """Build a consolidation engine with some pre-loaded unconsolidated episodes."""
    em = EpisodicMemory()
    sm = SemanticMemory()
    ep = Episode(episode_id="ep-1", content="User Alice asked about Python decorators")
    em.encode(ep)
    engine = ConsolidationEngine(episodic=em, semantic=sm, consolidation_interval=0.0)
    return engine, em, sm


# ---------------------------------------------------------------------------
# Parsing FACT lines
# ---------------------------------------------------------------------------

class TestParseFacts:
    """FACT lines should be parsed into semantic knowledge nodes."""

    def test_simple_fact(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "FACT: Alice prefers concise code examples | confidence: 0.8 | category: preference"
        counts = engine.process_consolidation_response(response)

        assert counts["facts"] == 1
        assert sm.node_count >= 1
        nodes = sm.query("Alice prefers concise", top_k=1)
        assert len(nodes) == 1
        assert nodes[0].category == "preference"
        assert nodes[0].confidence == pytest.approx(0.8)

    def test_fact_without_confidence_uses_default(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "FACT: Users enjoy interactive examples"
        counts = engine.process_consolidation_response(response)

        assert counts["facts"] == 1
        nodes = sm.query("Users enjoy interactive", top_k=1)
        assert len(nodes) == 1
        assert nodes[0].confidence == pytest.approx(0.5)  # default

    def test_fact_without_category_defaults_to_fact(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "FACT: Python is popular | confidence: 0.9"
        counts = engine.process_consolidation_response(response)

        assert counts["facts"] == 1
        nodes = sm.query("Python is popular", top_k=1)
        assert len(nodes) == 1
        assert nodes[0].category == "fact"

    def test_multiple_facts(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = (
            "FACT: Alice likes Python | confidence: 0.9 | category: preference\n"
            "FACT: Bob prefers Java | confidence: 0.7 | category: preference\n"
        )
        counts = engine.process_consolidation_response(response)
        assert counts["facts"] == 2
        assert sm.node_count >= 2


# ---------------------------------------------------------------------------
# Parsing RELATIONSHIP lines
# ---------------------------------------------------------------------------

class TestParseRelationships:
    """RELATIONSHIP lines should create both nodes and an edge."""

    def test_simple_relationship(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "RELATIONSHIP: Alice -> interested_in -> machine learning | strength: 0.8"
        counts = engine.process_consolidation_response(response)

        assert counts["relationships"] == 1
        # Both source and target nodes should exist
        assert sm.node_count >= 2
        assert sm.edge_count == 1

    def test_relationship_without_strength_uses_default(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "RELATIONSHIP: Alice -> likes -> Python"
        counts = engine.process_consolidation_response(response)

        assert counts["relationships"] == 1
        assert sm.edge_count == 1

    def test_relationship_nodes_are_queryable(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "RELATIONSHIP: Alice -> interested_in -> machine learning | strength: 0.8"
        engine.process_consolidation_response(response)

        rels = sm.get_relationships("Alice")
        assert len(rels) == 1
        edge, target = rels[0]
        assert edge.relationship == "interested_in"
        assert "machine learning" in target.content.lower()


# ---------------------------------------------------------------------------
# Parsing SELF and PATTERN lines
# ---------------------------------------------------------------------------

class TestParseSelfKnowledge:
    """SELF lines should produce nodes with category='self'."""

    def test_self_knowledge(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "SELF: I explain recursion well with tree metaphors | confidence: 0.85"
        counts = engine.process_consolidation_response(response)

        assert counts["self_knowledge"] == 1
        nodes = sm.query("explain recursion tree", category="self", top_k=1)
        assert len(nodes) >= 1
        assert nodes[0].confidence == pytest.approx(0.85)

    def test_self_knowledge_default_confidence(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "SELF: I tend to be verbose when excited"
        counts = engine.process_consolidation_response(response)

        assert counts["self_knowledge"] == 1
        # Should use default confidence 0.5
        nodes = sm.query("verbose excited", category="self", top_k=1)
        assert len(nodes) >= 1
        assert nodes[0].confidence == pytest.approx(0.5)


class TestParsePatterns:
    """PATTERN lines should produce nodes with category='pattern'."""

    def test_pattern(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "PATTERN: Users often need help with async Python | confidence: 0.75"
        counts = engine.process_consolidation_response(response)

        assert counts["patterns"] == 1
        nodes = sm.query("async Python", category="pattern", top_k=1)
        assert len(nodes) >= 1
        assert nodes[0].confidence == pytest.approx(0.75)

    def test_pattern_default_confidence(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "PATTERN: Error messages cluster on Mondays"
        counts = engine.process_consolidation_response(response)

        assert counts["patterns"] == 1


# ---------------------------------------------------------------------------
# Robustness to malformed lines
# ---------------------------------------------------------------------------

class TestMalformedLines:
    """The parser should skip malformed lines without crashing."""

    def test_empty_response(self):
        engine, em, sm = _fresh_engine_with_episodes()
        counts = engine.process_consolidation_response("")
        assert counts == {"facts": 0, "relationships": 0, "self_knowledge": 0, "patterns": 0}

    def test_unrecognized_lines_are_skipped(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = (
            "This is just some freeform text\n"
            "Another random line\n"
            "FACT: This one is valid | confidence: 0.7\n"
        )
        counts = engine.process_consolidation_response(response)
        assert counts["facts"] == 1
        # Other counts stay zero
        assert counts["relationships"] == 0
        assert counts["self_knowledge"] == 0
        assert counts["patterns"] == 0

    def test_relationship_with_too_few_arrows_does_not_create_edge(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "RELATIONSHIP: Alice -> Python"  # only 2 arrow parts, need 3
        counts = engine.process_consolidation_response(response)
        # Malformed relationship lines are rejected — counter stays 0,
        # and no edge is created in semantic memory.
        assert counts["relationships"] == 0  # malformed, not counted
        assert sm.edge_count == 0            # no edge stored

    def test_bad_confidence_value_uses_default(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = "FACT: some fact | confidence: not_a_number | category: fact"
        counts = engine.process_consolidation_response(response)
        assert counts["facts"] == 1
        # The confidence should fall back to 0.5 default
        nodes = sm.query("some fact", top_k=1)
        assert len(nodes) >= 1
        assert nodes[0].confidence == pytest.approx(0.5)

    def test_mixed_valid_and_invalid(self):
        engine, em, sm = _fresh_engine_with_episodes()
        response = (
            "FACT: Good fact | confidence: 0.8\n"
            "RELATIONSHIP: broken line\n"
            "SELF: Good self knowledge | confidence: 0.7\n"
            "PATTERN: Good pattern | confidence: 0.6\n"
            "GARBAGE LINE\n"
        )
        counts = engine.process_consolidation_response(response)
        assert counts["facts"] == 1
        # "RELATIONSHIP: broken line" is malformed (no arrow-parts),
        # so _process_relationship returns False and it is not counted.
        assert counts["relationships"] == 0
        assert sm.edge_count == 0
        assert counts["self_knowledge"] == 1
        assert counts["patterns"] == 1


# ---------------------------------------------------------------------------
# Marking episodes as consolidated
# ---------------------------------------------------------------------------

class TestMarkConsolidated:
    """process_consolidation_response should mark unconsolidated episodes."""

    def test_episodes_marked_after_processing(self):
        engine, em, sm = _fresh_engine_with_episodes()
        # Before consolidation, the episode is unconsolidated
        assert len(em.get_unconsolidated()) == 1

        engine.process_consolidation_response("FACT: test | confidence: 0.5")

        # After consolidation, it should be marked
        assert len(em.get_unconsolidated()) == 0

    def test_consolidated_episodes_not_reconsolidated(self):
        engine, em, sm = _fresh_engine_with_episodes()
        engine.process_consolidation_response("FACT: first pass | confidence: 0.5")

        # Add a new episode
        new_ep = Episode(episode_id="ep-2", content="new experience")
        em.encode(new_ep)

        # Only the new episode should be unconsolidated
        unconsolidated = em.get_unconsolidated()
        assert len(unconsolidated) == 1
        assert unconsolidated[0].episode_id == "ep-2"

    def test_consolidation_increments_total_count(self):
        engine, em, sm = _fresh_engine_with_episodes()
        assert engine.stats["total_consolidations"] == 0
        engine.process_consolidation_response("FACT: test | confidence: 0.5")
        assert engine.stats["total_consolidations"] == 1
        # Add another episode and consolidate again
        em.encode(Episode(episode_id="ep-new", content="newer"))
        engine.process_consolidation_response("FACT: another | confidence: 0.5")
        assert engine.stats["total_consolidations"] == 2


# ---------------------------------------------------------------------------
# Consolidation prompt generation
# ---------------------------------------------------------------------------

class TestConsolidationPrompt:
    """get_consolidation_prompt should format episodes into a prompt."""

    def test_prompt_contains_episode_content(self, consolidation_engine):
        prompt = consolidation_engine.get_consolidation_prompt()
        assert prompt is not None
        assert "Python decorators" in prompt

    def test_prompt_is_none_when_no_unconsolidated(self):
        em = EpisodicMemory()
        sm = SemanticMemory()
        engine = ConsolidationEngine(episodic=em, semantic=sm)
        prompt = engine.get_consolidation_prompt()
        assert prompt is None

    def test_should_consolidate_timing(self):
        em = EpisodicMemory()
        sm = SemanticMemory()
        engine = ConsolidationEngine(episodic=em, semantic=sm, consolidation_interval=9999.0)
        # Just initialized, _last_consolidation is 0.0, so time.time() - 0.0 > 9999
        # Actually, 0.0 means it's been "infinite" time, so should_consolidate is True
        assert engine.should_consolidate() is True
