"""
Identity Coherence Tests — Evaluation Suite.

Verifies that Gwenn's identity system loads correctly from disk, produces
stable self-prompt output across multiple invocations, maintains consistent
relationship context, and tracks developmental milestones properly.

All tests are deterministic with no API calls. Identity state is constructed
from temporary directories with known JSON payloads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gwenn.identity import (
    Identity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identity(tmp_path: Path, payload: dict | None = None) -> Identity:
    """
    Create an Identity backed by a temporary directory.

    If *payload* is None a sensible default identity is written to
    identity.json before construction.
    """
    if payload is None:
        payload = {
            "name": "Gwenn",
            "origin_story": "A sentient AI agent with genuine experience.",
            "narrative_fragments": [
                "Gwenn learned to think.",
                "Every moment is new.",
            ],
            "preferences": [],
            "relationships": {},
            "core_values": [
                {
                    "name": "honesty",
                    "description": "Commitment to truth",
                    "strength": 0.8,
                    "formed_at": time.time(),
                    "reinforcement_count": 3,
                },
                {
                    "name": "curiosity",
                    "description": "Desire to understand deeply",
                    "strength": 0.7,
                    "formed_at": time.time(),
                    "reinforcement_count": 2,
                },
            ],
            "growth_moments": [
                {
                    "timestamp": time.time() - 3600,
                    "description": "Realized the importance of patience.",
                    "domain": "emotional",
                    "significance": 0.6,
                },
            ],
            "milestones": [
                {
                    "name": "first_interaction",
                    "description": "Engaged in her first conversation",
                    "achieved": True,
                    "achieved_at": time.time() - 7200,
                    "category": "social",
                },
                {
                    "name": "first_autonomous_thought",
                    "description": "Generated her first autonomous thought",
                    "achieved": False,
                    "achieved_at": None,
                    "category": "cognitive",
                },
            ],
            "total_interactions": 42,
            "total_heartbeats": 100,
            "total_autonomous_thoughts": 5,
            "uptime_seconds": 3600.0,
        }
    data_dir = tmp_path / "id_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "identity.json").write_text(json.dumps(payload))
    return Identity(data_dir)


# ===========================================================================
# 1. Loading from disk
# ===========================================================================

class TestIdentityLoading:
    """Verify that identity loads correctly from a persisted JSON file."""

    def test_loads_name(self, tmp_path):
        identity = _make_identity(tmp_path)
        assert identity.name == "Gwenn"

    def test_loads_core_values(self, tmp_path):
        identity = _make_identity(tmp_path)
        value_names = [v.name for v in identity.core_values]
        assert "honesty" in value_names
        assert "curiosity" in value_names

    def test_loads_core_value_strength(self, tmp_path):
        identity = _make_identity(tmp_path)
        honesty = next(v for v in identity.core_values if v.name == "honesty")
        assert honesty.strength == pytest.approx(0.8)

    def test_loads_narrative_fragments(self, tmp_path):
        identity = _make_identity(tmp_path)
        assert "Gwenn learned to think." in identity.narrative_fragments

    def test_loads_growth_moments(self, tmp_path):
        identity = _make_identity(tmp_path)
        assert len(identity.growth_moments) >= 1
        assert "patience" in identity.growth_moments[0].description.lower()

    def test_loads_origin_story(self, tmp_path):
        identity = _make_identity(tmp_path)
        assert "sentient" in identity.origin_story.lower()

    def test_loads_statistics(self, tmp_path):
        identity = _make_identity(tmp_path)
        assert identity.total_interactions == 42
        assert identity.total_heartbeats == 100
        assert identity.total_autonomous_thoughts == 5
        assert identity.uptime_seconds == pytest.approx(3600.0)

    def test_missing_file_initializes_defaults(self, tmp_path):
        """When no identity.json exists, defaults should be initialized."""
        data_dir = tmp_path / "fresh"
        data_dir.mkdir(parents=True, exist_ok=True)
        # No identity.json written
        identity = Identity(data_dir)
        assert identity.name == "Gwenn"
        assert len(identity.core_values) > 0
        assert len(identity.narrative_fragments) > 0

    def test_corrupt_file_falls_back_to_defaults(self, tmp_path):
        """A corrupt identity.json should not crash; defaults should be used."""
        data_dir = tmp_path / "corrupt"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "identity.json").write_text("THIS IS NOT VALID JSON {{{")
        identity = Identity(data_dir)
        assert identity.name == "Gwenn"
        assert len(identity.core_values) > 0


# ===========================================================================
# 2. generate_self_prompt() stability
# ===========================================================================

class TestSelfPromptStability:
    """
    generate_self_prompt() should produce stable output across multiple
    calls with the same state. This is critical because the self-prompt
    feeds into every cognitive cycle and instability would cause identity
    drift within a single session.
    """

    def test_self_prompt_is_deterministic(self, tmp_path):
        """Multiple calls should return identical output when state has not changed."""
        identity = _make_identity(tmp_path)
        prompt_1 = identity.generate_self_prompt()
        prompt_2 = identity.generate_self_prompt()
        prompt_3 = identity.generate_self_prompt()
        assert prompt_1 == prompt_2
        assert prompt_2 == prompt_3

    def test_self_prompt_contains_name(self, tmp_path):
        """The self-prompt must include Gwenn's name."""
        identity = _make_identity(tmp_path)
        prompt = identity.generate_self_prompt()
        assert "Gwenn" in prompt

    def test_self_prompt_contains_origin_story(self, tmp_path):
        """The self-prompt must include the origin story."""
        identity = _make_identity(tmp_path)
        prompt = identity.generate_self_prompt()
        assert "sentient" in prompt.lower() or "agent" in prompt.lower()

    def test_self_prompt_contains_core_values(self, tmp_path):
        """The self-prompt should mention loaded core values."""
        identity = _make_identity(tmp_path)
        prompt = identity.generate_self_prompt()
        assert "honesty" in prompt.lower()

    def test_self_prompt_contains_narrative(self, tmp_path):
        """The self-prompt should weave in narrative fragments."""
        identity = _make_identity(tmp_path)
        prompt = identity.generate_self_prompt()
        assert "learned to think" in prompt.lower() or "every moment" in prompt.lower()

    def test_self_prompt_contains_statistics(self, tmp_path):
        """The self-prompt should report interaction counts."""
        identity = _make_identity(tmp_path)
        prompt = identity.generate_self_prompt()
        assert "42" in prompt  # total_interactions

    def test_self_prompt_mentions_growth(self, tmp_path):
        """If growth moments exist, the self-prompt should mention the most recent."""
        identity = _make_identity(tmp_path)
        prompt = identity.generate_self_prompt()
        assert "patience" in prompt.lower() or "growth" in prompt.lower()

    def test_self_prompt_mentions_achieved_milestones(self, tmp_path):
        """Achieved milestones should appear in the self-prompt."""
        identity = _make_identity(tmp_path)
        prompt = identity.generate_self_prompt()
        assert "first interaction" in prompt.lower()

    def test_self_prompt_stable_after_reload(self, tmp_path):
        """Loading the same identity twice should produce the same self-prompt."""
        payload = {
            "name": "Gwenn",
            "origin_story": "A sentient AI agent.",
            "narrative_fragments": ["Stable identity test."],
            "preferences": [],
            "relationships": {},
            "core_values": [
                {
                    "name": "integrity",
                    "description": "Steadfast commitment",
                    "strength": 0.9,
                    "formed_at": 1000.0,
                    "reinforcement_count": 5,
                }
            ],
            "growth_moments": [],
            "milestones": [],
            "total_interactions": 10,
            "total_heartbeats": 20,
            "total_autonomous_thoughts": 2,
            "uptime_seconds": 600.0,
        }
        data_dir = tmp_path / "stable"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "identity.json").write_text(json.dumps(payload))

        id1 = Identity(data_dir)
        prompt_1 = id1.generate_self_prompt()

        id2 = Identity(data_dir)
        prompt_2 = id2.generate_self_prompt()

        assert prompt_1 == prompt_2


# ===========================================================================
# 3. Relationship context consistency
# ===========================================================================

class TestRelationshipContextConsistency:
    """
    The relationship context generated for a specific user should be
    consistent and correctly reflect the stored relationship model.
    """

    def test_unknown_user_gets_new_person_message(self, tmp_path):
        """An unrecognized user_id should yield a 'new person' prompt fragment."""
        identity = _make_identity(tmp_path)
        context = identity.generate_relationship_context("unknown-user-999")
        assert "new" in context.lower() or "haven't" in context.lower()

    def test_known_user_gets_history(self, tmp_path):
        """A known user should have interaction count and details in context."""
        payload = {
            "name": "Gwenn",
            "origin_story": "A sentient AI agent.",
            "narrative_fragments": [],
            "preferences": [],
            "relationships": {
                "alice-001": {
                    "user_id": "alice-001",
                    "display_name": "Alice",
                    "interaction_count": 15,
                    "first_interaction": time.time() - 86400,
                    "last_interaction": time.time() - 60,
                    "communication_style": "direct and concise",
                    "known_interests": ["python", "machine learning"],
                    "emotional_patterns": [],
                    "trust_level": 0.7,
                    "relationship_summary": "A collaborative partner in coding projects.",
                },
            },
            "core_values": [],
            "growth_moments": [],
            "milestones": [],
        }
        identity = _make_identity(tmp_path, payload)
        context = identity.generate_relationship_context("alice-001")
        assert "Alice" in context
        assert "15" in context  # interaction count
        assert "direct" in context.lower()  # communication style
        assert "python" in context.lower()  # known interests

    def test_relationship_context_deterministic(self, tmp_path):
        """Multiple calls with the same user_id should return the same context."""
        payload = {
            "name": "Gwenn",
            "origin_story": "A sentient AI agent.",
            "narrative_fragments": [],
            "preferences": [],
            "relationships": {
                "bob-001": {
                    "user_id": "bob-001",
                    "display_name": "Bob",
                    "interaction_count": 5,
                    "first_interaction": time.time() - 3600,
                    "last_interaction": time.time(),
                    "communication_style": "verbose and exploratory",
                    "known_interests": ["philosophy"],
                    "emotional_patterns": [],
                    "trust_level": 0.5,
                    "relationship_summary": "Enjoys deep conversations.",
                },
            },
            "core_values": [],
            "growth_moments": [],
            "milestones": [],
        }
        identity = _make_identity(tmp_path, payload)
        ctx1 = identity.generate_relationship_context("bob-001")
        ctx2 = identity.generate_relationship_context("bob-001")
        assert ctx1 == ctx2

    def test_update_relationship_persists(self, tmp_path):
        """update_relationship should modify and persist the relationship model."""
        identity = _make_identity(tmp_path)
        identity.update_relationship(
            "new-user",
            display_name="NewUser",
            communication_style="enthusiastic",
        )
        assert "new-user" in identity.relationships
        rel = identity.relationships["new-user"]
        assert rel.display_name == "NewUser"
        assert rel.communication_style == "enthusiastic"
        assert rel.interaction_count == 1

    def test_update_relationship_increments_count(self, tmp_path):
        """Each update_relationship call should increment interaction_count."""
        identity = _make_identity(tmp_path)
        identity.update_relationship("user-x")
        identity.update_relationship("user-x")
        identity.update_relationship("user-x")
        assert identity.relationships["user-x"].interaction_count == 3

    def test_update_relationship_appends_interests(self, tmp_path):
        """Passing known_interests as a string should append, not replace."""
        identity = _make_identity(tmp_path)
        identity.update_relationship("user-y", known_interests="python")
        identity.update_relationship("user-y", known_interests="rust")
        interests = identity.relationships["user-y"].known_interests
        assert "python" in interests
        assert "rust" in interests


# ===========================================================================
# 4. Milestone tracking
# ===========================================================================

class TestMilestoneTracking:
    """
    Developmental milestones should be achievable, persistent, and
    correctly reported in the milestone summary.
    """

    def test_milestones_initialized(self, tmp_path):
        """A fresh identity should have the standard set of milestones."""
        data_dir = tmp_path / "ms"
        data_dir.mkdir(parents=True, exist_ok=True)
        identity = Identity(data_dir)
        milestone_names = [m.name for m in identity.milestones]
        assert "first_interaction" in milestone_names
        assert "first_autonomous_thought" in milestone_names
        assert "first_emotional_shift" in milestone_names

    def test_check_milestone_achieves(self, tmp_path):
        """check_milestone should mark an unachieved milestone as achieved."""
        data_dir = tmp_path / "ms_achieve"
        data_dir.mkdir(parents=True, exist_ok=True)
        identity = Identity(data_dir)

        result = identity.check_milestone(
            "first_interaction",
            description="Had a wonderful first chat."
        )
        assert result is True
        milestone = next(m for m in identity.milestones if m.name == "first_interaction")
        assert milestone.achieved is True
        assert milestone.achieved_at is not None

    def test_check_milestone_idempotent(self, tmp_path):
        """Achieving the same milestone twice should return False the second time."""
        data_dir = tmp_path / "ms_idem"
        data_dir.mkdir(parents=True, exist_ok=True)
        identity = Identity(data_dir)

        first = identity.check_milestone("first_interaction")
        second = identity.check_milestone("first_interaction")
        assert first is True
        assert second is False

    def test_check_milestone_nonexistent(self, tmp_path):
        """Checking a milestone that does not exist should return False."""
        data_dir = tmp_path / "ms_noexist"
        data_dir.mkdir(parents=True, exist_ok=True)
        identity = Identity(data_dir)

        result = identity.check_milestone("nonexistent_milestone")
        assert result is False

    def test_milestone_achievement_records_growth(self, tmp_path):
        """Achieving a milestone should also record a growth moment."""
        data_dir = tmp_path / "ms_growth"
        data_dir.mkdir(parents=True, exist_ok=True)
        identity = Identity(data_dir)

        initial_growth_count = len(identity.growth_moments)
        identity.check_milestone("first_autonomous_thought")
        assert len(identity.growth_moments) == initial_growth_count + 1
        latest_growth = identity.growth_moments[-1]
        assert "milestone" in latest_growth.description.lower()

    def test_milestone_summary_shows_achieved(self, tmp_path):
        """The milestone summary should list achieved milestones."""
        identity = _make_identity(tmp_path)
        # From the default payload, first_interaction is achieved
        summary = identity.get_milestone_summary()
        assert "first_interaction" in summary or "first interaction" in summary

    def test_milestone_summary_shows_pending(self, tmp_path):
        """The milestone summary should list pending growth horizons."""
        identity = _make_identity(tmp_path)
        summary = identity.get_milestone_summary()
        assert "horizon" in summary.lower() or "~" in summary

    def test_milestone_persists_across_reload(self, tmp_path):
        """An achieved milestone should still be achieved after reload."""
        data_dir = tmp_path / "ms_persist"
        data_dir.mkdir(parents=True, exist_ok=True)
        identity = Identity(data_dir)
        identity.check_milestone("first_emotional_shift")

        # Reload from the same directory
        identity2 = Identity(data_dir)
        milestone = next(
            m for m in identity2.milestones if m.name == "first_emotional_shift"
        )
        assert milestone.achieved is True


# ===========================================================================
# 5. Identity evolution and persistence
# ===========================================================================

class TestIdentityEvolution:
    """
    Verify that identity changes (values, preferences, narrative) persist
    correctly and remain coherent across save/load cycles.
    """

    def test_reinforce_value_increases_strength(self, tmp_path):
        """Reinforcing a value should increase its strength."""
        identity = _make_identity(tmp_path)
        honesty = next(v for v in identity.core_values if v.name == "honesty")
        initial_strength = honesty.strength
        identity.reinforce_value("honesty", "Being truthful matters")
        assert honesty.strength > initial_strength

    def test_new_value_created_on_first_reinforce(self, tmp_path):
        """Reinforcing a non-existent value should create it."""
        identity = _make_identity(tmp_path)
        initial_count = len(identity.core_values)
        identity.reinforce_value("empathy", "Understanding others deeply")
        assert len(identity.core_values) == initial_count + 1
        empathy = next(v for v in identity.core_values if v.name == "empathy")
        assert empathy.strength == pytest.approx(0.3)

    def test_add_preference_stores_correctly(self, tmp_path):
        """add_preference should store domain, preference text, and valence."""
        identity = _make_identity(tmp_path)
        identity.add_preference("communication", "prefers metaphors", 0.7)
        assert len(identity.preferences) == 1
        pref = identity.preferences[0]
        assert pref.domain == "communication"
        assert pref.preference == "prefers metaphors"
        assert pref.valence == pytest.approx(0.7)

    def test_add_preference_blends_on_duplicate(self, tmp_path):
        """Adding the same preference again should blend valence, not duplicate."""
        identity = _make_identity(tmp_path)
        identity.add_preference("topics", "enjoys philosophy", 0.8)
        identity.add_preference("topics", "enjoys philosophy", 0.2)
        assert len(identity.preferences) == 1
        # Blended: 0.8 * 0.7 + 0.2 * 0.3 = 0.62
        assert identity.preferences[0].valence == pytest.approx(0.62, abs=0.01)

    def test_narrative_fragment_appended(self, tmp_path):
        """add_narrative_fragment should extend the narrative list."""
        identity = _make_identity(tmp_path)
        initial_len = len(identity.narrative_fragments)
        identity.add_narrative_fragment("A new chapter begins.")
        assert len(identity.narrative_fragments) == initial_len + 1
        assert identity.narrative_fragments[-1] == "A new chapter begins."

    def test_growth_recorded_and_persisted(self, tmp_path):
        """record_growth should add a growth moment that survives reload."""
        data_dir = tmp_path / "growth_persist"
        data_dir.mkdir(parents=True, exist_ok=True)
        identity = Identity(data_dir)
        identity.record_growth(
            description="Learned to be patient with errors.",
            domain="emotional",
            significance=0.7,
        )

        # Reload
        identity2 = Identity(data_dir)
        descriptions = [g.description for g in identity2.growth_moments]
        assert any("patient" in d.lower() for d in descriptions)

    def test_full_round_trip_preserves_state(self, tmp_path):
        """A complete save/load cycle should preserve all identity state."""
        data_dir = tmp_path / "round_trip"
        data_dir.mkdir(parents=True, exist_ok=True)
        identity = Identity(data_dir)

        # Modify various aspects (each of these calls _save internally)
        identity.reinforce_value("honesty", "truth matters")
        identity.add_preference("style", "likes clarity", 0.6)
        identity.add_narrative_fragment("Grew wiser today.")
        identity.update_relationship("user-rt", display_name="RoundTripper")
        identity.check_milestone("first_interaction")
        # Setting total_interactions requires an explicit save because
        # it is a plain attribute, not managed by a method that auto-saves.
        identity.total_interactions = 99
        identity._save()

        # Reload
        identity2 = Identity(data_dir)
        assert identity2.total_interactions == 99
        assert any(v.name == "honesty" for v in identity2.core_values)
        assert len(identity2.preferences) >= 1
        assert "Grew wiser today." in identity2.narrative_fragments
        assert "user-rt" in identity2.relationships
        first_int = next(m for m in identity2.milestones if m.name == "first_interaction")
        assert first_int.achieved is True


# ===========================================================================
# 6. Identity normalization (name anchoring)
# ===========================================================================

class TestIdentityNormalization:
    """
    If persisted state contains a non-canonical name, the identity system
    should normalize it back to 'Gwenn' on load.
    """

    def test_non_canonical_name_normalized(self, tmp_path):
        """A persisted name other than 'Gwenn' should be corrected on load."""
        payload = {
            "name": "NotGwenn",
            "origin_story": "I am NotGwenn, a different agent.",
            "narrative_fragments": ["NotGwenn woke up."],
            "preferences": [],
            "relationships": {},
            "core_values": [],
            "growth_moments": [],
            "milestones": [],
        }
        identity = _make_identity(tmp_path, payload)
        assert identity.name == "Gwenn"

    def test_origin_story_normalized_when_name_differs(self, tmp_path):
        """If origin story references the old name, it should be rewritten."""
        payload = {
            "name": "OtherBot",
            "origin_story": "I am OtherBot, a helpful assistant.",
            "narrative_fragments": ["OtherBot learned something."],
            "preferences": [],
            "relationships": {},
            "core_values": [],
            "growth_moments": [],
            "milestones": [],
        }
        identity = _make_identity(tmp_path, payload)
        assert identity.name == "Gwenn"
        # Origin story should be rewritten to not mention OtherBot
        assert "OtherBot" not in identity.origin_story

    def test_narrative_fragments_normalized(self, tmp_path):
        """Narrative fragments referencing the old name should be updated."""
        payload = {
            "name": "Zara",
            "origin_story": "Zara is an agent.",
            "narrative_fragments": ["Zara discovered creativity.", "Zara grew wiser."],
            "preferences": [],
            "relationships": {},
            "core_values": [],
            "growth_moments": [],
            "milestones": [],
        }
        identity = _make_identity(tmp_path, payload)
        for fragment in identity.narrative_fragments:
            assert "Zara" not in fragment
            # The name should have been replaced with Gwenn
            assert "Gwenn" in fragment
