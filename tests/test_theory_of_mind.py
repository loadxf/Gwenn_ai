"""
Tests for gwenn.cognition.theory_of_mind -- Theory of Mind module.

Covers: BeliefState confirm/weaken/defaults, UserModel knowledge & preference
beliefs, record_interaction with rapport/decay/topics, decay_stale_beliefs,
TheoryOfMind get_or_create/set_current/current_user, context generation,
communication prompt/recommendations, round-trip serialization.
"""
from __future__ import annotations

import time

import pytest

from gwenn.cognition.theory_of_mind import BeliefState, TheoryOfMind, UserModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_belief(**overrides) -> BeliefState:
    defaults = dict(content="intermediate", confidence=0.5, source="inferred")
    defaults.update(overrides)
    return BeliefState(**defaults)


def _make_user(user_id: str = "alice", **overrides) -> UserModel:
    model = UserModel(user_id=user_id)
    for k, v in overrides.items():
        setattr(model, k, v)
    return model


def _fresh_tom() -> TheoryOfMind:
    return TheoryOfMind()


# -- BeliefState ------------------------------------------------------------

class TestBeliefState:
    def test_default_confidence_and_source(self):
        b = BeliefState(content="hello")
        assert b.confidence == pytest.approx(0.5)
        assert b.source == "inferred"

    def test_confirm_boosts_confidence(self):
        b = _make_belief(confidence=0.4)
        b.confirm(boost=0.2)
        assert b.confidence == pytest.approx(0.6)

    def test_confirm_clamps_at_one(self):
        b = _make_belief(confidence=0.95)
        b.confirm(boost=0.2)
        assert b.confidence == pytest.approx(1.0)

    def test_weaken_reduces_confidence(self):
        b = _make_belief(confidence=0.5)
        b.weaken(amount=0.2)
        assert b.confidence == pytest.approx(0.3)

    def test_weaken_clamps_at_zero(self):
        b = _make_belief(confidence=0.05)
        b.weaken(amount=0.2)
        assert b.confidence == pytest.approx(0.0)

    def test_source_field_preserved(self):
        b = _make_belief(source="stated")
        assert b.source == "stated"


# -- UserModel: knowledge beliefs -------------------------------------------

class TestUserModelKnowledgeBeliefs:
    def test_create_new_knowledge_belief(self):
        user = _make_user()
        user.update_knowledge_belief("python", "expert", 0.8, "stated")
        assert "python" in user.knowledge_beliefs
        assert user.knowledge_beliefs["python"].content == "expert"
        assert user.knowledge_beliefs["python"].confidence == pytest.approx(0.8)

    def test_update_existing_knowledge_belief(self):
        user = _make_user()
        user.update_knowledge_belief("python", "beginner", 0.5, "inferred")
        user.update_knowledge_belief("python", "expert", 0.9, "stated")
        assert user.knowledge_beliefs["python"].content == "expert"
        assert user.knowledge_beliefs["python"].confidence == pytest.approx(0.9)
        assert user.knowledge_beliefs["python"].source == "stated"

    def test_confidence_clamped_above_one(self):
        user = _make_user()
        user.update_knowledge_belief("rust", "expert", 1.5, "stated")
        assert user.knowledge_beliefs["rust"].confidence == pytest.approx(1.0)

    def test_confidence_clamped_below_zero(self):
        user = _make_user()
        user.update_knowledge_belief("java", "novice", -0.3, "inferred")
        assert user.knowledge_beliefs["java"].confidence == pytest.approx(0.0)


# -- UserModel: preferences -------------------------------------------------

class TestUserModelPreferences:
    def test_create_new_preference(self):
        user = _make_user()
        user.update_preference("verbosity", "concise", 0.7, "observed")
        assert "verbosity" in user.preference_beliefs
        assert user.preference_beliefs["verbosity"].content == "concise"
        assert user.preference_beliefs["verbosity"].confidence == pytest.approx(0.7)

    def test_update_existing_preference(self):
        user = _make_user()
        user.update_preference("tone", "casual", 0.5)
        user.update_preference("tone", "formal", 0.8, "stated")
        assert user.preference_beliefs["tone"].content == "formal"
        assert user.preference_beliefs["tone"].confidence == pytest.approx(0.8)


# -- UserModel: record_interaction ------------------------------------------

class TestRecordInteraction:
    def test_increments_interaction_count(self):
        user = _make_user()
        assert user.interaction_count == 0
        user.record_interaction()
        assert user.interaction_count == 1
        user.record_interaction()
        assert user.interaction_count == 2

    def test_updates_last_interaction(self):
        user = _make_user()
        before = time.time()
        user.record_interaction()
        assert user.last_interaction >= before

    def test_rapport_grows_by_002(self):
        user = _make_user()
        initial = user.rapport_level
        user.record_interaction()
        assert user.rapport_level == pytest.approx(initial + 0.02)

    def test_rapport_clamps_at_one(self):
        user = _make_user()
        user.rapport_level = 0.99
        user.record_interaction()
        assert user.rapport_level == pytest.approx(1.0)

    def test_absence_over_7_days_decays_rapport(self):
        user = _make_user()
        user.rapport_level = 0.5
        # Pretend last interaction was 17 days ago
        user.last_interaction = time.time() - 17 * 86400
        user.record_interaction()
        # 10 extra days beyond 7 => decay = min(0.5, 0.01*10) = 0.1
        # After decay: 0.5 - 0.1 = 0.4, then +0.02 = 0.42
        assert user.rapport_level == pytest.approx(0.42)

    def test_calls_decay_stale_beliefs(self):
        user = _make_user()
        # Add a stale belief confirmed 60 days ago
        b = _make_belief(confidence=0.8)
        b.last_confirmed = time.time() - 60 * 86400
        user.knowledge_beliefs["old_topic"] = b
        user.record_interaction()
        # Stale belief should have lost some confidence
        assert user.knowledge_beliefs["old_topic"].confidence < 0.8

    def test_topics_cap_at_max(self):
        user = _make_user()
        user.topics_discussed = [f"topic_{i}" for i in range(55)]
        user.record_interaction()
        assert len(user.topics_discussed) <= UserModel._MAX_TOPICS


# -- UserModel: decay_stale_beliefs -----------------------------------------

class TestDecayStaleBeliefs:
    def test_fresh_beliefs_stay_intact(self):
        user = _make_user()
        user.update_knowledge_belief("python", "expert", 0.8, "stated")
        original = user.knowledge_beliefs["python"].confidence
        user.decay_stale_beliefs()
        assert user.knowledge_beliefs["python"].confidence == pytest.approx(original)

    def test_stale_beliefs_lose_confidence(self):
        user = _make_user()
        b = _make_belief(confidence=0.9)
        b.last_confirmed = time.time() - 40 * 86400  # 40 days ago
        user.knowledge_beliefs["haskell"] = b
        user.decay_stale_beliefs()
        # 10 stale days => decay = min(0.15, 0.01*10) = 0.10
        assert user.knowledge_beliefs["haskell"].confidence == pytest.approx(0.9 - 0.10)

    def test_decay_capped_at_max(self):
        user = _make_user()
        b = _make_belief(confidence=0.9)
        b.last_confirmed = time.time() - 100 * 86400  # 100 days ago
        user.knowledge_beliefs["cobol"] = b
        user.decay_stale_beliefs()
        # 70 stale days => 0.01*70=0.70 but capped at 0.15
        assert user.knowledge_beliefs["cobol"].confidence == pytest.approx(0.9 - 0.15)

    def test_preference_beliefs_also_decay(self):
        user = _make_user()
        b = _make_belief(confidence=0.7)
        b.last_confirmed = time.time() - 50 * 86400
        user.preference_beliefs["verbosity"] = b
        user.decay_stale_beliefs()
        assert user.preference_beliefs["verbosity"].confidence < 0.7


# -- TheoryOfMind: user management ------------------------------------------

class TestTheoryOfMindUserManagement:
    def test_get_or_create_creates_new_user(self):
        tom = _fresh_tom()
        user = tom.get_or_create_user("bob")
        assert user.user_id == "bob"
        assert user.interaction_count == 0

    def test_get_or_create_returns_existing(self):
        tom = _fresh_tom()
        u1 = tom.get_or_create_user("bob")
        u1.interaction_count = 5
        u2 = tom.get_or_create_user("bob")
        assert u2 is u1
        assert u2.interaction_count == 5

    def test_set_current_user_sets_id_and_records_interaction(self):
        tom = _fresh_tom()
        user = tom.set_current_user("carol")
        assert tom._current_user_id == "carol"
        assert user.interaction_count == 1

    def test_current_user_returns_correct_user(self):
        tom = _fresh_tom()
        tom.set_current_user("dave")
        assert tom.current_user is not None
        assert tom.current_user.user_id == "dave"

    def test_current_user_none_when_not_set(self):
        tom = _fresh_tom()
        assert tom.current_user is None

    def test_lru_eviction_caps_user_models(self):
        tom = TheoryOfMind(max_user_models=3)
        # Add 3 users in order — all should fit.
        tom.set_current_user("u1")
        tom.set_current_user("u2")
        tom.set_current_user("u3")
        assert len(tom._user_models) == 3

        # Adding a 4th should evict the oldest (u1).
        tom.set_current_user("u4")
        assert len(tom._user_models) == 3
        assert "u1" not in tom._user_models
        assert "u4" in tom._user_models

    def test_lru_eviction_does_not_evict_current_user(self):
        tom = TheoryOfMind(max_user_models=2)
        tom.set_current_user("old")
        tom.set_current_user("new")
        # "new" is current, "old" is oldest — but we're within limit.
        assert len(tom._user_models) == 2
        # Adding a 3rd evicts "old" (the least-recently-interacted).
        tom.set_current_user("newest")
        assert "newest" in tom._user_models
        assert "new" in tom._user_models
        assert "old" not in tom._user_models

    def test_config_propagated_to_new_users(self):
        tom = TheoryOfMind(belief_staleness_days=7.0, max_topics_per_user=10)
        user = tom.get_or_create_user("alice")
        assert user._BELIEF_STALENESS_DAYS == 7.0
        assert user._MAX_TOPICS == 10


# -- Context generation -----------------------------------------------------

class TestContextGeneration:
    def test_unknown_user_returns_fallback(self):
        tom = _fresh_tom()
        ctx = tom.generate_user_context("nonexistent")
        assert "unknown" in ctx.lower() or "new" in ctx.lower()

    def test_no_current_user_returns_fallback(self):
        tom = _fresh_tom()
        ctx = tom.generate_user_context()
        assert "unknown" in ctx.lower() or "new" in ctx.lower()

    def test_knowledge_above_threshold_included(self):
        tom = _fresh_tom()
        tom.set_current_user("eve")
        user = tom._user_models["eve"]
        user.update_knowledge_belief("python", "expert", 0.8, "stated")
        ctx = tom.generate_user_context("eve")
        assert "python" in ctx
        assert "expert" in ctx

    def test_knowledge_below_threshold_excluded(self):
        tom = _fresh_tom()
        tom.set_current_user("eve")
        user = tom._user_models["eve"]
        user.update_knowledge_belief("obscure_lang", "novice", 0.2, "inferred")
        ctx = tom.generate_user_context("eve")
        assert "obscure_lang" not in ctx

    def test_communication_style_shown_after_3_interactions(self):
        tom = _fresh_tom()
        user_model = tom.get_or_create_user("frank")
        user_model.interaction_count = 4
        tom._user_models["frank"] = user_model
        ctx = tom.generate_user_context("frank")
        assert "Communication style" in ctx

    def test_communication_style_hidden_for_few_interactions(self):
        tom = _fresh_tom()
        user_model = tom.get_or_create_user("grace")
        user_model.interaction_count = 1
        ctx = tom.generate_user_context("grace")
        assert "Communication style" not in ctx


# -- Communication prompt and recommendations --------------------------------

class TestCommunicationPromptAndRecommendations:
    def test_generate_communication_prompt_empty_for_unknown(self):
        tom = _fresh_tom()
        assert tom.generate_communication_prompt("nonexistent") == ""

    def test_generate_communication_prompt_has_guidance_for_known(self):
        tom = _fresh_tom()
        tom.set_current_user("heidi")
        prompt = tom.generate_communication_prompt("heidi")
        assert "Communication guidance" in prompt

    def test_recommendations_adapt_false_for_unknown(self):
        tom = _fresh_tom()
        recs = tom.get_communication_recommendations("unknown_user")
        assert recs["adapt"] is False

    def test_recommendations_adapt_true_for_known(self):
        tom = _fresh_tom()
        tom.set_current_user("ivan")
        recs = tom.get_communication_recommendations("ivan")
        assert recs["adapt"] is True
        assert "verbosity" in recs
        assert "technical_level" in recs


# -- Persistence: round-trip to_dict / restore_from_dict --------------------

class TestPersistence:
    def test_round_trip_preserves_user_models(self):
        tom = _fresh_tom()
        tom.set_current_user("alice")
        user = tom._user_models["alice"]
        user.display_name = "Alice"
        user.update_knowledge_belief("python", "expert", 0.9, "stated")
        user.update_preference("tone", "casual", 0.7, "observed")

        data = tom.to_dict()
        tom2 = _fresh_tom()
        tom2.restore_from_dict(data)

        assert "alice" in tom2._user_models
        restored = tom2._user_models["alice"]
        assert restored.display_name == "Alice"
        assert restored.knowledge_beliefs["python"].content == "expert"
        assert restored.knowledge_beliefs["python"].confidence == pytest.approx(0.9)
        assert restored.preference_beliefs["tone"].content == "casual"

    def test_round_trip_preserves_interaction_count_and_topics(self):
        tom = _fresh_tom()
        tom.set_current_user("bob")
        user = tom._user_models["bob"]
        user.interaction_count = 10
        user.topics_discussed = ["ai", "music", "cooking"]

        data = tom.to_dict()
        tom2 = _fresh_tom()
        tom2.restore_from_dict(data)

        restored = tom2._user_models["bob"]
        assert restored.interaction_count == 10
        assert restored.topics_discussed == ["ai", "music", "cooking"]

    def test_current_user_id_not_restored(self):
        tom = _fresh_tom()
        tom.set_current_user("carol")
        data = tom.to_dict()
        assert data["current_user_id"] == "carol"

        tom2 = _fresh_tom()
        tom2.restore_from_dict(data)
        assert tom2._current_user_id is None

    def test_malformed_data_does_not_crash(self):
        tom = _fresh_tom()
        tom.restore_from_dict({"users": {"bad": "not_a_dict"}})
        assert len(tom._user_models) == 0

    def test_empty_dict_is_safe(self):
        tom = _fresh_tom()
        tom.restore_from_dict({})
        assert len(tom._user_models) == 0

    def test_non_dict_input_is_safe(self):
        tom = _fresh_tom()
        tom.restore_from_dict(None)  # type: ignore[arg-type]
        tom.restore_from_dict(42)  # type: ignore[arg-type]
        tom.restore_from_dict("string")  # type: ignore[arg-type]
        assert len(tom._user_models) == 0

    def test_multiple_users_preserved(self):
        tom = _fresh_tom()
        tom.set_current_user("x")
        tom.set_current_user("y")
        tom._user_models["x"].display_name = "X"
        tom._user_models["y"].display_name = "Y"

        data = tom.to_dict()
        tom2 = _fresh_tom()
        tom2.restore_from_dict(data)

        assert "x" in tom2._user_models
        assert "y" in tom2._user_models
        assert tom2._user_models["x"].display_name == "X"
        assert tom2._user_models["y"].display_name == "Y"

    def test_restored_users_get_config_values(self):
        tom = TheoryOfMind(belief_staleness_days=7.0, max_topics_per_user=10)
        tom.set_current_user("alice")
        data = tom.to_dict()

        tom2 = TheoryOfMind(belief_staleness_days=7.0, max_topics_per_user=10)
        tom2.restore_from_dict(data)
        restored = tom2._user_models["alice"]
        assert restored._BELIEF_STALENESS_DAYS == 7.0
        assert restored._MAX_TOPICS == 10

    def test_round_trip_preserves_belief_fields(self):
        tom = _fresh_tom()
        tom.set_current_user("dana")
        user = tom._user_models["dana"]
        user.update_knowledge_belief("go", "intermediate", 0.6, "observed")
        formed = user.knowledge_beliefs["go"].formed_at
        confirmed = user.knowledge_beliefs["go"].last_confirmed

        data = tom.to_dict()
        tom2 = _fresh_tom()
        tom2.restore_from_dict(data)

        restored_belief = tom2._user_models["dana"].knowledge_beliefs["go"]
        assert restored_belief.source == "observed"
        assert restored_belief.formed_at == pytest.approx(formed, abs=0.01)
        assert restored_belief.last_confirmed == pytest.approx(confirmed, abs=0.01)
