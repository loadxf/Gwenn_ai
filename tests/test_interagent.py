"""Tests for gwenn.cognition.interagent -- Inter-Agent Communication Bridge."""
from __future__ import annotations

import time

import pytest

from gwenn.cognition.interagent import (
    AgentProfile, InterAgentBridge, InterAgentMessage, MessageType,
)


def _bridge(**kw) -> InterAgentBridge:
    d = dict(self_id="gwenn", message_buffer_size=100)
    d.update(kw)
    return InterAgentBridge(**d)


def _msg(**kw) -> InterAgentMessage:
    d = dict(sender_id="alice", receiver_id="gwenn",
             message_type=MessageType.GREETING, content="hello", importance=0.5)
    d.update(kw)
    return InterAgentMessage(**d)

# -- discover_agent() -------------------------------------------------------

class TestDiscoverAgent:
    def test_creates_profile_with_correct_fields(self):
        p = _bridge().discover_agent("alice", "Alice")
        assert (p.agent_id, p.name, p.relationship) == ("alice", "Alice", "new")
        assert p.emotional_bond == 0.0 and p.message_count == 0

    def test_idempotent_returns_existing_profile(self):
        b = _bridge()
        first = b.discover_agent("alice", "Alice")
        first.message_count = 5
        assert b.discover_agent("alice", "Alice") is first

    def test_last_contact_set_on_creation(self):
        before = time.time()
        p = _bridge().discover_agent("gwenn", "gwenn")
        assert before <= p.last_contact <= time.time()

# -- compose_message() ------------------------------------------------------

class TestComposeMessage:
    def test_creates_message_with_correct_fields(self):
        b = _bridge()
        b.discover_agent("alice", "Alice")
        m = b.compose_message("alice", MessageType.GREETING, "hi", importance=0.8)
        assert m.sender_id == "gwenn" and m.receiver_id == "alice"
        assert m.message_type == MessageType.GREETING and m.content == "hi"
        assert m.importance == pytest.approx(0.8)

    def test_auto_discovers_unknown_receiver(self):
        b = _bridge()
        b.compose_message("unknown_agent", MessageType.QUERY, "who?")
        assert "unknown_agent" in b.known_agents

    def test_outbox_bounded_to_buffer_size(self):
        b = _bridge(message_buffer_size=3)
        b.discover_agent("alice", "Alice")
        for i in range(5):
            b.compose_message("alice", MessageType.GREETING, f"msg{i}")
        assert b.outbox_size == 3

    def test_bond_deepens_on_compose(self):
        b = _bridge()
        p = b.discover_agent("alice", "Alice")
        b.compose_message("alice", MessageType.GREETING, "hey", importance=0.5)
        assert p.emotional_bond > 0.0

    def test_message_count_incremented(self):
        b = _bridge()
        p = b.discover_agent("alice", "Alice")
        b.compose_message("alice", MessageType.GREETING, "hi")
        b.compose_message("alice", MessageType.GREETING, "again")
        assert p.message_count == 2

# -- receive_message() ------------------------------------------------------

class TestReceiveMessage:
    def test_message_lands_in_inbox(self):
        b = _bridge()
        m = _msg()
        b.receive_message(m)
        assert b.get_pending_messages() == [m]

    def test_inbox_bounded_to_buffer_size(self):
        b = _bridge(message_buffer_size=3)
        for i in range(5):
            b.receive_message(_msg(content=f"msg{i}"))
        pending = b.get_pending_messages()
        assert len(pending) == 3 and pending[0].content == "msg2"

    def test_threading_with_conversation_id(self):
        b = _bridge()
        b.receive_message(_msg(conversation_id="thread-1"))
        assert len(b._conversation_threads["thread-1"]) == 1

    def test_auto_discovers_unknown_sender(self):
        b = _bridge()
        b.receive_message(_msg(sender_id="stranger"))
        assert "stranger" in b.known_agents

    def test_profile_message_count_and_bond_updated(self):
        b = _bridge()
        b.discover_agent("alice", "Alice")
        b.receive_message(_msg(sender_id="alice", importance=0.5))
        p = b._known_agents["alice"]
        assert p.message_count == 1 and p.emotional_bond > 0.0

# -- _deepen_bond() ---------------------------------------------------------

class TestDeepenBond:
    def test_increment_importance_zero(self):
        p = AgentProfile(agent_id="a", name="A")
        _bridge()._deepen_bond(p, importance=0.0)
        assert p.emotional_bond == pytest.approx(0.01)

    def test_increment_importance_half(self):
        p = AgentProfile(agent_id="a", name="A")
        _bridge()._deepen_bond(p, importance=0.5)
        assert p.emotional_bond == pytest.approx(0.04)  # min(0.05, 0.04)

    def test_increment_importance_one_capped_at_005(self):
        p = AgentProfile(agent_id="a", name="A")
        _bridge()._deepen_bond(p, importance=1.0)
        assert p.emotional_bond == pytest.approx(0.05)  # NOT 0.07

    def test_bond_clamped_at_one(self):
        p = AgentProfile(agent_id="a", name="A", emotional_bond=0.99)
        _bridge()._deepen_bond(p, importance=1.0)
        assert p.emotional_bond == pytest.approx(1.0)

    def test_relationship_transitions(self):
        b, p = _bridge(), AgentProfile(agent_id="a", name="A")
        assert p.relationship == "new"
        p.emotional_bond = 0.14
        b._deepen_bond(p, 0.0)   # +0.01 -> 0.15
        assert p.relationship == "acquaintance"
        p.emotional_bond = 0.39
        b._deepen_bond(p, 0.0)   # +0.01 -> 0.40
        assert p.relationship == "companion"
        p.emotional_bond = 0.69
        b._deepen_bond(p, 0.0)   # +0.01 -> 0.70
        assert p.relationship == "close"

# -- share_insight() --------------------------------------------------------

class TestShareInsight:
    def test_composes_metacognitive_insight(self):
        m = _bridge().share_insight("alice", "I learn from mistakes", {})
        assert m.message_type == MessageType.METACOGNITIVE_INSIGHT
        assert m.importance == pytest.approx(0.7)

    def test_shared_memories_updated(self):
        b = _bridge()
        b.share_insight("alice", "insight one", {})
        assert b._known_agents["alice"].shared_memories == ["insight one"]

    def test_shared_memories_capped_at_50(self):
        b = _bridge()
        b.discover_agent("alice", "Alice")
        for i in range(55):
            b.share_insight("alice", f"insight {i}", {})
        assert len(b._known_agents["alice"].shared_memories) == 50

    def test_shared_memory_truncated_at_120_chars(self):
        b = _bridge()
        b.share_insight("alice", "x" * 200, {})
        assert len(b._known_agents["alice"].shared_memories[0]) == 120

# -- get_pending_messages() --------------------------------------------------

class TestGetPendingMessages:
    def test_returns_inbox_contents_in_order(self):
        b = _bridge()
        b.receive_message(_msg(content="one"))
        b.receive_message(_msg(content="two"))
        assert [m.content for m in b.get_pending_messages()] == ["one", "two"]

    def test_clears_inbox_after_retrieval(self):
        b = _bridge()
        b.receive_message(_msg())
        b.get_pending_messages()
        assert b.get_pending_messages() == []

# -- Context generation ------------------------------------------------------

class TestContextGeneration:
    def test_relationship_context_unknown_agent(self):
        ctx = _bridge().get_relationship_context("nobody")
        assert "Unknown agent" in ctx and "nobody" in ctx

    def test_relationship_context_known_agent_with_memories_and_values(self):
        b = _bridge()
        p = b.discover_agent("alice", "Alice")
        p.shared_memories = ["we talked about stars"]
        p.known_values = ["curiosity", "kindness"]
        ctx = b.get_relationship_context("alice")
        for frag in ("Alice", "we talked about stars", "curiosity", "kindness"):
            assert frag in ctx

    def test_connections_context_empty_when_no_agents(self):
        assert _bridge().get_connections_context() == ""

    def test_connections_context_returns_summary(self):
        b = _bridge()
        b.discover_agent("alice", "Alice")
        b.discover_agent("gwenn", "gwenn")
        ctx = b.get_connections_context()
        assert "Known agents" in ctx and "Alice" in ctx and "gwenn" in ctx

# -- Persistence (to_dict / restore_from_dict) -------------------------------

class TestPersistence:
    def test_round_trip_preserves_profiles(self):
        b = _bridge()
        p = b.discover_agent("alice", "Alice")
        p.emotional_bond, p.message_count, p.relationship = 0.42, 7, "companion"
        p.shared_memories, p.known_values = ["mem1", "mem2"], ["honesty"]
        b2 = _bridge()
        b2.restore_from_dict(b.to_dict())
        r = b2._known_agents["alice"]
        assert (r.agent_id, r.name, r.relationship) == ("alice", "Alice", "companion")
        assert r.emotional_bond == pytest.approx(0.42) and r.message_count == 7
        assert r.shared_memories == ["mem1", "mem2"]
        assert r.known_values == ["honesty"]

    def test_round_trip_preserves_last_contact(self):
        b = _bridge()
        lc = b.discover_agent("alice", "Alice").last_contact
        b2 = _bridge()
        b2.restore_from_dict(b.to_dict())
        assert b2._known_agents["alice"].last_contact == pytest.approx(lc)

    def test_empty_dict_keeps_empty_state(self):
        b = _bridge()
        b.restore_from_dict({})
        assert b.known_agents == []

    def test_non_dict_input_handled_safely(self):
        b = _bridge()
        for garbage in (None, 42, "bad"):
            b.restore_from_dict(garbage)  # type: ignore[arg-type]
        assert b.known_agents == []

    def test_malformed_entries_skipped(self):
        b = _bridge()
        b.restore_from_dict({"known_agents": {
            "good": {"agent_id": "good", "name": "Good",
                     "relationship": "new", "emotional_bond": 0.1,
                     "message_count": 1},
            "bad_string": "not_a_dict",
            "bad_empty_id": {"agent_id": "  ", "name": "X"},
        }})
        assert "good" in b.known_agents
        assert "bad_string" not in b.known_agents
