"""
Runtime behavior tests for gwenn.agent.SentientAgent.

These tests use lightweight stubs to verify wiring and sequencing logic in the
agent runtime without making network calls.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from gwenn.agent import SentientAgent
from gwenn.privacy.redaction import PIIRedactor


class _HeartbeatStub:
    def __init__(self):
        self.notified = False
        self.stopped = False

    def notify_user_activity(self) -> None:
        self.notified = True

    async def stop(self) -> None:
        self.stopped = True


class _LoopStub:
    def __init__(self):
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="ok")


class _IdentityStub:
    def __init__(self):
        self.total_interactions = 0
        self.total_heartbeats = 0
        self.uptime_seconds = 0.0
        self.last_user_id = None
        self.saved = False

    def update_relationship(self, user_id: str) -> None:
        self.last_user_id = user_id

    def _save(self) -> None:
        self.saved = True


class _WorkingMemoryStub:
    def __init__(self):
        self.items = []

    def generate_id(self, prefix: str) -> str:
        return f"{prefix}_1"

    def attend(self, item) -> None:
        self.items.append(item)


class _MemoryStoreRecorder:
    def __init__(self, events: list[str]):
        self._events = events

    def save_affect_snapshot(self, **kwargs) -> None:
        self._events.append("save_affect")

    def save_episode(self, episode) -> None:
        self._events.append("save_episode")

    def save_knowledge_node(self, **kwargs) -> None:
        self._events.append("save_knowledge_node")

    def clear_knowledge_edges(self) -> None:
        self._events.append("clear_edges")

    def save_knowledge_edge(self, **kwargs) -> None:
        self._events.append("save_knowledge_edge")

    def close(self) -> None:
        self._events.append("close")


@pytest.mark.asyncio
async def test_respond_passes_tools_and_redacts_api_payload():
    agent = object.__new__(SentientAgent)

    dims = SimpleNamespace(
        valence=0.1,
        arousal=0.3,
        dominance=0.0,
        certainty=0.0,
        goal_congruence=0.0,
    )
    agent.affect_state = SimpleNamespace(
        dimensions=dims,
        current_emotion=SimpleNamespace(value="neutral"),
    )
    agent._config = SimpleNamespace(
        privacy=SimpleNamespace(
            redact_before_api=True,
            redact_before_persist=False,
        )
    )
    agent._conversation_history = []
    agent._current_user_id = None
    agent.heartbeat = _HeartbeatStub()
    agent.identity = _IdentityStub()
    agent.sensory = SimpleNamespace(
        ground_social=lambda user_id, text: None,
        ground_temporal=lambda event_description: None,
    )
    agent.ethics = SimpleNamespace(
        detect_ethical_dimensions=lambda text: [],
    )
    agent.episodic_memory = SimpleNamespace(
        retrieve=lambda **kwargs: [],
    )
    agent.semantic_memory = SimpleNamespace(
        query=lambda **kwargs: [],
    )
    agent.working_memory = _WorkingMemoryStub()
    agent.safety = SimpleNamespace(
        reset_iteration_count=lambda: None,
    )
    agent.tool_registry = SimpleNamespace(
        get_api_tools=lambda **kwargs: [{"name": "echo"}],
    )
    agent.goal_system = SimpleNamespace()
    agent.theory_of_mind = SimpleNamespace()
    agent.metacognition = SimpleNamespace()
    agent.interagent = SimpleNamespace()
    agent.memory_store = SimpleNamespace()
    agent.redactor = PIIRedactor(enabled=True)
    agent.process_appraisal = lambda event: None
    agent._assemble_system_prompt = (
        lambda **kwargs: "Call me at alice@example.com for details."
    )

    async def _noop_integrate(*args, **kwargs):
        return None

    agent._integrate_exchange = _noop_integrate
    loop = _LoopStub()
    agent.agentic_loop = loop

    response = await SentientAgent.respond(
        agent,
        "Here is my email: alice@example.com",
        user_id="user-1",
    )

    assert response == "ok"
    assert len(loop.calls) == 1
    call = loop.calls[0]
    assert call["tools"] == [{"name": "echo"}]
    assert "[REDACTED_EMAIL]" in call["system_prompt"]
    assert "alice@example.com" not in call["system_prompt"]
    assert "[REDACTED_EMAIL]" in call["messages"][0]["content"]
    assert "alice@example.com" not in call["messages"][0]["content"]
    # Internal conversation state remains unredacted; only API payload is redacted.
    assert "alice@example.com" in agent._conversation_history[0]["content"]


@pytest.mark.asyncio
async def test_shutdown_consolidates_before_persisting_memory():
    events: list[str] = []

    agent = object.__new__(SentientAgent)
    agent.heartbeat = _HeartbeatStub()

    dims = SimpleNamespace(
        valence=0.0,
        arousal=0.3,
        dominance=0.0,
        certainty=0.0,
        goal_congruence=0.0,
    )
    agent.affect_state = SimpleNamespace(
        dimensions=dims,
        current_emotion=SimpleNamespace(value="neutral"),
    )

    agent.memory_store = _MemoryStoreRecorder(events)
    agent.episodic_memory = SimpleNamespace(
        retrieve_recent=lambda n=100: [SimpleNamespace(episode_id="ep-1")],
    )
    node = SimpleNamespace(
        node_id="node-1",
        label="node",
        category="concept",
        content="content",
        confidence=0.8,
        source_episodes=["ep-1"],
        created_at=1.0,
        last_updated=2.0,
        access_count=1,
    )
    edge = SimpleNamespace(
        source_id="node-1",
        target_id="node-1",
        relationship="related_to",
        strength=0.5,
        context="ctx",
        created_at=1.0,
    )
    agent.semantic_memory = SimpleNamespace(
        _nodes={"node-1": node},
        _edges=[edge],
    )
    agent.identity = _IdentityStub()
    agent.identity.total_interactions = 3
    agent._start_time = time.time() - 5.0

    async def _consolidate():
        events.append("consolidate")

    agent.consolidate_memories = _consolidate

    await SentientAgent.shutdown(agent)

    assert agent.heartbeat.stopped is True
    assert "consolidate" in events
    assert events.index("consolidate") < events.index("save_episode")
    assert events.index("consolidate") < events.index("save_knowledge_node")
    assert events[-1] == "close"

