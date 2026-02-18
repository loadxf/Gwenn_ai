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
from gwenn.memory.working import WorkingMemoryItem
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
        self.onboarding_completed = False
        self.onboarding_profile = {}
        self.last_relationship_kwargs = None

    def update_relationship(self, user_id: str, **kwargs) -> None:
        self.last_user_id = user_id
        self.last_relationship_kwargs = kwargs

    def _save(self) -> None:
        self.saved = True

    def mark_onboarding_completed(self, profile: dict[str, str]) -> None:
        self.onboarding_completed = True
        self.onboarding_profile = profile


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


class _MemoryStoreOnboardingStub:
    def __init__(self):
        self.context = ""
        self.saved_episode = None

    def load_persistent_context(self) -> str:
        return self.context

    def save_persistent_context(self, content: str) -> None:
        self.context = content

    def save_episode(self, episode) -> None:
        self.saved_episode = episode


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


def test_upsert_marked_context_block_replaces_existing_section():
    existing = (
        "## Existing Notes\n"
        "- keep this\n\n"
        "<!-- onboarding_profile_start -->\n"
        "old section\n"
        "<!-- onboarding_profile_end -->\n\n"
        "## Tail\n"
        "- keep that"
    )
    new_block = (
        "<!-- onboarding_profile_start -->\n"
        "new section\n"
        "<!-- onboarding_profile_end -->"
    )

    merged = SentientAgent._upsert_marked_context_block(
        existing,
        "<!-- onboarding_profile_start -->",
        "<!-- onboarding_profile_end -->",
        new_block,
    )

    assert "old section" not in merged
    assert "new section" in merged
    assert "## Existing Notes" in merged
    assert "## Tail" in merged


def test_apply_startup_onboarding_persists_context_and_identity_profile():
    agent = object.__new__(SentientAgent)
    agent.identity = _IdentityStub()
    agent.memory_store = _MemoryStoreOnboardingStub()
    agent.episodic_memory = SimpleNamespace(encode=lambda episode: None)
    dims = SimpleNamespace(valence=0.1, arousal=0.3)
    agent.affect_state = SimpleNamespace(dimensions=dims)

    profile = {
        "name": "Bob",
        "role": "coding partner",
        "needs": "debugging and architecture guidance",
        "communication_style": "concise and direct",
        "boundaries": "do not use destructive git commands",
    }

    SentientAgent.apply_startup_onboarding(agent, profile, user_id="default_user")

    assert agent.identity.onboarding_completed is True
    assert agent.identity.onboarding_profile["name"] == "Bob"
    assert agent.identity.last_user_id == "default_user"
    assert agent.identity.last_relationship_kwargs["display_name"] == "Bob"
    assert "Primary User Onboarding" in agent.memory_store.context
    assert "Desired Gwenn role: coding partner" in agent.memory_store.context
    assert agent.memory_store.saved_episode is not None
    assert agent.memory_store.saved_episode.category == "onboarding"


@pytest.mark.asyncio
async def test_recall_tool_forwards_category_filter():
    captured: dict = {}

    class _Registry:
        def __init__(self):
            self._tools = {"recall": SimpleNamespace(handler=None)}

        def get(self, name: str):
            return self._tools.get(name)

    def _retrieve(**kwargs):
        captured.update(kwargs)
        return [(SimpleNamespace(content="match"), 0.9)]

    agent = object.__new__(SentientAgent)
    agent.tool_registry = _Registry()
    agent.episodic_memory = SimpleNamespace(retrieve=_retrieve)
    dims = SimpleNamespace(valence=0.2)
    agent.affect_state = SimpleNamespace(dimensions=dims)

    SentientAgent._wire_builtin_tool_handlers(agent)
    recall = agent.tool_registry.get("recall")
    result = await recall.handler(query="python", category="conversation", max_results=3)

    assert "match" in result
    assert captured["category"] == "conversation"
    assert captured["top_k"] == 3


@pytest.mark.asyncio
async def test_consolidate_memories_persists_semantic_and_episode_flags():
    events: list[str] = []

    class _Consolidator:
        last_processed_episode_ids = ["ep-1"]

        def get_consolidation_prompt(self):
            return "prompt"

        def process_consolidation_response(self, _response_text: str):
            return {"facts": 1, "relationships": 0, "self_knowledge": 0, "patterns": 0}

    class _Store:
        def save_knowledge_node(self, **kwargs):
            events.append("save_knowledge_node")

        def clear_knowledge_edges(self):
            events.append("clear_knowledge_edges")

        def save_knowledge_edge(self, **kwargs):
            events.append("save_knowledge_edge")

        def save_episode(self, _episode):
            events.append("save_episode")

    async def _reflect(**kwargs):
        return {"content": "FACT: remembered detail | confidence: 0.9"}

    agent = object.__new__(SentientAgent)
    agent._config = SimpleNamespace(
        memory=SimpleNamespace(persist_semantic_after_consolidation=True),
    )
    agent.consolidator = _Consolidator()
    agent.memory_store = _Store()
    agent.engine = SimpleNamespace(
        reflect=_reflect,
        extract_text=lambda _resp: "FACT: remembered detail | confidence: 0.9",
    )
    node = SimpleNamespace(
        node_id="node-1",
        label="label",
        category="fact",
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
    agent.semantic_memory = SimpleNamespace(_nodes={"node-1": node}, _edges=[edge])
    agent.episodic_memory = SimpleNamespace(
        get_episode=lambda episode_id: SimpleNamespace(episode_id=episode_id),
    )
    agent.identity = SimpleNamespace(total_heartbeats=0)

    await SentientAgent.consolidate_memories(agent)

    assert "save_knowledge_node" in events
    assert "clear_knowledge_edges" in events
    assert "save_knowledge_edge" in events
    assert "save_episode" in events


def test_capture_evicted_working_memory_records_episode():
    encoded = []
    persisted = []

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent.episodic_memory = SimpleNamespace(encode=lambda ep: encoded.append(ep))
    agent.memory_store = SimpleNamespace(save_episode=lambda ep: persisted.append(ep))

    item = WorkingMemoryItem(
        item_id="wm-1",
        content="Critical short-term detail",
        category="task_state",
        salience=0.7,
        emotional_valence=0.2,
    )

    SentientAgent._capture_evicted_working_memory(agent, item)

    assert len(encoded) == 1
    assert len(persisted) == 1
    assert encoded[0].category == "working_memory_eviction"
    assert "Critical short-term detail" in encoded[0].content
