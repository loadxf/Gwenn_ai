"""
Runtime behavior tests for gwenn.agent.SentientAgent.

These tests use lightweight stubs to verify wiring and sequencing logic in the
agent runtime without making network calls.
"""

from __future__ import annotations

import time
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gwenn.agent import SentientAgent
from gwenn.config import GwennConfig
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
        self.core_values = []
        self.growth_moments = []
        self.growth_events: list[tuple[str, str, float]] = []

    def update_relationship(self, user_id: str, **kwargs) -> None:
        self.last_user_id = user_id
        self.last_relationship_kwargs = kwargs

    def _save(self) -> None:
        self.saved = True

    def mark_onboarding_completed(self, profile: dict[str, str]) -> None:
        self.onboarding_completed = True
        self.onboarding_profile = profile

    def generate_self_prompt(self) -> str:
        return "I am Gwenn."

    def record_growth(self, description: str, domain: str, significance: float = 0.5) -> None:
        self.growth_events.append((description, domain, significance))


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

    def save_working_memory(self, items: list) -> None:
        self._events.append("save_working_memory")

    def prune_old_episodes(self, **kwargs) -> int:
        self._events.append("prune_old_episodes")
        return 0

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
    agent._initialized = True

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
    agent._initialized = True
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
    agent.working_memory = SimpleNamespace(
        to_dict=lambda: {"items": []},
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


def test_fetch_url_tool_blocks_localhost_targets():
    class _Registry:
        def __init__(self):
            self._tools = {"fetch_url": SimpleNamespace(handler=None)}

        def get(self, name: str):
            return self._tools.get(name)

    agent = object.__new__(SentientAgent)
    agent.tool_registry = _Registry()
    SentientAgent._wire_builtin_tool_handlers(agent)
    fetch = agent.tool_registry.get("fetch_url")

    result = fetch.handler(url="http://127.0.0.1:8000")
    assert "blocked by network safety policy" in result


def test_calculate_tool_evaluates_safe_expression():
    class _Registry:
        def __init__(self):
            self._tools = {"calculate": SimpleNamespace(handler=None)}

        def get(self, name: str):
            return self._tools.get(name)

    agent = object.__new__(SentientAgent)
    agent.tool_registry = _Registry()
    SentientAgent._wire_builtin_tool_handlers(agent)
    calc = agent.tool_registry.get("calculate")

    result = calc.handler(expression="round(pi * 2, 3)")
    assert "6.283" in result


def test_calculate_tool_rejects_code_execution_payloads():
    class _Registry:
        def __init__(self):
            self._tools = {"calculate": SimpleNamespace(handler=None)}

        def get(self, name: str):
            return self._tools.get(name)

    agent = object.__new__(SentientAgent)
    agent.tool_registry = _Registry()
    SentientAgent._wire_builtin_tool_handlers(agent)
    calc = agent.tool_registry.get("calculate")

    result = calc.handler(expression="__import__('os').system('id')")
    assert "Error evaluating" in result


def test_fetch_url_tool_blocks_private_dns_resolution(monkeypatch):
    class _Registry:
        def __init__(self):
            self._tools = {"fetch_url": SimpleNamespace(handler=None)}

        def get(self, name: str):
            return self._tools.get(name)

    def _fake_getaddrinfo(_host: str, port: int, **_kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.5", port)),
        ]

    monkeypatch.setattr("socket.getaddrinfo", _fake_getaddrinfo)

    agent = object.__new__(SentientAgent)
    agent.tool_registry = _Registry()
    SentientAgent._wire_builtin_tool_handlers(agent)
    fetch = agent.tool_registry.get("fetch_url")

    result = fetch.handler(url="https://example.com")
    assert "blocked by network safety policy" in result


def test_fetch_url_tool_allows_public_targets(monkeypatch):
    class _Registry:
        def __init__(self):
            self._tools = {"fetch_url": SimpleNamespace(handler=None)}

        def get(self, name: str):
            return self._tools.get(name)

    class _FakeSocket:
        def __init__(self) -> None:
            self.sent = b""

        def sendall(self, payload: bytes) -> None:
            self.sent += payload

        def close(self) -> None:
            return None

    class _FakeHTTPResponse:
        status = 200
        reason = "OK"

        def __init__(self, _sock) -> None:
            pass

        def begin(self) -> None:
            return None

        def getheader(self, name: str, default: str = "") -> str:
            return "text/plain" if name == "Content-Type" else default

        def read(self, _size: int = -1) -> bytes:
            return b"public payload"

    def _fake_getaddrinfo(_host: str, port: int, **_kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port)),
        ]

    captured: dict[str, tuple[str, int]] = {}

    def _fake_create_connection(address: tuple[str, int], timeout: int = 10):
        captured["address"] = address
        return _FakeSocket()

    monkeypatch.setattr("socket.getaddrinfo", _fake_getaddrinfo)
    monkeypatch.setattr("socket.create_connection", _fake_create_connection)
    monkeypatch.setattr("http.client.HTTPResponse", _FakeHTTPResponse)

    agent = object.__new__(SentientAgent)
    agent.tool_registry = _Registry()
    SentientAgent._wire_builtin_tool_handlers(agent)
    fetch = agent.tool_registry.get("fetch_url")

    result = fetch.handler(url="http://example.com/data", max_chars=200)
    assert "URL: http://example.com/data" in result
    assert "public payload" in result
    assert captured["address"] == ("93.184.216.34", 80)


def test_fetch_url_tool_truncates_large_responses(monkeypatch):
    class _Registry:
        def __init__(self):
            self._tools = {"fetch_url": SimpleNamespace(handler=None)}

        def get(self, name: str):
            return self._tools.get(name)

    class _FakeSocket:
        def sendall(self, _payload: bytes) -> None:
            return None

        def close(self) -> None:
            return None

    class _ChunkedHTTPResponse:
        status = 200
        reason = "OK"

        def __init__(self, _sock) -> None:
            self._chunks = [
                b"A" * 9000,
                b"B" * 9000,
                b"C" * 9000,
                b"",
            ]

        def begin(self) -> None:
            return None

        def getheader(self, name: str, default: str = "") -> str:
            return "text/plain" if name == "Content-Type" else default

        def read(self, _size: int = -1) -> bytes:
            return self._chunks.pop(0)

    def _fake_getaddrinfo(_host: str, port: int, **_kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port)),
        ]

    monkeypatch.setattr("socket.getaddrinfo", _fake_getaddrinfo)
    monkeypatch.setattr("socket.create_connection", lambda *_args, **_kwargs: _FakeSocket())
    monkeypatch.setattr("http.client.HTTPResponse", _ChunkedHTTPResponse)

    agent = object.__new__(SentientAgent)
    agent.tool_registry = _Registry()
    SentientAgent._wire_builtin_tool_handlers(agent)
    fetch = agent.tool_registry.get("fetch_url")

    result = fetch.handler(url="http://example.com/huge", max_chars=200)
    assert "Truncated" in result


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

        def prune_old_episodes(self, **kwargs) -> int:
            return 0

    async def _reflect(**kwargs):
        return {"content": "FACT: remembered detail | confidence: 0.9"}

    agent = object.__new__(SentientAgent)
    agent._initialized = True
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
        decay=lambda rate=0.001: None,
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


@pytest.mark.asyncio
async def test_respond_requires_initialize():
    agent = object.__new__(SentientAgent)
    agent._initialized = False

    with pytest.raises(RuntimeError, match="initialized"):
        await SentientAgent.respond(agent, "hello")


@pytest.mark.asyncio
async def test_shutdown_before_initialize_is_safe():
    events: list[str] = []
    agent = object.__new__(SentientAgent)
    agent._initialized = False
    agent.memory_store = SimpleNamespace(close=lambda: events.append("close"))

    await SentientAgent.shutdown(agent)
    assert events == ["close"]


@pytest.mark.asyncio
async def test_consolidate_memories_no_prompt_marks_checked():
    marker = {"called": False}

    class _Consolidator:
        def get_consolidation_prompt(self):
            return None

        def mark_checked_no_work(self):
            marker["called"] = True

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent.consolidator = _Consolidator()

    await SentientAgent.consolidate_memories(agent)
    assert marker["called"] is True


@pytest.mark.asyncio
async def test_consolidate_memories_persists_episode_flags_when_semantic_flush_disabled():
    events: list[str] = []

    class _Consolidator:
        last_processed_episode_ids = ["ep-1"]

        def get_consolidation_prompt(self):
            return "prompt"

        def process_consolidation_response(self, _response_text: str):
            return {"facts": 1, "relationships": 0, "self_knowledge": 0, "patterns": 0}

    class _Store:
        def save_episode(self, _episode):
            events.append("save_episode")

        def save_knowledge_node(self, **kwargs):
            events.append("save_knowledge_node")

        def clear_knowledge_edges(self):
            events.append("clear_knowledge_edges")

        def save_knowledge_edge(self, **kwargs):
            events.append("save_knowledge_edge")

    async def _reflect(**kwargs):
        return {"content": "FACT: remembered detail | confidence: 0.9"}

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent._config = SimpleNamespace(
        memory=SimpleNamespace(persist_semantic_after_consolidation=False),
    )
    agent.consolidator = _Consolidator()
    agent.memory_store = _Store()
    agent.engine = SimpleNamespace(
        reflect=_reflect,
        extract_text=lambda _resp: "FACT: remembered detail | confidence: 0.9",
    )
    agent.semantic_memory = SimpleNamespace(_nodes={}, _edges=[])
    agent.episodic_memory = SimpleNamespace(
        get_episode=lambda episode_id: SimpleNamespace(episode_id=episode_id),
    )

    await SentientAgent.consolidate_memories(agent)

    assert "save_episode" in events
    assert "save_knowledge_node" not in events


@pytest.mark.asyncio
async def test_shutdown_persists_identity_snapshot_when_supported():
    events: list[str] = []

    class _Store:
        def save_affect_snapshot(self, **kwargs):
            events.append("save_affect")

        def save_episode(self, _episode):
            events.append("save_episode")

        def save_knowledge_node(self, **kwargs):
            events.append("save_knowledge_node")

        def clear_knowledge_edges(self):
            events.append("clear_knowledge_edges")

        def save_knowledge_edge(self, **kwargs):
            events.append("save_knowledge_edge")

        def save_identity_snapshot(self, **kwargs):
            events.append(f"snapshot:{kwargs.get('trigger')}")

        def save_working_memory(self, items: list) -> None:
            pass

        def close(self):
            events.append("close")

    agent = object.__new__(SentientAgent)
    agent._initialized = True
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
    agent.memory_store = _Store()
    agent.episodic_memory = SimpleNamespace(retrieve_recent=lambda n=100: [])
    agent.semantic_memory = SimpleNamespace(_nodes={}, _edges=[])
    agent.working_memory = SimpleNamespace(to_dict=lambda: {"items": []})
    agent.identity = _IdentityStub()
    agent._start_time = time.time() - 2.0

    async def _consolidate():
        events.append("consolidate")

    agent.consolidate_memories = _consolidate

    await SentientAgent.shutdown(agent)

    assert "snapshot:shutdown" in events


@pytest.mark.asyncio
async def test_consolidate_memories_persists_identity_emotional_insights():
    class _Consolidator:
        last_processed_episode_ids = ["ep-1"]
        last_emotional_insights = [
            {"content": "Creative tasks increase engagement", "confidence": 0.7},
        ]

        def get_consolidation_prompt(self):
            return "prompt"

        def process_consolidation_response(self, _response_text: str):
            return {
                "facts": 0,
                "relationships": 0,
                "self_knowledge": 0,
                "patterns": 0,
                "emotional_insights": 1,
            }

    events: list[str] = []

    class _Store:
        def save_episode(self, _episode):
            events.append("save_episode")

        def save_identity_snapshot(self, **kwargs):
            events.append(f"snapshot:{kwargs.get('trigger')}")

    async def _reflect(**kwargs):
        return {"content": "EMOTIONAL_INSIGHT: Creative tasks increase engagement | confidence: 0.7"}

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent._config = SimpleNamespace(
        memory=SimpleNamespace(persist_semantic_after_consolidation=False),
    )
    agent.consolidator = _Consolidator()
    agent.memory_store = _Store()
    agent.engine = SimpleNamespace(
        reflect=_reflect,
        extract_text=lambda _resp: "EMOTIONAL_INSIGHT: Creative tasks increase engagement | confidence: 0.7",
    )
    agent.semantic_memory = SimpleNamespace(_nodes={}, _edges=[])
    agent.episodic_memory = SimpleNamespace(
        get_episode=lambda episode_id: SimpleNamespace(episode_id=episode_id),
    )
    agent.identity = _IdentityStub()

    await SentientAgent.consolidate_memories(agent)

    assert agent.identity.growth_events
    assert "snapshot:consolidation" in events


def test_decay_and_prune_removes_fully_decayed_nodes():
    """_decay_and_prune_semantic_nodes prunes nodes at floor confidence (<=0.05)."""
    from gwenn.memory.semantic import KnowledgeNode, KnowledgeEdge

    agent = object.__new__(SentientAgent)

    # Two nodes: one healthy, one at floor confidence
    healthy = KnowledgeNode(node_id="healthy", label="healthy", confidence=0.8)
    stale = KnowledgeNode(node_id="stale", label="stale", confidence=0.05)
    edge_between = KnowledgeEdge(
        source_id="healthy", target_id="stale", relationship="related_to"
    )
    edge_internal = KnowledgeEdge(
        source_id="healthy", target_id="healthy", relationship="self_ref"
    )

    from gwenn.memory.semantic import SemanticMemory
    agent.semantic_memory = SemanticMemory()
    agent.semantic_memory._nodes = {"healthy": healthy, "stale": stale}
    agent.semantic_memory._label_index = {"healthy": "healthy", "stale": "stale"}
    agent.semantic_memory._edges = [edge_between, edge_internal]

    SentientAgent._decay_and_prune_semantic_nodes(agent)

    # 'stale' should be pruned (confidence 0.05 <= 0.05 after decay)
    assert "stale" not in agent.semantic_memory._nodes
    assert "stale" not in agent.semantic_memory._label_index
    # Edge to stale node removed; self-ref edge on healthy kept
    assert all(
        e.target_id != "stale" and e.source_id != "stale"
        for e in agent.semantic_memory._edges
    )
    assert "healthy" in agent.semantic_memory._nodes


def test_upsert_context_section_second_note_prepended():
    """A second note to the same section appears directly after the header."""
    from gwenn.agent import _upsert_context_section
    after_first = _upsert_context_section("", "reminders", "first note")
    after_second = _upsert_context_section(after_first, "reminders", "second note")
    assert "## Reminders" in after_second
    assert "- first note" in after_second
    assert "- second note" in after_second
    assert after_second.count("## Reminders") == 1


def test_agent_init_wires_sensory_ethics_interagent_from_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GWENN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GWENN_EPISODIC_DB", str(tmp_path / "gwenn.db"))
    monkeypatch.setenv("GWENN_SEMANTIC_DB", str(tmp_path / "semantic_vectors"))
    monkeypatch.setenv("GWENN_MAX_PERCEPTS_PER_CHANNEL", "3")
    monkeypatch.setenv("GWENN_PERCEPT_EXPIRY", "12")
    monkeypatch.setenv("GWENN_ETHICS_HISTORY_SIZE", "7")
    monkeypatch.setenv("GWENN_ETHICS_CONCERN_THRESHOLD", "0.65")
    monkeypatch.setenv("GWENN_AGENT_ID", "gwenn-test")
    monkeypatch.setenv("GWENN_INTERAGENT_BUFFER_SIZE", "9")
    monkeypatch.setenv("GWENN_MCP_SERVERS", "[]")

    captured: dict[str, object] = {}

    class _SensoryStub:
        def __init__(self, max_percepts_per_channel: int, percept_expiry_seconds: float):
            captured["sensory_max"] = max_percepts_per_channel
            captured["sensory_expiry"] = percept_expiry_seconds

    class _EthicsStub:
        def __init__(self, assessment_history_size: int, concern_threshold: float):
            captured["ethics_history_size"] = assessment_history_size
            captured["ethics_threshold"] = concern_threshold

    class _InterAgentStub:
        def __init__(self, self_id: str, message_buffer_size: int):
            captured["interagent_self_id"] = self_id
            captured["interagent_buffer_size"] = message_buffer_size

    monkeypatch.setattr("gwenn.agent.CognitiveEngine", lambda _cfg: SimpleNamespace())
    monkeypatch.setattr("gwenn.agent.SensoryIntegrator", _SensoryStub)
    monkeypatch.setattr("gwenn.agent.EthicalReasoner", _EthicsStub)
    monkeypatch.setattr("gwenn.agent.InterAgentBridge", _InterAgentStub)

    config = GwennConfig()
    SentientAgent(config)

    assert captured["sensory_max"] == 3
    assert captured["sensory_expiry"] == 12.0
    assert captured["ethics_history_size"] == 7
    assert captured["ethics_threshold"] == 0.65
    assert captured["interagent_self_id"] == "gwenn-test"
    assert captured["interagent_buffer_size"] == 9


@pytest.mark.asyncio
async def test_initialize_mcp_tools_uses_configured_servers():
    agent = object.__new__(SentientAgent)
    servers = [{"name": "demo", "transport": "stdio", "command": "echo"}]
    agent._config = SimpleNamespace(
        mcp=SimpleNamespace(get_server_list=lambda: servers),
    )
    agent._mcp_client = SimpleNamespace(
        initialize=AsyncMock(),
        discover_tools=AsyncMock(return_value=[]),
        register_tools=AsyncMock(return_value=0),
    )

    await SentientAgent._initialize_mcp_tools(agent)

    agent._mcp_client.initialize.assert_awaited_once_with(servers)
    agent._mcp_client.discover_tools.assert_awaited_once()
    agent._mcp_client.register_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_mcp_tools_skips_when_no_servers():
    agent = object.__new__(SentientAgent)
    agent._config = SimpleNamespace(
        mcp=SimpleNamespace(get_server_list=lambda: []),
    )
    agent._mcp_client = SimpleNamespace(
        initialize=AsyncMock(),
        discover_tools=AsyncMock(),
        register_tools=AsyncMock(),
    )

    await SentientAgent._initialize_mcp_tools(agent)

    agent._mcp_client.initialize.assert_not_awaited()
    agent._mcp_client.discover_tools.assert_not_awaited()
    agent._mcp_client.register_tools.assert_not_awaited()


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
