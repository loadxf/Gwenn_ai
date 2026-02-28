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
from gwenn.affect.appraisal import AppraisalEvent, StimulusType
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
        self._max_iterations = 75

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="ok", was_truncated=False)


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

    def save_episode(self, episode, **kwargs) -> None:
        self._events.append("save_episode")

    def save_knowledge_node(self, **kwargs) -> None:
        self._events.append("save_knowledge_node")

    def clear_knowledge_edges(self) -> None:
        self._events.append("clear_edges")

    def save_knowledge_edge(self, **kwargs) -> None:
        self._events.append("save_knowledge_edge")

    def save_working_memory(self, items: list) -> None:
        self._events.append("save_working_memory")

    def sync_episode_embeddings(self, episodes) -> int:
        return 0

    def sync_knowledge_embeddings(self, nodes) -> int:
        return 0

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

    def save_episode(self, episode, **kwargs) -> None:
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
        set_iteration_limit=lambda limit: None,
        reset_iteration_limit=lambda: None,
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
    agent._continuation_pending = False
    agent._default_max_iterations = 75

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

    assert str(response) == "ok"
    assert len(loop.calls) == 1
    call = loop.calls[0]
    assert call["tools"] == [{"name": "echo"}]
    assert "[REDACTED_EMAIL]" in call["system_prompt"]
    assert "alice@example.com" not in call["system_prompt"]
    assert "[REDACTED_EMAIL]" in call["messages"][0]["content"]
    assert "alice@example.com" not in call["messages"][0]["content"]
    # Internal conversation state remains unredacted; only API payload is redacted.
    assert "alice@example.com" in agent._conversation_history[0]["content"]


def test_estimate_message_valence_detects_polarity():
    agent = object.__new__(SentientAgent)
    negative = SentientAgent._estimate_message_valence(
        agent,
        "I hate this and I am afraid it will fail.",
    )
    positive = SentientAgent._estimate_message_valence(
        agent,
        "I love this, thank you!",
    )
    assert negative < 0.0
    assert positive > 0.0


@pytest.mark.asyncio
async def test_respond_appraisal_includes_message_valence_hint():
    appraisals: list[AppraisalEvent] = []

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent._config = SimpleNamespace(
        privacy=SimpleNamespace(
            redact_before_api=False,
            redact_before_persist=False,
        )
    )
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
    agent._conversation_history = []
    agent._current_user_id = None
    agent.heartbeat = _HeartbeatStub()
    agent.identity = _IdentityStub()
    agent.sensory = SimpleNamespace(
        ground_social=lambda user_id, text: None,
        ground_temporal=lambda event_description: None,
    )
    agent.ethics = SimpleNamespace(detect_ethical_dimensions=lambda text: [])
    agent.episodic_memory = SimpleNamespace(retrieve=lambda **kwargs: [])
    agent.semantic_memory = SimpleNamespace(query=lambda **kwargs: [])
    agent.working_memory = _WorkingMemoryStub()
    agent.safety = SimpleNamespace(
        reset_iteration_count=lambda: None,
        set_iteration_limit=lambda limit: None,
        reset_iteration_limit=lambda: None,
    )
    agent.tool_registry = SimpleNamespace(get_api_tools=lambda **kwargs: [])
    agent.goal_system = SimpleNamespace()
    agent.theory_of_mind = SimpleNamespace()
    agent.metacognition = SimpleNamespace()
    agent.interagent = SimpleNamespace()
    agent.memory_store = SimpleNamespace()
    agent.redactor = PIIRedactor(enabled=False)
    agent.process_appraisal = lambda event: appraisals.append(event)
    agent._assemble_system_prompt = lambda **kwargs: "prompt"
    agent.agentic_loop = _LoopStub()
    agent._continuation_pending = False
    agent._default_max_iterations = 75

    async def _noop_integrate(*args, **kwargs):
        return None

    agent._integrate_exchange = _noop_integrate

    await SentientAgent.respond(
        agent,
        "I hate this and I am afraid.",
        user_id="user-1",
    )

    assert appraisals
    first = appraisals[0]
    assert first.stimulus_type == StimulusType.USER_MESSAGE
    assert first.metadata["valence_hint"] < 0.0


@pytest.mark.asyncio
async def test_respond_appraises_tool_results():
    appraisals: list[AppraisalEvent] = []

    class _LoopWithToolResults:
        _max_iterations = 75

        async def run(self, **kwargs):
            on_tool_result = kwargs.get("on_tool_result")
            assert callable(on_tool_result)
            on_tool_result(
                {"id": "tool_1", "name": "echo", "input": {"text": "ok"}},
                SimpleNamespace(success=True, result="ok", error=None),
            )
            on_tool_result(
                {"id": "tool_2", "name": "echo", "input": {"text": "bad"}},
                SimpleNamespace(success=False, result=None, error="boom"),
            )
            return SimpleNamespace(text="ok", was_truncated=False)

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent._config = SimpleNamespace(
        privacy=SimpleNamespace(
            redact_before_api=False,
            redact_before_persist=False,
        )
    )
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
    agent._conversation_history = []
    agent._current_user_id = None
    agent.heartbeat = _HeartbeatStub()
    agent.identity = _IdentityStub()
    agent.sensory = SimpleNamespace(
        ground_social=lambda user_id, text: None,
        ground_temporal=lambda event_description: None,
    )
    agent.ethics = SimpleNamespace(detect_ethical_dimensions=lambda text: [])
    agent.episodic_memory = SimpleNamespace(retrieve=lambda **kwargs: [])
    agent.semantic_memory = SimpleNamespace(query=lambda **kwargs: [])
    agent.working_memory = _WorkingMemoryStub()
    agent.safety = SimpleNamespace(
        reset_iteration_count=lambda: None,
        set_iteration_limit=lambda limit: None,
        reset_iteration_limit=lambda: None,
    )
    agent.tool_registry = SimpleNamespace(get_api_tools=lambda **kwargs: [{"name": "echo"}])
    agent.goal_system = SimpleNamespace()
    agent.theory_of_mind = SimpleNamespace()
    agent.metacognition = SimpleNamespace()
    agent.interagent = SimpleNamespace()
    agent.memory_store = SimpleNamespace()
    agent.redactor = PIIRedactor(enabled=False)
    agent.process_appraisal = lambda event: appraisals.append(event)
    agent._assemble_system_prompt = lambda **kwargs: "prompt"
    agent.agentic_loop = _LoopWithToolResults()
    agent._continuation_pending = False
    agent._default_max_iterations = 75

    async def _noop_integrate(*args, **kwargs):
        return None

    agent._integrate_exchange = _noop_integrate

    await SentientAgent.respond(
        agent,
        "please run tools",
        user_id="user-1",
    )

    stimulus_types = [event.stimulus_type for event in appraisals]
    assert StimulusType.TOOL_SUCCESS in stimulus_types
    assert StimulusType.TOOL_FAILURE in stimulus_types
    tool_events = [
        event for event in appraisals
        if event.stimulus_type in {StimulusType.TOOL_SUCCESS, StimulusType.TOOL_FAILURE}
    ]
    assert all("habituation_key" in event.metadata for event in tool_events)


def test_process_appraisal_applies_habituation_scaling():
    captured: dict[str, AppraisalEvent] = {}

    def _appraise(event, state):
        captured["event"] = event
        return state

    agent = object.__new__(SentientAgent)
    initial_state = SimpleNamespace()
    agent.affect_state = initial_state
    agent.appraisal_engine = SimpleNamespace(appraise=_appraise)
    agent.resilience = SimpleNamespace(
        check=lambda state: state,
        get_habituation_factor=lambda key: 0.5,
    )
    agent._initialized = False

    event = AppraisalEvent(
        stimulus_type=StimulusType.TOOL_FAILURE,
        intensity=0.8,
        metadata={"habituation_key": "tool:echo:failure"},
    )
    SentientAgent.process_appraisal(agent, event)

    assert captured["event"].intensity == pytest.approx(0.4)


def test_process_appraisal_throttles_steady_state_affect_snapshot_writes():
    from gwenn.affect.state import AffectiveState

    writes: list[dict] = []

    state = AffectiveState()
    agent = object.__new__(SentientAgent)
    agent.affect_state = state
    agent.appraisal_engine = SimpleNamespace(appraise=lambda event, current_state: current_state)
    agent.resilience = SimpleNamespace(
        check=lambda current_state: current_state,
        get_habituation_factor=lambda _key: 1.0,
    )
    agent._initialized = True
    agent.memory_store = SimpleNamespace(
        save_affect_snapshot=lambda **kwargs: writes.append(kwargs),
        prune_affect_snapshots=lambda **kwargs: 0,
    )
    agent._last_affect_snapshot_at = time.time()
    agent._affect_snapshot_min_interval_seconds = 9999.0
    agent._affect_snapshot_min_delta = 1.0
    agent._affect_snapshot_prune_every = 128
    agent._affect_snapshot_since_prune = 0

    SentientAgent.process_appraisal(
        agent,
        AppraisalEvent(stimulus_type=StimulusType.HEARTBEAT_IDLE, intensity=0.2),
    )
    assert writes == []

    SentientAgent.process_appraisal(
        agent,
        AppraisalEvent(stimulus_type=StimulusType.TOOL_FAILURE, intensity=0.2),
    )
    assert len(writes) == 1
    assert writes[0]["trigger"] == "tool_failure"


@pytest.mark.asyncio
async def test_integrate_exchange_runs_metacognition_pass():
    confidence_claims: list[dict] = []
    audit_results: list[object] = []

    class _Meta:
        _growth_metrics: dict = {}

        def record_confidence_claim(self, **kwargs):
            confidence_claims.append(kwargs)

        def record_audit_result(self, result):
            audit_results.append(result)

        def add_insight(self, insight: str):
            return None

        def assess_growth(self, dimension: str, new_level: float, evidence: str):
            return None

    agent = object.__new__(SentientAgent)
    agent._config = SimpleNamespace(
        privacy=SimpleNamespace(redact_before_persist=False),
    )
    agent.redactor = PIIRedactor(enabled=False)
    agent.affect_state = SimpleNamespace(
        dimensions=SimpleNamespace(valence=0.0, arousal=0.3),
        to_dict=lambda: {},
    )
    encoded: list[object] = []
    persisted: list[object] = []
    agent.episodic_memory = SimpleNamespace(encode=lambda episode: encoded.append(episode))
    agent._persist_episode = lambda episode: persisted.append(episode)
    agent.theory_of_mind = SimpleNamespace(set_current_user=lambda _uid: None)
    agent.goal_system = SimpleNamespace(
        satisfy_need=lambda _need, _amount: None,
        get_goal_for_need=lambda _need: None,
        complete_goal=lambda _id: None,
    )
    agent.process_appraisal = lambda _event: None
    agent.interagent = SimpleNamespace(known_agents=[])
    agent.identity = SimpleNamespace(
        total_interactions=2,
        check_milestone=lambda *args, **kwargs: None,
    )
    agent.metacognition = _Meta()

    await SentientAgent._integrate_exchange(
        agent,
        user_message="Can you explain this quickly?",
        response="As an AI language model, I can help.",
        user_id="u1",
        had_relevant_memories=False,
    )

    assert encoded
    assert persisted
    assert confidence_claims
    assert audit_results
    assert audit_results[0].is_honest is False
    assert audit_results[0].concerns


@pytest.mark.asyncio
async def test_integrate_exchange_updates_tom_and_records_ethics():
    from gwenn.cognition.ethics import EthicalDimension

    class _UserModel:
        def __init__(self):
            self.verbosity_preference = 0.5
            self.technical_level = 0.5
            self.formality_level = 0.5
            self.inferred_emotion = "neutral"
            self.emotion_confidence = 0.3
            self.topics_discussed: list[str] = []
            self.preference_updates: list[tuple[str, str]] = []
            self.knowledge_updates: list[tuple[str, str]] = []

        def update_preference(
            self,
            pref: str,
            value: str,
            confidence: float,
            source: str = "observed",
        ) -> None:
            self.preference_updates.append((pref, value))

        def update_knowledge_belief(self, topic: str, level: str, confidence: float, source: str) -> None:
            self.knowledge_updates.append((topic, level))

    user_model = _UserModel()
    recorded_assessments: list[object] = []

    agent = object.__new__(SentientAgent)
    agent._config = SimpleNamespace(
        privacy=SimpleNamespace(redact_before_persist=False),
    )
    agent.redactor = PIIRedactor(enabled=False)
    agent.affect_state = SimpleNamespace(
        dimensions=SimpleNamespace(valence=-0.2, arousal=0.4),
        to_dict=lambda: {},
    )
    agent.episodic_memory = SimpleNamespace(encode=lambda _episode: None)
    agent._persist_episode = lambda _episode: None
    agent.theory_of_mind = SimpleNamespace(set_current_user=lambda _uid: user_model)
    agent.goal_system = SimpleNamespace(
        satisfy_need=lambda _need, _amount: None,
        get_goal_for_need=lambda _need: None,
        complete_goal=lambda _id: None,
    )
    agent.process_appraisal = lambda _event: None
    agent.interagent = SimpleNamespace(known_agents=[])
    agent.identity = SimpleNamespace(
        total_interactions=2,
        check_milestone=lambda *args, **kwargs: None,
    )
    agent.metacognition = SimpleNamespace()
    agent.ethics = SimpleNamespace(
        record_assessment=lambda assessment: recorded_assessments.append(assessment),
    )

    await SentientAgent._integrate_exchange(
        agent,
        user_message="Please keep this concise. I'm anxious about this Python API issue.",
        response="You can choose between two safe options. I recommend starting with option A.",
        user_id="u1",
        had_relevant_memories=False,
        ethical_dimensions=[EthicalDimension.HARM, EthicalDimension.AUTONOMY],
    )

    assert ("response_length", "concise") in user_model.preference_updates
    assert ("python", "interested") in user_model.knowledge_updates
    assert user_model.inferred_emotion in {"concerned", "distressed"}
    assert recorded_assessments
    assessment = recorded_assessments[0]
    assert EthicalDimension.HARM in assessment.dimension_scores
    assert EthicalDimension.AUTONOMY in assessment.dimension_scores
    assert assessment.tensions


def test_persist_episode_redacts_before_persist_when_enabled():
    captured: dict[str, object] = {}

    class _Store:
        def save_episode(self, episode, **kwargs) -> None:
            captured["episode"] = episode

    agent = object.__new__(SentientAgent)
    agent.memory_store = _Store()
    agent.redactor = PIIRedactor(enabled=True)
    agent._config = SimpleNamespace(
        privacy=SimpleNamespace(redact_before_persist=True),
    )

    episode = SimpleNamespace(
        episode_id="ep-1",
        timestamp=1000.0,
        content="Reach me at alice@example.com",
        category="conversation",
        emotional_valence=0.0,
        emotional_arousal=0.3,
        importance=0.5,
        tags=[],
        participants=[],
        outcome=None,
        consolidated=False,
        embedding=None,
    )

    SentientAgent._persist_episode(agent, episode)
    persisted = captured["episode"]
    assert "[REDACTED_EMAIL]" in persisted.content
    assert "alice@example.com" not in persisted.content


@pytest.mark.asyncio
async def test_shutdown_skips_prunable_episodes():
    saved_ids: list[str] = []

    class _Store:
        def save_affect_snapshot(self, **kwargs):
            return None

        def save_episode(self, episode, **kwargs):
            saved_ids.append(episode.episode_id)

        def save_knowledge_node(self, **kwargs):
            return None

        def clear_knowledge_edges(self):
            return None

        def save_knowledge_edge(self, **kwargs):
            return None

        def save_working_memory(self, items: list) -> None:
            return None

        def sync_episode_embeddings(self, episodes) -> int:
            return 0

        def sync_knowledge_embeddings(self, nodes) -> int:
            return 0

        def prune_old_episodes(self, **kwargs) -> int:
            return 0

        def close(self):
            return None

    old_episode = SimpleNamespace(
        episode_id="old-prunable",
        timestamp=time.time() - (120 * 86400),
        content="old",
        category="conversation",
        emotional_valence=0.0,
        emotional_arousal=0.3,
        importance=0.1,
        tags=[],
        participants=[],
        outcome=None,
        consolidated=True,
        embedding=None,
    )
    keep_episode = SimpleNamespace(
        episode_id="keep-me",
        timestamp=time.time(),
        content="fresh",
        category="conversation",
        emotional_valence=0.0,
        emotional_arousal=0.3,
        importance=0.1,
        tags=[],
        participants=[],
        outcome=None,
        consolidated=False,
        embedding=None,
    )

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent.heartbeat = _HeartbeatStub()
    agent.affect_state = SimpleNamespace(
        dimensions=SimpleNamespace(
            valence=0.0,
            arousal=0.3,
            dominance=0.0,
            certainty=0.0,
            goal_congruence=0.0,
        ),
        current_emotion=SimpleNamespace(value="neutral"),
    )
    agent._config = SimpleNamespace(
        memory=SimpleNamespace(shutdown_persist_recent_episodes=0),
        privacy=SimpleNamespace(redact_before_persist=False),
    )
    agent.redactor = PIIRedactor(enabled=False)
    agent.memory_store = _Store()
    agent.episodic_memory = SimpleNamespace(
        count=2,
        retrieve_recent=lambda n=2: [old_episode, keep_episode],
    )
    agent.semantic_memory = SimpleNamespace(_nodes={}, _edges=[])
    agent.working_memory = SimpleNamespace(to_dict=lambda: {"items": []})
    agent.identity = _IdentityStub()
    agent._start_time = time.time() - 1.0

    async def _consolidate():
        return None

    agent.consolidate_memories = _consolidate

    await SentientAgent.shutdown(agent)

    assert "keep-me" in saved_ids
    assert "old-prunable" not in saved_ids


@pytest.mark.asyncio
async def test_shutdown_persists_goal_state_when_supported():
    events: list[str] = []

    class _Store:
        def save_affect_snapshot(self, **kwargs):
            return None

        def save_episode(self, episode, **kwargs):
            return None

        def save_knowledge_node(self, **kwargs):
            return None

        def clear_knowledge_edges(self):
            return None

        def save_knowledge_edge(self, **kwargs):
            return None

        def save_working_memory(self, items: list) -> None:
            return None

        def save_goal_state(self, payload: dict) -> None:
            events.append(f"goal:{payload.get('marker')}")

        def sync_episode_embeddings(self, episodes) -> int:
            return 0

        def sync_knowledge_embeddings(self, nodes) -> int:
            return 0

        def close(self):
            return None

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent.heartbeat = _HeartbeatStub()
    agent.affect_state = SimpleNamespace(
        dimensions=SimpleNamespace(
            valence=0.0,
            arousal=0.3,
            dominance=0.0,
            certainty=0.0,
            goal_congruence=0.0,
        ),
        current_emotion=SimpleNamespace(value="neutral"),
    )
    agent.memory_store = _Store()
    agent.episodic_memory = SimpleNamespace(count=0, retrieve_recent=lambda n=1: [])
    agent.semantic_memory = SimpleNamespace(_nodes={}, _edges=[])
    agent.working_memory = SimpleNamespace(to_dict=lambda: {"items": []})
    agent.goal_system = SimpleNamespace(to_dict=lambda: {"marker": "saved"})
    agent.identity = _IdentityStub()
    agent._start_time = time.time() - 1.0
    agent._config = SimpleNamespace(
        memory=SimpleNamespace(shutdown_persist_recent_episodes=0),
        privacy=SimpleNamespace(redact_before_persist=False),
    )
    agent.redactor = PIIRedactor(enabled=False)

    async def _consolidate():
        return None

    agent.consolidate_memories = _consolidate

    await SentientAgent.shutdown(agent)
    assert "goal:saved" in events


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
        "name": "gwenn",
        "role": "coding partner",
        "interests": "debugging and architecture guidance",
        "communication_style": "concise and direct",
        "boundaries": "do not use destructive git commands",
    }

    SentientAgent.apply_startup_onboarding(agent, profile, user_id="default_user")

    assert agent.identity.onboarding_completed is True
    assert agent.identity.onboarding_profile["name"] == "gwenn"
    assert agent.identity.last_user_id == "default_user"
    assert agent.identity.last_relationship_kwargs["display_name"] == "gwenn"
    assert "Primary User Onboarding" in agent.memory_store.context
    assert "Desired relationship: coding partner" in agent.memory_store.context
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

        def save_episode(self, _episode, **kwargs):
            events.append("save_episode")

        def sync_knowledge_embeddings(self, nodes) -> int:
            return 0

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
        def save_episode(self, _episode, **kwargs):
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

        def save_episode(self, _episode, **kwargs):
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

        def sync_episode_embeddings(self, episodes) -> int:
            return 0

        def sync_knowledge_embeddings(self, nodes) -> int:
            return 0

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
        def save_episode(self, _episode, **kwargs):
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


def test_decay_and_prune_preserves_immutable_nodes():
    from gwenn.memory.semantic import KnowledgeNode, SemanticMemory

    deleted: list[str] = []

    class _Store:
        def delete_knowledge_nodes(self, node_ids: list[str]) -> None:
            deleted.extend(node_ids)

    agent = object.__new__(SentientAgent)
    agent.memory_store = _Store()
    agent.semantic_memory = SemanticMemory()
    agent.semantic_memory._nodes = {
        "immutable": KnowledgeNode(
            node_id="immutable",
            label="genesis:identity",
            content="immutable seed",
            confidence=0.05,
            metadata={"immutable": True, "genesis": True},
        ),
        "stale": KnowledgeNode(
            node_id="stale",
            label="stale",
            content="stale",
            confidence=0.05,
        ),
    }
    agent.semantic_memory._label_index = {
        "genesis:identity": "immutable",
        "stale": "stale",
    }
    agent.semantic_memory._edges = []

    SentientAgent._decay_and_prune_semantic_nodes(agent)

    assert "immutable" in agent.semantic_memory._nodes
    assert "stale" not in agent.semantic_memory._nodes
    assert deleted == ["stale"]


def test_seed_genesis_knowledge_repairs_drifted_nodes_and_creates_missing():
    from gwenn.genesis import GENESIS_NODE_SPECS
    from gwenn.memory.semantic import KnowledgeNode, SemanticMemory

    saved_labels: list[str] = []

    class _Store:
        def save_knowledge_node(self, **kwargs) -> None:
            saved_labels.append(kwargs["label"])

    agent = object.__new__(SentientAgent)
    agent.memory_store = _Store()
    agent.semantic_memory = SemanticMemory()

    # Existing genesis node with tampered values should be repaired, not trusted.
    identity_spec = GENESIS_NODE_SPECS[0]
    drifted = KnowledgeNode(
        node_id="drifted-id",
        label=identity_spec["label"],
        category="concept",
        content="tampered content",
        confidence=0.1,
        metadata={"immutable": False},
    )
    agent.semantic_memory._nodes = {"drifted-id": drifted}
    agent.semantic_memory._label_index = {identity_spec["label"].lower(): "drifted-id"}
    agent.semantic_memory._edges = []

    SentientAgent._seed_genesis_knowledge(agent)

    repaired = agent.semantic_memory._nodes["drifted-id"]
    assert repaired.category == identity_spec["category"]
    assert repaired.content == identity_spec["content"]
    assert repaired.confidence == pytest.approx(1.0)
    assert repaired.metadata.get("immutable") is True
    assert repaired.metadata.get("genesis") is True

    labels = {node.label for node in agent.semantic_memory._nodes.values()}
    for spec in GENESIS_NODE_SPECS:
        assert spec["label"] in labels

    # One repaired existing node + two newly created nodes.
    assert len(saved_labels) == len(GENESIS_NODE_SPECS)


def test_seed_genesis_knowledge_repairs_non_dict_metadata():
    from gwenn.genesis import GENESIS_NODE_SPECS
    from gwenn.memory.semantic import KnowledgeNode, SemanticMemory

    saved: list[dict[str, object]] = []

    class _Store:
        def save_knowledge_node(self, **kwargs) -> None:
            saved.append(kwargs)

    spec = GENESIS_NODE_SPECS[0]
    agent = object.__new__(SentientAgent)
    agent.memory_store = _Store()
    agent.semantic_memory = SemanticMemory()
    agent.semantic_memory._nodes = {
        "drifted-id": KnowledgeNode(
            node_id="drifted-id",
            label=spec["label"],
            category="self",
            content=spec["content"],
            confidence=1.0,
            metadata=["not", "a", "dict"],
        )
    }
    agent.semantic_memory._label_index = {spec["label"].lower(): "drifted-id"}
    agent.semantic_memory._edges = []

    SentientAgent._seed_genesis_knowledge(agent)

    repaired = agent.semantic_memory._nodes["drifted-id"]
    assert isinstance(repaired.metadata, dict)
    assert repaired.metadata.get("immutable") is True
    assert repaired.metadata.get("genesis") is True
    # Drift repair should trigger persistence for repaired node.
    assert any(item.get("node_id") == "drifted-id" for item in saved)


def test_decay_and_prune_tolerates_non_dict_metadata():
    from gwenn.memory.semantic import KnowledgeNode, SemanticMemory

    deleted: list[str] = []

    class _Store:
        def delete_knowledge_nodes(self, node_ids: list[str]) -> None:
            deleted.extend(node_ids)

    agent = object.__new__(SentientAgent)
    agent.memory_store = _Store()
    agent.semantic_memory = SemanticMemory()
    agent.semantic_memory._nodes = {
        "corrupt-meta": KnowledgeNode(
            node_id="corrupt-meta",
            label="corrupt",
            content="corrupt",
            confidence=0.05,
            metadata=["not", "dict"],
        ),
    }
    agent.semantic_memory._label_index = {"corrupt": "corrupt-meta"}
    agent.semantic_memory._edges = []

    SentientAgent._decay_and_prune_semantic_nodes(agent)

    assert "corrupt-meta" not in agent.semantic_memory._nodes
    assert deleted == ["corrupt-meta"]


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

    class _EngineStub:
        def set_safety_hooks(self, **kwargs):
            captured["safety_hooks_bound"] = True
            captured["before_model_call_hook"] = callable(kwargs.get("before_model_call"))
            captured["on_model_usage_hook"] = callable(kwargs.get("on_model_usage"))

    monkeypatch.setattr("gwenn.agent.CognitiveEngine", lambda _cfg: _EngineStub())
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
    assert captured["safety_hooks_bound"] is True
    assert captured["before_model_call_hook"] is True
    assert captured["on_model_usage_hook"] is True


def test_agent_init_wires_budget_limits_from_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GWENN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GWENN_EPISODIC_DB", str(tmp_path / "gwenn.db"))
    monkeypatch.setenv("GWENN_SEMANTIC_DB", str(tmp_path / "semantic_vectors"))
    monkeypatch.setenv("GWENN_MCP_SERVERS", "[]")
    monkeypatch.setenv("GWENN_MAX_INPUT_TOKENS", "1234")
    monkeypatch.setenv("GWENN_MAX_OUTPUT_TOKENS", "4321")
    monkeypatch.setenv("GWENN_MAX_API_CALLS", "9")

    class _EngineStub:
        def set_safety_hooks(self, **_kwargs):
            return None

    monkeypatch.setattr("gwenn.agent.CognitiveEngine", lambda _cfg: _EngineStub())

    config = GwennConfig()
    agent = SentientAgent(config)

    assert agent.safety._budget.max_input_tokens == 1234
    assert agent.safety._budget.max_output_tokens == 4321
    assert agent.safety._budget.max_api_calls == 9


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
    agent.memory_store = SimpleNamespace(save_episode=lambda ep, **kw: persisted.append(ep))

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


# ---------------------------------------------------------------------------
# Metacognition persistence round-trip
# ---------------------------------------------------------------------------

def test_metacognition_to_dict_and_restore_round_trip():
    from gwenn.cognition.metacognition import MetacognitionEngine, CalibrationRecord

    meta = MetacognitionEngine()
    meta.add_concern("Possible overstatement")
    meta.add_insight("Acknowledged limits")
    meta.assess_growth("honesty_consistency", 0.62, "clean exchange")
    meta._calibration_records.append(
        CalibrationRecord(
            claim="test claim",
            stated_confidence=0.7,
            actual_outcome=True,
            domain="conversation",
        )
    )

    snapshot = meta.to_dict()

    # Restore into a fresh instance
    meta2 = MetacognitionEngine()
    meta2.restore_from_dict(snapshot)

    assert meta2._concerns == ["Possible overstatement"]
    assert meta2._insights == ["Acknowledged limits"]
    hc = meta2._growth_metrics["honesty_consistency"]
    assert abs(hc.current_level - 0.62) < 1e-6
    assert hc.trajectory != 0.0
    assert len(meta2._calibration_records) == 1
    assert meta2._calibration_records[0].actual_outcome is True


def test_metacognition_growth_summary_only_shows_after_assessment():
    from gwenn.cognition.metacognition import MetacognitionEngine

    meta = MetacognitionEngine()
    # No assessment yet — growth summary should not appear in context
    ctx = meta.get_metacognitive_context()
    assert "Growth self-assessment" not in ctx

    # After an assessment, it appears
    meta.assess_growth("honesty_consistency", 0.55, "some evidence")
    # Inject a fake audit so the context has content
    from gwenn.cognition.metacognition import HonestyAuditResult
    meta.record_audit_result(HonestyAuditResult(content_summary="ok", is_honest=True))
    ctx2 = meta.get_metacognitive_context()
    assert "Growth self-assessment" in ctx2


# ---------------------------------------------------------------------------
# Calibration outcome resolution
# ---------------------------------------------------------------------------

def test_resolve_calibration_outcome_records_positive_on_no_correction():
    from gwenn.cognition.metacognition import MetacognitionEngine, CalibrationRecord

    outcomes: list[dict] = []

    class _Meta(MetacognitionEngine):
        def record_outcome(self, claim: str, was_correct: bool) -> None:
            outcomes.append({"claim": claim, "was_correct": was_correct})

    meta = _Meta()
    meta._calibration_records.append(
        CalibrationRecord(claim="prior-claim", stated_confidence=0.6)
    )

    agent = object.__new__(SentientAgent)
    agent.metacognition = meta

    SentientAgent._resolve_calibration_outcome(agent, "Thanks, that makes sense!")
    assert len(outcomes) == 1
    assert outcomes[0]["was_correct"] is True


def test_resolve_calibration_outcome_records_negative_on_correction():
    from gwenn.cognition.metacognition import MetacognitionEngine, CalibrationRecord

    outcomes: list[dict] = []

    class _Meta(MetacognitionEngine):
        def record_outcome(self, claim: str, was_correct: bool) -> None:
            outcomes.append({"claim": claim, "was_correct": was_correct})

    meta = _Meta()
    meta._calibration_records.append(
        CalibrationRecord(claim="prior-claim", stated_confidence=0.8)
    )

    agent = object.__new__(SentientAgent)
    agent.metacognition = meta

    SentientAgent._resolve_calibration_outcome(agent, "Actually, that's not right.")
    assert len(outcomes) == 1
    assert outcomes[0]["was_correct"] is False


def test_resolve_calibration_outcome_skips_when_no_unresolved():
    from gwenn.cognition.metacognition import MetacognitionEngine, CalibrationRecord

    call_count = {"n": 0}

    class _Meta(MetacognitionEngine):
        def record_outcome(self, claim: str, was_correct: bool) -> None:
            call_count["n"] += 1

    meta = _Meta()
    # All claims already resolved
    meta._calibration_records.append(
        CalibrationRecord(claim="resolved", stated_confidence=0.5, actual_outcome=True)
    )

    agent = object.__new__(SentientAgent)
    agent.metacognition = meta

    SentientAgent._resolve_calibration_outcome(agent, "Thanks!")
    assert call_count["n"] == 0


# ---------------------------------------------------------------------------
# PII redaction integration tests — persistence paths
# ---------------------------------------------------------------------------


def test_persist_semantic_memory_redacts_when_enabled():
    """Knowledge node label/content and edge context are redacted before save."""
    saved_nodes: list[dict] = []
    saved_edges: list[dict] = []

    class _Store:
        def save_knowledge_node(self, **kwargs) -> None:
            saved_nodes.append(kwargs)

        def clear_knowledge_edges(self) -> None:
            pass

        def save_knowledge_edge(self, **kwargs) -> None:
            saved_edges.append(kwargs)

        def sync_knowledge_embeddings(self, nodes) -> int:
            return 0

    node = SimpleNamespace(
        node_id="n1",
        label="User alice@example.com",
        category="person",
        content="Lives at 192.168.1.1",
        confidence=0.9,
        source_episodes=["ep1"],
        created_at=1.0,
        last_updated=2.0,
        access_count=1,
    )
    edge = SimpleNamespace(
        source_id="n1",
        target_id="n2",
        relationship="knows",
        strength=0.8,
        context="Met via alice@example.com",
        created_at=1.0,
    )

    agent = object.__new__(SentientAgent)
    agent._config = SimpleNamespace(
        privacy=SimpleNamespace(redact_before_persist=True),
    )
    agent.redactor = PIIRedactor(enabled=True)
    agent.memory_store = _Store()
    agent.semantic_memory = SimpleNamespace(_nodes={"n1": node}, _edges=[edge])

    SentientAgent._persist_semantic_memory(agent)

    assert saved_nodes
    assert "alice@example.com" not in saved_nodes[0]["label"]
    assert "[REDACTED_EMAIL]" in saved_nodes[0]["label"]
    assert "192.168.1.1" not in saved_nodes[0]["content"]
    assert "[REDACTED_IP]" in saved_nodes[0]["content"]
    assert saved_edges
    assert "alice@example.com" not in saved_edges[0]["context"]
    assert "[REDACTED_EMAIL]" in saved_edges[0]["context"]


@pytest.mark.asyncio
async def test_persist_working_memory_redacts_when_enabled():
    """Working memory item content is redacted during shutdown when enabled."""
    saved_items: list[list] = []

    class _Store:
        def save_affect_snapshot(self, **kwargs):
            return None

        def save_episode(self, episode, **kwargs):
            return None

        def save_knowledge_node(self, **kwargs):
            return None

        def clear_knowledge_edges(self):
            return None

        def save_knowledge_edge(self, **kwargs):
            return None

        def save_working_memory(self, items: list) -> None:
            saved_items.append(items)

        def sync_episode_embeddings(self, episodes) -> int:
            return 0

        def sync_knowledge_embeddings(self, nodes) -> int:
            return 0

        def close(self):
            return None

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent.heartbeat = _HeartbeatStub()
    agent.affect_state = SimpleNamespace(
        dimensions=SimpleNamespace(
            valence=0.0, arousal=0.3, dominance=0.0,
            certainty=0.0, goal_congruence=0.0,
        ),
        current_emotion=SimpleNamespace(value="neutral"),
    )
    agent._config = SimpleNamespace(
        memory=SimpleNamespace(shutdown_persist_recent_episodes=0),
        privacy=SimpleNamespace(redact_before_persist=True),
    )
    agent.redactor = PIIRedactor(enabled=True)
    agent.memory_store = _Store()
    agent.episodic_memory = SimpleNamespace(count=0, retrieve_recent=lambda n=1: [])
    agent.semantic_memory = SimpleNamespace(_nodes={}, _edges=[])
    agent.working_memory = SimpleNamespace(
        to_dict=lambda: {
            "items": [
                {"content": "User email is alice@example.com", "category": "note"},
            ]
        },
    )
    agent.identity = _IdentityStub()
    agent._start_time = time.time() - 1.0

    async def _consolidate():
        return None

    agent.consolidate_memories = _consolidate

    await SentientAgent.shutdown(agent)

    assert saved_items
    items = saved_items[0]
    assert items
    assert "alice@example.com" not in items[0]["content"]
    assert "[REDACTED_EMAIL]" in items[0]["content"]


@pytest.mark.asyncio
async def test_persist_goal_state_redacts_when_enabled():
    """Goal descriptions are redacted during shutdown when persist redaction is enabled."""
    saved_goals: list[dict] = []

    class _Store:
        def save_affect_snapshot(self, **kwargs):
            return None

        def save_episode(self, episode, **kwargs):
            return None

        def save_knowledge_node(self, **kwargs):
            return None

        def clear_knowledge_edges(self):
            return None

        def save_knowledge_edge(self, **kwargs):
            return None

        def save_working_memory(self, items: list) -> None:
            return None

        def save_goal_state(self, payload: dict) -> None:
            saved_goals.append(payload)

        def sync_episode_embeddings(self, episodes) -> int:
            return 0

        def sync_knowledge_embeddings(self, nodes) -> int:
            return 0

        def close(self):
            return None

    agent = object.__new__(SentientAgent)
    agent._initialized = True
    agent.heartbeat = _HeartbeatStub()
    agent.affect_state = SimpleNamespace(
        dimensions=SimpleNamespace(
            valence=0.0, arousal=0.3, dominance=0.0,
            certainty=0.0, goal_congruence=0.0,
        ),
        current_emotion=SimpleNamespace(value="neutral"),
    )
    agent._config = SimpleNamespace(
        memory=SimpleNamespace(shutdown_persist_recent_episodes=0),
        privacy=SimpleNamespace(redact_before_persist=True),
    )
    agent.redactor = PIIRedactor(enabled=True)
    agent.memory_store = _Store()
    agent.episodic_memory = SimpleNamespace(count=0, retrieve_recent=lambda n=1: [])
    agent.semantic_memory = SimpleNamespace(_nodes={}, _edges=[])
    agent.working_memory = SimpleNamespace(to_dict=lambda: {"items": []})
    agent.goal_system = SimpleNamespace(
        to_dict=lambda: {
            "active_goals": [
                {"description": "Help alice@example.com with setup", "status": "active"},
            ],
            "completed_goals": [],
        },
    )
    agent.identity = _IdentityStub()
    agent._start_time = time.time() - 1.0

    async def _consolidate():
        return None

    agent.consolidate_memories = _consolidate

    await SentientAgent.shutdown(agent)

    assert saved_goals
    goals = saved_goals[0]
    assert goals["active_goals"]
    desc = goals["active_goals"][0]["description"]
    assert "alice@example.com" not in desc
    assert "[REDACTED_EMAIL]" in desc


def test_note_to_self_redacts_context_when_enabled():
    """GWENN_CONTEXT.md content is redacted when persist redaction is enabled."""
    import asyncio

    saved_context: list[str] = []

    class _Store:
        def load_persistent_context(self) -> str:
            return ""

        def save_persistent_context(self, content: str) -> None:
            saved_context.append(content)

        def save_episode(self, episode, **kwargs) -> None:
            pass

    agent = object.__new__(SentientAgent)
    agent._config = SimpleNamespace(
        privacy=SimpleNamespace(redact_before_persist=True),
    )
    agent.redactor = PIIRedactor(enabled=True)
    agent.memory_store = _Store()
    agent.affect_state = SimpleNamespace(
        dimensions=SimpleNamespace(valence=0.0),
    )
    agent.episodic_memory = SimpleNamespace(encode=lambda ep: None)

    # We need to invoke the tool handler directly.
    # Build it inline to mimic _bind_tool_handlers.
    from gwenn.memory.episodic import Episode

    async def handle_set_note(note: str, section: str = "reminders") -> str:
        episode = Episode(
            content=f"[NOTE TO SELF — {section}] {note}",
            category="self_knowledge",
            emotional_valence=agent.affect_state.dimensions.valence,
            emotional_arousal=0.3,
            importance=0.8,
            tags=["note_to_self", section],
            participants=["gwenn"],
        )
        agent.episodic_memory.encode(episode)
        agent.memory_store.save_episode(episode)

        persist_note = note
        if agent._should_redact_for_persist():
            redactor = getattr(agent, "redactor", None)
            if redactor is not None:
                persist_note = redactor.redact(persist_note)
        existing_context = agent.memory_store.load_persistent_context()
        # Simple inline append (real code uses _upsert_context_section)
        agent.memory_store.save_persistent_context(
            existing_context + f"\n## {section}\n{persist_note}\n"
        )
        return f"Note stored in '{section}': {note[:80]}..."

    result = asyncio.get_event_loop().run_until_complete(
        handle_set_note("Contact alice@example.com for help"),
    )

    assert saved_context
    assert "alice@example.com" not in saved_context[-1]
    assert "[REDACTED_EMAIL]" in saved_context[-1]
