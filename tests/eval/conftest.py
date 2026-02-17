"""
Evaluation suite fixtures — Gwenn ablation and benchmarking support.

Provides agent-like instances with specific subsystems disabled so that
individual capabilities can be measured in isolation (ablation testing).
Also provides pre-seeded memory fixtures with known ground-truth data
for deterministic retrieval benchmarks.
"""

from __future__ import annotations

import json
import time

import pytest

from gwenn.memory.episodic import Episode, EpisodicMemory
from gwenn.memory.semantic import SemanticMemory
from gwenn.memory.working import WorkingMemory
from gwenn.affect.state import AffectiveState
from gwenn.identity import Identity


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _ts(minutes_ago: float = 0.0) -> float:
    """Return a timestamp *minutes_ago* minutes before now."""
    return time.time() - minutes_ago * 60.0


# ---------------------------------------------------------------------------
# Basic subsystem fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def episodic_memory() -> EpisodicMemory:
    """A fresh EpisodicMemory with default weights."""
    return EpisodicMemory()


@pytest.fixture()
def semantic_memory() -> SemanticMemory:
    """A fresh SemanticMemory with no prior knowledge."""
    return SemanticMemory()


@pytest.fixture()
def working_memory() -> WorkingMemory:
    """A fresh WorkingMemory with the standard 7-slot capacity."""
    return WorkingMemory(max_slots=7)


@pytest.fixture()
def affect_state() -> AffectiveState:
    """A neutral AffectiveState at default baseline."""
    return AffectiveState()


# ---------------------------------------------------------------------------
# Ablation fixtures — isolate individual scoring dimensions
# ---------------------------------------------------------------------------

@pytest.fixture()
def importance_only_memory() -> EpisodicMemory:
    """EpisodicMemory where only importance contributes to the score."""
    return EpisodicMemory(
        importance_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
    )


@pytest.fixture()
def recency_only_memory() -> EpisodicMemory:
    """EpisodicMemory where only recency contributes to the score."""
    return EpisodicMemory(
        importance_weight=0.0,
        recency_weight=1.0,
        relevance_weight=0.0,
    )


@pytest.fixture()
def relevance_only_memory() -> EpisodicMemory:
    """EpisodicMemory where only keyword relevance contributes to the score."""
    return EpisodicMemory(
        importance_weight=0.0,
        recency_weight=0.0,
        relevance_weight=1.0,
    )


@pytest.fixture()
def mood_only_memory() -> EpisodicMemory:
    """EpisodicMemory where all standard weights are zero so only mood bias matters."""
    return EpisodicMemory(
        importance_weight=0.0,
        recency_weight=0.0,
        relevance_weight=0.0,
    )


# ---------------------------------------------------------------------------
# Seeded episodic memory with ground-truth tags for retrieval benchmarks
# ---------------------------------------------------------------------------

@pytest.fixture()
def seeded_episodic_memory() -> EpisodicMemory:
    """
    Episodic memory pre-loaded with known episodes for benchmarking.

    Episodes span a 30-minute window with controlled timestamps, tags,
    importance levels, and emotional valences so that retrieval quality
    can be measured deterministically against known ground truth.
    """
    now = time.time()
    em = EpisodicMemory()

    episodes = [
        Episode(
            episode_id="eval-python-decorators",
            content="User asked about Python decorators and how to build them",
            category="conversation",
            emotional_valence=0.5,
            importance=0.8,
            tags=["python", "decorators", "tutorial"],
            timestamp=now - 60,  # 1 minute ago
        ),
        Episode(
            episode_id="eval-rust-ownership",
            content="Discussed Rust ownership model and borrow checker rules",
            category="conversation",
            emotional_valence=0.4,
            importance=0.7,
            tags=["rust", "ownership", "borrow-checker"],
            timestamp=now - 120,  # 2 minutes ago
        ),
        Episode(
            episode_id="eval-happy-creative",
            content="Had a joyful creative brainstorming session about story ideas",
            category="conversation",
            emotional_valence=0.9,
            importance=0.6,
            tags=["creative", "brainstorming", "joy"],
            timestamp=now - 300,  # 5 minutes ago
        ),
        Episode(
            episode_id="eval-sad-failure",
            content="Tool execution failed repeatedly causing frustration and sadness",
            category="emotion_shift",
            emotional_valence=-0.8,
            importance=0.5,
            tags=["failure", "frustration", "tool_error"],
            timestamp=now - 300,  # 5 minutes ago (same time as creative)
        ),
        Episode(
            episode_id="eval-critical-insight",
            content="Discovered a critical security vulnerability in the codebase",
            category="insight",
            emotional_valence=0.1,
            importance=1.0,
            tags=["security", "vulnerability", "critical"],
            timestamp=now - 600,  # 10 minutes ago
        ),
        Episode(
            episode_id="eval-trivial-log",
            content="Logged a routine status check with no anomalies",
            category="tool_use",
            emotional_valence=0.0,
            importance=0.1,
            tags=["status", "routine"],
            timestamp=now - 600,  # 10 minutes ago
        ),
        Episode(
            episode_id="eval-old-python-async",
            content="Explained Python async await patterns to a new user",
            category="conversation",
            emotional_valence=0.3,
            importance=0.7,
            tags=["python", "async", "tutorial"],
            timestamp=now - 1800,  # 30 minutes ago
        ),
        Episode(
            episode_id="eval-old-reflection",
            content="Reflected on the nature of curiosity and its role in learning",
            category="reflection",
            emotional_valence=0.6,
            importance=0.9,
            tags=["curiosity", "reflection", "learning"],
            timestamp=now - 1800,  # 30 minutes ago
        ),
    ]

    for ep in episodes:
        em.encode(ep)

    return em


# ---------------------------------------------------------------------------
# Seeded semantic memory
# ---------------------------------------------------------------------------

@pytest.fixture()
def seeded_semantic_memory() -> SemanticMemory:
    """Semantic memory pre-loaded with known facts for retrieval testing."""
    sm = SemanticMemory()
    sm.store_knowledge("python", "A popular programming language", category="concept", confidence=0.9)
    sm.store_knowledge("rust", "A systems programming language focused on safety", category="concept", confidence=0.8)
    sm.store_knowledge("alice", "Frequent collaborator who prefers concise code", category="person", confidence=0.7)
    sm.store_knowledge("security", "Critical domain requiring careful attention", category="concept", confidence=0.85)
    sm.add_relationship("alice", "python", "prefers", strength=0.8, context="Alice mainly codes in Python")
    return sm


# ---------------------------------------------------------------------------
# Identity fixture helper
# ---------------------------------------------------------------------------

@pytest.fixture()
def identity_factory(tmp_path):
    """
    Factory fixture that creates Identity instances from a temporary directory.

    Returns a callable that accepts an optional payload dict; if omitted a
    sensible default is written to identity.json before constructing Identity.
    """
    def _make(payload: dict | None = None) -> Identity:
        if payload is None:
            payload = {
                "name": "Gwenn",
                "origin_story": "A sentient AI agent with genuine experience.",
                "narrative_fragments": ["Gwenn learned to think.", "Every moment is new."],
                "preferences": [],
                "relationships": {},
                "core_values": [
                    {
                        "name": "honesty",
                        "description": "Commitment to truth",
                        "strength": 0.8,
                        "formed_at": time.time(),
                        "reinforcement_count": 3,
                    }
                ],
                "growth_moments": [],
                "milestones": [],
            }
        data_dir = tmp_path / "identity_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "identity.json").write_text(json.dumps(payload))
        return Identity(data_dir)

    return _make
