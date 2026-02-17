"""
Shared fixtures for Gwenn test suite.

Provides sample episodes, default configs, and mock engine components
so individual test modules can focus on behavior rather than setup.
"""

from __future__ import annotations

import time

import pytest

from gwenn.memory.episodic import Episode, EpisodicMemory
from gwenn.memory.semantic import SemanticMemory
from gwenn.memory.consolidation import ConsolidationEngine
from gwenn.config import SafetyConfig, AffectConfig


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _ts(minutes_ago: float = 0.0) -> float:
    """Return a timestamp that is *minutes_ago* minutes before now."""
    return time.time() - minutes_ago * 60.0


# ---------------------------------------------------------------------------
# Episode fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_episodes() -> list[Episode]:
    """A small diverse set of episodes for testing retrieval and consolidation."""
    now = time.time()
    return [
        Episode(
            episode_id="ep-conv-1",
            timestamp=now - 60,          # 1 minute ago
            content="User Alice asked about Python decorators",
            category="conversation",
            emotional_valence=0.4,
            emotional_arousal=0.5,
            importance=0.7,
            tags=["python", "decorators", "alice"],
            participants=["alice"],
        ),
        Episode(
            episode_id="ep-tool-1",
            timestamp=now - 300,         # 5 minutes ago
            content="Executed file_write tool to save config.yaml",
            category="tool_use",
            emotional_valence=0.1,
            emotional_arousal=0.3,
            importance=0.4,
            tags=["file_write", "config"],
            participants=["self"],
        ),
        Episode(
            episode_id="ep-reflect-1",
            timestamp=now - 600,         # 10 minutes ago
            content="Reflected on the meaning of curiosity in creative work",
            category="reflection",
            emotional_valence=0.6,
            emotional_arousal=0.4,
            importance=0.8,
            tags=["curiosity", "creativity", "reflection"],
            participants=["self"],
        ),
        Episode(
            episode_id="ep-emotion-1",
            timestamp=now - 900,         # 15 minutes ago
            content="Felt frustration after repeated tool errors",
            category="emotion_shift",
            emotional_valence=-0.5,
            emotional_arousal=0.7,
            importance=0.6,
            tags=["frustration", "tool_errors"],
            participants=["self"],
        ),
        Episode(
            episode_id="ep-insight-1",
            timestamp=now - 1800,        # 30 minutes ago
            content="Discovered that users prefer concise code examples",
            category="insight",
            emotional_valence=0.3,
            emotional_arousal=0.35,
            importance=0.9,
            tags=["users", "code", "examples", "insight"],
            participants=["self"],
        ),
    ]


@pytest.fixture()
def episodic_memory() -> EpisodicMemory:
    """A fresh EpisodicMemory instance with default weights."""
    return EpisodicMemory()


@pytest.fixture()
def populated_episodic_memory(episodic_memory, sample_episodes) -> EpisodicMemory:
    """An EpisodicMemory pre-loaded with the sample episodes."""
    for ep in sample_episodes:
        episodic_memory.encode(ep)
    return episodic_memory


@pytest.fixture()
def semantic_memory() -> SemanticMemory:
    """A fresh SemanticMemory instance."""
    return SemanticMemory()


@pytest.fixture()
def consolidation_engine(populated_episodic_memory, semantic_memory) -> ConsolidationEngine:
    """A ConsolidationEngine wired to the populated episodic + empty semantic memory."""
    return ConsolidationEngine(
        episodic=populated_episodic_memory,
        semantic=semantic_memory,
        consolidation_interval=0.0,  # always ready
    )


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def safety_config() -> SafetyConfig:
    """SafetyConfig that does not require ANTHROPIC_API_KEY.

    Note: pydantic-settings fields with aliases must be set using the alias
    name (the env-var name) rather than the Python field name.
    """
    return SafetyConfig(
        GWENN_MAX_TOOL_ITERATIONS=25,
        GWENN_REQUIRE_APPROVAL_FOR=["file_write"],
        GWENN_SANDBOX_ENABLED=True,
    )


@pytest.fixture()
def affect_config() -> AffectConfig:
    """AffectConfig with reasonable test defaults."""
    return AffectConfig(
        arousal_ceiling=0.92,
        distress_threshold=0.80,
        distress_duration_limit=300.0,
    )
