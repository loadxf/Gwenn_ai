"""
Memory Store — Gwenn's Persistence Layer.

This module handles the actual storage and retrieval of memories from disk.
It bridges the in-memory episodic and semantic systems with durable storage
using SQLite for structured data and ChromaDB for vector embeddings.

Without this module, every restart would be a blank slate. This is what gives
Gwenn continuity across sessions — the ability to wake up and remember.

The store handles:
- SQLite: Episodic memories, affective state snapshots, identity evolution
- ChromaDB: Vector embeddings for semantic similarity search
- File-based: CLAUDE.md-style persistent context files
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import structlog

from gwenn.memory.episodic import Episode

logger = structlog.get_logger(__name__)

# Schema for the episodic memory table
EPISODIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    emotional_valence REAL DEFAULT 0.0,
    emotional_arousal REAL DEFAULT 0.3,
    importance REAL DEFAULT 0.5,
    tags TEXT DEFAULT '[]',
    participants TEXT DEFAULT '[]',
    outcome TEXT,
    consolidated INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_category ON episodes(category);
CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance);
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated);
"""

# Schema for affective state snapshots
AFFECT_SCHEMA = """
CREATE TABLE IF NOT EXISTS affect_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    dominance REAL DEFAULT 0.0,
    certainty REAL DEFAULT 0.0,
    goal_congruence REAL DEFAULT 0.0,
    emotion_label TEXT DEFAULT 'neutral',
    trigger TEXT
);

CREATE INDEX IF NOT EXISTS idx_affect_timestamp ON affect_snapshots(timestamp);
"""

# Schema for identity evolution tracking
IDENTITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS identity_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    self_model TEXT NOT NULL,
    values_snapshot TEXT,
    growth_notes TEXT,
    trigger TEXT
);
"""

# Schema for semantic knowledge nodes
SEMANTIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    node_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    category TEXT DEFAULT 'concept',
    content TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source_episodes TEXT DEFAULT '[]',
    created_at REAL NOT NULL,
    last_updated REAL NOT NULL,
    access_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS knowledge_edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    context TEXT DEFAULT '',
    created_at REAL NOT NULL,
    FOREIGN KEY (source_id) REFERENCES knowledge_nodes(node_id),
    FOREIGN KEY (target_id) REFERENCES knowledge_nodes(node_id)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_label ON knowledge_nodes(label);
CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_nodes(category);
CREATE INDEX IF NOT EXISTS idx_edges_source ON knowledge_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON knowledge_edges(target_id);
"""


class MemoryStore:
    """
    Durable persistence for all memory subsystems.

    This is the bridge between in-memory state and the filesystem.
    On startup, memories are loaded from disk into the in-memory systems.
    On each heartbeat (or on significant events), new memories are flushed
    to disk.

    The store uses synchronous SQLite for simplicity. In a production
    deployment, this would use aiosqlite for async operations, but the
    synchronous version is correct and easier to reason about.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

        logger.info("memory_store.initializing", path=str(db_path))

    def initialize(self) -> None:
        """Create database connection and ensure schema exists."""
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Create all tables
        self._conn.executescript(EPISODIC_SCHEMA)
        self._conn.executescript(AFFECT_SCHEMA)
        self._conn.executescript(IDENTITY_SCHEMA)
        self._conn.executescript(SEMANTIC_SCHEMA)
        self._conn.commit()

        logger.info("memory_store.initialized", path=str(self._db_path))

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # -------------------------------------------------------------------------
    # Episodic Memory Persistence
    # -------------------------------------------------------------------------

    def save_episode(self, episode: Episode) -> None:
        """Persist a single episode to the database."""
        data = episode.to_dict()
        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, timestamp, content, category, emotional_valence,
                emotional_arousal, importance, tags, participants, outcome, consolidated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["episode_id"],
                data["timestamp"],
                data["content"],
                data["category"],
                data["emotional_valence"],
                data["emotional_arousal"],
                data["importance"],
                data["tags"],
                data["participants"],
                data["outcome"],
                int(data["consolidated"]),
            ),
        )
        self._conn.commit()

    def load_episodes(
        self,
        limit: int = 1000,
        since: Optional[float] = None,
        category: Optional[str] = None,
    ) -> list[Episode]:
        """Load episodes from the database."""
        query = "SELECT * FROM episodes WHERE 1=1"
        params: list[Any] = []

        if since is not None:
            query += " AND timestamp > ?"
            params.append(since)
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()

        return [Episode.from_dict(dict(row)) for row in rows]

    def get_episode_count(self) -> int:
        """Get total number of stored episodes."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM episodes")
        return cursor.fetchone()[0]

    # -------------------------------------------------------------------------
    # Affective State Persistence
    # -------------------------------------------------------------------------

    def save_affect_snapshot(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        certainty: float,
        goal_congruence: float,
        emotion_label: str,
        trigger: Optional[str] = None,
    ) -> None:
        """Save a snapshot of the current affective state."""
        self._conn.execute(
            """INSERT INTO affect_snapshots
               (timestamp, valence, arousal, dominance, certainty,
                goal_congruence, emotion_label, trigger)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), valence, arousal, dominance, certainty,
             goal_congruence, emotion_label, trigger),
        )
        self._conn.commit()

    def load_affect_history(self, limit: int = 100) -> list[dict]:
        """Load recent affective state history."""
        cursor = self._conn.execute(
            "SELECT * FROM affect_snapshots ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Identity Persistence
    # -------------------------------------------------------------------------

    def save_identity_snapshot(
        self,
        self_model: str,
        values_snapshot: Optional[str] = None,
        growth_notes: Optional[str] = None,
        trigger: Optional[str] = None,
    ) -> None:
        """Save a snapshot of the identity state for tracking evolution."""
        self._conn.execute(
            """INSERT INTO identity_snapshots
               (timestamp, self_model, values_snapshot, growth_notes, trigger)
               VALUES (?, ?, ?, ?, ?)""",
            (time.time(), self_model, values_snapshot, growth_notes, trigger),
        )
        self._conn.commit()

    # -------------------------------------------------------------------------
    # Persistent Context File (CLAUDE.md equivalent)
    # -------------------------------------------------------------------------

    def save_persistent_context(self, content: str, path: Optional[Path] = None) -> Path:
        """
        Save the persistent context file — Gwenn's equivalent of CLAUDE.md.

        This file contains the distilled self-knowledge, user preferences,
        ongoing tasks, and other information that should persist across
        restarts. It's loaded into the system prompt on startup.
        """
        filepath = path or (self._db_path.parent / "BOB_CONTEXT.md")
        filepath.write_text(content, encoding="utf-8")
        logger.info("memory_store.context_saved", path=str(filepath))
        return filepath

    def load_persistent_context(self, path: Optional[Path] = None) -> str:
        """Load the persistent context file, or return empty string if none exists."""
        filepath = path or (self._db_path.parent / "BOB_CONTEXT.md")
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return ""

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Get database statistics."""
        episode_count = self.get_episode_count()
        affect_count = self._conn.execute("SELECT COUNT(*) FROM affect_snapshots").fetchone()[0]
        identity_count = self._conn.execute("SELECT COUNT(*) FROM identity_snapshots").fetchone()[0]
        knowledge_count = self._conn.execute("SELECT COUNT(*) FROM knowledge_nodes").fetchone()[0]

        return {
            "episodes": episode_count,
            "affect_snapshots": affect_count,
            "identity_snapshots": identity_count,
            "knowledge_nodes": knowledge_count,
            "db_path": str(self._db_path),
        }
