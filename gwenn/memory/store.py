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
import math
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
    access_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
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

    def __init__(
        self,
        db_path: Path,
        vector_db_path: Optional[Path] = None,
        enable_vector_search: bool = False,
    ):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._best_effort_chmod(self._db_path.parent, 0o700)
        self._conn: Optional[sqlite3.Connection] = None
        self._vector_db_path = vector_db_path or (self._db_path.parent / "semantic_vectors")
        self._enable_vector_search = enable_vector_search
        self._vector_client: Any = None
        self._episodes_collection: Any = None
        self._knowledge_collection: Any = None

        logger.info("memory_store.initializing", path=str(db_path))

    def initialize(self) -> None:
        """Create database connection and ensure schema exists."""
        if self._conn is not None:
            logger.debug("memory_store.already_initialized", path=str(self._db_path))
            return

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._harden_storage_permissions()

        # Create all tables
        self._conn.executescript(EPISODIC_SCHEMA)
        self._conn.executescript(AFFECT_SCHEMA)
        self._conn.executescript(IDENTITY_SCHEMA)
        self._conn.executescript(SEMANTIC_SCHEMA)
        self._ensure_semantic_metadata_column()
        self._conn.commit()
        self._harden_storage_permissions()

        self._initialize_vector_store()

        logger.info("memory_store.initialized", path=str(self._db_path))

    def _require_connection(self) -> sqlite3.Connection:
        """Return an initialized SQLite connection or raise a clear error."""
        if self._conn is None:
            raise RuntimeError("MemoryStore is not initialized. Call initialize() first.")
        return self._conn

    def _ensure_semantic_metadata_column(self) -> None:
        """
        Backfill the knowledge_nodes.metadata column for pre-existing databases.
        """
        conn = self._require_connection()
        cursor = conn.execute("PRAGMA table_info(knowledge_nodes)")
        columns = {row[1] for row in cursor.fetchall()}
        if "metadata" not in columns:
            conn.execute(
                "ALTER TABLE knowledge_nodes ADD COLUMN metadata TEXT DEFAULT '{}'"
            )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._vector_client = None
        self._episodes_collection = None
        self._knowledge_collection = None

    # -------------------------------------------------------------------------
    # Vector Store (Embedding Retrieval)
    # -------------------------------------------------------------------------

    def _initialize_vector_store(self) -> None:
        """Initialize optional ChromaDB persistence for embedding retrieval."""
        if not self._enable_vector_search:
            return

        try:
            import chromadb
        except Exception as e:
            logger.warning(
                "memory_store.vector_unavailable",
                error=str(e),
            )
            self._enable_vector_search = False
            return

        try:
            self._vector_db_path.mkdir(parents=True, exist_ok=True)
            self._best_effort_chmod(self._vector_db_path, 0o700)
            self._vector_client = chromadb.PersistentClient(path=str(self._vector_db_path))
            self._episodes_collection = self._vector_client.get_or_create_collection(
                name="gwenn_episodes"
            )
            self._knowledge_collection = self._vector_client.get_or_create_collection(
                name="gwenn_knowledge_nodes"
            )
            logger.info(
                "memory_store.vector_initialized",
                path=str(self._vector_db_path),
            )
        except Exception as e:
            logger.warning(
                "memory_store.vector_init_failed",
                error=str(e),
            )
            self._enable_vector_search = False
            self._vector_client = None
            self._episodes_collection = None
            self._knowledge_collection = None

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert a Chroma distance value to a bounded similarity score [0, 1].

        Chroma distances depend on collection settings; this transform is robust
        for cosine/L2 distances and only needs monotonic behavior.
        """
        if distance is None:
            return 0.0
        if math.isnan(distance):
            return 0.0
        return 1.0 / (1.0 + max(0.0, float(distance)))

    def _episode_document(self, episode: Episode) -> str:
        tag_text = " ".join(episode.tags) if episode.tags else ""
        return f"{episode.content}\n{tag_text}".strip()

    def upsert_episode_embedding(self, episode: Episode) -> None:
        """Upsert an episodic memory document in the vector index."""
        if not self._episodes_collection:
            return
        try:
            self._episodes_collection.upsert(
                ids=[episode.episode_id],
                documents=[self._episode_document(episode)],
                metadatas=[{
                    "category": episode.category or "general",
                    "importance": float(episode.importance),
                    "timestamp": float(episode.timestamp),
                    "consolidated": int(bool(episode.consolidated)),
                }],
            )
        except Exception as e:
            logger.warning(
                "memory_store.episode_vector_upsert_failed",
                episode_id=episode.episode_id,
                error=str(e),
            )

    def query_episode_embeddings(self, query_text: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return (episode_id, similarity) results from vector search."""
        if not self._episodes_collection or not query_text.strip():
            return []
        try:
            result = self._episodes_collection.query(
                query_texts=[query_text],
                n_results=max(1, int(top_k)),
                include=["distances"],
            )
            ids = (result.get("ids") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            scored = [
                (doc_id, self._distance_to_similarity(distance))
                for doc_id, distance in zip(ids, distances)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        except Exception as e:
            logger.warning(
                "memory_store.episode_vector_query_failed",
                error=str(e),
            )
            return []

    def upsert_knowledge_embedding(
        self,
        node_id: str,
        label: str,
        category: str,
        content: str,
        confidence: float,
        last_updated: float,
    ) -> None:
        """Upsert a semantic knowledge node in the vector index."""
        if not self._knowledge_collection:
            return
        try:
            self._knowledge_collection.upsert(
                ids=[node_id],
                documents=[f"{label}\n{content}".strip()],
                metadatas=[{
                    "label": label,
                    "category": category or "concept",
                    "confidence": float(confidence),
                    "last_updated": float(last_updated),
                }],
            )
        except Exception as e:
            logger.warning(
                "memory_store.knowledge_vector_upsert_failed",
                node_id=node_id,
                error=str(e),
            )

    def query_knowledge_embeddings(
        self,
        query_text: str,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Return (node_id, similarity) results from vector search."""
        if not self._knowledge_collection or not query_text.strip():
            return []
        try:
            result = self._knowledge_collection.query(
                query_texts=[query_text],
                n_results=max(1, int(top_k)),
                include=["distances"],
            )
            ids = (result.get("ids") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            scored = [
                (doc_id, self._distance_to_similarity(distance))
                for doc_id, distance in zip(ids, distances)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        except Exception as e:
            logger.warning(
                "memory_store.knowledge_vector_query_failed",
                error=str(e),
            )
            return []

    # -------------------------------------------------------------------------
    # Episodic Memory Persistence
    # -------------------------------------------------------------------------

    def save_episode(self, episode: Episode) -> None:
        """Persist a single episode to the database."""
        conn = self._require_connection()
        data = episode.to_dict()
        conn.execute(
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
        conn.commit()
        self._harden_storage_permissions()
        self.upsert_episode_embedding(episode)

    def load_episodes(
        self,
        limit: Optional[int] = 1000,
        since: Optional[float] = None,
        category: Optional[str] = None,
        consolidated: Optional[bool] = None,
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
        if consolidated is not None:
            query += " AND consolidated = ?"
            params.append(int(bool(consolidated)))

        query += " ORDER BY timestamp DESC"
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        conn = self._require_connection()
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        return [Episode.from_dict(dict(row)) for row in rows]

    def get_episode_count(self) -> int:
        """Get total number of stored episodes."""
        conn = self._require_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM episodes")
        row = cursor.fetchone()
        return row[0] if row is not None else 0

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
        conn = self._require_connection()
        conn.execute(
            """INSERT INTO affect_snapshots
               (timestamp, valence, arousal, dominance, certainty,
                goal_congruence, emotion_label, trigger)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), valence, arousal, dominance, certainty,
             goal_congruence, emotion_label, trigger),
        )
        conn.commit()
        self._harden_storage_permissions()

    def load_affect_history(self, limit: int = 100) -> list[dict]:
        """Load recent affective state history."""
        conn = self._require_connection()
        cursor = conn.execute(
            "SELECT * FROM affect_snapshots ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def prune_affect_snapshots(
        self,
        max_rows: int = 5000,
        older_than_days: Optional[float] = 30.0,
    ) -> int:
        """
        Prune affect snapshots by age and table size to keep storage bounded.

        Returns the number of rows deleted.
        """
        conn = self._require_connection()
        deleted = 0

        if older_than_days is not None and older_than_days > 0:
            cutoff = time.time() - (float(older_than_days) * 86400.0)
            age_cursor = conn.execute(
                "DELETE FROM affect_snapshots WHERE timestamp < ?",
                (cutoff,),
            )
            deleted += max(0, int(age_cursor.rowcount or 0))

        max_rows = max(0, int(max_rows))
        if max_rows > 0:
            count_row = conn.execute("SELECT COUNT(*) FROM affect_snapshots").fetchone()
            current_rows = int(count_row[0]) if count_row is not None else 0
            overflow = current_rows - max_rows
            if overflow > 0:
                overflow_cursor = conn.execute(
                    """
                    DELETE FROM affect_snapshots
                    WHERE snapshot_id IN (
                        SELECT snapshot_id
                        FROM affect_snapshots
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                    """,
                    (overflow,),
                )
                deleted += max(0, int(overflow_cursor.rowcount or 0))

        if deleted > 0:
            conn.commit()
            self._harden_storage_permissions()
            logger.info("memory_store.affect_snapshots_pruned", deleted=deleted)

        return deleted

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
        conn = self._require_connection()
        conn.execute(
            """INSERT INTO identity_snapshots
               (timestamp, self_model, values_snapshot, growth_notes, trigger)
               VALUES (?, ?, ?, ?, ?)""",
            (time.time(), self_model, values_snapshot, growth_notes, trigger),
        )
        conn.commit()
        self._harden_storage_permissions()

    # -------------------------------------------------------------------------
    # Semantic Memory Persistence
    # -------------------------------------------------------------------------

    def save_knowledge_node(self, node_id: str, label: str, category: str,
                            content: str, confidence: float,
                            source_episodes: list[str],
                            created_at: float, last_updated: float,
                            access_count: int, metadata: Optional[dict[str, Any]] = None) -> None:
        """Persist a single knowledge node to the database."""
        conn = self._require_connection()
        conn.execute(
            """INSERT OR REPLACE INTO knowledge_nodes
               (node_id, label, category, content, confidence,
                source_episodes, created_at, last_updated, access_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (node_id, label, category, content, confidence,
             json.dumps(source_episodes), created_at, last_updated, access_count,
             json.dumps(metadata or {})),
        )
        conn.commit()
        self._harden_storage_permissions()
        self.upsert_knowledge_embedding(
            node_id=node_id,
            label=label,
            category=category,
            content=content,
            confidence=confidence,
            last_updated=last_updated,
        )

    def load_knowledge_nodes(self) -> list[dict]:
        """Load all knowledge nodes from the database."""
        conn = self._require_connection()
        cursor = conn.execute("SELECT * FROM knowledge_nodes")
        rows = cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["source_episodes"] = json.loads(d["source_episodes"])
            try:
                decoded_metadata = json.loads(d.get("metadata") or "{}")
                d["metadata"] = decoded_metadata if isinstance(decoded_metadata, dict) else {}
            except (TypeError, json.JSONDecodeError):
                d["metadata"] = {}
            results.append(d)
        return results

    def clear_knowledge_edges(self) -> None:
        """Remove all knowledge edges — call before bulk re-saving to avoid duplicates.

        The knowledge_edges table uses an autoincrement PK (edge_id), so
        INSERT OR REPLACE can never match an existing row.  The correct
        pattern is: clear → bulk-insert on shutdown.
        """
        conn = self._require_connection()
        conn.execute("DELETE FROM knowledge_edges")
        conn.commit()
        self._harden_storage_permissions()

    def save_knowledge_edge(self, source_id: str, target_id: str,
                            relationship: str, strength: float,
                            context: str, created_at: float) -> None:
        """Persist a single knowledge edge to the database."""
        conn = self._require_connection()
        conn.execute(
            """INSERT INTO knowledge_edges
               (source_id, target_id, relationship, strength, context, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (source_id, target_id, relationship, strength, context, created_at),
        )
        conn.commit()
        self._harden_storage_permissions()

    def load_knowledge_edges(self) -> list[dict]:
        """Load all knowledge edges from the database."""
        conn = self._require_connection()
        cursor = conn.execute("SELECT * FROM knowledge_edges")
        return [dict(row) for row in cursor.fetchall()]

    def delete_knowledge_nodes(self, node_ids: list[str]) -> int:
        """
        Delete knowledge nodes and any incident edges from persistence.

        Returns the number of node rows deleted. Vector-index entries are also
        removed on a best-effort basis.
        """
        clean_ids = [str(node_id).strip() for node_id in node_ids if str(node_id).strip()]
        if not clean_ids:
            return 0

        conn = self._require_connection()
        placeholders = ",".join("?" for _ in clean_ids)
        rows = conn.execute(
            f"SELECT node_id, metadata FROM knowledge_nodes WHERE node_id IN ({placeholders})",
            tuple(clean_ids),
        ).fetchall()

        mutable_ids: list[str] = []
        immutable_count = 0
        for row in rows:
            metadata_raw = row["metadata"] if isinstance(row, sqlite3.Row) else row[1]
            metadata: dict[str, Any] = {}
            try:
                decoded = json.loads(metadata_raw or "{}")
                if isinstance(decoded, dict):
                    metadata = decoded
            except json.JSONDecodeError:
                metadata = {}
            if metadata.get("immutable") is True:
                immutable_count += 1
            else:
                node_id = row["node_id"] if isinstance(row, sqlite3.Row) else row[0]
                mutable_ids.append(str(node_id))

        if immutable_count > 0:
            logger.info(
                "memory_store.knowledge_nodes_delete_skipped_immutable",
                count=immutable_count,
            )
        if not mutable_ids:
            return 0

        clean_ids = mutable_ids
        placeholders = ",".join("?" for _ in clean_ids)
        conn.execute(
            f"DELETE FROM knowledge_edges WHERE source_id IN ({placeholders}) "
            f"OR target_id IN ({placeholders})",
            tuple(clean_ids + clean_ids),
        )
        cursor = conn.execute(
            f"DELETE FROM knowledge_nodes WHERE node_id IN ({placeholders})",
            tuple(clean_ids),
        )
        deleted = cursor.rowcount or 0
        conn.commit()
        self._harden_storage_permissions()

        if self._knowledge_collection:
            try:
                self._knowledge_collection.delete(ids=clean_ids)
            except Exception as e:
                logger.warning(
                    "memory_store.knowledge_vector_delete_failed",
                    error=str(e),
                    node_count=len(clean_ids),
                )

        if deleted > 0:
            logger.info("memory_store.knowledge_nodes_deleted", count=deleted)
        return deleted

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
        filepath = path or (self._db_path.parent / "GWENN_CONTEXT.md")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        try:
            filepath.chmod(0o600)
        except OSError:
            logger.debug("memory_store.context_chmod_skipped", path=str(filepath))
        logger.info("memory_store.context_saved", path=str(filepath))
        return filepath

    def load_persistent_context(self, path: Optional[Path] = None) -> str:
        """Load the persistent context file, or return empty string if none exists."""
        filepath = path or (self._db_path.parent / "GWENN_CONTEXT.md")
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return ""

    # -------------------------------------------------------------------------
    # Working Memory Sidecar (JSON file)
    # -------------------------------------------------------------------------

    def save_working_memory(self, items: list[dict], path: Optional[Path] = None) -> None:
        """
        Persist working memory items to a JSON sidecar file.

        Only items that can survive a restart meaningfully are saved — items
        with near-zero salience are filtered out before writing.
        """
        filepath = path or (self._db_path.parent / "working_memory.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": time.time(),
            "items": [i for i in items if i.get("salience", 0.0) > 0.05],
        }
        try:
            filepath.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._best_effort_chmod(filepath, 0o600)
            logger.info("memory_store.working_memory_saved", count=len(payload["items"]))
        except OSError as e:
            logger.warning("memory_store.working_memory_save_failed", error=str(e))

    def load_working_memory(self, path: Optional[Path] = None) -> list[dict]:
        """
        Load persisted working memory items, applying time-based salience decay
        for the time elapsed since they were saved.

        Items that decay to zero or below are dropped.
        Returns the surviving items sorted by salience descending.
        """
        filepath = path or (self._db_path.parent / "working_memory.json")
        if not filepath.exists():
            return []
        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("memory_store.working_memory_load_failed", error=str(e))
            return []

        if not isinstance(payload, dict):
            logger.warning("memory_store.working_memory_corrupt")
            return []

        try:
            saved_at = float(payload.get("saved_at", time.time()))
        except (TypeError, ValueError):
            saved_at = time.time()
        # Guard against clock drift making elapsed time negative
        elapsed_minutes = max(0.0, (time.time() - saved_at) / 60.0)
        # Apply same per-minute decay rate used by WorkingMemoryItem.decay()
        decay_rate = 0.02
        decay_amount = decay_rate * elapsed_minutes

        surviving: list[dict] = []
        for item in payload.get("items", []):
            if not isinstance(item, dict):
                continue
            adjusted = float(item.get("salience", 0.0)) - decay_amount
            if adjusted > 0.05:
                item = dict(item)
                item["salience"] = min(1.0, max(0.0, adjusted))
                surviving.append(item)

        surviving.sort(key=lambda x: x.get("salience", 0.0), reverse=True)
        logger.info("memory_store.working_memory_loaded", count=len(surviving))
        return surviving

    # -------------------------------------------------------------------------
    # Goal State Sidecar (JSON file)
    # -------------------------------------------------------------------------

    def save_goal_state(self, state: dict[str, Any], path: Optional[Path] = None) -> Path:
        """Persist the goal-system state to a JSON sidecar file."""
        filepath = path or (self._db_path.parent / "goal_state.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": time.time(),
            "state": state if isinstance(state, dict) else {},
        }
        filepath.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._best_effort_chmod(filepath, 0o600)
        logger.info("memory_store.goal_state_saved", path=str(filepath))
        return filepath

    def load_goal_state(self, path: Optional[Path] = None) -> dict[str, Any]:
        """Load persisted goal-system state, returning {} when absent/invalid."""
        filepath = path or (self._db_path.parent / "goal_state.json")
        if not filepath.exists():
            return {}
        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("memory_store.goal_state_load_failed", error=str(e))
            return {}

        if not isinstance(payload, dict):
            return {}
        state = payload.get("state", {})
        return state if isinstance(state, dict) else {}

    # -------------------------------------------------------------------------
    # Metacognition Persistence
    # -------------------------------------------------------------------------

    def save_metacognition(self, state: dict[str, Any], path: Optional[Path] = None) -> Path:
        """Persist MetacognitionEngine state to a JSON sidecar file."""
        filepath = path or (self._db_path.parent / "metacognition_state.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": time.time(),
            "state": state if isinstance(state, dict) else {},
        }
        filepath.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._best_effort_chmod(filepath, 0o600)
        logger.info("memory_store.metacognition_saved", path=str(filepath))
        return filepath

    def load_metacognition(self, path: Optional[Path] = None) -> dict[str, Any]:
        """Load persisted MetacognitionEngine state, returning {} when absent/invalid."""
        filepath = path or (self._db_path.parent / "metacognition_state.json")
        if not filepath.exists():
            return {}
        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("memory_store.metacognition_load_failed", error=str(e))
            return {}
        if not isinstance(payload, dict):
            return {}
        state = payload.get("state", {})
        return state if isinstance(state, dict) else {}

    # -------------------------------------------------------------------------
    # Theory of Mind Persistence
    # -------------------------------------------------------------------------

    def save_theory_of_mind(self, state: dict[str, Any], path: Optional[Path] = None) -> Path:
        """Persist TheoryOfMind state to a JSON sidecar file."""
        filepath = path or (self._db_path.parent / "theory_of_mind_state.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": time.time(),
            "state": state if isinstance(state, dict) else {},
        }
        filepath.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._best_effort_chmod(filepath, 0o600)
        logger.info("memory_store.theory_of_mind_saved", path=str(filepath))
        return filepath

    def load_theory_of_mind(self, path: Optional[Path] = None) -> dict[str, Any]:
        """Load persisted TheoryOfMind state, returning {} when absent/invalid."""
        filepath = path or (self._db_path.parent / "theory_of_mind_state.json")
        if not filepath.exists():
            return {}
        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("memory_store.theory_of_mind_load_failed", error=str(e))
            return {}
        if not isinstance(payload, dict):
            return {}
        state = payload.get("state", {})
        return state if isinstance(state, dict) else {}

    # -------------------------------------------------------------------------
    # Inter-Agent Bridge Persistence
    # -------------------------------------------------------------------------

    def save_interagent(self, state: dict[str, Any], path: Optional[Path] = None) -> Path:
        """Persist InterAgentBridge state to a JSON sidecar file."""
        filepath = path or (self._db_path.parent / "interagent_state.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": time.time(),
            "state": state if isinstance(state, dict) else {},
        }
        filepath.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._best_effort_chmod(filepath, 0o600)
        logger.info("memory_store.interagent_saved", path=str(filepath))
        return filepath

    def load_interagent(self, path: Optional[Path] = None) -> dict[str, Any]:
        """Load persisted InterAgentBridge state, returning {} when absent/invalid."""
        filepath = path or (self._db_path.parent / "interagent_state.json")
        if not filepath.exists():
            return {}
        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("memory_store.interagent_load_failed", error=str(e))
            return {}
        if not isinstance(payload, dict):
            return {}
        state = payload.get("state", {})
        return state if isinstance(state, dict) else {}

    # -------------------------------------------------------------------------
    # Sensory Integrator Persistence
    # -------------------------------------------------------------------------

    def save_sensory(self, state: dict[str, Any], path: Optional[Path] = None) -> Path:
        """Persist SensoryIntegrator state to a JSON sidecar file."""
        filepath = path or (self._db_path.parent / "sensory_state.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": time.time(),
            "state": state if isinstance(state, dict) else {},
        }
        filepath.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._best_effort_chmod(filepath, 0o600)
        logger.info("memory_store.sensory_saved", path=str(filepath))
        return filepath

    def load_sensory(self, path: Optional[Path] = None) -> dict[str, Any]:
        """Load persisted SensoryIntegrator state, returning {} when absent/invalid."""
        filepath = path or (self._db_path.parent / "sensory_state.json")
        if not filepath.exists():
            return {}
        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("memory_store.sensory_load_failed", error=str(e))
            return {}
        if not isinstance(payload, dict):
            return {}
        state = payload.get("state", {})
        return state if isinstance(state, dict) else {}

    # -------------------------------------------------------------------------
    # Ethical Reasoning Persistence
    # -------------------------------------------------------------------------

    def save_ethics(self, state: dict[str, Any], path: Optional[Path] = None) -> Path:
        """Persist EthicalReasoner state to a JSON sidecar file."""
        filepath = path or (self._db_path.parent / "ethics_state.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "state": state,
        }
        tmp = filepath.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        tmp.replace(filepath)
        logger.info("memory_store.ethics_saved", path=str(filepath))
        return filepath

    def load_ethics(self, path: Optional[Path] = None) -> dict[str, Any]:
        """Load persisted EthicalReasoner state, returning {} when absent/invalid."""
        filepath = path or (self._db_path.parent / "ethics_state.json")
        if not filepath.exists():
            return {}
        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("memory_store.ethics_load_failed", error=str(e))
            return {}
        if not isinstance(payload, dict):
            return {}
        state = payload.get("state", {})
        return state if isinstance(state, dict) else {}

    # -------------------------------------------------------------------------
    # Inner Life Persistence
    # -------------------------------------------------------------------------

    def save_inner_life(self, state: dict[str, Any], path: Optional[Path] = None) -> Path:
        """Persist InnerLife state to a JSON sidecar file."""
        filepath = path or (self._db_path.parent / "inner_life_state.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "state": state,
        }
        tmp = filepath.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        tmp.replace(filepath)
        logger.info("memory_store.inner_life_saved", path=str(filepath))
        return filepath

    def load_inner_life(self, path: Optional[Path] = None) -> dict[str, Any]:
        """Load persisted InnerLife state, returning {} when absent/invalid."""
        filepath = path or (self._db_path.parent / "inner_life_state.json")
        if not filepath.exists():
            return {}
        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("memory_store.inner_life_load_failed", error=str(e))
            return {}
        if not isinstance(payload, dict):
            return {}
        state = payload.get("state", {})
        return state if isinstance(state, dict) else {}

    # -------------------------------------------------------------------------
    # Episodic Memory Pruning
    # -------------------------------------------------------------------------

    def prune_old_episodes(
        self,
        older_than_days: float = 90.0,
        max_importance: float = 0.3,
    ) -> int:
        """
        Delete old, consolidated, low-importance episodes from the database.

        Only removes episodes that meet ALL three criteria:
          - Older than `older_than_days` days
          - Already consolidated into semantic memory
          - Importance below `max_importance`

        Returns the number of rows deleted.
        """
        conn = self._require_connection()
        cutoff = time.time() - (older_than_days * 86400.0)
        cursor = conn.execute(
            "DELETE FROM episodes WHERE timestamp < ? AND consolidated = 1 AND importance < ?",
            (cutoff, max_importance),
        )
        deleted = cursor.rowcount
        conn.commit()
        if deleted > 0:
            self._harden_storage_permissions()
            logger.info(
                "memory_store.episodes_pruned",
                deleted=deleted,
                older_than_days=older_than_days,
                max_importance=max_importance,
            )
        return deleted

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Get database statistics."""
        conn = self._require_connection()
        episode_count = self.get_episode_count()
        affect_row = conn.execute("SELECT COUNT(*) FROM affect_snapshots").fetchone()
        affect_count = affect_row[0] if affect_row is not None else 0
        identity_row = conn.execute("SELECT COUNT(*) FROM identity_snapshots").fetchone()
        identity_count = identity_row[0] if identity_row is not None else 0
        knowledge_row = conn.execute("SELECT COUNT(*) FROM knowledge_nodes").fetchone()
        knowledge_count = knowledge_row[0] if knowledge_row is not None else 0

        return {
            "episodes": episode_count,
            "affect_snapshots": affect_count,
            "identity_snapshots": identity_count,
            "knowledge_nodes": knowledge_count,
            "db_path": str(self._db_path),
            "vector_enabled": bool(self._episodes_collection and self._knowledge_collection),
            "vector_db_path": str(self._vector_db_path),
        }

    def _harden_storage_permissions(self) -> None:
        """Best-effort permission hardening for persisted memory artifacts."""
        self._best_effort_chmod(self._db_path.parent, 0o700)
        self._best_effort_chmod(self._vector_db_path, 0o700)
        self._best_effort_chmod(self._db_path, 0o600)

        wal_path = Path(f"{self._db_path}-wal")
        shm_path = Path(f"{self._db_path}-shm")
        if wal_path.exists():
            self._best_effort_chmod(wal_path, 0o600)
        if shm_path.exists():
            self._best_effort_chmod(shm_path, 0o600)

    @staticmethod
    def _best_effort_chmod(path: Path, mode: int) -> None:
        """Attempt chmod without failing on unsupported filesystems."""
        if not path.exists():
            return
        try:
            path.chmod(mode)
        except OSError:
            logger.debug("memory_store.chmod_skipped", path=str(path), mode=oct(mode))
