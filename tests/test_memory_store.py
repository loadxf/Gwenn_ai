"""
Tests for gwenn.memory.store — MemoryStore persistence layer.

Covers:
- Database initialization and schema creation
- Episode CRUD: save_episode, load_episodes, get_episode_count
- Episode filtering by timestamp (since) and category
- Affect snapshot: save_affect_snapshot, load_affect_history, ordering
- Identity snapshot: save_identity_snapshot
- Knowledge nodes: save_knowledge_node, load_knowledge_nodes, source_episodes JSON round-trip
- Knowledge edges: save_knowledge_edge, load_knowledge_edges, clear_knowledge_edges
- Persistent context: save_persistent_context, load_persistent_context
- Stats property
- Edge cases: empty tables, duplicate IDs (INSERT OR REPLACE), close and reopen
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gwenn.memory.episodic import Episode
from gwenn.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(
    episode_id: str = "ep-1",
    timestamp: float | None = None,
    content: str = "test content",
    category: str = "general",
    emotional_valence: float = 0.0,
    emotional_arousal: float = 0.3,
    importance: float = 0.5,
    tags: list[str] | None = None,
    participants: list[str] | None = None,
    outcome: str | None = None,
    consolidated: bool = False,
) -> Episode:
    """Build an Episode with sensible defaults, overridable per-field."""
    return Episode(
        episode_id=episode_id,
        timestamp=timestamp if timestamp is not None else time.time(),
        content=content,
        category=category,
        emotional_valence=emotional_valence,
        emotional_arousal=emotional_arousal,
        importance=importance,
        tags=tags if tags is not None else [],
        participants=participants if participants is not None else [],
        outcome=outcome,
        consolidated=consolidated,
    )


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    """A fully initialised MemoryStore backed by a temp-directory SQLite file."""
    ms = MemoryStore(tmp_path / "gwenn_test.db")
    ms.initialize()
    yield ms
    ms.close()


@pytest.fixture()
def memory_db_path(tmp_path: Path) -> Path:
    """Return a path (but do NOT create the database yet)."""
    return tmp_path / "sub" / "gwenn_test.db"


# ---------------------------------------------------------------------------
# 1. Database initialisation and schema creation
# ---------------------------------------------------------------------------

class TestInitialisation:
    """Database file creation, schema tables, and pragmas."""

    def test_db_file_created(self, tmp_path: Path):
        db_path = tmp_path / "init_test.db"
        ms = MemoryStore(db_path)
        ms.initialize()
        assert db_path.exists()
        ms.close()

    def test_parent_directories_created(self, memory_db_path: Path):
        """MemoryStore.__init__ should mkdir parents that do not yet exist."""
        ms = MemoryStore(memory_db_path)
        assert memory_db_path.parent.exists()
        ms.close()

    def test_tables_exist_after_initialise(self, store: MemoryStore):
        """All four tables should be present after initialize()."""
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = sorted(row[0] for row in cursor.fetchall())
        for expected in ["affect_snapshots", "episodes", "identity_snapshots",
                         "knowledge_edges", "knowledge_nodes"]:
            assert expected in table_names, f"Missing table: {expected}"

    def test_wal_journal_mode(self, store: MemoryStore):
        """WAL journal mode should be active."""
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_foreign_keys_enabled(self, store: MemoryStore):
        """Foreign key enforcement should be ON."""
        fk = store._conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_indices_created(self, store: MemoryStore):
        """Key indices should be present."""
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        index_names = {row[0] for row in cursor.fetchall()}
        for idx in [
            "idx_episodes_timestamp",
            "idx_episodes_category",
            "idx_episodes_importance",
            "idx_episodes_consolidated",
            "idx_affect_timestamp",
            "idx_knowledge_label",
            "idx_knowledge_category",
            "idx_edges_source",
            "idx_edges_target",
        ]:
            assert idx in index_names, f"Missing index: {idx}"

    def test_double_initialise_is_safe(self, tmp_path: Path):
        """Calling initialize() twice must not raise (CREATE IF NOT EXISTS)."""
        db_path = tmp_path / "double_init.db"
        ms = MemoryStore(db_path)
        ms.initialize()
        first_conn = ms._conn
        ms.initialize()  # should be idempotent
        assert ms._conn is first_conn
        ms.close()

    def test_permissions_hardened_for_db_files(self, tmp_path: Path):
        if os.name == "nt":
            pytest.skip("Permission bits are not POSIX-stable on Windows")

        db_path = tmp_path / "perm_test.db"
        ms = MemoryStore(db_path)
        ms.initialize()
        ms.save_episode(_make_episode(episode_id="perm-ep"))

        db_mode = os.stat(db_path).st_mode & 0o777
        parent_mode = os.stat(db_path.parent).st_mode & 0o777
        assert db_mode & 0o077 == 0
        assert parent_mode & 0o077 == 0

        wal_path = Path(f"{db_path}-wal")
        shm_path = Path(f"{db_path}-shm")
        for sidecar in (wal_path, shm_path):
            if sidecar.exists():
                mode = os.stat(sidecar).st_mode & 0o777
                assert mode & 0o077 == 0

        ms.close()


# ---------------------------------------------------------------------------
# 2. Episode CRUD
# ---------------------------------------------------------------------------

class TestEpisodeCRUD:
    """save_episode, load_episodes, get_episode_count."""

    def test_save_and_load_single_episode(self, store: MemoryStore):
        ep = _make_episode(episode_id="ep-save-1", content="hello world")
        store.save_episode(ep)

        loaded = store.load_episodes(limit=10)
        assert len(loaded) == 1
        assert loaded[0].episode_id == "ep-save-1"
        assert loaded[0].content == "hello world"

    def test_round_trip_preserves_all_fields(self, store: MemoryStore):
        ep = _make_episode(
            episode_id="ep-rt",
            timestamp=1_700_000_000.0,
            content="round-trip content",
            category="conversation",
            emotional_valence=-0.4,
            emotional_arousal=0.7,
            importance=0.85,
            tags=["alpha", "beta"],
            participants=["alice", "self"],
            outcome="success",
            consolidated=True,
        )
        store.save_episode(ep)
        loaded = store.load_episodes(limit=1)
        got = loaded[0]

        assert got.episode_id == ep.episode_id
        assert got.timestamp == pytest.approx(ep.timestamp)
        assert got.content == ep.content
        assert got.category == ep.category
        assert got.emotional_valence == pytest.approx(ep.emotional_valence)
        assert got.emotional_arousal == pytest.approx(ep.emotional_arousal)
        assert got.importance == pytest.approx(ep.importance)
        assert got.tags == ep.tags
        assert got.participants == ep.participants
        assert got.outcome == ep.outcome
        assert got.consolidated == ep.consolidated

    def test_get_episode_count_zero(self, store: MemoryStore):
        assert store.get_episode_count() == 0

    def test_get_episode_count_after_inserts(self, store: MemoryStore):
        for i in range(5):
            store.save_episode(_make_episode(episode_id=f"ep-{i}"))
        assert store.get_episode_count() == 5

    def test_load_episodes_limit(self, store: MemoryStore):
        now = time.time()
        for i in range(10):
            store.save_episode(
                _make_episode(episode_id=f"ep-{i}", timestamp=now - i)
            )
        loaded = store.load_episodes(limit=3)
        assert len(loaded) == 3

    def test_load_episodes_ordered_desc_by_timestamp(self, store: MemoryStore):
        now = time.time()
        for i in range(5):
            store.save_episode(
                _make_episode(episode_id=f"ep-{i}", timestamp=now - i * 100)
            )
        loaded = store.load_episodes(limit=10)
        for i in range(len(loaded) - 1):
            assert loaded[i].timestamp >= loaded[i + 1].timestamp

    def test_load_episodes_without_limit_returns_all(self, store: MemoryStore):
        for i in range(4):
            store.save_episode(_make_episode(episode_id=f"ep-all-{i}"))
        loaded = store.load_episodes(limit=None)
        assert len(loaded) == 4


# ---------------------------------------------------------------------------
# 3. Episode filtering by timestamp (since) and category
# ---------------------------------------------------------------------------

class TestEpisodeFiltering:
    """load_episodes with since and category filters."""

    def test_filter_by_since(self, store: MemoryStore):
        now = time.time()
        store.save_episode(_make_episode(episode_id="old", timestamp=now - 3600))
        store.save_episode(_make_episode(episode_id="new", timestamp=now - 10))

        loaded = store.load_episodes(since=now - 60)
        assert len(loaded) == 1
        assert loaded[0].episode_id == "new"

    def test_filter_by_category(self, store: MemoryStore):
        store.save_episode(_make_episode(episode_id="conv", category="conversation"))
        store.save_episode(_make_episode(episode_id="tool", category="tool_use"))

        loaded = store.load_episodes(category="conversation")
        assert len(loaded) == 1
        assert loaded[0].episode_id == "conv"

    def test_filter_by_since_and_category(self, store: MemoryStore):
        now = time.time()
        store.save_episode(
            _make_episode(episode_id="old-conv", timestamp=now - 7200, category="conversation")
        )
        store.save_episode(
            _make_episode(episode_id="new-conv", timestamp=now - 10, category="conversation")
        )
        store.save_episode(
            _make_episode(episode_id="new-tool", timestamp=now - 10, category="tool_use")
        )

        loaded = store.load_episodes(since=now - 60, category="conversation")
        assert len(loaded) == 1
        assert loaded[0].episode_id == "new-conv"

    def test_filter_nonexistent_category_returns_empty(self, store: MemoryStore):
        store.save_episode(_make_episode(episode_id="ep-1"))
        loaded = store.load_episodes(category="nonexistent")
        assert loaded == []

    def test_since_boundary_exclusive(self, store: MemoryStore):
        """since uses strict '>' — an episode exactly at 'since' is excluded."""
        boundary = 1_700_000_000.0
        store.save_episode(_make_episode(episode_id="exact", timestamp=boundary))
        store.save_episode(_make_episode(episode_id="after", timestamp=boundary + 1))

        loaded = store.load_episodes(since=boundary)
        ids = [e.episode_id for e in loaded]
        assert "exact" not in ids
        assert "after" in ids

    def test_filter_by_consolidated_flag(self, store: MemoryStore):
        store.save_episode(_make_episode(episode_id="c-yes", consolidated=True))
        store.save_episode(_make_episode(episode_id="c-no", consolidated=False))

        consolidated_rows = store.load_episodes(limit=None, consolidated=True)
        unconsolidated_rows = store.load_episodes(limit=None, consolidated=False)

        assert {e.episode_id for e in consolidated_rows} == {"c-yes"}
        assert {e.episode_id for e in unconsolidated_rows} == {"c-no"}


# ---------------------------------------------------------------------------
# 4. Affect snapshot
# ---------------------------------------------------------------------------

class TestAffectSnapshot:
    """save_affect_snapshot, load_affect_history, ordering."""

    def test_save_and_load_single_snapshot(self, store: MemoryStore):
        store.save_affect_snapshot(
            valence=0.5, arousal=0.6, dominance=0.3,
            certainty=0.7, goal_congruence=0.8,
            emotion_label="curious", trigger="new topic",
        )
        history = store.load_affect_history(limit=10)
        assert len(history) == 1
        snap = history[0]
        assert snap["valence"] == pytest.approx(0.5)
        assert snap["arousal"] == pytest.approx(0.6)
        assert snap["dominance"] == pytest.approx(0.3)
        assert snap["certainty"] == pytest.approx(0.7)
        assert snap["goal_congruence"] == pytest.approx(0.8)
        assert snap["emotion_label"] == "curious"
        assert snap["trigger"] == "new topic"

    def test_trigger_none_stored_as_null(self, store: MemoryStore):
        store.save_affect_snapshot(
            valence=0.0, arousal=0.0, dominance=0.0,
            certainty=0.0, goal_congruence=0.0,
            emotion_label="neutral",
        )
        history = store.load_affect_history()
        assert history[0]["trigger"] is None

    def test_history_ordered_desc_by_timestamp(self, store: MemoryStore):
        for i in range(5):
            store.save_affect_snapshot(
                valence=float(i) / 5, arousal=0.5, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
                emotion_label=f"label-{i}",
            )
        history = store.load_affect_history(limit=10)
        for i in range(len(history) - 1):
            assert history[i]["timestamp"] >= history[i + 1]["timestamp"]

    def test_history_respects_limit(self, store: MemoryStore):
        for i in range(10):
            store.save_affect_snapshot(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
                emotion_label=f"l-{i}",
            )
        history = store.load_affect_history(limit=3)
        assert len(history) == 3

    def test_autoincrement_snapshot_id(self, store: MemoryStore):
        for _ in range(3):
            store.save_affect_snapshot(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
                emotion_label="neutral",
            )
        history = store.load_affect_history(limit=10)
        ids = [snap["snapshot_id"] for snap in history]
        assert len(set(ids)) == 3  # all unique


# ---------------------------------------------------------------------------
# 5. Identity snapshot
# ---------------------------------------------------------------------------

class TestIdentitySnapshot:
    """save_identity_snapshot and database verification."""

    def test_save_identity_snapshot_all_fields(self, store: MemoryStore):
        store.save_identity_snapshot(
            self_model="I am curious and helpful",
            values_snapshot='{"honesty": 0.9}',
            growth_notes="Improved patience today",
            trigger="deep conversation",
        )
        cursor = store._conn.execute("SELECT * FROM identity_snapshots")
        rows = [dict(row) for row in cursor.fetchall()]
        assert len(rows) == 1
        row = rows[0]
        assert row["self_model"] == "I am curious and helpful"
        assert row["values_snapshot"] == '{"honesty": 0.9}'
        assert row["growth_notes"] == "Improved patience today"
        assert row["trigger"] == "deep conversation"
        assert row["timestamp"] > 0

    def test_save_identity_snapshot_optional_none(self, store: MemoryStore):
        store.save_identity_snapshot(self_model="basic self model")
        cursor = store._conn.execute("SELECT * FROM identity_snapshots")
        row = dict(cursor.fetchone())
        assert row["values_snapshot"] is None
        assert row["growth_notes"] is None
        assert row["trigger"] is None

    def test_multiple_identity_snapshots(self, store: MemoryStore):
        for i in range(4):
            store.save_identity_snapshot(self_model=f"self model v{i}")
        cursor = store._conn.execute("SELECT COUNT(*) FROM identity_snapshots")
        assert cursor.fetchone()[0] == 4


# ---------------------------------------------------------------------------
# 6. Knowledge nodes
# ---------------------------------------------------------------------------

class TestKnowledgeNodes:
    """save_knowledge_node, load_knowledge_nodes, source_episodes JSON round-trip."""

    def _save_sample_node(self, store: MemoryStore, node_id: str = "node-1",
                          source_episodes: list[str] | None = None) -> None:
        store.save_knowledge_node(
            node_id=node_id,
            label="Python",
            category="concept",
            content="A programming language",
            confidence=0.9,
            source_episodes=source_episodes if source_episodes is not None else ["ep-1", "ep-2"],
            created_at=1_700_000_000.0,
            last_updated=1_700_001_000.0,
            access_count=5,
            metadata={"source": "test"},
        )

    def test_save_and_load_single_node(self, store: MemoryStore):
        self._save_sample_node(store)
        nodes = store.load_knowledge_nodes()
        assert len(nodes) == 1
        n = nodes[0]
        assert n["node_id"] == "node-1"
        assert n["label"] == "Python"
        assert n["category"] == "concept"
        assert n["content"] == "A programming language"
        assert n["confidence"] == pytest.approx(0.9)
        assert n["created_at"] == pytest.approx(1_700_000_000.0)
        assert n["last_updated"] == pytest.approx(1_700_001_000.0)
        assert n["access_count"] == 5
        assert n["metadata"] == {"source": "test"}

    def test_source_episodes_json_round_trip(self, store: MemoryStore):
        episodes = ["ep-alpha", "ep-beta", "ep-gamma"]
        self._save_sample_node(store, source_episodes=episodes)
        nodes = store.load_knowledge_nodes()
        assert nodes[0]["source_episodes"] == episodes

    def test_source_episodes_empty_list(self, store: MemoryStore):
        self._save_sample_node(store, source_episodes=[])
        nodes = store.load_knowledge_nodes()
        assert nodes[0]["source_episodes"] == []

    def test_load_multiple_nodes(self, store: MemoryStore):
        for i in range(5):
            store.save_knowledge_node(
                node_id=f"node-{i}", label=f"label-{i}", category="concept",
                content=f"content-{i}", confidence=0.5,
                source_episodes=[], created_at=time.time(),
                last_updated=time.time(), access_count=0,
            )
        nodes = store.load_knowledge_nodes()
        assert len(nodes) == 5


# ---------------------------------------------------------------------------
# 7. Knowledge edges
# ---------------------------------------------------------------------------

class TestKnowledgeEdges:
    """save_knowledge_edge, load_knowledge_edges, clear_knowledge_edges."""

    def _seed_nodes(self, store: MemoryStore) -> None:
        """Insert two minimal knowledge nodes so foreign keys are satisfied."""
        now = time.time()
        for nid in ("src", "tgt"):
            store.save_knowledge_node(
                node_id=nid, label=nid, category="concept",
                content="placeholder", confidence=0.5,
                source_episodes=[], created_at=now, last_updated=now,
                access_count=0,
            )

    def test_save_and_load_edge(self, store: MemoryStore):
        self._seed_nodes(store)
        store.save_knowledge_edge(
            source_id="src", target_id="tgt",
            relationship="related_to", strength=0.8,
            context="testing", created_at=1_700_000_000.0,
        )
        edges = store.load_knowledge_edges()
        assert len(edges) == 1
        e = edges[0]
        assert e["source_id"] == "src"
        assert e["target_id"] == "tgt"
        assert e["relationship"] == "related_to"
        assert e["strength"] == pytest.approx(0.8)
        assert e["context"] == "testing"
        assert e["created_at"] == pytest.approx(1_700_000_000.0)

    def test_clear_knowledge_edges(self, store: MemoryStore):
        self._seed_nodes(store)
        for i in range(3):
            store.save_knowledge_edge(
                source_id="src", target_id="tgt",
                relationship=f"rel-{i}", strength=0.5,
                context="", created_at=time.time(),
            )
        assert len(store.load_knowledge_edges()) == 3

        store.clear_knowledge_edges()
        assert len(store.load_knowledge_edges()) == 0

    def test_clear_then_re_add(self, store: MemoryStore):
        self._seed_nodes(store)
        store.save_knowledge_edge(
            source_id="src", target_id="tgt",
            relationship="old", strength=0.1,
            context="", created_at=time.time(),
        )
        store.clear_knowledge_edges()

        store.save_knowledge_edge(
            source_id="src", target_id="tgt",
            relationship="new", strength=0.9,
            context="fresh", created_at=time.time(),
        )
        edges = store.load_knowledge_edges()
        assert len(edges) == 1
        assert edges[0]["relationship"] == "new"

    def test_edge_autoincrement_ids_are_unique(self, store: MemoryStore):
        self._seed_nodes(store)
        for i in range(4):
            store.save_knowledge_edge(
                source_id="src", target_id="tgt",
                relationship=f"rel-{i}", strength=0.5,
                context="", created_at=time.time(),
            )
        edges = store.load_knowledge_edges()
        ids = [e["edge_id"] for e in edges]
        assert len(set(ids)) == 4


# ---------------------------------------------------------------------------
# 8. Persistent context
# ---------------------------------------------------------------------------

class TestPersistentContext:
    """save_persistent_context and load_persistent_context (file-based)."""

    def test_save_and_load_default_path(self, store: MemoryStore):
        content = "# Gwenn Context\nI remember things."
        saved_path = store.save_persistent_context(content)
        assert saved_path.exists()
        assert saved_path.name == "GWENN_CONTEXT.md"

        loaded = store.load_persistent_context()
        assert loaded == content

    def test_save_and_load_custom_path(self, store: MemoryStore, tmp_path: Path):
        custom = tmp_path / "custom_ctx.md"
        content = "custom context data"
        saved_path = store.save_persistent_context(content, path=custom)
        assert saved_path == custom
        assert custom.exists()

        loaded = store.load_persistent_context(path=custom)
        assert loaded == content

    def test_load_nonexistent_returns_empty_string(self, store: MemoryStore, tmp_path: Path):
        missing = tmp_path / "does_not_exist.md"
        loaded = store.load_persistent_context(path=missing)
        assert loaded == ""

    def test_overwrite_existing_context(self, store: MemoryStore):
        store.save_persistent_context("version 1")
        store.save_persistent_context("version 2")
        loaded = store.load_persistent_context()
        assert loaded == "version 2"

    def test_unicode_content(self, store: MemoryStore):
        content = "Gwenn remembers: cafe\u0301, na\u00efve, \u2014 dashes and \u201csmart quotes.\u201d"
        store.save_persistent_context(content)
        assert store.load_persistent_context() == content

    def test_empty_string_context(self, store: MemoryStore):
        store.save_persistent_context("")
        loaded = store.load_persistent_context()
        assert loaded == ""

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
    def test_persistent_context_file_is_owner_only(self, store: MemoryStore):
        path = store.save_persistent_context("private")
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# 9. Stats property
# ---------------------------------------------------------------------------

class TestStats:
    """The stats property aggregates counts from all tables."""

    def test_stats_empty_database(self, store: MemoryStore):
        s = store.stats
        assert s["episodes"] == 0
        assert s["affect_snapshots"] == 0
        assert s["identity_snapshots"] == 0
        assert s["knowledge_nodes"] == 0
        assert "db_path" in s

    def test_stats_after_populating(self, store: MemoryStore):
        # Episodes
        for i in range(3):
            store.save_episode(_make_episode(episode_id=f"ep-{i}"))

        # Affect snapshots
        for _ in range(2):
            store.save_affect_snapshot(
                valence=0.0, arousal=0.0, dominance=0.0,
                certainty=0.0, goal_congruence=0.0,
                emotion_label="neutral",
            )

        # Identity snapshots
        store.save_identity_snapshot(self_model="test model")

        # Knowledge nodes
        store.save_knowledge_node(
            node_id="kn-1", label="test", category="concept",
            content="data", confidence=0.5, source_episodes=[],
            created_at=time.time(), last_updated=time.time(),
            access_count=0,
        )

        s = store.stats
        assert s["episodes"] == 3
        assert s["affect_snapshots"] == 2
        assert s["identity_snapshots"] == 1
        assert s["knowledge_nodes"] == 1

    def test_stats_db_path_matches(self, store: MemoryStore):
        s = store.stats
        assert s["db_path"] == str(store._db_path)


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Empty tables, duplicate IDs (INSERT OR REPLACE), close and reopen."""

    def test_load_episodes_from_empty_table(self, store: MemoryStore):
        assert store.load_episodes() == []

    def test_load_affect_history_from_empty_table(self, store: MemoryStore):
        assert store.load_affect_history() == []

    def test_load_knowledge_nodes_from_empty_table(self, store: MemoryStore):
        assert store.load_knowledge_nodes() == []

    def test_load_knowledge_edges_from_empty_table(self, store: MemoryStore):
        assert store.load_knowledge_edges() == []

    def test_clear_knowledge_edges_on_empty_table(self, store: MemoryStore):
        """clear_knowledge_edges should not raise on an already-empty table."""
        store.clear_knowledge_edges()  # no error
        assert store.load_knowledge_edges() == []

    def test_vector_queries_empty_when_vector_store_disabled(self, store: MemoryStore):
        assert store.query_episode_embeddings("python", top_k=5) == []
        assert store.query_knowledge_embeddings("python", top_k=5) == []

    # -- INSERT OR REPLACE behaviour for episodes --

    def test_episode_insert_or_replace_same_id(self, store: MemoryStore):
        """Saving an episode with the same ID should replace, not duplicate."""
        ep_v1 = _make_episode(episode_id="dup", content="original")
        ep_v2 = _make_episode(episode_id="dup", content="updated")

        store.save_episode(ep_v1)
        store.save_episode(ep_v2)

        assert store.get_episode_count() == 1
        loaded = store.load_episodes()
        assert loaded[0].content == "updated"

    # -- INSERT OR REPLACE behaviour for knowledge nodes --

    def test_knowledge_node_insert_or_replace_same_id(self, store: MemoryStore):
        """Saving a knowledge node with the same node_id should replace."""
        now = time.time()
        store.save_knowledge_node(
            node_id="kn-dup", label="v1", category="concept",
            content="original", confidence=0.5, source_episodes=[],
            created_at=now, last_updated=now, access_count=0,
        )
        store.save_knowledge_node(
            node_id="kn-dup", label="v2", category="fact",
            content="replaced", confidence=0.9, source_episodes=["ep-x"],
            created_at=now, last_updated=now + 1, access_count=10,
        )

        nodes = store.load_knowledge_nodes()
        assert len(nodes) == 1
        assert nodes[0]["label"] == "v2"
        assert nodes[0]["content"] == "replaced"
        assert nodes[0]["confidence"] == pytest.approx(0.9)
        assert nodes[0]["source_episodes"] == ["ep-x"]

    # -- Close and reopen --

    def test_close_sets_conn_to_none(self, tmp_path: Path):
        ms = MemoryStore(tmp_path / "close_test.db")
        ms.initialize()
        assert ms._conn is not None
        ms.close()
        assert ms._conn is None

    def test_close_twice_is_safe(self, tmp_path: Path):
        ms = MemoryStore(tmp_path / "close_twice.db")
        ms.initialize()
        ms.close()
        ms.close()  # should not raise

    def test_data_persists_after_close_and_reopen(self, tmp_path: Path):
        db_path = tmp_path / "reopen.db"

        # First session: write data
        ms1 = MemoryStore(db_path)
        ms1.initialize()
        ms1.save_episode(_make_episode(episode_id="persist-ep", content="I survive restarts"))
        ms1.save_affect_snapshot(
            valence=0.2, arousal=0.3, dominance=0.1,
            certainty=0.4, goal_congruence=0.5,
            emotion_label="serene",
        )
        ms1.save_identity_snapshot(self_model="persistent self")
        ms1.save_knowledge_node(
            node_id="persist-kn", label="durable", category="concept",
            content="I persist", confidence=0.8, source_episodes=["persist-ep"],
            created_at=time.time(), last_updated=time.time(), access_count=1,
        )
        ms1.close()

        # Second session: read data back
        ms2 = MemoryStore(db_path)
        ms2.initialize()

        episodes = ms2.load_episodes()
        assert len(episodes) == 1
        assert episodes[0].episode_id == "persist-ep"

        affect = ms2.load_affect_history()
        assert len(affect) == 1
        assert affect[0]["emotion_label"] == "serene"

        nodes = ms2.load_knowledge_nodes()
        assert len(nodes) == 1
        assert nodes[0]["node_id"] == "persist-kn"
        assert nodes[0]["source_episodes"] == ["persist-ep"]

        s = ms2.stats
        assert s["episodes"] == 1
        assert s["affect_snapshots"] == 1
        assert s["identity_snapshots"] == 1
        assert s["knowledge_nodes"] == 1
        assert s["vector_enabled"] is False

        ms2.close()

    def test_episode_with_none_outcome(self, store: MemoryStore):
        ep = _make_episode(episode_id="none-outcome", outcome=None)
        store.save_episode(ep)
        loaded = store.load_episodes()
        assert loaded[0].outcome is None

    def test_episode_tags_json_round_trip(self, store: MemoryStore):
        """Tags serialise as JSON; ensure complex tag lists survive the trip."""
        ep = _make_episode(
            episode_id="tagged",
            tags=["alpha", "beta gamma", "special-chars_123"],
        )
        store.save_episode(ep)
        loaded = store.load_episodes()
        assert loaded[0].tags == ["alpha", "beta gamma", "special-chars_123"]

    def test_episode_participants_json_round_trip(self, store: MemoryStore):
        ep = _make_episode(
            episode_id="partic",
            participants=["alice", "bob", "self"],
        )
        store.save_episode(ep)
        loaded = store.load_episodes()
        assert loaded[0].participants == ["alice", "bob", "self"]

    def test_episode_consolidated_bool_round_trip(self, store: MemoryStore):
        """consolidated is stored as INTEGER; ensure it comes back as bool."""
        ep_true = _make_episode(episode_id="cons-true", consolidated=True)
        ep_false = _make_episode(episode_id="cons-false", consolidated=False)
        store.save_episode(ep_true)
        store.save_episode(ep_false)

        loaded = {e.episode_id: e for e in store.load_episodes()}
        # from_dict may return int or bool depending on implementation;
        # verify truthiness is correct
        assert loaded["cons-true"].consolidated
        assert not loaded["cons-false"].consolidated


class TestWorkingMemorySidecar:
    """save_working_memory / load_working_memory round-trip and decay."""

    def test_round_trip_preserves_high_salience_items(self, store: MemoryStore):
        items = [
            {
                "item_id": "wm-1",
                "content": "thinking about something",
                "category": "general",
                "salience": 0.9,
                "entered_at": time.time(),
                "emotional_valence": 0.1,
                "access_count": 2,
            }
        ]
        store.save_working_memory(items)
        loaded = store.load_working_memory()
        assert len(loaded) == 1
        assert loaded[0]["item_id"] == "wm-1"
        assert loaded[0]["salience"] > 0.0

    def test_low_salience_items_filtered_on_save(self, store: MemoryStore):
        items = [
            {"item_id": "low", "content": "x", "salience": 0.03},
            {"item_id": "high", "content": "y", "salience": 0.8},
        ]
        store.save_working_memory(items)
        loaded = store.load_working_memory()
        ids = [i["item_id"] for i in loaded]
        assert "high" in ids
        assert "low" not in ids

    def test_missing_file_returns_empty_list(self, store: MemoryStore, tmp_path):
        result = store.load_working_memory(path=tmp_path / "nonexistent.json")
        assert result == []

    def test_decay_applied_to_old_items(self, store: MemoryStore, tmp_path):
        """Items saved a long time ago have their salience reduced on load."""
        import json as _json
        path = tmp_path / "wm_old.json"
        # Simulate saving 60 minutes ago (decay_rate=0.02 * 60 = 1.2 => salience 0.6 → drops below 0.05)
        payload = {
            "saved_at": time.time() - 3600,
            "items": [
                {"item_id": "old-low", "salience": 0.5, "content": "x"},
                {"item_id": "old-high", "salience": 0.99, "content": "y"},
            ],
        }
        path.write_text(_json.dumps(payload), encoding="utf-8")
        loaded = store.load_working_memory(path=path)
        ids = [i["item_id"] for i in loaded]
        # old-low (0.5 - 1.2 = -0.7) should be pruned; old-high (0.99 - 1.2 = -0.21) too
        # Both items should be gone after 60 minutes at rate 0.02/min
        assert "old-low" not in ids
        assert "old-high" not in ids

    def test_empty_items_list_saves_and_loads(self, store: MemoryStore):
        store.save_working_memory([])
        assert store.load_working_memory() == []


class TestPruneOldEpisodes:
    """prune_old_episodes deletes old consolidated low-importance episodes."""

    def test_prunes_qualifying_episodes(self, store: MemoryStore):
        old_ts = time.time() - (100 * 86400)  # 100 days ago
        ep_old_cons = _make_episode(
            episode_id="old-cons",
            timestamp=old_ts,
            importance=0.1,
            consolidated=True,
        )
        ep_old_unconsolidated = _make_episode(
            episode_id="old-uncons",
            timestamp=old_ts,
            importance=0.1,
            consolidated=False,
        )
        ep_recent = _make_episode(
            episode_id="recent",
            timestamp=time.time(),
            importance=0.1,
            consolidated=True,
        )
        ep_high_importance = _make_episode(
            episode_id="high-imp",
            timestamp=old_ts,
            importance=0.9,
            consolidated=True,
        )
        for ep in [ep_old_cons, ep_old_unconsolidated, ep_recent, ep_high_importance]:
            store.save_episode(ep)

        deleted = store.prune_old_episodes(older_than_days=90.0, max_importance=0.3)
        assert deleted == 1  # Only ep_old_cons qualifies

        remaining_ids = {e.episode_id for e in store.load_episodes(limit=None)}
        assert "old-cons" not in remaining_ids
        assert "old-uncons" in remaining_ids
        assert "recent" in remaining_ids
        assert "high-imp" in remaining_ids

    def test_no_deletions_when_nothing_qualifies(self, store: MemoryStore):
        ep = _make_episode(episode_id="safe", timestamp=time.time(), importance=0.5)
        store.save_episode(ep)
        deleted = store.prune_old_episodes()
        assert deleted == 0


class TestUpsertContextSection:
    """Tests for agent._upsert_context_section helper."""

    def test_creates_new_section_in_empty_string(self):
        from gwenn.agent import _upsert_context_section
        result = _upsert_context_section("", "reminders", "buy milk")
        assert "## Reminders" in result
        assert "- buy milk" in result

    def test_appends_to_existing_section(self):
        from gwenn.agent import _upsert_context_section
        content = "## Reminders\n- first note"
        result = _upsert_context_section(content, "reminders", "second note")
        assert "- first note" in result
        assert "- second note" in result
        assert result.count("## Reminders") == 1

    def test_creates_new_section_when_header_absent(self):
        from gwenn.agent import _upsert_context_section
        content = "## Other Section\n- something"
        result = _upsert_context_section(content, "reminders", "new note")
        assert "## Other Section" in result
        assert "## Reminders" in result
        assert "- new note" in result

    def test_header_in_body_text_does_not_trigger_false_match(self):
        """A partial match like '## Reminders extra' must NOT count as the section."""
        from gwenn.agent import _upsert_context_section
        content = "## Reminders Extra\n- unrelated"
        result = _upsert_context_section(content, "reminders", "my note")
        # A new '## Reminders' section should be appended, not inserted into the wrong section
        assert result.count("## Reminders\n") >= 1
        assert "## Reminders Extra" in result

    def test_section_name_with_underscores_titlecased(self):
        from gwenn.agent import _upsert_context_section
        result = _upsert_context_section("", "long_term_goals", "finish the project")
        assert "## Long Term Goals" in result
