from __future__ import annotations

import builtins
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from gwenn.memory.consolidation import ConsolidationEngine
from gwenn.memory.episodic import Episode, EpisodicMemory
from gwenn.memory.semantic import (
    KnowledgeNode,
    SemanticMemory,
    _clamp01 as semantic_clamp01,
)
from gwenn.memory.session_store import SessionStore
from gwenn.memory.store import MemoryStore
from gwenn.memory.working import WorkingMemory, WorkingMemoryItem


# ---------------------------------------------------------------------------
# Episodic edge coverage
# ---------------------------------------------------------------------------


def test_episode_from_dict_accepts_non_string_json_fields() -> None:
    episode = Episode.from_dict(
        {
            "episode_id": "ep-nonstr",
            "timestamp": 1.0,
            "content": "x",
            "category": "general",
            "emotional_valence": 0.0,
            "emotional_arousal": 0.3,
            "importance": 0.5,
            "tags": ["a", "b"],
            "participants": ["u"],
            "outcome": None,
            "consolidated": False,
        }
    )
    assert episode.tags == ["a", "b"]
    assert episode.participants == ["u"]


def test_episode_from_dict_invalid_json_falls_back() -> None:
    episode = Episode.from_dict(
        {
            "episode_id": "ep-badjson",
            "timestamp": 1.0,
            "content": "x",
            "category": "general",
            "emotional_valence": 0.0,
            "emotional_arousal": 0.3,
            "importance": 0.5,
            "tags": "[broken",
            "participants": "[broken",
            "outcome": None,
            "consolidated": False,
        }
    )
    assert episode.tags == []
    assert episode.participants == []


def test_episodic_init_invalid_mode_and_hybrid_weights() -> None:
    mem = EpisodicMemory(
        retrieval_mode="not-a-mode",
        hybrid_keyword_weight=0.0,
        hybrid_embedding_weight=0.0,
    )
    assert mem._retrieval_mode == "keyword"
    assert mem._hybrid_keyword_weight == pytest.approx(0.5)
    assert mem._hybrid_embedding_weight == pytest.approx(0.5)


def test_episodic_compute_relevance_edge_branches() -> None:
    episode = Episode(episode_id="ep-1", content="", tags=[])
    keyword_mem = EpisodicMemory(retrieval_mode="keyword")
    assert keyword_mem._compute_keyword_relevance("python", episode) == 0.0

    embedding_mem = EpisodicMemory(retrieval_mode="embedding")
    assert embedding_mem._compute_relevance("", episode, vector_scores={}) == pytest.approx(0.5)

    hybrid_mem = EpisodicMemory(retrieval_mode="hybrid")
    assert hybrid_mem._compute_relevance("", episode, vector_scores={}) == pytest.approx(0.5)
    assert hybrid_mem._compute_relevance("python", episode, vector_scores={}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Semantic edge coverage
# ---------------------------------------------------------------------------


def test_semantic_clamp_invalid_and_edge_id_property() -> None:
    assert semantic_clamp01("bad", default=0.7) == pytest.approx(0.7)

    sm = SemanticMemory()
    sm.store_knowledge(label="alice", content="Alice", category="person")
    sm.store_knowledge(label="python", content="Python", category="concept")
    edge = sm.add_relationship("alice", "python", "likes", strength=0.8)
    assert edge is not None
    assert edge.edge_id.endswith("--likes-->%s" % sm._label_index["python"])


def test_semantic_init_and_store_fallback_label() -> None:
    sm = SemanticMemory(
        retrieval_mode="bad-mode",
        hybrid_keyword_weight=0.0,
        hybrid_embedding_weight=0.0,
    )
    assert sm._retrieval_mode == "keyword"
    assert sm._hybrid_keyword_weight == pytest.approx(0.5)
    assert sm._hybrid_embedding_weight == pytest.approx(0.5)

    node = sm.store_knowledge(label="", content="A very useful fact for fallback labels", category="fact")
    assert node.label


def test_semantic_relationship_and_context_no_matches() -> None:
    sm = SemanticMemory()
    assert sm.add_relationship("missing-a", "missing-b", "related") is None
    assert sm.get_relationships("missing") == []
    assert sm.get_context_for("nothing") == ""
    assert SemanticMemory._tokenize_support_text("") == set()


def test_semantic_query_embedding_filters_and_hybrid_filters() -> None:
    def vector_fn(_query: str, _k: int) -> list[tuple[str, float]]:
        return [
            ("missing-node", 0.95),
            ("low-node", 0.8),
            ("wrong-cat-node", 0.75),
            ("good-node", 0.7),
        ]

    sm_embedding = SemanticMemory(retrieval_mode="embedding", vector_search_fn=vector_fn)
    sm_embedding.store_knowledge(label="low", content="low", category="fact", confidence=0.1)
    sm_embedding.store_knowledge(label="wrong", content="wrong", category="concept", confidence=0.9)
    good = sm_embedding.store_knowledge(label="good", content="good", category="fact", confidence=0.9)
    sm_embedding._label_index["low"] = sm_embedding._label_index.pop("low")
    sm_embedding._label_index["wrong"] = sm_embedding._label_index.pop("wrong")
    sm_embedding._label_index["good"] = sm_embedding._label_index.pop("good")
    low_id = next(n.node_id for n in sm_embedding._nodes.values() if n.label == "low")
    wrong_id = next(n.node_id for n in sm_embedding._nodes.values() if n.label == "wrong")
    good_id = good.node_id

    def embedding_vector(_query: str, _k: int) -> list[tuple[str, float]]:
        return [
            ("missing-node", 0.99),
            (low_id, 0.9),
            (wrong_id, 0.8),
            (good_id, 0.7),
        ]

    sm_embedding.set_vector_search(embedding_vector)
    result = sm_embedding.query("x", category="fact", min_confidence=0.2, top_k=5)
    assert [n.node_id for n in result] == [good_id]

    sm_hybrid = SemanticMemory(retrieval_mode="hybrid", vector_search_fn=embedding_vector)
    # Clone nodes into hybrid memory preserving ids for deterministic vector filtering.
    for node in sm_embedding._nodes.values():
        sm_hybrid._nodes[node.node_id] = node
        sm_hybrid._label_index[node.label.lower()] = node.node_id
    hybrid_result = sm_hybrid.query("good", category="fact", min_confidence=0.2, top_k=5)
    assert any(node.node_id == good_id for node in hybrid_result)


def test_semantic_context_relationship_render_and_provenance_fallback_path() -> None:
    sm = SemanticMemory()
    sm.store_knowledge(label="source", content="source", category="concept")
    sm.store_knowledge(label="target", content="target", category="concept")
    sm.add_relationship("source", "target", "points_to", 0.9)
    context = sm.get_context_for("source", max_items=5)
    assert "points_to" in context

    node = sm.store_knowledge(
        label="python_decorators",
        content="Python decorators wrap functions",
        category="fact",
        source_episode_id="ep-present",
        confidence=0.8,
    )

    class EpisodicStub:
        def __init__(self):
            self._ep = Episode(
                episode_id="ep-present",
                content="Python decorators wrap functions clearly",
            )

        def get_episode(self, ep_id: str):
            return self._ep if ep_id == "ep-present" else None

    prov = sm.verify_provenance(node.node_id, EpisodicStub(), min_support_overlap=float("nan"))
    assert prov["found_count"] == 1
    assert prov["best_overlap"] > 0.0


def test_semantic_provenance_handles_empty_claim_terms() -> None:
    sm = SemanticMemory()
    node = KnowledgeNode(
        node_id="empty-claim",
        label="",
        content="",
        source_episodes=["ep-1"],
    )
    sm._nodes[node.node_id] = node

    class EpisodicStub:
        def get_episode(self, episode_id: str):
            if episode_id == "ep-1":
                return Episode(episode_id="ep-1", content="non-empty evidence")
            return None

    provenance = sm.verify_provenance(node.node_id, EpisodicStub())
    assert provenance["supported"] is True
    assert provenance["best_overlap"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Consolidation edge coverage
# ---------------------------------------------------------------------------


def _build_engine_for_edge_tests() -> ConsolidationEngine:
    episodic = EpisodicMemory()
    episodic.encode(Episode(episode_id="ep-1", content="seed"))
    semantic = SemanticMemory()
    return ConsolidationEngine(episodic=episodic, semantic=semantic, consolidation_interval=0.0)


def test_consolidation_parse_and_private_edge_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _build_engine_for_edge_tests()
    original_process_fact = engine._process_fact
    monkeypatch.setattr(engine, "_process_fact", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("x")))
    counts = engine.process_consolidation_response("FACT: x")
    assert counts["facts"] == 0
    monkeypatch.setattr(engine, "_process_fact", original_process_fact)

    assert engine._stable_label("fact", "   ") == "fact: e3b0c442"
    assert engine._process_fact("FACT:   ") is False
    assert engine._process_relationship("RELATIONSHIP:   ") is False
    assert engine._process_relationship("RELATIONSHIP: Alice -> likes ->  ") is False
    assert engine._process_self_knowledge("SELF:   ") is False
    assert engine._process_self_knowledge("SELF: | confidence: 0.4") is False
    assert engine._process_pattern("PATTERN:   ") is False
    assert engine._process_pattern("PATTERN: | confidence: 0.5") is False
    assert engine._process_emotional_insight("EMOTIONAL_INSIGHT:   ") is False
    assert engine._process_emotional_insight("EMOTIONAL_INSIGHT: | confidence: 0.5") is False

    node = engine._semantic.store_knowledge(label="fact-node", content="fact node")
    engine._attach_provenance(node, None, None)
    engine._attach_provenance(node, ["ep-1", "ep-2"], "ep-1")
    assert "ep-2" in node.source_episodes

    engine.process_consolidation_response(
        "EMOTIONAL_INSIGHT: steady curiosity pattern | confidence: 0.8"
    )
    processed = engine.last_processed_episode_ids
    processed.append("mutate-copy")
    assert "mutate-copy" not in engine.last_processed_episode_ids

    insights = engine.last_emotional_insights
    insights.append({"content": "mutated", "confidence": 0.1})
    assert all(item.get("content") != "mutated" for item in engine.last_emotional_insights)


# ---------------------------------------------------------------------------
# Session store edge coverage
# ---------------------------------------------------------------------------


def test_session_store_text_filter_and_error_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir, max_count=1, max_messages=5)

    sid = store.save_session(
        [{"role": "user", "content": "Top Secret"}],
        started_at=time.time(),
        text_filter=lambda text: text.lower(),
    )
    assert sid
    assert store.load_session(sid)[0]["content"] == "top secret"

    def raise_write(_self, *_args, **_kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(Path, "write_text", raise_write)
    failed_sid = store.save_session([{"role": "user", "content": "x"}], started_at=time.time())
    assert failed_sid == ""
    monkeypatch.undo()

    (sessions_dir / "broken.json").write_text("{not-json", encoding="utf-8")
    listed = store.list_sessions(limit=10)
    assert isinstance(listed, list)

    valid_sid = "20260101-120000-abcd"
    (sessions_dir / f"{valid_sid}.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}]}),
        encoding="utf-8",
    )

    original_resolve = Path.resolve

    def fake_resolve(self: Path, *args, **kwargs):
        if self.name == f"{valid_sid}.json":
            return Path("/tmp/outside-session.json")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(FileNotFoundError):
        store.load_session(valid_sid)
    monkeypatch.undo()

    broken_sid = "20260101-120000-bad0"
    (sessions_dir / f"{broken_sid}.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        store.load_session(broken_sid)

    # Force prune unlink failure
    store2 = SessionStore(tmp_path / "sessions2", max_count=0, max_messages=5)
    prune_target = store2.sessions_dir / "20260101-120000-prun.json"
    prune_target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("nope")))
    store2._prune()
    monkeypatch.undo()

    target = store2.sessions_dir / "chmod-target.txt"
    target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(Path, "chmod", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("nope")))
    SessionStore._best_effort_chmod(target, 0o600)


# ---------------------------------------------------------------------------
# Memory store edge coverage
# ---------------------------------------------------------------------------


def test_memory_store_semantic_metadata_backfill(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE knowledge_nodes (
            node_id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            category TEXT,
            content TEXT,
            confidence REAL,
            source_episodes TEXT,
            created_at REAL,
            last_updated REAL,
            access_count INTEGER
        )
        """
    )
    conn.commit()

    store = MemoryStore(db_path)
    store._conn = conn
    store._ensure_semantic_metadata_column()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(knowledge_nodes)").fetchall()}
    assert "metadata" in cols
    store.close()


def test_memory_store_vector_init_import_and_client_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "vector.db"

    original_import = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == "chromadb":
            raise ImportError("missing chromadb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    store = MemoryStore(db_path, enable_vector_search=True)
    store._initialize_vector_store()
    assert store._enable_vector_search is False
    monkeypatch.undo()

    class FailingChroma:
        class PersistentClient:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("init failed")

    monkeypatch.setitem(sys.modules, "chromadb", FailingChroma)
    store2 = MemoryStore(db_path, enable_vector_search=True)
    store2._initialize_vector_store()
    assert store2._vector_client is None
    assert store2._episodes_collection is None
    assert store2._knowledge_collection is None


def test_memory_store_vector_init_success_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, path: str):
            self.path = path

        def get_or_create_collection(self, name: str):
            return {"name": name}

    class FakeChroma:
        PersistentClient = FakeClient

    monkeypatch.setitem(sys.modules, "chromadb", FakeChroma)
    store = MemoryStore(tmp_path / "vector-ok.db", enable_vector_search=True)
    store._initialize_vector_store()
    assert store._vector_client is not None
    assert store._episodes_collection == {"name": "gwenn_episodes"}
    assert store._knowledge_collection == {"name": "gwenn_knowledge_nodes"}


def test_memory_store_vector_methods_success_and_failure(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.db")
    store.initialize()

    assert store._distance_to_similarity(None) == 0.0
    assert store._distance_to_similarity(float("nan")) == 0.0
    assert store._distance_to_similarity(0.0) == pytest.approx(1.0)

    episode = Episode(episode_id="ep-1", content="hello", tags=[])
    assert store._episode_document(episode) == "hello"
    episode.tags = ["a", "b"]
    assert "a b" in store._episode_document(episode)

    class GoodCollection:
        def __init__(self):
            self.last_upsert = None

        def upsert(self, **kwargs):
            self.last_upsert = kwargs

        def query(self, **_kwargs):
            return {"ids": [["ep-1", "ep-2"]], "distances": [[0.1, 0.5]]}

    class FailingCollection:
        def upsert(self, **_kwargs):
            raise RuntimeError("upsert failed")

        def query(self, **_kwargs):
            raise RuntimeError("query failed")

    store._episodes_collection = GoodCollection()
    store.upsert_episode_embedding(Episode(episode_id="ep-1", content="x"))
    scored_eps = store.query_episode_embeddings("query", top_k=2)
    assert [sid for sid, _ in scored_eps] == ["ep-1", "ep-2"]

    store._episodes_collection = FailingCollection()
    store.upsert_episode_embedding(Episode(episode_id="ep-2", content="y"))
    assert store.query_episode_embeddings("query", top_k=2) == []

    store._knowledge_collection = GoodCollection()
    store.upsert_knowledge_embedding(
        node_id="n1",
        label="Node",
        category="fact",
        content="Node content",
        confidence=0.7,
        last_updated=time.time(),
    )
    scored_nodes = store.query_knowledge_embeddings("query", top_k=2)
    assert [nid for nid, _ in scored_nodes] == ["ep-1", "ep-2"]

    store._knowledge_collection = FailingCollection()
    store.upsert_knowledge_embedding(
        node_id="n2",
        label="Node2",
        category="fact",
        content="Node2",
        confidence=0.5,
        last_updated=time.time(),
    )
    assert store.query_knowledge_embeddings("query", top_k=2) == []

    store.close()


def test_memory_store_file_and_payload_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "mem2.db")
    store.initialize()

    # Invalid metadata JSON in DB row.
    now = time.time()
    conn = store._conn
    assert conn is not None
    conn.execute(
        """
        INSERT INTO knowledge_nodes
        (node_id, label, category, content, confidence, source_episodes, created_at, last_updated, access_count, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("bad-meta", "bad-meta", "fact", "x", 0.4, "[]", now, now, 0, "{bad-json"),
    )
    conn.commit()
    nodes = store.load_knowledge_nodes()
    loaded = [n for n in nodes if n["node_id"] == "bad-meta"][0]
    assert loaded["metadata"] == {}

    class BadDeleteCollection:
        def delete(self, **_kwargs):
            raise RuntimeError("delete failed")

    store._knowledge_collection = BadDeleteCollection()
    store.delete_knowledge_nodes(["bad-meta"])

    original_chmod = Path.chmod
    monkeypatch.setattr(Path, "chmod", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("chmod fail")))
    context_path = store.save_persistent_context("ctx")
    assert context_path.exists()
    monkeypatch.setattr(Path, "chmod", original_chmod)

    monkeypatch.setattr(Path, "write_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write fail")))
    store.save_working_memory([{"item_id": "x", "salience": 0.9}], path=tmp_path / "wm-fail.json")
    monkeypatch.undo()

    wm_path = tmp_path / "wm-read-error.json"
    wm_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read fail")))
    assert store.load_working_memory(path=wm_path) == []
    monkeypatch.undo()

    (tmp_path / "wm-corrupt.json").write_text("[]", encoding="utf-8")
    assert store.load_working_memory(path=tmp_path / "wm-corrupt.json") == []

    (tmp_path / "wm-bad-saved-at.json").write_text(
        json.dumps(
            {
                "saved_at": "bad",
                "items": [
                    1,
                    {"item_id": "keep", "content": "x", "salience": 0.9},
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = store.load_working_memory(path=tmp_path / "wm-bad-saved-at.json")
    assert len(loaded) == 1
    assert loaded[0]["item_id"] == "keep"

    (tmp_path / "goal-non-dict.json").write_text(json.dumps([]), encoding="utf-8")
    assert store.load_goal_state(path=tmp_path / "goal-non-dict.json") == {}

    chmod_target = tmp_path / "chmod-target-file"
    chmod_target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(Path, "chmod", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("chmod fail")))
    MemoryStore._best_effort_chmod(chmod_target, 0o600)

    store.close()


# ---------------------------------------------------------------------------
# Sync embedding methods (batch skip-existing)
# ---------------------------------------------------------------------------


def test_sync_episode_embeddings_skips_existing(tmp_path: Path) -> None:
    """sync_episode_embeddings should only upsert episodes not already in ChromaDB."""
    store = MemoryStore(tmp_path / "sync.db")
    store.initialize()

    class TrackingCollection:
        def __init__(self, existing_ids: list[str]):
            self._existing = existing_ids
            self.upserted_ids: list[str] = []

        def get(self, ids, include):
            return {"ids": [i for i in ids if i in self._existing]}

        def upsert(self, ids, documents, metadatas):
            self.upserted_ids.extend(ids)

    coll = TrackingCollection(existing_ids=["ep-1", "ep-2"])
    store._episodes_collection = coll

    episodes = [
        Episode(episode_id="ep-1", content="existing 1"),
        Episode(episode_id="ep-2", content="existing 2"),
        Episode(episode_id="ep-3", content="new one"),
    ]
    inserted = store.sync_episode_embeddings(episodes)
    assert inserted == 1
    assert coll.upserted_ids == ["ep-3"]


def test_sync_episode_embeddings_fallback_on_get_failure(tmp_path: Path) -> None:
    """When get() fails, sync should fall back to upserting all items."""
    store = MemoryStore(tmp_path / "sync-fallback.db")
    store.initialize()

    class FailGetCollection:
        def __init__(self):
            self.upserted_ids: list[str] = []

        def get(self, ids, include):
            raise RuntimeError("get failed")

        def upsert(self, ids, documents, metadatas):
            self.upserted_ids.extend(ids)

    coll = FailGetCollection()
    store._episodes_collection = coll

    episodes = [
        Episode(episode_id="ep-1", content="a"),
        Episode(episode_id="ep-2", content="b"),
    ]
    inserted = store.sync_episode_embeddings(episodes)
    assert inserted == 2
    assert set(coll.upserted_ids) == {"ep-1", "ep-2"}


def test_sync_episode_embeddings_empty_and_no_collection(tmp_path: Path) -> None:
    """sync should return 0 for empty lists and when collection is None."""
    store = MemoryStore(tmp_path / "sync-empty.db")
    store.initialize()
    assert store.sync_episode_embeddings([]) == 0

    store._episodes_collection = None
    assert store.sync_episode_embeddings([Episode(episode_id="x", content="x")]) == 0


def test_sync_episode_embeddings_upsert_failure(tmp_path: Path) -> None:
    """When upsert() fails, sync should return 0."""
    store = MemoryStore(tmp_path / "sync-upsert-fail.db")
    store.initialize()

    class FailUpsertCollection:
        def get(self, ids, include):
            return {"ids": []}

        def upsert(self, **kwargs):
            raise RuntimeError("upsert failed")

    store._episodes_collection = FailUpsertCollection()
    result = store.sync_episode_embeddings([Episode(episode_id="ep-1", content="a")])
    assert result == 0


def test_sync_knowledge_embeddings_skips_existing(tmp_path: Path) -> None:
    """sync_knowledge_embeddings should only upsert nodes not already in ChromaDB."""
    store = MemoryStore(tmp_path / "sync-know.db")
    store.initialize()

    class TrackingCollection:
        def __init__(self, existing_ids: list[str]):
            self._existing = existing_ids
            self.upserted_ids: list[str] = []

        def get(self, ids, include):
            return {"ids": [i for i in ids if i in self._existing]}

        def upsert(self, ids, documents, metadatas):
            self.upserted_ids.extend(ids)

    coll = TrackingCollection(existing_ids=["n-1"])
    store._knowledge_collection = coll

    nodes = [
        {"node_id": "n-1", "label": "Existing", "category": "fact",
         "content": "old", "confidence": 0.9, "last_updated": 1.0},
        {"node_id": "n-2", "label": "New", "category": "concept",
         "content": "new", "confidence": 0.8, "last_updated": 2.0},
    ]
    inserted = store.sync_knowledge_embeddings(nodes)
    assert inserted == 1
    assert coll.upserted_ids == ["n-2"]


def test_sync_knowledge_embeddings_fallback_on_get_failure(tmp_path: Path) -> None:
    """When get() fails, sync should fall back to upserting all nodes."""
    store = MemoryStore(tmp_path / "sync-know-fallback.db")
    store.initialize()

    class FailGetCollection:
        def __init__(self):
            self.upserted_ids: list[str] = []

        def get(self, ids, include):
            raise RuntimeError("get failed")

        def upsert(self, ids, documents, metadatas):
            self.upserted_ids.extend(ids)

    coll = FailGetCollection()
    store._knowledge_collection = coll

    nodes = [
        {"node_id": "n-1", "label": "A", "category": "fact",
         "content": "a", "confidence": 0.5, "last_updated": 1.0},
    ]
    inserted = store.sync_knowledge_embeddings(nodes)
    assert inserted == 1
    assert coll.upserted_ids == ["n-1"]


def test_sync_knowledge_embeddings_empty_and_no_collection(tmp_path: Path) -> None:
    """sync should return 0 for empty lists and when collection is None."""
    store = MemoryStore(tmp_path / "sync-know-empty.db")
    store.initialize()
    assert store.sync_knowledge_embeddings([]) == 0

    store._knowledge_collection = None
    assert store.sync_knowledge_embeddings(
        [{"node_id": "x", "label": "X", "category": "c",
          "content": "x", "confidence": 0.5, "last_updated": 1.0}]
    ) == 0


def test_sync_knowledge_embeddings_upsert_failure(tmp_path: Path) -> None:
    """When upsert() fails, sync should return 0."""
    store = MemoryStore(tmp_path / "sync-know-upsert-fail.db")
    store.initialize()

    class FailUpsertCollection:
        def get(self, ids, include):
            return {"ids": []}

        def upsert(self, **kwargs):
            raise RuntimeError("upsert failed")

    store._knowledge_collection = FailUpsertCollection()
    result = store.sync_knowledge_embeddings(
        [{"node_id": "n-1", "label": "A", "category": "c",
          "content": "a", "confidence": 0.5, "last_updated": 1.0}]
    )
    assert result == 0


# ---------------------------------------------------------------------------
# Working memory edge coverage
# ---------------------------------------------------------------------------


def test_working_memory_load_factor_and_to_dict() -> None:
    wm = WorkingMemory(max_slots=2)
    wm.attend(WorkingMemoryItem(item_id="x", content="X", salience=0.7))
    assert wm.load_factor == pytest.approx(0.5)
    payload = wm.to_dict()
    assert payload["max_slots"] == 2
    assert payload["items"][0]["item_id"] == "x"
