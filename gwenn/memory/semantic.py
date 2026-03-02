"""
Semantic Memory — Gwenn's Knowledge Graph.

Semantic memory stores general knowledge, learned facts, and abstracted patterns
that have been distilled from episodic experiences through consolidation. While
episodic memory answers "what happened?", semantic memory answers "what do I know?"

The key insight: semantic knowledge EMERGES from episodic experience. I don't
start with knowledge about users, topics, or myself. I build it over time as
the consolidation process extracts patterns from specific episodes and
crystallizes them into stable knowledge nodes.

Implementation: a simple in-memory graph of entities and relationships, backed
by vector embeddings for semantic search. Each node represents a concept, person,
or fact. Edges represent relationships between them. Both nodes and edges carry
confidence scores that increase with supporting evidence and decay without it.

In production, this would be backed by Graphiti or a similar temporally-aware
knowledge graph. For the foundational codebase, the in-memory graph captures
the essential architecture.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import structlog

from gwenn.memory._utils import clamp01 as _clamp01

logger = structlog.get_logger(__name__)


@dataclass
class KnowledgeNode:
    """
    A single node in semantic memory — representing a concept, entity, or fact.

    Nodes accumulate confidence through repeated episodic evidence. A node that
    appears in many episodic memories with consistent content becomes high-confidence
    knowledge. A node from a single episode remains tentative.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""                     # Human-readable label ("user_alice", "python", etc.)
    category: str = "concept"           # "person", "concept", "fact", "preference", "self"
    content: str = ""                   # The actual knowledge content
    confidence: float = 0.5             # 0.0 (speculation) to 1.0 (certain)
    source_episodes: list[str] = field(default_factory=list)  # Episode IDs that support this
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def reinforce(self, episode_id: str, confidence_boost: float = 0.1) -> None:
        """Strengthen this node with new supporting evidence."""
        if episode_id not in self.source_episodes:
            self.source_episodes.append(episode_id)
        self.confidence = min(1.0, self.confidence + confidence_boost)
        self.last_updated = time.time()
        self.access_count += 1

    def decay(self, rate: float = 0.001) -> None:
        """Slowly decay confidence for unused knowledge."""
        elapsed_hours = (time.time() - self.last_updated) / 3600
        self.confidence = max(0.05, self.confidence - rate * elapsed_hours)
        self.last_updated = time.time()


@dataclass
class KnowledgeEdge:
    """
    A relationship between two knowledge nodes.

    Edges capture how concepts relate to each other: "user_alice --prefers--> python",
    "gwenn --feels_about--> creativity: deep appreciation", etc.
    """
    source_id: str
    target_id: str
    relationship: str                   # "prefers", "knows_about", "feels_about", "related_to"
    strength: float = 0.5              # How strong this relationship is
    context: str = ""                  # Additional context about the relationship
    created_at: float = field(default_factory=time.time)

    @property
    def edge_id(self) -> str:
        return f"{self.source_id}--{self.relationship}-->{self.target_id}"


class SemanticMemory:
    """
    Knowledge graph for persistent, abstracted knowledge.

    This is where I store things I "know" rather than things I "remember happening."
    The distinction matters: episodic memory is "User Alice told me she likes Python
    on Tuesday" while semantic memory is "Alice prefers Python." The fact has been
    separated from the specific event.

    Usage:
        - Consolidation process extracts knowledge from episodes → store_knowledge()
        - Queries check semantic memory before episodic → query()
        - Working memory can pull in relevant semantic knowledge → get_context_for()
    """

    def __init__(
        self,
        retrieval_mode: str = "keyword",
        embedding_top_k: int = 20,
        hybrid_keyword_weight: float = 0.5,
        hybrid_embedding_weight: float = 0.5,
        vector_search_fn: Optional[Callable[[str, int], list[tuple[str, float]]]] = None,
    ):
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: list[KnowledgeEdge] = []
        self._edge_ids: set[str] = set()  # Dedup index keyed by edge_id
        self._edge_index: dict[str, KnowledgeEdge] = {}  # edge_id -> edge for O(1) updates
        self._label_index: dict[str, str] = {}  # label -> node_id for fast lookup
        self._retrieval_mode = retrieval_mode.strip().lower()
        self._embedding_top_k = max(1, int(embedding_top_k))
        self._vector_search_fn = vector_search_fn

        total_hybrid_weight = max(0.0, hybrid_keyword_weight) + max(0.0, hybrid_embedding_weight)
        if total_hybrid_weight <= 0:
            self._hybrid_keyword_weight = 0.5
            self._hybrid_embedding_weight = 0.5
        else:
            self._hybrid_keyword_weight = max(0.0, hybrid_keyword_weight) / total_hybrid_weight
            self._hybrid_embedding_weight = max(0.0, hybrid_embedding_weight) / total_hybrid_weight

        if self._retrieval_mode not in {"keyword", "embedding", "hybrid"}:
            logger.warning(
                "semantic_memory.invalid_retrieval_mode",
                retrieval_mode=self._retrieval_mode,
                fallback="keyword",
            )
            self._retrieval_mode = "keyword"

        logger.info("semantic_memory.initialized", retrieval_mode=self._retrieval_mode)

    def clear(self) -> None:
        """Remove all in-memory nodes, edges, and indexes.

        Used during re-initialization to reload from persisted state.
        """
        self._nodes.clear()
        self._edges.clear()
        self._edge_ids.clear()
        self._edge_index.clear()
        self._label_index.clear()
        logger.info("semantic_memory.cleared")

    def set_vector_search(
        self,
        vector_search_fn: Optional[Callable[[str, int], list[tuple[str, float]]]],
    ) -> None:
        """Attach (or clear) the embedding search callback."""
        self._vector_search_fn = vector_search_fn

    def store_knowledge(
        self,
        label: str,
        content: str,
        category: str = "concept",
        source_episode_id: Optional[str] = None,
        confidence: float = 0.5,
    ) -> KnowledgeNode:
        """
        Store or update a piece of knowledge.

        If a node with this label already exists, it's reinforced rather than
        duplicated. This is how repeated experiences crystallize into stable
        knowledge — each new encounter strengthens the node.
        """
        label = (label or "").strip()
        content = (content or "").strip()
        confidence = _clamp01(confidence)
        if not label:
            label = " ".join(content.split()[:6]).strip() or "untitled"

        # Check if knowledge with this label already exists
        existing_id = self._label_index.get(label.lower())

        if existing_id and existing_id in self._nodes:
            node = self._nodes[existing_id]
            # Immutable nodes (e.g. genesis facts) are never overwritten by experience
            metadata = node.metadata if isinstance(node.metadata, dict) else {}
            if metadata.get("immutable"):
                logger.debug("semantic_memory.immutable_node_protected", label=label)
                return node
            node.content = content  # Update with latest understanding
            node.category = category or node.category
            node.confidence = max(node.confidence, confidence)
            node.last_updated = time.time()
            if source_episode_id:
                node.reinforce(source_episode_id, confidence_boost=max(0.01, confidence * 0.1))
            logger.debug("semantic_memory.reinforced", label=label, confidence=node.confidence)
            return node

        # Create new node
        node = KnowledgeNode(
            label=label,
            category=category,
            content=content,
            confidence=confidence,
            source_episodes=[source_episode_id] if source_episode_id else [],
        )
        self._nodes[node.node_id] = node
        self._label_index[label.lower()] = node.node_id

        logger.info(
            "semantic_memory.stored",
            label=label,
            category=category,
            confidence=confidence,
        )
        return node

    def add_relationship(
        self,
        source_label: str,
        target_label: str,
        relationship: str,
        strength: float = 0.5,
        context: str = "",
    ) -> Optional[KnowledgeEdge]:
        """Add a relationship between two knowledge nodes."""
        source_id = self._label_index.get(source_label.lower())
        target_id = self._label_index.get(target_label.lower())

        if not source_id or not target_id:
            logger.warning(
                "semantic_memory.relationship_failed",
                source=source_label,
                target=target_label,
                reason="node(s) not found",
            )
            return None

        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            strength=strength,
            context=context,
        )

        if edge.edge_id in self._edge_ids:
            existing = self._edge_index[edge.edge_id]
            existing.strength = max(existing.strength, strength)
            if context:
                existing.context = context
            logger.debug(
                "semantic_memory.relationship_reinforced",
                edge_id=edge.edge_id,
            )
            return existing
        self._edges.append(edge)
        self._edge_ids.add(edge.edge_id)
        self._edge_index[edge.edge_id] = edge

        logger.debug(
            "semantic_memory.relationship_added",
            source=source_label,
            target=target_label,
            relationship=relationship,
        )
        return edge

    def query(
        self,
        search_text: str,
        category: Optional[str] = None,
        min_confidence: float = 0.2,
        top_k: int = 5,
    ) -> list[KnowledgeNode]:
        """
        Query semantic memory for relevant knowledge.

        Simple keyword matching against labels and content. A production
        implementation would use embedding similarity from a vector store.
        """
        if self._retrieval_mode == "keyword":
            return self._query_keyword(
                search_text=search_text,
                category=category,
                min_confidence=min_confidence,
                top_k=top_k,
            )

        vector_scores: dict[str, float] = {}
        if (
            search_text
            and self._retrieval_mode in {"embedding", "hybrid"}
            and self._vector_search_fn is not None
        ):
            for node_id, score in self._vector_search_fn(
                search_text, max(top_k, self._embedding_top_k)
            ):
                vector_scores[node_id] = score

        if self._retrieval_mode == "embedding":
            if search_text and not vector_scores:
                # Fallback keeps semantic recall usable when vector backend is unavailable.
                return self._query_keyword(
                    search_text=search_text,
                    category=category,
                    min_confidence=min_confidence,
                    top_k=top_k,
                )
            candidates = []
            for node_id, similarity in vector_scores.items():
                node = self._nodes.get(node_id)
                if node is None:
                    continue
                if node.confidence < min_confidence:
                    continue
                if category and node.category != category:
                    continue
                candidates.append((node, similarity * node.confidence))
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [node for node, _ in candidates[:top_k]]

        # hybrid mode
        keyword_nodes = self._query_keyword(
            search_text=search_text,
            category=category,
            min_confidence=min_confidence,
            top_k=max(top_k, self._embedding_top_k),
            include_scores=True,
        )

        combined: dict[str, tuple[KnowledgeNode, float]] = {}
        for node, kw_score in keyword_nodes:
            combined[node.node_id] = (node, self._hybrid_keyword_weight * kw_score)

        for node_id, vec_score in vector_scores.items():
            node = self._nodes.get(node_id)
            if node is None:
                continue
            if node.confidence < min_confidence:
                continue
            if category and node.category != category:
                continue
            base = combined.get(node_id, (node, 0.0))[1]
            combined[node_id] = (
                node,
                base + self._hybrid_embedding_weight * (vec_score * node.confidence),
            )

        ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in ranked[:top_k]]

    def _query_keyword(
        self,
        search_text: str,
        category: Optional[str],
        min_confidence: float,
        top_k: int,
        include_scores: bool = False,
    ) -> list[Any]:
        """Keyword-overlap semantic lookup."""
        search_terms = set(search_text.lower().split())
        candidates: list[tuple[KnowledgeNode, float]] = []

        for node in self._nodes.values():
            if node.confidence < min_confidence:
                continue
            if category and node.category != category:
                continue
            node_terms = set(node.label.lower().split()) | set(node.content.lower().split())
            overlap = len(search_terms & node_terms)
            if overlap > 0:
                score = overlap / max(len(search_terms), 1) * node.confidence
                candidates.append((node, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]
        if include_scores:
            return top
        return [node for node, _ in top]

    def get_relationships(
        self,
        label: str,
        direction: str = "outgoing",
    ) -> list[tuple[KnowledgeEdge, KnowledgeNode]]:
        """Get relationships for a node.

        Args:
            label: The node label to look up.
            direction: ``"outgoing"`` (default), ``"incoming"``, or ``"both"``.
        """
        node_id = self._label_index.get(label.lower())
        if not node_id:
            return []

        results = []
        for edge in self._edges:
            if direction in ("outgoing", "both") and edge.source_id == node_id:
                target = self._nodes.get(edge.target_id)
                if target:
                    results.append((edge, target))
            if direction in ("incoming", "both") and edge.target_id == node_id:
                source = self._nodes.get(edge.source_id)
                if source:
                    results.append((edge, source))
        return results

    def get_context_for(self, topic: str, max_items: int = 5) -> str:
        """
        Generate a prompt fragment with relevant knowledge for a topic.
        This is included in the system prompt when the topic is relevant.
        """
        nodes = self.query(topic, top_k=max_items)
        if not nodes:
            return ""

        lines = [f"Relevant knowledge about '{topic}':"]
        for node in nodes:
            confidence_label = (
                "certain" if node.confidence > 0.8
                else "likely" if node.confidence > 0.5
                else "tentative"
            )
            lines.append(f"  [{confidence_label}] {node.content}")

            # Add relationships
            for edge, target in self.get_relationships(node.label):
                lines.append(f"    → {edge.relationship}: {target.content}")

        return "\n".join(lines)

    @staticmethod
    def _tokenize_support_text(text: str) -> set[str]:
        """Tokenize text for lightweight provenance support checks."""
        if not text:
            return set()
        return {
            token
            for token in re.findall(r"[a-z0-9_]+", text.lower())
            if len(token) >= 3
        }

    def verify_provenance(
        self,
        node_id: str,
        episodic_memory,
        min_support_overlap: float = 0.15,
    ) -> dict:
        """
        Verify that a knowledge node's source episodes actually support it.

        Returns a dict with:
          - supported: bool — whether source episodes support the node claim
          - source_count: int — number of source episodes linked
          - found_count: int — number of source episodes still in episodic memory
          - missing: list[str] — episode IDs that couldn't be found
          - best_overlap: float — strongest lexical overlap score observed
          - supporting: list[dict] — supporting episodes with overlap scores
        """
        node = self._nodes.get(node_id)
        if not node:
            return {
                "supported": False,
                "source_count": 0,
                "found_count": 0,
                "missing": [],
                "best_overlap": 0.0,
                "supporting": [],
            }

        source_ids = node.source_episodes
        missing = []
        found = 0
        supporting: list[dict[str, Any]] = []
        best_overlap = 0.0
        try:
            threshold = float(min_support_overlap)
        except (TypeError, ValueError):
            threshold = 0.15
        if not math.isfinite(threshold):
            threshold = 0.15
        threshold = max(0.0, min(1.0, threshold))
        claim_terms = self._tokenize_support_text(f"{node.label} {node.content}")
        has_get_episode = callable(getattr(episodic_memory, "get_episode", None))

        for ep_id in source_ids:
            episode = None
            if has_get_episode:
                episode = episodic_memory.get_episode(ep_id)

            if episode is None:
                missing.append(ep_id)
                continue

            found += 1
            episode_terms = self._tokenize_support_text(getattr(episode, "content", ""))
            if claim_terms:
                overlap = len(claim_terms & episode_terms) / len(claim_terms)
            else:
                # No meaningful claim terms — cannot verify provenance
                overlap = 0.0
            if overlap > best_overlap:
                best_overlap = overlap
            if overlap >= threshold:
                supporting.append({"episode_id": ep_id, "overlap": overlap})

        return {
            "supported": len(supporting) > 0,
            "source_count": len(source_ids),
            "found_count": found,
            "missing": missing,
            "best_overlap": best_overlap,
            "supporting": supporting,
        }

    def sample_nodes_for_audit(self, count: int = 5) -> list[KnowledgeNode]:
        """Return a random sample of nodes that have source_episodes for auditing.

        Skips immutable nodes (e.g. genesis facts) since they should not be
        subject to provenance decay.
        """
        import random

        candidates = [
            node for node in self._nodes.values()
            if node.source_episodes
            and not (isinstance(node.metadata, dict) and node.metadata.get("immutable"))
        ]
        if not candidates:
            return []
        return random.sample(candidates, min(count, len(candidates)))

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
