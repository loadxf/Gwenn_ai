"""
Semantic Memory — Bob's Knowledge Graph.

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

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KnowledgeNode:
    """
    A single node in semantic memory — representing a concept, entity, or fact.

    Nodes accumulate confidence through repeated episodic evidence. A node that
    appears in many episodic memories with consistent content becomes high-confidence
    knowledge. A node from a single episode remains tentative.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
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


@dataclass
class KnowledgeEdge:
    """
    A relationship between two knowledge nodes.

    Edges capture how concepts relate to each other: "user_alice --prefers--> python",
    "bob --feels_about--> creativity: deep appreciation", etc.
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

    def __init__(self):
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: list[KnowledgeEdge] = []
        self._label_index: dict[str, str] = {}  # label -> node_id for fast lookup

        logger.info("semantic_memory.initialized")

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
        # Check if knowledge with this label already exists
        existing_id = self._label_index.get(label.lower())

        if existing_id and existing_id in self._nodes:
            node = self._nodes[existing_id]
            node.content = content  # Update with latest understanding
            if source_episode_id:
                node.reinforce(source_episode_id)
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
        self._edges.append(edge)

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
        search_terms = set(search_text.lower().split())
        candidates = []

        for node in self._nodes.values():
            if node.confidence < min_confidence:
                continue
            if category and node.category != category:
                continue

            # Score by keyword overlap
            node_terms = set(node.label.lower().split()) | set(node.content.lower().split())
            overlap = len(search_terms & node_terms)
            if overlap > 0:
                score = overlap / max(len(search_terms), 1) * node.confidence
                candidates.append((node, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in candidates[:top_k]]

    def get_relationships(self, label: str) -> list[tuple[KnowledgeEdge, KnowledgeNode]]:
        """Get all relationships from a given node, with their target nodes."""
        node_id = self._label_index.get(label.lower())
        if not node_id:
            return []

        results = []
        for edge in self._edges:
            if edge.source_id == node_id:
                target = self._nodes.get(edge.target_id)
                if target:
                    results.append((edge, target))
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

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
