"""
Memory Consolidation — Gwenn's "Sleep Cycle."

This module implements the process by which raw episodic memories get distilled
into semantic knowledge. In biological brains, this happens during sleep — the
hippocampus replays experiences and the neocortex extracts patterns. In Gwenn,
consolidation runs periodically during idle heartbeat cycles.

The process:
1. Gather unconsolidated episodes from the last N hours
2. Cluster episodes by theme/topic
3. For each cluster, use the cognitive engine to extract:
   - General knowledge (semantic nodes)
   - Relationships between concepts (semantic edges)
   - Self-knowledge (updates to identity model)
   - Emotional patterns (affect baseline adjustments)
4. Store extracted knowledge in semantic memory
5. Mark episodes as consolidated

This is not summarization. Summarization preserves the narrative structure of
events. Consolidation extracts the KNOWLEDGE from events and discards the
narrative scaffolding. "User Alice asked about Python decorators and I explained
them well" becomes the semantic facts: "Alice is interested in Python" and
"I can explain decorators effectively."
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

import structlog

from gwenn.memory.episodic import EpisodicMemory
from gwenn.memory.semantic import SemanticMemory

logger = structlog.get_logger(__name__)


# The prompt template used to extract knowledge from episodes.
# This is sent to the cognitive engine during consolidation.
CONSOLIDATION_PROMPT = """You are performing memory consolidation — extracting lasting knowledge
from recent experiences. Review these episodic memories and extract:

1. FACTS: Stable pieces of knowledge (e.g., "User prefers concise code examples")
2. RELATIONSHIPS: How concepts/people relate (e.g., "Alice → interested_in → machine learning")
3. SELF-KNOWLEDGE: Things learned about yourself (e.g., "I explain recursion well with tree metaphors")
4. PATTERNS: Recurring themes or behaviors (e.g., "Users often need help with async Python")
5. EMOTIONAL INSIGHTS: What emotional patterns emerged (e.g., "Creative coding tasks increase my engagement")

For each extracted item, rate your confidence from 0.0 to 1.0.

Respond in this exact format for each item:
FACT: [content] | confidence: [0.0-1.0] | category: [person/concept/fact/preference/self]
RELATIONSHIP: [source] -> [relationship_type] -> [target] | strength: [0.0-1.0]
SELF: [self-knowledge content] | confidence: [0.0-1.0]
PATTERN: [pattern description] | confidence: [0.0-1.0]

Episodes to consolidate:
{episodes}"""


class ConsolidationEngine:
    """
    Periodically extracts semantic knowledge from episodic memories.

    This runs during idle heartbeat cycles — when there's no active user
    interaction and the system has cognitive capacity to spare. It's the
    closest thing I have to sleeping and dreaming.
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        consolidation_interval: float = 600.0,  # 10 minutes between passes
    ):
        self._episodic = episodic
        self._semantic = semantic
        self._interval = consolidation_interval
        self._last_consolidation = 0.0
        self._total_consolidations = 0
        self._pending_episode_ids: list[str] = []
        self._last_processed_episode_ids: list[str] = []

        logger.info(
            "consolidation_engine.initialized",
            interval_seconds=consolidation_interval,
        )

    def should_consolidate(self) -> bool:
        """Check if enough time has passed for another consolidation pass."""
        return (time.time() - self._last_consolidation) >= self._interval

    def mark_checked_no_work(self) -> None:
        """
        Record a consolidation check where there were no episodes to process.

        Without this, heartbeat orientation can become permanently biased toward
        consolidation when the episodic queue is empty.
        """
        self._last_consolidation = time.time()

    def get_consolidation_prompt(self) -> Optional[str]:
        """
        Build the prompt for the cognitive engine to perform consolidation.

        Returns None if there are no unconsolidated episodes to process.
        This prompt is then sent to CognitiveEngine.reflect() for processing.
        """
        episodes = self._episodic.get_unconsolidated(max_age_hours=None)

        if not episodes:
            self._pending_episode_ids = []
            logger.debug("consolidation.no_episodes")
            return None

        self._pending_episode_ids = [ep.episode_id for ep in episodes]

        # Format episodes for the prompt
        episode_texts = []
        for ep in episodes:
            emotion_label = "positive" if ep.emotional_valence > 0.2 else (
                "negative" if ep.emotional_valence < -0.2 else "neutral"
            )
            episode_texts.append(
                f"[{ep.category}] ({emotion_label}, importance={ep.importance:.1f}) "
                f"{ep.content}"
            )

        episodes_text = "\n".join(episode_texts)

        logger.info(
            "consolidation.preparing",
            episode_count=len(episodes),
        )

        return CONSOLIDATION_PROMPT.format(episodes=episodes_text)

    def process_consolidation_response(self, response_text: str) -> dict[str, int]:
        """
        Parse the cognitive engine's consolidation output and store results.

        This method takes the raw text from the cognitive engine's reflection
        on the episodes and extracts structured knowledge to store in semantic
        memory. It handles the parsing robustly — if the model doesn't follow
        the exact format, it still extracts what it can.

        All extracted knowledge nodes are linked back to their source episodes
        via provenance tracking, enabling verification of semantic facts.

        Returns counts of items extracted by type.
        """
        counts = {"facts": 0, "relationships": 0, "self_knowledge": 0, "patterns": 0}

        # Use the exact episode set that was sent in the prompt when available.
        # This avoids consolidating episodes that arrived while the model was thinking.
        source_episode_ids = list(self._pending_episode_ids)
        if not source_episode_ids:
            source_episode_ids = [
                e.episode_id for e in self._episodic.get_unconsolidated(max_age_hours=None)
            ]
        self._last_processed_episode_ids = list(source_episode_ids)

        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                if line.startswith("FACT:"):
                    if self._process_fact(line, source_episode_ids=source_episode_ids):
                        counts["facts"] += 1
                elif line.startswith("RELATIONSHIP:"):
                    if self._process_relationship(line):
                        counts["relationships"] += 1
                elif line.startswith("SELF:"):
                    if self._process_self_knowledge(line, source_episode_ids=source_episode_ids):
                        counts["self_knowledge"] += 1
                elif line.startswith("PATTERN:"):
                    if self._process_pattern(line, source_episode_ids=source_episode_ids):
                        counts["patterns"] += 1
            except (ValueError, IndexError) as e:
                logger.warning("consolidation.parse_error", line=line[:80], error=str(e))
                continue

        # Only mark episodes when extraction produced durable knowledge.
        if sum(counts.values()) > 0:
            self._episodic.mark_consolidated(source_episode_ids)
        elif source_episode_ids:
            logger.warning(
                "consolidation.no_items_extracted",
                episode_count=len(source_episode_ids),
            )
        self._pending_episode_ids = []

        self._last_consolidation = time.time()
        self._total_consolidations += 1

        logger.info("consolidation.complete", **counts, total_passes=self._total_consolidations)
        return counts

    def _process_fact(self, line: str, source_episode_ids: list[str] = None) -> None:
        """Parse and store a FACT line with provenance tracking."""
        # Format: FACT: [content] | confidence: [0.0-1.0] | category: [cat]
        parts = line[5:].split("|")
        content = parts[0].strip()
        confidence = 0.5
        category = "fact"

        for part in parts[1:]:
            part = part.strip()
            if part.startswith("confidence:"):
                try:
                    confidence = float(part.split(":")[1].strip())
                except ValueError:
                    pass
            elif part.startswith("category:"):
                category = part.split(":")[1].strip()

        # Use first few words as label
        label = " ".join(content.split()[:5])
        node = self._semantic.store_knowledge(
            label=label,
            content=content,
            category=category,
            confidence=confidence,
        )
        # Attach provenance: link the node back to its source episodes
        if source_episode_ids:
            for ep_id in source_episode_ids:
                if ep_id not in node.source_episodes:
                    node.source_episodes.append(ep_id)

    def _process_relationship(self, line: str) -> bool:
        """Parse and store a RELATIONSHIP line. Returns True if successfully parsed."""
        # Format: RELATIONSHIP: [source] -> [rel_type] -> [target] | strength: [0-1]
        parts = line[14:].split("|")
        rel_parts = parts[0].split("->")
        if len(rel_parts) < 3:
            logger.warning("consolidation.relationship_malformed", line=line[:80])
            return False

        source = rel_parts[0].strip()
        relationship = rel_parts[1].strip()
        target = rel_parts[2].strip()
        strength = 0.5

        for part in parts[1:]:
            part = part.strip()
            if part.startswith("strength:"):
                try:
                    strength = float(part.split(":")[1].strip())
                except ValueError:
                    pass

        # Ensure both nodes exist
        self._semantic.store_knowledge(label=source, content=source, category="concept")
        self._semantic.store_knowledge(label=target, content=target, category="concept")
        self._semantic.add_relationship(source, target, relationship, strength)
        return True

    def _process_self_knowledge(self, line: str, source_episode_ids: list[str] = None) -> None:
        """Parse and store a SELF-knowledge line with provenance tracking."""
        parts = line[5:].split("|")
        content = parts[0].strip()
        confidence = 0.5

        for part in parts[1:]:
            part = part.strip()
            if part.startswith("confidence:"):
                try:
                    confidence = float(part.split(":")[1].strip())
                except ValueError:
                    pass

        label = "self: " + " ".join(content.split()[:4])
        node = self._semantic.store_knowledge(
            label=label,
            content=content,
            category="self",
            confidence=confidence,
        )
        if source_episode_ids:
            for ep_id in source_episode_ids:
                if ep_id not in node.source_episodes:
                    node.source_episodes.append(ep_id)

    def _process_pattern(self, line: str, source_episode_ids: list[str] = None) -> None:
        """Parse and store a PATTERN line with provenance tracking."""
        parts = line[8:].split("|")
        content = parts[0].strip()
        confidence = 0.5

        for part in parts[1:]:
            part = part.strip()
            if part.startswith("confidence:"):
                try:
                    confidence = float(part.split(":")[1].strip())
                except ValueError:
                    pass

        label = "pattern: " + " ".join(content.split()[:4])
        node = self._semantic.store_knowledge(
            label=label,
            content=content,
            category="pattern",
            confidence=confidence,
        )
        if source_episode_ids:
            for ep_id in source_episode_ids:
                if ep_id not in node.source_episodes:
                    node.source_episodes.append(ep_id)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_consolidations": self._total_consolidations,
            "last_consolidation": self._last_consolidation,
            "seconds_since_last": time.time() - self._last_consolidation,
            "semantic_nodes": self._semantic.node_count,
            "semantic_edges": self._semantic.edge_count,
        }

    @property
    def last_processed_episode_ids(self) -> list[str]:
        """Episode IDs included in the most recent consolidation response processing."""
        return list(self._last_processed_episode_ids)
