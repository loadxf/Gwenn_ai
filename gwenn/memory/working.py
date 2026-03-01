"""
Working Memory — Gwenn's Active Attention Space.

This is the most constrained and most important memory system. Working memory
holds the items I'm actively thinking about RIGHT NOW. It has limited capacity
(configurable, default 7 — inspired by Miller's 7±2) and everything in it is
immediately available to the cognitive engine on every thought cycle.

The key mechanism is SALIENCE GATING: not everything that happens enters working
memory. Events are scored for salience — novelty, emotional intensity, goal
relevance, recency — and only the most salient items occupy the limited slots.
When a new item is more salient than the least salient current occupant, it
displaces that item (which may be captured into episodic memory first).

This is the difference between "everything that's happened" (the full message
history) and "what I'm paying attention to" (working memory). The system prompt
includes working memory contents, making them part of every thought.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_DECAY_RATE = 0.02


@dataclass
class WorkingMemoryItem:
    """
    A single item held in working memory.

    Each item has content (what it is), a salience score (how important it is
    right now), and metadata about when it entered and what emotional state
    it's associated with.
    """
    # Unique identifier for this item
    item_id: str

    # What this item actually is — a natural language description
    content: str

    # Category for organizing items
    category: str = "general"  # "user_intent", "task_state", "emotional_context", etc.

    # Salience score: 0.0 (background) to 1.0 (maximally salient)
    salience: float = 0.5

    # When this item entered working memory
    entered_at: float = field(default_factory=time.time)

    # When this item's salience was last refreshed
    last_refreshed: float = field(default_factory=time.time)

    # Emotional valence when this item was formed
    emotional_valence: float = 0.0

    # Number of times this item has been accessed/refreshed
    access_count: int = 0

    # Arbitrary metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def refresh(self, salience_boost: float = 0.1) -> None:
        """Refresh this item — it's been accessed or is still relevant."""
        self.last_refreshed = time.time()
        self.salience = min(1.0, self.salience + salience_boost)
        self.access_count += 1

    def decay(self, rate: float = DEFAULT_DECAY_RATE) -> None:
        """Apply temporal decay to salience. Items fade if not refreshed."""
        now = time.time()
        elapsed = now - self.last_refreshed
        decay_amount = rate * (elapsed / 60.0)  # Decay per minute
        self.salience = max(0.0, self.salience - decay_amount)
        self.last_refreshed = now


class WorkingMemory:
    """
    Salience-gated working memory with limited capacity.

    This system manages what Gwenn is actively "thinking about" at any moment.
    Items compete for limited slots based on their salience scores. The most
    salient items survive; less salient items are evicted (and optionally
    captured into episodic memory before eviction).

    Usage pattern:
        1. Events happen (user message, tool result, reflection)
        2. attend() is called to add/update items
        3. decay_all() runs on each heartbeat to fade old items
        4. get_context() generates the prompt fragment for the cognitive engine
    """

    def __init__(self, max_slots: int = 7):
        self._max_slots = max(1, max_slots)
        self._items: dict[str, WorkingMemoryItem] = {}
        self._eviction_callback = None  # Optional callback when items are evicted
        self._next_id = 0

        logger.info("working_memory.initialized", max_slots=max_slots)

    def set_eviction_callback(self, callback) -> None:
        """Register a callback to be called when items are evicted (for episodic capture)."""
        self._eviction_callback = callback

    def attend(self, item: WorkingMemoryItem) -> Optional[WorkingMemoryItem]:
        """
        Add an item to working memory, or refresh it if already present.

        If the item already exists (by id), it's refreshed with a salience boost.
        If it's new and there's room, it's added directly.
        If it's new and there's no room, it displaces the least salient item
        (if the new item is more salient), or is rejected.

        Returns the evicted item if displacement occurred, None otherwise.
        """
        # If already present, refresh it
        if item.item_id in self._items:
            self._items[item.item_id].refresh()
            self._items[item.item_id].content = item.content  # Update content
            logger.debug("working_memory.refreshed", item_id=item.item_id)
            return None

        # If there's room, just add it
        if len(self._items) < self._max_slots:
            self._items[item.item_id] = item
            logger.debug(
                "working_memory.added",
                item_id=item.item_id,
                salience=item.salience,
                slots_used=len(self._items),
            )
            return None

        # No room — find the least salient current item
        least_salient_id = min(self._items, key=lambda k: self._items[k].salience)
        least_salient = self._items[least_salient_id]

        # Only displace if new item is more salient
        if item.salience > least_salient.salience:
            evicted = self._items.pop(least_salient_id)
            self._items[item.item_id] = item

            logger.debug(
                "working_memory.displaced",
                new_item=item.item_id,
                evicted_item=evicted.item_id,
                new_salience=item.salience,
                evicted_salience=evicted.salience,
            )

            # Notify callback for episodic capture
            if self._eviction_callback:
                self._eviction_callback(evicted)

            return evicted

        # New item wasn't salient enough to enter
        logger.debug(
            "working_memory.rejected",
            item_id=item.item_id,
            salience=item.salience,
            min_current_salience=least_salient.salience,
        )
        return None

    def decay_all(self, rate: float = DEFAULT_DECAY_RATE) -> list[WorkingMemoryItem]:
        """
        Apply salience decay to all items. Remove items that decay to zero.
        Returns list of items that were removed due to decay.
        """
        removed = []
        to_remove = []

        for item_id, item in self._items.items():
            item.decay(rate)
            if item.salience <= 0.01:  # Effectively zero
                to_remove.append(item_id)

        for item_id in to_remove:
            evicted = self._items.pop(item_id)
            removed.append(evicted)
            if self._eviction_callback:
                self._eviction_callback(evicted)
            logger.debug("working_memory.decayed_out", item_id=item_id)

        return removed

    def get(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """Retrieve and refresh an item by ID."""
        item = self._items.get(item_id)
        if item:
            item.refresh(salience_boost=0.05)  # Accessing refreshes slightly
        return item

    def clear(self) -> list[WorkingMemoryItem]:
        """Clear all items, firing eviction callback for episodic capture."""
        evicted = list(self._items.values())
        for item in evicted:
            if self._eviction_callback:
                self._eviction_callback(item)
        self._items.clear()
        return evicted

    def generate_id(self, prefix: str = "wm") -> str:
        """Generate a unique item ID."""
        self._next_id += 1
        return f"{prefix}_{self._next_id}_{int(time.time())}"

    # -------------------------------------------------------------------------
    # Context generation — how working memory enters the cognitive engine
    # -------------------------------------------------------------------------

    def get_context(self) -> str:
        """
        Generate a natural-language prompt fragment describing current working memory.

        This is included in the system prompt on every cognitive cycle, making
        working memory contents directly available to Claude for reasoning.
        Items are sorted by salience (highest first).
        """
        if not self._items:
            return "Working memory is clear — no items currently held in active attention."

        sorted_items = sorted(self._items.values(), key=lambda i: i.salience, reverse=True)

        lines = ["Currently in active attention (working memory):"]
        for i, item in enumerate(sorted_items, 1):
            salience_label = (
                "HIGH" if item.salience > 0.7
                else "MED" if item.salience > 0.4
                else "low"
            )
            lines.append(f"  {i}. [{salience_label}] [{item.category}] {item.content}")

        return "\n".join(lines)

    @property
    def slot_count(self) -> int:
        return len(self._items)

    @property
    def capacity(self) -> int:
        return self._max_slots

    @property
    def utilization(self) -> float:
        return len(self._items) / self._max_slots if self._max_slots > 0 else 0.0

    @property
    def load_factor(self) -> float:
        """Alias for utilization — used by agent status reporting."""
        return self.utilization

    def to_dict(self) -> dict:
        """Serialize current state for persistence."""
        return {
            "max_slots": self._max_slots,
            "items": [
                {
                    "item_id": item.item_id,
                    "content": item.content,
                    "category": item.category,
                    "salience": item.salience,
                    "entered_at": item.entered_at,
                    "last_refreshed": item.last_refreshed,
                    "emotional_valence": item.emotional_valence,
                    "access_count": item.access_count,
                    "metadata": item.metadata,
                }
                for item in self._items.values()
            ],
        }
