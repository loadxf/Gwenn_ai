"""
Tests for gwenn.memory.working — Working Memory.

Covers:
- Slot management (capacity limit)
- Eviction by lowest salience
- Decay rate
- Item refresh
- generate_id uniqueness
"""

from __future__ import annotations

import time

import pytest

from gwenn.memory.working import WorkingMemory, WorkingMemoryItem


# ---------------------------------------------------------------------------
# Helper to build items quickly
# ---------------------------------------------------------------------------

def _item(item_id: str, salience: float = 0.5, content: str = "") -> WorkingMemoryItem:
    return WorkingMemoryItem(
        item_id=item_id,
        content=content or f"content-{item_id}",
        salience=salience,
    )


# ---------------------------------------------------------------------------
# Slot management (capacity limit)
# ---------------------------------------------------------------------------

class TestSlotManagement:
    """Working memory should respect max_slots capacity."""

    def test_default_capacity_is_seven(self):
        wm = WorkingMemory()
        assert wm.capacity == 7

    def test_custom_capacity(self):
        wm = WorkingMemory(max_slots=3)
        assert wm.capacity == 3

    def test_items_fit_within_capacity(self):
        wm = WorkingMemory(max_slots=3)
        for i in range(3):
            wm.attend(_item(f"item-{i}", salience=0.5))
        assert wm.slot_count == 3

    def test_utilization_reflects_usage(self):
        wm = WorkingMemory(max_slots=4)
        wm.attend(_item("a", salience=0.5))
        wm.attend(_item("b", salience=0.5))
        assert wm.utilization == pytest.approx(0.5)

    def test_cannot_exceed_capacity_with_low_salience(self):
        """A new item that is less salient than all current items is rejected."""
        wm = WorkingMemory(max_slots=2)
        wm.attend(_item("high1", salience=0.8))
        wm.attend(_item("high2", salience=0.7))

        # Try adding a low-salience item — should be rejected
        evicted = wm.attend(_item("low", salience=0.3))
        assert evicted is None  # nothing evicted; item was rejected
        assert wm.slot_count == 2
        assert wm.get("low") is None

    def test_clear_empties_memory(self):
        wm = WorkingMemory(max_slots=3)
        for i in range(3):
            wm.attend(_item(f"item-{i}"))
        evicted = wm.clear()
        assert len(evicted) == 3
        assert wm.slot_count == 0


# ---------------------------------------------------------------------------
# Eviction by lowest salience
# ---------------------------------------------------------------------------

class TestEviction:
    """When full, the least salient item should be displaced by a more salient newcomer."""

    def test_least_salient_is_evicted(self):
        wm = WorkingMemory(max_slots=2)
        wm.attend(_item("a", salience=0.6))
        wm.attend(_item("b", salience=0.4))  # b is least salient

        evicted = wm.attend(_item("c", salience=0.5))
        assert evicted is not None
        assert evicted.item_id == "b"
        assert wm.slot_count == 2
        assert wm.get("c") is not None

    def test_eviction_callback_is_called(self):
        captured = []
        wm = WorkingMemory(max_slots=1)
        wm.set_eviction_callback(lambda item: captured.append(item.item_id))

        wm.attend(_item("first", salience=0.3))
        wm.attend(_item("second", salience=0.6))

        assert captured == ["first"]

    def test_eviction_preserves_higher_salience_items(self):
        wm = WorkingMemory(max_slots=3)
        wm.attend(_item("a", salience=0.9))
        wm.attend(_item("b", salience=0.7))
        wm.attend(_item("c", salience=0.2))

        # New item should evict "c" (salience 0.2)
        evicted = wm.attend(_item("d", salience=0.5))
        assert evicted.item_id == "c"
        # "a" and "b" should still be there
        assert wm.get("a") is not None
        assert wm.get("b") is not None


# ---------------------------------------------------------------------------
# Decay rate
# ---------------------------------------------------------------------------

class TestDecayRate:
    """Salience decays over time; items at zero are removed."""

    def test_decay_reduces_salience(self):
        wm = WorkingMemory(max_slots=3)
        item = _item("decaying", salience=0.5)
        # Backdate the last_refreshed to simulate time passing
        item.last_refreshed = time.time() - 120  # 2 minutes ago
        wm.attend(item)

        # Retrieve the stored item (attend stores a reference)
        stored = wm._items["decaying"]
        # Manually set last_refreshed back again because attend may have refreshed
        stored.last_refreshed = time.time() - 120

        wm.decay_all(rate=0.02)
        assert stored.salience < 0.5

    def test_decay_to_zero_removes_item(self):
        wm = WorkingMemory(max_slots=3)
        item = _item("ephemeral", salience=0.01)
        wm.attend(item)
        # Backdate so decay pushes below threshold
        stored = wm._items["ephemeral"]
        stored.last_refreshed = time.time() - 600  # 10 minutes ago
        stored.salience = 0.01

        removed = wm.decay_all(rate=0.02)
        assert len(removed) == 1
        assert removed[0].item_id == "ephemeral"
        assert wm.slot_count == 0

    def test_decay_callback_on_removal(self):
        captured = []
        wm = WorkingMemory(max_slots=3)
        wm.set_eviction_callback(lambda item: captured.append(item.item_id))

        item = _item("will-decay", salience=0.01)
        wm.attend(item)
        stored = wm._items["will-decay"]
        stored.last_refreshed = time.time() - 600
        stored.salience = 0.01

        wm.decay_all(rate=0.02)
        assert "will-decay" in captured

    def test_decay_does_not_go_negative(self):
        item = WorkingMemoryItem(item_id="test", content="test", salience=0.0)
        item.last_refreshed = time.time() - 600
        item.decay(rate=0.5)
        assert item.salience >= 0.0


# ---------------------------------------------------------------------------
# Item refresh
# ---------------------------------------------------------------------------

class TestItemRefresh:
    """Refreshing an item boosts its salience and updates access_count."""

    def test_refresh_boosts_salience(self):
        item = WorkingMemoryItem(item_id="r1", content="refreshable", salience=0.5)
        original_salience = item.salience
        item.refresh(salience_boost=0.1)
        assert item.salience == pytest.approx(original_salience + 0.1)

    def test_refresh_caps_at_one(self):
        item = WorkingMemoryItem(item_id="r2", content="almost full", salience=0.95)
        item.refresh(salience_boost=0.2)
        assert item.salience == 1.0

    def test_refresh_increments_access_count(self):
        item = WorkingMemoryItem(item_id="r3", content="counted", salience=0.5)
        assert item.access_count == 0
        item.refresh()
        assert item.access_count == 1
        item.refresh()
        assert item.access_count == 2

    def test_attend_existing_item_refreshes(self):
        """Calling attend with an existing item_id refreshes rather than adding."""
        wm = WorkingMemory(max_slots=3)
        wm.attend(_item("x", salience=0.5, content="original"))
        original_count = wm._items["x"].access_count

        # Attend with the same id and new content
        wm.attend(WorkingMemoryItem(item_id="x", content="updated", salience=0.5))
        assert wm.slot_count == 1  # no additional slot used
        assert wm._items["x"].access_count == original_count + 1
        assert wm._items["x"].content == "updated"

    def test_get_refreshes_item(self):
        wm = WorkingMemory(max_slots=3)
        wm.attend(_item("g1", salience=0.5))
        initial_salience = wm._items["g1"].salience
        item = wm.get("g1")
        assert item is not None
        assert item.salience > initial_salience  # get applies a small boost
        assert item.access_count >= 1


# ---------------------------------------------------------------------------
# generate_id uniqueness
# ---------------------------------------------------------------------------

class TestGenerateId:
    """generate_id should produce unique IDs with the given prefix."""

    def test_ids_have_prefix(self):
        wm = WorkingMemory()
        id1 = wm.generate_id(prefix="test")
        assert id1.startswith("test_")

    def test_ids_are_unique(self):
        wm = WorkingMemory()
        ids = {wm.generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_default_prefix_is_wm(self):
        wm = WorkingMemory()
        id1 = wm.generate_id()
        assert id1.startswith("wm_")

    def test_sequential_counter_increments(self):
        wm = WorkingMemory()
        id1 = wm.generate_id()
        id2 = wm.generate_id()
        # The counter portion (second segment) should differ
        counter1 = int(id1.split("_")[1])
        counter2 = int(id2.split("_")[1])
        assert counter2 == counter1 + 1


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------

class TestContextGeneration:
    """get_context produces a readable prompt fragment."""

    def test_empty_memory_context(self):
        wm = WorkingMemory()
        ctx = wm.get_context()
        assert "clear" in ctx.lower() or "no items" in ctx.lower()

    def test_context_contains_item_content(self):
        wm = WorkingMemory()
        wm.attend(WorkingMemoryItem(item_id="ctx1", content="important task", salience=0.8))
        ctx = wm.get_context()
        assert "important task" in ctx

    def test_context_orders_by_salience_descending(self):
        wm = WorkingMemory(max_slots=3)
        wm.attend(_item("low", salience=0.2, content="low item"))
        wm.attend(_item("high", salience=0.9, content="high item"))
        wm.attend(_item("mid", salience=0.5, content="mid item"))

        ctx = wm.get_context()
        high_pos = ctx.index("high item")
        mid_pos = ctx.index("mid item")
        low_pos = ctx.index("low item")
        assert high_pos < mid_pos < low_pos


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip (P0-1 validation)
# ---------------------------------------------------------------------------

class TestToDictRoundTrip:
    """to_dict must serialize metadata and last_refreshed."""

    def test_to_dict_includes_metadata(self):
        wm = WorkingMemory(max_slots=3)
        item = WorkingMemoryItem(
            item_id="meta-item",
            content="has metadata",
            salience=0.7,
            metadata={"source": "test", "priority": 1},
        )
        wm.attend(item)
        data = wm.to_dict()
        serialized = data["items"][0]
        assert serialized["metadata"] == {"source": "test", "priority": 1}

    def test_to_dict_includes_last_refreshed(self):
        wm = WorkingMemory(max_slots=3)
        item = WorkingMemoryItem(item_id="ref-item", content="refreshed", salience=0.5)
        expected_time = item.last_refreshed
        wm.attend(item)
        data = wm.to_dict()
        assert data["items"][0]["last_refreshed"] == pytest.approx(expected_time, abs=1.0)

    def test_to_dict_round_trip_preserves_all_fields(self):
        wm = WorkingMemory(max_slots=3)
        item = WorkingMemoryItem(
            item_id="rt-item",
            content="round trip",
            category="task_state",
            salience=0.8,
            emotional_valence=0.3,
            access_count=5,
            metadata={"key": "value"},
        )
        wm.attend(item)
        data = wm.to_dict()
        serialized = data["items"][0]
        assert serialized["item_id"] == "rt-item"
        assert serialized["content"] == "round trip"
        assert serialized["category"] == "task_state"
        assert serialized["salience"] == pytest.approx(0.8)
        assert serialized["emotional_valence"] == pytest.approx(0.3)
        assert serialized["access_count"] == 5
        assert serialized["metadata"] == {"key": "value"}
        assert "last_refreshed" in serialized
        assert "entered_at" in serialized


# ---------------------------------------------------------------------------
# clear() fires eviction callback (P0-2 validation)
# ---------------------------------------------------------------------------

class TestClearEvictionCallback:
    """clear() must fire the eviction callback for each item."""

    def test_clear_fires_eviction_callback(self):
        captured = []
        wm = WorkingMemory(max_slots=3)
        wm.set_eviction_callback(lambda item: captured.append(item.item_id))

        wm.attend(_item("a", salience=0.5))
        wm.attend(_item("b", salience=0.6))
        wm.attend(_item("c", salience=0.7))

        evicted = wm.clear()
        assert len(evicted) == 3
        assert set(captured) == {"a", "b", "c"}

    def test_clear_without_callback_still_returns_items(self):
        wm = WorkingMemory(max_slots=2)
        wm.attend(_item("x", salience=0.5))
        evicted = wm.clear()
        assert len(evicted) == 1
        assert evicted[0].item_id == "x"
