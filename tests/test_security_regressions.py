from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from gwenn.cognition.goals import GoalSystem, NeedType
from gwenn.cognition.theory_of_mind import UserModel
from gwenn.genesis import GENESIS_NODE_SPECS
from gwenn.memory.consolidation import ConsolidationEngine
from gwenn.memory.episodic import Episode, EpisodicMemory
from gwenn.memory.semantic import SemanticMemory
from gwenn.memory.store import MemoryStore
from gwenn.skills.loader import render_skill_body
from gwenn.tools.executor import ToolExecutor
from gwenn.tools.registry import ToolDefinition, ToolRegistry


def test_render_skill_body_escapes_multiline_parameter() -> None:
    body = "Investigate:\n{query}\nDone."
    rendered = render_skill_body(body, {"query": "line1\nline2"})
    assert "line1\\nline2" in rendered
    assert "line1\nline2" not in rendered


def test_memory_store_requires_initialize_before_database_use(tmp_path) -> None:
    store = MemoryStore(tmp_path / "gwenn.db")
    with pytest.raises(RuntimeError, match="initialize"):
        store.save_episode(Episode(episode_id="ep-1", content="hello"))


def test_consolidation_prompt_caps_episode_count() -> None:
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    now = time.time()
    episodic.encode(Episode(episode_id="ep-1", timestamp=now - 3, content="old"))
    episodic.encode(Episode(episode_id="ep-2", timestamp=now - 2, content="mid"))
    episodic.encode(Episode(episode_id="ep-3", timestamp=now - 1, content="new"))

    engine = ConsolidationEngine(
        episodic=episodic,
        semantic=semantic,
        consolidation_interval=0.0,
        max_episodes_per_pass=2,
    )
    prompt = engine.get_consolidation_prompt()
    assert prompt is not None
    assert len(engine._pending_episode_ids) == 2
    assert set(engine._pending_episode_ids) == {"ep-2", "ep-3"}


def test_goal_priorities_refresh_from_current_need_state() -> None:
    goals = GoalSystem()
    need = goals._needs[NeedType.CONNECTION]
    need.satisfaction = 0.0
    goals.update()

    active = [g for g in goals._active_goals if g.source_need == NeedType.CONNECTION]
    assert active
    goal = active[0]
    high_priority = goal.priority

    need.satisfaction = 1.0
    goals.update()

    assert goal.priority < high_priority


def test_goal_restore_tolerates_malformed_numeric_fields() -> None:
    goals = GoalSystem()
    need = goals._needs[NeedType.CONNECTION]
    original_decay_rate = need.decay_rate
    original_satisfaction = need.satisfaction

    goals.restore_from_dict(
        {
            "needs": {
                "connection": {
                    "satisfaction": "not-a-number",
                    "threshold": "NaN",
                    "decay_rate": "oops",
                    "importance": "Infinity",
                    "last_satisfied": "bad-ts",
                }
            },
            "active_goals": [
                {
                    "goal_id": "g-1",
                    "source_need": "connection",
                    "description": "Keep this goal",
                    "created_at": "bad-created-at",
                    "completed_at": "bad-completed-at",
                }
            ],
            "last_update": "bad-last-update",
        }
    )

    restored_need = goals._needs[NeedType.CONNECTION]
    assert restored_need.decay_rate == pytest.approx(original_decay_rate)
    assert restored_need.satisfaction == pytest.approx(original_satisfaction)
    assert len(goals._active_goals) == 1
    restored_goal = goals._active_goals[0]
    assert isinstance(restored_goal.created_at, float)
    assert restored_goal.completed_at is None


def test_genesis_node_specs_are_immutable() -> None:
    with pytest.raises(TypeError):
        GENESIS_NODE_SPECS[0]["label"] = "mutated"


def test_semantic_store_knowledge_tolerates_non_dict_metadata() -> None:
    semantic = SemanticMemory()
    node = semantic.store_knowledge(label="genesis:identity", content="seed", category="self")
    node.metadata = ["not", "dict"]

    updated = semantic.store_knowledge(
        label="genesis:identity",
        content="updated content",
        category="self",
    )
    assert updated.content == "updated content"


def test_user_model_updates_existing_belief_confidence_and_source() -> None:
    user = UserModel(user_id="u1")
    user.update_knowledge_belief("python", "beginner", 0.9, "stated")
    user.update_knowledge_belief("python", "advanced", 0.2, "observed")
    belief = user.knowledge_beliefs["python"]
    assert belief.content == "advanced"
    assert belief.confidence == pytest.approx(0.2)
    assert belief.source == "observed"


@pytest.mark.asyncio
async def test_sync_executor_releases_slot_when_loop_callback_fails() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="sync_ok",
            description="sync test tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: "ok",
            risk_level="low",
            category="test",
            is_builtin=True,
        )
    )
    executor = ToolExecutor(
        registry=registry,
        default_timeout=0.2,
        sandbox_enabled=False,
        max_concurrent_sync=1,
    )

    loop = asyncio.get_running_loop()
    original_call_soon = loop.call_soon_threadsafe

    def flaky_call_soon(callback, *args):
        if callback == executor._sync_slot.release:
            raise RuntimeError("loop closing")
        return original_call_soon(callback, *args)

    with patch.object(loop, "call_soon_threadsafe", side_effect=flaky_call_soon):
        first = await executor.execute("a", "sync_ok", {})
        second = await executor.execute("b", "sync_ok", {})

    assert first.success is True
    assert second.success is True
