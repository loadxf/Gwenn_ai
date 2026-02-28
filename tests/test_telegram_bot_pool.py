"""Tests for gwenn.channels.telegram_bot_pool — Bot Pool Manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gwenn.channels.telegram_bot_pool import TelegramBotPool, TelegramBotSlot
from gwenn.orchestration.models import SubagentPersona


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persona(name: str = "Researcher", role: str = "Research specialist") -> SubagentPersona:
    return SubagentPersona(name=name, role=role, style="formal")


def _pool(n: int = 3) -> TelegramBotPool:
    """Create a pool with *n* fake tokens."""
    tokens = [f"fake_token_{i}" for i in range(n)]
    return TelegramBotPool(bot_tokens=tokens)


# ---------------------------------------------------------------------------
# TelegramBotSlot
# ---------------------------------------------------------------------------


class TestTelegramBotSlot:
    def test_defaults(self) -> None:
        slot = TelegramBotSlot(bot_token="tok123")
        assert slot.bot_token == "tok123"
        assert slot.bot_id == 0
        assert slot.bot_username == ""
        assert slot.current_persona is None
        assert slot.current_task_id is None
        assert slot.is_active is False

    def test_assignment(self) -> None:
        slot = TelegramBotSlot(bot_token="tok123")
        persona = _persona()
        slot.current_persona = persona
        slot.current_task_id = "task-1"
        slot.is_active = True
        assert slot.current_persona.name == "Researcher"
        assert slot.is_active is True


# ---------------------------------------------------------------------------
# TelegramBotPool — Construction & Properties
# ---------------------------------------------------------------------------


class TestBotPoolProperties:
    def test_pool_size(self) -> None:
        pool = _pool(3)
        assert pool.pool_size == 3

    def test_available_count_all_idle(self) -> None:
        pool = _pool(3)
        assert pool.available_count == 3
        assert pool.active_count == 0

    def test_empty_pool(self) -> None:
        pool = TelegramBotPool(bot_tokens=[])
        assert pool.pool_size == 0
        assert pool.available_count == 0


# ---------------------------------------------------------------------------
# TelegramBotPool — Initialize
# ---------------------------------------------------------------------------


class TestBotPoolInitialize:
    @pytest.mark.asyncio
    async def test_initialize_fetches_bot_info(self) -> None:
        pool = _pool(2)

        mock_bot_instance = AsyncMock()
        mock_me = MagicMock()
        mock_me.id = 12345
        mock_me.username = "gwenn_agent_1"
        mock_bot_instance.get_me = AsyncMock(return_value=mock_me)
        mock_bot_instance.__aenter__ = AsyncMock(return_value=mock_bot_instance)
        mock_bot_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("gwenn.channels.telegram_bot_pool.Bot", return_value=mock_bot_instance):
            await pool.initialize()

        assert pool._slots[0].bot_id == 12345
        assert pool._slots[0].bot_username == "gwenn_agent_1"

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        pool = _pool(1)

        mock_bot = AsyncMock()
        mock_me = MagicMock(id=1, username="bot1")
        mock_bot.get_me = AsyncMock(return_value=mock_me)
        mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
        mock_bot.__aexit__ = AsyncMock(return_value=False)

        with patch("gwenn.channels.telegram_bot_pool.Bot", return_value=mock_bot):
            await pool.initialize()
            await pool.initialize()  # second call should be a no-op

        # get_me called only once (for the single slot)
        assert mock_bot.get_me.call_count == 1

    @pytest.mark.asyncio
    async def test_initialize_handles_failure(self) -> None:
        pool = _pool(1)

        mock_bot = AsyncMock()
        mock_bot.get_me = AsyncMock(side_effect=Exception("network error"))
        mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
        mock_bot.__aexit__ = AsyncMock(return_value=False)

        with patch("gwenn.channels.telegram_bot_pool.Bot", return_value=mock_bot):
            await pool.initialize()  # should not raise

        # Slot remains with default values
        assert pool._slots[0].bot_id == 0
        assert pool._initialized is True


# ---------------------------------------------------------------------------
# TelegramBotPool — Acquire / Release
# ---------------------------------------------------------------------------


class TestBotPoolAcquireRelease:
    @pytest.mark.asyncio
    async def test_acquire_assigns_slot(self) -> None:
        pool = _pool(2)
        persona = _persona()

        # Patch _apply_persona to avoid Telegram API calls
        with patch.object(pool, "_apply_persona", new_callable=AsyncMock):
            slot = await pool.acquire(persona, "task-1")

        assert slot is not None
        assert slot.is_active is True
        assert slot.current_persona == persona
        assert slot.current_task_id == "task-1"
        assert pool.available_count == 1
        assert pool.active_count == 1

    @pytest.mark.asyncio
    async def test_acquire_returns_none_when_exhausted(self) -> None:
        pool = _pool(1)
        persona = _persona()

        with patch.object(pool, "_apply_persona", new_callable=AsyncMock):
            slot1 = await pool.acquire(persona, "task-1")
            slot2 = await pool.acquire(persona, "task-2")

        assert slot1 is not None
        assert slot2 is None
        assert pool.available_count == 0

    @pytest.mark.asyncio
    async def test_release_returns_slot_to_pool(self) -> None:
        pool = _pool(1)
        persona = _persona()

        with patch.object(pool, "_apply_persona", new_callable=AsyncMock):
            slot = await pool.acquire(persona, "task-1")
            assert pool.available_count == 0

            await pool.release(slot)
            assert pool.available_count == 1
            assert slot.is_active is False
            assert slot.current_persona is None
            assert slot.current_task_id is None

    @pytest.mark.asyncio
    async def test_acquire_after_release(self) -> None:
        pool = _pool(1)
        p1 = _persona("Alpha", "role A")
        p2 = _persona("Beta", "role B")

        with patch.object(pool, "_apply_persona", new_callable=AsyncMock):
            slot = await pool.acquire(p1, "task-1")
            await pool.release(slot)
            slot2 = await pool.acquire(p2, "task-2")

        assert slot2 is not None
        assert slot2.current_persona.name == "Beta"
        assert slot2.current_task_id == "task-2"

    @pytest.mark.asyncio
    async def test_release_all(self) -> None:
        pool = _pool(3)
        persona = _persona()

        with patch.object(pool, "_apply_persona", new_callable=AsyncMock):
            for i in range(3):
                await pool.acquire(persona, f"task-{i}")
            assert pool.active_count == 3

            await pool.release_all()
            assert pool.active_count == 0
            assert pool.available_count == 3


# ---------------------------------------------------------------------------
# TelegramBotPool — get_slot_for_task
# ---------------------------------------------------------------------------


class TestGetSlotForTask:
    @pytest.mark.asyncio
    async def test_find_assigned_slot(self) -> None:
        pool = _pool(2)
        persona = _persona()

        with patch.object(pool, "_apply_persona", new_callable=AsyncMock):
            await pool.acquire(persona, "task-a")
            await pool.acquire(persona, "task-b")

        slot = pool.get_slot_for_task("task-b")
        assert slot is not None
        assert slot.current_task_id == "task-b"

    def test_returns_none_for_unknown_task(self) -> None:
        pool = _pool(2)
        assert pool.get_slot_for_task("unknown") is None


# ---------------------------------------------------------------------------
# TelegramBotPool — send_as
# ---------------------------------------------------------------------------


class TestBotPoolSendAs:
    @pytest.mark.asyncio
    async def test_send_as_calls_bot(self) -> None:
        pool = _pool(1)
        slot = pool._slots[0]
        slot.bot_token = "test_token"

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
        mock_bot.__aexit__ = AsyncMock(return_value=False)

        with patch("gwenn.channels.telegram_bot_pool.Bot", return_value=mock_bot):
            await pool.send_as(slot, chat_id=123, thread_id=456, text="Hello!")

        mock_bot.send_message.assert_called_once_with(
            chat_id=123, text="Hello!", message_thread_id=456
        )

    @pytest.mark.asyncio
    async def test_send_as_without_thread(self) -> None:
        pool = _pool(1)
        slot = pool._slots[0]

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
        mock_bot.__aexit__ = AsyncMock(return_value=False)

        with patch("gwenn.channels.telegram_bot_pool.Bot", return_value=mock_bot):
            await pool.send_as(slot, chat_id=123, thread_id=None, text="Hi!")

        mock_bot.send_message.assert_called_once_with(chat_id=123, text="Hi!")

    @pytest.mark.asyncio
    async def test_send_as_handles_error(self) -> None:
        pool = _pool(1)
        slot = pool._slots[0]

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(side_effect=Exception("send failed"))
        mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
        mock_bot.__aexit__ = AsyncMock(return_value=False)

        with patch("gwenn.channels.telegram_bot_pool.Bot", return_value=mock_bot):
            await pool.send_as(slot, chat_id=123, thread_id=None, text="Hi!")
            # Should not raise


# ---------------------------------------------------------------------------
# TelegramBotPool — _apply_persona (caching)
# ---------------------------------------------------------------------------


class TestApplyPersona:
    @pytest.mark.asyncio
    async def test_persona_caching_skips_api_when_unchanged(self) -> None:
        pool = _pool(1)
        slot = pool._slots[0]
        persona = _persona("Agent1", "role1")

        mock_bot = AsyncMock()
        mock_bot.set_my_name = AsyncMock()
        mock_bot.set_my_description = AsyncMock()
        mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
        mock_bot.__aexit__ = AsyncMock(return_value=False)

        with patch("gwenn.channels.telegram_bot_pool.Bot", return_value=mock_bot):
            # First call — should set name and description.
            await pool._apply_persona(slot, persona)
            assert mock_bot.set_my_name.call_count == 1
            assert mock_bot.set_my_description.call_count == 1

            # Second call with same persona — should skip API calls.
            mock_bot.set_my_name.reset_mock()
            mock_bot.set_my_description.reset_mock()
            await pool._apply_persona(slot, persona)
            assert mock_bot.set_my_name.call_count == 0
            assert mock_bot.set_my_description.call_count == 0

    @pytest.mark.asyncio
    async def test_persona_change_triggers_api(self) -> None:
        pool = _pool(1)
        slot = pool._slots[0]

        mock_bot = AsyncMock()
        mock_bot.set_my_name = AsyncMock()
        mock_bot.set_my_description = AsyncMock()
        mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
        mock_bot.__aexit__ = AsyncMock(return_value=False)

        with patch("gwenn.channels.telegram_bot_pool.Bot", return_value=mock_bot):
            await pool._apply_persona(slot, _persona("A", "roleA"))
            mock_bot.set_my_name.reset_mock()
            mock_bot.set_my_description.reset_mock()

            await pool._apply_persona(slot, _persona("B", "roleB"))
            assert mock_bot.set_my_name.call_count == 1
            assert mock_bot.set_my_description.call_count == 1

    @pytest.mark.asyncio
    async def test_apply_persona_handles_import_error(self) -> None:
        pool = _pool(1)
        slot = pool._slots[0]
        persona = _persona()

        with patch("gwenn.channels.telegram_bot_pool.Bot", side_effect=ImportError("no telegram")):
            # Should not raise — gracefully handles ImportError.
            await pool._apply_persona(slot, persona)


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_acquire_no_double_assign(self) -> None:
        """Multiple concurrent acquires should not assign the same slot twice."""
        pool = _pool(2)

        with patch.object(pool, "_apply_persona", new_callable=AsyncMock):
            results = await asyncio.gather(
                pool.acquire(_persona("A"), "t1"),
                pool.acquire(_persona("B"), "t2"),
                pool.acquire(_persona("C"), "t3"),  # should get None
            )

        assigned = [r for r in results if r is not None]
        assert len(assigned) == 2
        # Ensure no duplicates
        task_ids = {s.current_task_id for s in assigned}
        assert len(task_ids) == 2
