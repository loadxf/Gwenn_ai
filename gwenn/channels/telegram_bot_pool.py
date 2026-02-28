"""Telegram Bot Pool — Visible Subagent Swarm Visualization.

Manages a pool of pre-registered Telegram bots. When Gwenn spawns a swarm,
each subagent is assigned a bot from the pool, giving it a visible persona
in the Telegram group topic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from gwenn.orchestration.models import SubagentPersona

try:
    from telegram import Bot
except ImportError:  # pragma: no cover
    Bot = None  # type: ignore[assignment, misc]

logger = structlog.get_logger(__name__)


@dataclass
class TelegramBotSlot:
    """A pre-registered Telegram bot available for subagent assignment."""

    bot_token: str = field(repr=False)
    bot_id: int = 0
    bot_username: str = ""
    current_persona: SubagentPersona | None = None
    current_task_id: str | None = None
    application: Any = None  # python-telegram-bot Application
    is_active: bool = False
    _cached_name: str = ""
    _cached_description: str = ""


class TelegramBotPool:
    """Manages a pool of Telegram bots for swarm visualization.

    Bots are pre-registered via BotFather and stored as tokens in config.
    When a swarm spawns, bots are acquired and given personas. When the
    swarm completes, bots are released back to the pool.
    """

    def __init__(self, bot_tokens: list[str]) -> None:
        self._slots: list[TelegramBotSlot] = [
            TelegramBotSlot(bot_token=token) for token in bot_tokens
        ]
        self._lock = asyncio.Lock()
        self._initialized = False

    @property
    def pool_size(self) -> int:
        """Total number of bots in the pool."""
        return len(self._slots)

    @property
    def available_count(self) -> int:
        """Number of idle (unassigned) bots."""
        return sum(1 for s in self._slots if not s.is_active)

    @property
    def active_count(self) -> int:
        """Number of currently assigned bots."""
        return sum(1 for s in self._slots if s.is_active)

    async def initialize(self) -> None:
        """Initialize bot identities by fetching bot info from Telegram.

        Call once at startup. Populates bot_id and bot_username for
        each slot by calling getMe().
        """
        async with self._lock:
            if self._initialized:
                return
            self._initialized = True  # Set early to prevent re-entry

        for slot in self._slots:
            try:
                bot = Bot(token=slot.bot_token)
                async with bot:
                    me = await bot.get_me()
                    slot.bot_id = me.id
                    slot.bot_username = me.username or ""
                logger.info(
                    "bot_pool.slot_initialized",
                    bot_id=slot.bot_id,
                    username=slot.bot_username,
                )
            except Exception as e:
                logger.warning(
                    "bot_pool.slot_init_failed",
                    token_prefix=slot.bot_token[:4] + "...",
                    error=str(e),
                )

    async def acquire(
        self,
        persona: SubagentPersona,
        task_id: str,
    ) -> TelegramBotSlot | None:
        """Assign an idle bot to a subagent with the given persona.

        Sets the bot's display name and description via the Telegram API
        (only if changed from the cached values). Returns None if no bots
        are available.
        """
        async with self._lock:
            slot = next(
                (s for s in self._slots if not s.is_active),
                None,
            )
            if slot is None:
                logger.warning("bot_pool.exhausted", task_id=task_id)
                return None

            slot.is_active = True
            slot.current_persona = persona
            slot.current_task_id = task_id

        # Set persona via Telegram API (outside lock to avoid holding it)
        await self._apply_persona(slot, persona)

        logger.info(
            "bot_pool.acquired",
            task_id=task_id,
            persona=persona.name,
            bot_username=slot.bot_username,
        )
        return slot

    async def release(self, slot: TelegramBotSlot) -> None:
        """Return a bot to the pool. Resets to idle state."""
        async with self._lock:
            slot.is_active = False
            slot.current_persona = None
            slot.current_task_id = None

        logger.info(
            "bot_pool.released",
            bot_username=slot.bot_username,
        )

    async def send_as(
        self,
        slot: TelegramBotSlot,
        chat_id: int,
        thread_id: int | None,
        text: str,
    ) -> None:
        """Send a message to a chat/topic as the assigned bot."""
        try:
            bot = Bot(token=slot.bot_token)
            async with bot:
                kwargs: dict[str, Any] = {
                    "chat_id": chat_id,
                    "text": text,
                }
                if thread_id is not None:
                    kwargs["message_thread_id"] = thread_id
                await bot.send_message(**kwargs)
        except Exception as e:
            logger.warning(
                "bot_pool.send_failed",
                bot_username=slot.bot_username,
                error=str(e),
            )

    async def _apply_persona(
        self,
        slot: TelegramBotSlot,
        persona: SubagentPersona,
    ) -> None:
        """Set bot name and description via Telegram API.

        Skips API calls if the persona hasn't changed (caching).
        Stagger calls with small delays to respect rate limits.
        """
        try:
            bot = Bot(token=slot.bot_token)
            async with bot:
                # Set name (only if changed)
                if persona.name != slot._cached_name:
                    try:
                        await bot.set_my_name(persona.name)
                        slot._cached_name = persona.name
                        await asyncio.sleep(1.0)  # Rate limit spacing
                    except Exception as e:
                        logger.debug("bot_pool.set_name_failed", error=str(e))

                # Set description (only if changed)
                desc = persona.role[:512]
                if desc != slot._cached_description:
                    try:
                        await bot.set_my_description(desc)
                        slot._cached_description = desc
                    except Exception as e:
                        logger.debug("bot_pool.set_description_failed", error=str(e))
        except (ImportError, TypeError):
            logger.warning("bot_pool.telegram_not_installed")
        except Exception as e:
            logger.warning("bot_pool.apply_persona_failed", error=str(e))

    def get_slot_for_task(self, task_id: str) -> TelegramBotSlot | None:
        """Find the bot slot assigned to a specific task."""
        return next(
            (s for s in self._slots if s.current_task_id == task_id),
            None,
        )

    async def release_all(self) -> None:
        """Release all active bots back to the pool."""
        async with self._lock:
            active_slots = [s for s in self._slots if s.is_active]
        for slot in active_slots:
            await self.release(slot)
