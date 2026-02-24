"""
Base channel abstract class.

Every platform adapter (Telegram, Discord, …) inherits from BaseChannel and
implements start(), stop(), send_message(), and channel_name.

The shared handle_message() method is the bridge between a platform callback
and SentientAgent.respond().  It:
  1. Constructs a canonical user_id prefixed with the channel name.
  2. Fetches (or creates) the user's conversation history from the SessionManager.
  3. Calls agent.respond() with that history, mutating it in place.
  4. Returns the AgentResponse for the subclass to deliver.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

from gwenn.config import _normalize_session_scope_mode
from gwenn.types import AgentResponse, UserMessage

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent
    from gwenn.channels.session import SessionManager

logger = structlog.get_logger(__name__)

# Default minimum interval (seconds) between messages from the same user.
_DEFAULT_RATE_LIMIT_INTERVAL: float = 1.0


class BaseChannel(ABC):
    """Abstract base for all Gwenn platform channel adapters."""

    def __init__(
        self,
        agent: "SentientAgent",
        sessions: "SessionManager",
        *,
        user_lock_cache_size: int = 512,
        rate_limit_interval: float = _DEFAULT_RATE_LIMIT_INTERVAL,
    ) -> None:
        self._agent = agent
        self._sessions = sessions
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._user_lock_last_used: dict[str, float] = {}
        self._user_lock_cache_size: int = max(1, int(user_lock_cache_size))
        # Per-user rate limiting — minimum seconds between accepted messages.
        self._rate_limit_interval: float = max(0.0, float(rate_limit_interval))
        self._user_last_message: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Start the channel (connect to platform, begin polling, …)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel gracefully."""

    @abstractmethod
    async def send_message(self, platform_user_id: str, text: str) -> None:
        """
        Send an unsolicited message to a platform user (e.g. heartbeat
        notifications, proactive messages).
        """

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Short platform identifier: 'telegram', 'discord', …"""

    async def send_proactive(self, text: str) -> None:
        """
        Broadcast a proactive message to the channel's owner/primary users.

        Called by ``SentientAgent.broadcast_to_channels()`` when the heartbeat
        or other autonomous system wants to share a thought.  Subclasses
        override this to determine *who* receives the message.

        Default implementation is a no-op (safe for channels that have no
        concept of an owner).
        """

    # ------------------------------------------------------------------
    # Shared helpers — subclasses use these
    # ------------------------------------------------------------------

    def make_user_id(self, raw_id: str) -> str:
        """
        Build a namespaced user identifier that is unique across platforms.

        e.g. "telegram_123456789" or "discord_987654321"
        """
        return f"{self.channel_name}_{raw_id}"

    @staticmethod
    def _normalize_scope_mode(scope_mode: str | None, default: str = "per_user") -> str:
        return _normalize_session_scope_mode(scope_mode, default)

    def make_session_scope_key(
        self,
        *,
        raw_user_id: str,
        raw_chat_id: str | None = None,
        raw_thread_id: str | None = None,
        scope_mode: str = "per_user",
    ) -> str:
        """
        Build a raw session scope key before channel namespacing.

        Modes:
        - per_user: one history per user
        - per_chat: one history per chat/channel
        - per_thread: one history per thread (fallback chat → user)
        """
        mode = self._normalize_scope_mode(scope_mode, default="per_user")
        if mode == "per_chat":
            return f"chat:{raw_chat_id}" if raw_chat_id else f"user:{raw_user_id}"
        if mode == "per_thread":
            if raw_thread_id:
                return f"thread:{raw_thread_id}"
            if raw_chat_id:
                return f"chat:{raw_chat_id}"
            return f"user:{raw_user_id}"
        return f"user:{raw_user_id}"

    def make_session_id(self, session_scope_key: str) -> str:
        """Build a namespaced session id for SessionManager storage."""
        return f"{self.channel_name}_{session_scope_key}"

    @staticmethod
    def _normalize_optional_id(value: object) -> str | None:
        """Coerce a platform id (int or str) to a stripped string, or None."""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return None

    @staticmethod
    def _id_set(values: object) -> set[str]:
        """Coerce a list/tuple/set of platform IDs into a set of stripped strings."""
        if not isinstance(values, (list, tuple, set)):
            return set()
        return {str(item).strip() for item in values if str(item).strip()}

    @staticmethod
    def _validate_platform_id(platform_user_id: str) -> int | None:
        """Validate and convert a platform user ID to int.

        Returns the integer ID, or None if the value is not a valid numeric ID.
        """
        try:
            return int(platform_user_id)
        except (ValueError, TypeError):
            logger.warning(
                "channel.invalid_platform_user_id",
                platform_user_id=platform_user_id,
            )
            return None

    def _check_rate_limit(self, raw_user_id: str) -> bool:
        """Return True if the user is within rate limits, False if throttled.

        Uses a simple minimum-interval approach: each user must wait at least
        ``_rate_limit_interval`` seconds between accepted messages.
        """
        if self._rate_limit_interval <= 0:
            return True
        now = time.monotonic()
        last = self._user_last_message.get(raw_user_id)
        if last is not None and (now - last) < self._rate_limit_interval:
            return False
        self._user_last_message[raw_user_id] = now
        self._evict_stale_rate_entries(now)
        return True

    def _evict_stale_rate_entries(self, now: float) -> None:
        """Remove rate-limit entries older than the interval to prevent unbounded growth."""
        if len(self._user_last_message) <= self._user_lock_cache_size:
            return
        cutoff = now - self._rate_limit_interval * 10
        stale = [uid for uid, ts in self._user_last_message.items() if ts < cutoff]
        for uid in stale:
            del self._user_last_message[uid]

    def _get_user_lock(self, raw_user_id: str) -> asyncio.Lock:
        """Return (or create) the per-user asyncio lock and refresh its LRU timestamp."""
        lock = self._user_locks.get(raw_user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._user_locks[raw_user_id] = lock
        self._user_lock_last_used[raw_user_id] = time.monotonic()
        self._evict_user_locks()
        return lock

    def _evict_user_locks(self) -> None:
        """Evict the least-recently-used unlocked entries when the cache is full."""
        if len(self._user_locks) <= self._user_lock_cache_size:
            return
        excess = len(self._user_locks) - self._user_lock_cache_size
        for raw_user_id, _ in sorted(self._user_lock_last_used.items(), key=lambda item: item[1]):
            lock = self._user_locks.get(raw_user_id)
            if lock is None:
                self._user_lock_last_used.pop(raw_user_id, None)
                continue
            if lock.locked():
                continue
            self._user_locks.pop(raw_user_id, None)
            self._user_lock_last_used.pop(raw_user_id, None)
            excess -= 1
            if excess <= 0:
                break

    def _release_user_lock(self, raw_user_id: str) -> None:
        """Refresh LRU timestamp and run eviction after releasing a user lock."""
        self._user_lock_last_used[raw_user_id] = time.monotonic()
        self._evict_user_locks()

    def _get_session_respond_lock(self, session_id: str) -> asyncio.Lock:
        """
        Return a per-session lock for serialising respond() calls.

        Different sessions (e.g. different Telegram group topics) can run
        concurrently.  Within a single session, calls are serialised to
        prevent history corruption.

        Locks are stored on the agent so they are shared across all channel
        adapters that reference the same agent instance.
        """
        locks: dict[str, asyncio.Lock] | None = getattr(
            self._agent, "_session_respond_locks", None
        )
        if locks is None:
            locks = {}
            self._agent._session_respond_locks = locks  # type: ignore[attr-defined]
        if session_id not in locks:
            locks[session_id] = asyncio.Lock()
        return locks[session_id]

    async def handle_message(
        self,
        raw_user_id: str,
        message: UserMessage | str,
        *,
        session_scope_key: str | None = None,
    ) -> AgentResponse:
        """
        Bridge between a platform callback and SentientAgent.

        Fetches the per-user conversation history and passes it to respond(),
        which appends the new turn in-place.  Returns an ``AgentResponse``.

        *message* may be a plain string (backward-compatible) or a
        ``UserMessage`` carrying optional image data.
        """
        if isinstance(message, str):
            message = UserMessage(text=message)
        user_id = self.make_user_id(raw_user_id)
        if session_scope_key:
            session_id = self.make_session_id(session_scope_key)
        else:
            session_id = self.make_session_id(f"user:{raw_user_id}")
        history = self._sessions.get_or_create(session_id)
        async with self._get_session_respond_lock(session_id):
            return await self._agent.respond(
                user_message=message,
                user_id=user_id,
                conversation_history=history,
                session_id=session_id,
            )
