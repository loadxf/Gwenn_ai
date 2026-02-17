"""
Base channel abstract class.

Every platform adapter (Telegram, Discord, …) inherits from BaseChannel and
implements start(), stop(), send_message(), and channel_name.

The shared handle_message() method is the bridge between a platform callback
and SentientAgent.respond().  It:
  1. Constructs a canonical user_id prefixed with the channel name.
  2. Fetches (or creates) the user's conversation history from the SessionManager.
  3. Calls agent.respond() with that history, mutating it in place.
  4. Returns the response string for the subclass to deliver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent
    from gwenn.channels.session import SessionManager


class BaseChannel(ABC):
    """Abstract base for all Gwenn platform channel adapters."""

    def __init__(self, agent: "SentientAgent", sessions: "SessionManager") -> None:
        self._agent = agent
        self._sessions = sessions

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

    # ------------------------------------------------------------------
    # Shared helpers — subclasses use these
    # ------------------------------------------------------------------

    def make_user_id(self, raw_id: str) -> str:
        """
        Build a namespaced user identifier that is unique across platforms.

        e.g. "telegram_123456789" or "discord_987654321"
        """
        return f"{self.channel_name}_{raw_id}"

    async def handle_message(self, raw_user_id: str, text: str) -> str:
        """
        Bridge between a platform callback and SentientAgent.

        Fetches the per-user conversation history and passes it to respond(),
        which appends the new turn in-place.  Returns Gwenn's response string.
        """
        user_id = self.make_user_id(raw_user_id)
        history = self._sessions.get_or_create(user_id)
        return await self._agent.respond(
            user_message=text,
            user_id=user_id,
            conversation_history=history,
        )
