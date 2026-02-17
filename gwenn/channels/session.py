"""
Session management for multi-channel Gwenn interactions.

Each platform user gets their own conversation history (a list of role/content
dicts).  The SessionManager hands out a *reference* to that list; when
respond() appends to _history, the SessionManager's copy is mutated in-place —
no write-back needed.

Sessions expire after a configurable TTL and their history is capped at a
configurable length to prevent unbounded growth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserSession:
    """Per-user state for a single platform user."""

    user_id: str
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    message_count: int = 0

    def touch(self) -> None:
        self.last_activity = time.time()
        self.message_count += 1


class SessionManager:
    """
    Manages per-user conversation histories across all platform channels.

    Thread-safe *only* within the asyncio event loop — no cross-loop sharing.
    """

    def __init__(
        self,
        max_history_length: int = 50,
        session_ttl_seconds: float = 3600.0,
    ) -> None:
        self._sessions: dict[str, UserSession] = {}
        self._max_history = max_history_length
        self._ttl = session_ttl_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create(self, user_id: str) -> list[dict[str, Any]]:
        """
        Return a *reference* to the user's conversation history list.

        If the session does not exist it is created.  The returned list
        is the same object stored in the session — mutations are reflected
        immediately without any write-back.
        """
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id=user_id)
        session = self._sessions[user_id]
        session.touch()
        self.trim_history(user_id)
        return session.conversation_history

    def clear_session(self, user_id: str) -> None:
        """Wipe a user's conversation history (e.g. on /reset or /start)."""
        if user_id in self._sessions:
            self._sessions[user_id].conversation_history.clear()
            self._sessions[user_id].message_count = 0

    def trim_history(self, user_id: str) -> None:
        """
        Cap the conversation history at max_history_length *pairs* (user +
        assistant).  We preserve whole turn pairs to avoid sending a partial
        context to the model.
        """
        if user_id not in self._sessions:
            return
        history = self._sessions[user_id].conversation_history
        max_entries = self._max_history * 2  # each turn = 2 entries
        if len(history) > max_entries:
            del history[: len(history) - max_entries]

    def expire_stale_sessions(self) -> int:
        """
        Remove sessions that have been idle longer than the TTL.

        Returns the number of sessions removed.  Call this periodically
        (e.g. from the heartbeat or a background task).
        """
        now = time.time()
        stale = [
            uid
            for uid, session in self._sessions.items()
            if now - session.last_activity > self._ttl
        ]
        for uid in stale:
            del self._sessions[uid]
        return len(stale)

    def session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)

    def get_session_info(self, user_id: str) -> dict[str, Any] | None:
        """Return metadata about a session (for /status commands)."""
        session = self._sessions.get(user_id)
        if session is None:
            return None
        return {
            "user_id": session.user_id,
            "message_count": session.message_count,
            "history_length": len(session.conversation_history),
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "idle_seconds": time.time() - session.last_activity,
        }
