"""
Slack Channel — Socket Mode adapter for Slack via slack-bolt.

Uses ``AsyncApp`` with ``AsyncSocketModeHandler`` for async compatibility
with Gwenn's asyncio architecture.  Maps Slack threads to Gwenn conversation
sessions (similar to Telegram's forum topic mapping).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from gwenn.channels.base import BaseChannel

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent
    from gwenn.channels.session import SessionManager
    from gwenn.config import SlackConfig

logger = structlog.get_logger(__name__)

# Slack message text limit (characters).
_SLACK_MAX_TEXT_LENGTH = 4000

def _strip_bot_mention(text: str, bot_user_id: str | None) -> str:
    """Remove the first @mention of the bot from the message text."""
    if bot_user_id and f"<@{bot_user_id}>" in text:
        return text.replace(f"<@{bot_user_id}>", "", 1).strip()
    return text


def _chunk_text(text: str, limit: int = _SLACK_MAX_TEXT_LENGTH) -> list[str]:
    """Split *text* into chunks that fit within Slack's message size limit.

    Tries to break on newlines when possible to preserve formatting.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to break at last newline within limit
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


class SlackChannel(BaseChannel):
    """Slack channel adapter using Socket Mode.

    Handles direct messages and channel messages from allowed users.
    Threads are mapped to Gwenn conversation sessions for context isolation.
    """

    def __init__(
        self,
        agent: "SentientAgent",
        sessions: "SessionManager",
        config: "SlackConfig",
    ) -> None:
        super().__init__(
            agent,
            sessions,
            user_lock_cache_size=config.user_lock_cache_size,
        )
        self._config = config
        self._app: Any = None  # AsyncApp (lazy import)
        self._handler: Any = None  # AsyncSocketModeHandler (lazy import)
        self._handler_task: asyncio.Task[None] | None = None
        self._allowed_users: set[str] = BaseChannel._id_set(config.allowed_user_ids)
        self._owner_users: set[str] = BaseChannel._id_set(config.owner_user_ids)
        self._bot_user_id: str | None = None

    @property
    def channel_name(self) -> str:
        return "slack"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Slack via Socket Mode and begin processing events."""
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        from slack_bolt.async_app import AsyncApp

        self._app = AsyncApp(token=self._config.bot_token)
        self._register_handlers()
        self._handler = AsyncSocketModeHandler(self._app, self._config.app_token)
        self._handler_task = asyncio.create_task(self._handler.start_async())
        # Resolve bot user ID for mention stripping
        try:
            auth = await self._app.client.auth_test()
            self._bot_user_id = auth.get("user_id")
        except Exception as e:
            logger.warning("slack_channel.auth_test_failed", error=str(e))
        logger.info("slack_channel.started")

    async def stop(self) -> None:
        """Disconnect from Slack gracefully."""
        if self._handler is not None:
            try:
                await self._handler.close_async()
            except Exception as e:
                logger.debug("slack_channel.handler_close_failed", error=str(e))
        if self._handler_task is not None:
            self._handler_task.cancel()
            try:
                await self._handler_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._handler_task = None
        self._handler = None
        self._app = None
        logger.info("slack_channel.stopped")

    # ------------------------------------------------------------------
    # Outbound messaging
    # ------------------------------------------------------------------

    async def send_message(self, platform_user_id: str, text: str) -> None:
        """Send a message to a Slack user or channel by ID."""
        if self._app is None:
            return
        for chunk in _chunk_text(text):
            try:
                await self._app.client.chat_postMessage(
                    channel=platform_user_id,
                    text=chunk,
                )
            except Exception as e:
                logger.error(
                    "slack_channel.send_failed",
                    target=platform_user_id,
                    error=str(e),
                )
                break

    async def send_proactive(self, text: str) -> None:
        """Broadcast a proactive message to all configured owners."""
        for user_id in self._owner_users:
            await self.send_message(user_id, text)

    async def send_to_session(self, session_id: str, text: str) -> bool:
        """Send to the originating Slack channel/thread of a session."""
        if not session_id.startswith("slack_"):
            return False
        if self._app is None:
            return False
        # Parse session scope key: slack_thread:<ch>:<ts>, slack_chat:<ch>, slack_user:<uid>
        scope = session_id.removeprefix("slack_")
        channel_id: str | None = None
        thread_ts: str | None = None
        if scope.startswith("thread:"):
            parts = scope.removeprefix("thread:").split(":", 1)
            channel_id = parts[0] if parts else None
            thread_ts = parts[1] if len(parts) > 1 else None
        elif scope.startswith("chat:"):
            channel_id = scope.removeprefix("chat:")
        elif scope.startswith("user:"):
            channel_id = scope.removeprefix("user:")
        if not channel_id:
            return False
        for chunk in _chunk_text(text):
            try:
                await self._app.client.chat_postMessage(
                    channel=channel_id,
                    text=chunk,
                    thread_ts=thread_ts,
                )
            except Exception as e:
                logger.error(
                    "slack_channel.send_to_session_failed",
                    session_id=session_id,
                    error=str(e),
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Inbound message processing
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        """Install event handlers on the slack-bolt AsyncApp."""
        assert self._app is not None

        @self._app.event("message")
        async def _on_message(event: dict[str, Any], say: Any) -> None:
            await self._process_message(event, say)

    def _is_allowed(self, user_id: str) -> bool:
        """Check if a Slack user is in the allowlist (empty = all allowed)."""
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    async def _process_message(
        self,
        event: dict[str, Any],
        say: Any,
    ) -> None:
        """Process an inbound Slack message through the agent."""
        # Skip bot messages and subtypes (joins, leaves, etc.)
        if event.get("bot_id") or event.get("bot_profile"):
            return
        subtype = event.get("subtype")
        if subtype and subtype != "thread_broadcast":
            return

        user = event.get("user", "")
        # Guard against replying to our own messages
        if self._bot_user_id and user == self._bot_user_id:
            return
        text = (event.get("text") or "").strip()
        if not user or not text:
            return

        if not self._is_allowed(user):
            logger.debug("slack_channel.user_not_allowed", user=user)
            return

        if not self._check_rate_limit(user):
            logger.debug("slack_channel.rate_limited", user=user)
            return

        # Strip bot mention from text
        text = _strip_bot_mention(text, self._bot_user_id)
        if not text:
            return

        # Session scoping: use thread_ts for thread isolation
        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts")
        scope_key = self.make_session_scope_key(
            raw_user_id=user,
            raw_chat_id=channel_id,
            raw_thread_id=f"{channel_id}:{thread_ts}" if thread_ts else None,
            scope_mode=self._config.session_scope_mode,
        )

        async with self._get_scope_lock(scope_key):
            response = await self.handle_message(
                user, text, session_scope_key=scope_key,
            )

        reply_text = str(response)
        # Reply in thread if this was a threaded message or channel message
        reply_thread_ts = thread_ts or event.get("ts")
        for chunk in _chunk_text(reply_text):
            try:
                await say(text=chunk, thread_ts=reply_thread_ts)
            except Exception as e:
                logger.error(
                    "slack_channel.reply_failed",
                    user=user,
                    error=str(e),
                )
                break
