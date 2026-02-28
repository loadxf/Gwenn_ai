"""Tests for gwenn/channels/slack_channel.py."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gwenn.channels.slack_channel import (
    SlackChannel,
    _chunk_text,
    _strip_bot_mention,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> SimpleNamespace:
    """Build a minimal SlackConfig-like namespace."""
    defaults = dict(
        bot_token="xoxb-fake",
        app_token="xapp-fake",
        allowed_user_ids=[],
        owner_user_ids=["U_OWNER"],
        max_history_length=50,
        session_ttl_seconds=3600.0,
        session_scope_mode="per_thread",
        user_lock_cache_size=512,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.respond = AsyncMock(return_value="agent says hi")
    agent.register_channel = MagicMock()
    agent.unregister_channel = MagicMock()
    return agent


def _make_sessions() -> MagicMock:
    sessions = MagicMock()
    sessions.get_or_create = MagicMock(return_value=[])
    return sessions


def _make_event(
    *,
    user: str = "U123",
    text: str = "hello",
    channel: str = "C456",
    thread_ts: str | None = None,
    bot_id: str | None = None,
    subtype: str | None = None,
    ts: str = "1234567890.123456",
) -> dict[str, Any]:
    """Build a Slack message event dict."""
    event: dict[str, Any] = {
        "user": user,
        "text": text,
        "channel": channel,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    if bot_id is not None:
        event["bot_id"] = bot_id
    if subtype is not None:
        event["subtype"] = subtype
    return event


# ---------------------------------------------------------------------------
# Unit: helper functions
# ---------------------------------------------------------------------------


class TestStripBotMention:
    def test_strips_bot_mention(self) -> None:
        assert _strip_bot_mention("<@U_BOT> hello", "U_BOT") == "hello"

    def test_strips_only_first_mention(self) -> None:
        result = _strip_bot_mention("<@U_BOT> hey <@U_BOT>", "U_BOT")
        assert result == "hey <@U_BOT>"

    def test_no_mention_unchanged(self) -> None:
        assert _strip_bot_mention("hello world", "U_BOT") == "hello world"

    def test_no_bot_user_id(self) -> None:
        assert _strip_bot_mention("<@U_BOT> hello", None) == "<@U_BOT> hello"

    def test_different_bot_id(self) -> None:
        assert _strip_bot_mention("<@U_OTHER> hello", "U_BOT") == "<@U_OTHER> hello"


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        assert _chunk_text("hello", 10) == ["hello"]

    def test_exact_limit(self) -> None:
        assert _chunk_text("12345", 5) == ["12345"]

    def test_splits_on_newline(self) -> None:
        text = "line1\nline2\nline3"
        chunks = _chunk_text(text, 12)
        assert chunks == ["line1\nline2", "line3"]

    def test_splits_at_limit_when_no_newline(self) -> None:
        text = "abcdefghij"
        chunks = _chunk_text(text, 5)
        assert chunks == ["abcde", "fghij"]

    def test_empty_text(self) -> None:
        assert _chunk_text("", 10) == [""]


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_initial_state(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        assert ch.channel_name == "slack"
        assert ch._app is None
        assert ch._handler is None
        assert ch._handler_task is None
        assert ch._bot_user_id is None

    def test_allowed_users_populated(self) -> None:
        ch = SlackChannel(
            _make_agent(), _make_sessions(),
            _make_config(allowed_user_ids=["U1", "U2"]),
        )
        assert ch._allowed_users == {"U1", "U2"}

    def test_owner_users_populated(self) -> None:
        ch = SlackChannel(
            _make_agent(), _make_sessions(),
            _make_config(owner_user_ids=["U_OWNER", "U_ADMIN"]),
        )
        assert ch._owner_users == {"U_OWNER", "U_ADMIN"}


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_app_and_handler(self) -> None:
        mock_app = MagicMock()
        mock_app.event = MagicMock(return_value=lambda fn: fn)
        mock_app.client.auth_test = AsyncMock(return_value={"user_id": "U_BOT"})
        mock_handler = MagicMock()
        mock_handler.start_async = AsyncMock()
        mock_handler.close_async = AsyncMock()

        with (
            patch("slack_bolt.async_app.AsyncApp", return_value=mock_app) as app_cls,
            patch("slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler", return_value=mock_handler),
        ):
            ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
            await ch.start()
            app_cls.assert_called_once_with(token="xoxb-fake")
            assert ch._bot_user_id == "U_BOT"
            assert ch._handler_task is not None
            await ch.stop()
            assert ch._app is None
            assert ch._handler is None
            assert ch._handler_task is None

    @pytest.mark.asyncio
    async def test_start_auth_test_failure_non_fatal(self) -> None:
        mock_app = MagicMock()
        mock_app.event = MagicMock(return_value=lambda fn: fn)
        mock_app.client.auth_test = AsyncMock(side_effect=RuntimeError("auth fail"))
        mock_handler = MagicMock()
        mock_handler.start_async = AsyncMock()
        mock_handler.close_async = AsyncMock()

        with (
            patch("slack_bolt.async_app.AsyncApp", return_value=mock_app),
            patch("slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler", return_value=mock_handler),
        ):
            ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
            await ch.start()
            assert ch._bot_user_id is None
            await ch.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        await ch.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_handler_close_error_swallowed(self) -> None:
        mock_app = MagicMock()
        mock_app.event = MagicMock(return_value=lambda fn: fn)
        mock_app.client.auth_test = AsyncMock(return_value={})
        mock_handler = MagicMock()
        mock_handler.start_async = AsyncMock()
        mock_handler.close_async = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch("slack_bolt.async_app.AsyncApp", return_value=mock_app),
            patch("slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler", return_value=mock_handler),
        ):
            ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
            await ch.start()
            await ch.stop()  # Should not raise
            assert ch._handler is None


# ---------------------------------------------------------------------------
# Outbound messaging
# ---------------------------------------------------------------------------


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        mock_client = AsyncMock()
        ch._app = MagicMock()
        ch._app.client.chat_postMessage = mock_client
        await ch.send_message("U123", "hello")
        mock_client.assert_awaited_once_with(channel="U123", text="hello")

    @pytest.mark.asyncio
    async def test_send_message_chunked(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        mock_client = AsyncMock()
        ch._app = MagicMock()
        ch._app.client.chat_postMessage = mock_client
        long_text = "a" * 5000
        await ch.send_message("U123", long_text)
        assert mock_client.await_count == 2

    @pytest.mark.asyncio
    async def test_send_message_error_stops_chunks(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        mock_client = AsyncMock(side_effect=RuntimeError("fail"))
        ch._app = MagicMock()
        ch._app.client.chat_postMessage = mock_client
        await ch.send_message("U123", "hello")  # Should not raise

    @pytest.mark.asyncio
    async def test_send_message_when_not_started(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        await ch.send_message("U123", "hello")  # No-op, no crash


class TestSendProactive:
    @pytest.mark.asyncio
    async def test_sends_to_all_owners(self) -> None:
        config = _make_config(owner_user_ids=["U_A", "U_B"])
        ch = SlackChannel(_make_agent(), _make_sessions(), config)
        mock_client = AsyncMock()
        ch._app = MagicMock()
        ch._app.client.chat_postMessage = mock_client
        await ch.send_proactive("proactive thought")
        assert mock_client.await_count == 2

    @pytest.mark.asyncio
    async def test_no_owners_no_send(self) -> None:
        config = _make_config(owner_user_ids=[])
        ch = SlackChannel(_make_agent(), _make_sessions(), config)
        ch._app = MagicMock()
        await ch.send_proactive("thought")
        # No calls made


class TestSendToSession:
    @pytest.mark.asyncio
    async def test_send_to_thread_session(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        mock_client = AsyncMock()
        ch._app = MagicMock()
        ch._app.client.chat_postMessage = mock_client
        result = await ch.send_to_session("slack_thread:C123:1234.5678", "reply")
        assert result is True
        mock_client.assert_awaited_once_with(
            channel="C123", text="reply", thread_ts="1234.5678",
        )

    @pytest.mark.asyncio
    async def test_send_to_chat_session(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        mock_client = AsyncMock()
        ch._app = MagicMock()
        ch._app.client.chat_postMessage = mock_client
        result = await ch.send_to_session("slack_chat:C123", "reply")
        assert result is True
        mock_client.assert_awaited_once_with(
            channel="C123", text="reply", thread_ts=None,
        )

    @pytest.mark.asyncio
    async def test_non_slack_session_returns_false(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        result = await ch.send_to_session("telegram_user:123", "reply")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_error_returns_false(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        mock_client = AsyncMock(side_effect=RuntimeError("fail"))
        ch._app = MagicMock()
        ch._app.client.chat_postMessage = mock_client
        result = await ch.send_to_session("slack_thread:C123:ts", "msg")
        assert result is False

    @pytest.mark.asyncio
    async def test_user_scope_sends_dm(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        mock_client = AsyncMock()
        ch._app = MagicMock()
        ch._app.client.chat_postMessage = mock_client
        result = await ch.send_to_session("slack_user:U123", "msg")
        assert result is True
        mock_client.assert_awaited_once_with(
            channel="U123", text="msg", thread_ts=None,
        )

    @pytest.mark.asyncio
    async def test_unparsable_scope_returns_false(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        ch._app = MagicMock()
        result = await ch.send_to_session("slack_unknown:xyz", "msg")
        assert result is False

    @pytest.mark.asyncio
    async def test_not_started_returns_false(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        result = await ch.send_to_session("slack_thread:C123:ts", "msg")
        assert result is False


# ---------------------------------------------------------------------------
# Inbound message processing
# ---------------------------------------------------------------------------


class TestProcessMessage:
    @pytest.mark.asyncio
    async def test_basic_message(self) -> None:
        agent = _make_agent()
        sessions = _make_sessions()
        ch = SlackChannel(agent, sessions, _make_config())
        say = AsyncMock()
        event = _make_event()
        await ch._process_message(event, say)
        agent.respond.assert_awaited_once()
        say.assert_awaited_once()
        # Verify reply is threaded
        assert say.call_args.kwargs["thread_ts"] == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_threaded_message_scope(self) -> None:
        agent = _make_agent()
        sessions = _make_sessions()
        ch = SlackChannel(agent, sessions, _make_config())
        say = AsyncMock()
        event = _make_event(thread_ts="parent_ts")
        await ch._process_message(event, say)
        agent.respond.assert_awaited_once()
        # Reply should use the parent thread_ts
        assert say.call_args.kwargs["thread_ts"] == "parent_ts"

    @pytest.mark.asyncio
    async def test_bot_message_ignored(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        say = AsyncMock()
        event = _make_event(bot_id="B_BOT")
        await ch._process_message(event, say)
        agent.respond.assert_not_awaited()
        say.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_subtype_message_ignored(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        say = AsyncMock()
        event = _make_event(subtype="channel_join")
        await ch._process_message(event, say)
        agent.respond.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_thread_broadcast_allowed(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        say = AsyncMock()
        event = _make_event(subtype="thread_broadcast")
        await ch._process_message(event, say)
        agent.respond.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        say = AsyncMock()
        event = _make_event(text="")
        await ch._process_message(event, say)
        agent.respond.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_user_ignored(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        say = AsyncMock()
        event = _make_event(user="")
        await ch._process_message(event, say)
        agent.respond.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_not_allowed(self) -> None:
        agent = _make_agent()
        config = _make_config(allowed_user_ids=["U999"])
        ch = SlackChannel(agent, _make_sessions(), config)
        say = AsyncMock()
        event = _make_event(user="U123")
        await ch._process_message(event, say)
        agent.respond.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_allowed_when_list_empty(self) -> None:
        agent = _make_agent()
        config = _make_config(allowed_user_ids=[])
        ch = SlackChannel(agent, _make_sessions(), config)
        say = AsyncMock()
        event = _make_event()
        await ch._process_message(event, say)
        agent.respond.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_strips_bot_mention(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        ch._bot_user_id = "U_BOT"
        say = AsyncMock()
        event = _make_event(text="<@U_BOT> hello")
        await ch._process_message(event, say)
        # Verify the text passed to respond was stripped
        call_args = agent.respond.call_args
        user_message = call_args.kwargs.get("user_message") or call_args.args[0]
        assert "U_BOT" not in str(user_message)

    @pytest.mark.asyncio
    async def test_mention_only_text_ignored(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        ch._bot_user_id = "U_BOT"
        say = AsyncMock()
        event = _make_event(text="<@U_BOT>")
        await ch._process_message(event, say)
        agent.respond.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reply_error_swallowed(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        say = AsyncMock(side_effect=RuntimeError("send fail"))
        event = _make_event()
        await ch._process_message(event, say)
        # No crash

    @pytest.mark.asyncio
    async def test_rate_limited(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(
            agent, _make_sessions(),
            _make_config(),
        )
        ch._rate_limit_interval = 10.0
        say = AsyncMock()
        event = _make_event()
        # First message goes through
        await ch._process_message(event, say)
        assert agent.respond.await_count == 1
        # Second message within rate limit is dropped
        await ch._process_message(event, say)
        assert agent.respond.await_count == 1

    @pytest.mark.asyncio
    async def test_self_message_ignored(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        ch._bot_user_id = "U_BOT"
        say = AsyncMock()
        event = _make_event(user="U_BOT")
        await ch._process_message(event, say)
        agent.respond.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bot_profile_ignored(self) -> None:
        agent = _make_agent()
        ch = SlackChannel(agent, _make_sessions(), _make_config())
        say = AsyncMock()
        event = _make_event()
        event["bot_profile"] = {"name": "some-bot"}
        await ch._process_message(event, say)
        agent.respond.assert_not_awaited()


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


class TestIsAllowed:
    def test_empty_allowlist_allows_all(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config(allowed_user_ids=[]))
        assert ch._is_allowed("U_ANY") is True

    def test_allowlist_allows_listed_user(self) -> None:
        ch = SlackChannel(
            _make_agent(), _make_sessions(),
            _make_config(allowed_user_ids=["U1", "U2"]),
        )
        assert ch._is_allowed("U1") is True

    def test_allowlist_rejects_unlisted_user(self) -> None:
        ch = SlackChannel(
            _make_agent(), _make_sessions(),
            _make_config(allowed_user_ids=["U1"]),
        )
        assert ch._is_allowed("U_OTHER") is False


# ---------------------------------------------------------------------------
# Register handlers
# ---------------------------------------------------------------------------


class TestRegisterHandlers:
    def test_registers_message_handler(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        mock_app = MagicMock()
        registered_events: list[str] = []
        mock_app.event = MagicMock(
            side_effect=lambda event_type: (
                registered_events.append(event_type) or (lambda fn: fn)
            )
        )
        ch._app = mock_app
        ch._register_handlers()
        assert "message" in registered_events


# ---------------------------------------------------------------------------
# Channel name
# ---------------------------------------------------------------------------


class TestChannelName:
    def test_channel_name_is_slack(self) -> None:
        ch = SlackChannel(_make_agent(), _make_sessions(), _make_config())
        assert ch.channel_name == "slack"


# ---------------------------------------------------------------------------
# Session scope key
# ---------------------------------------------------------------------------


class TestSessionScope:
    def test_per_thread_with_thread_ts(self) -> None:
        ch = SlackChannel(
            _make_agent(), _make_sessions(),
            _make_config(session_scope_mode="per_thread"),
        )
        key = ch.make_session_scope_key(
            raw_user_id="U1",
            raw_chat_id="C1",
            raw_thread_id="C1:ts123",
            scope_mode="per_thread",
        )
        assert key == "thread:C1:ts123"

    def test_per_user_mode(self) -> None:
        ch = SlackChannel(
            _make_agent(), _make_sessions(),
            _make_config(session_scope_mode="per_user"),
        )
        key = ch.make_session_scope_key(
            raw_user_id="U1",
            raw_chat_id="C1",
            scope_mode="per_user",
        )
        assert key == "user:U1"

    def test_per_chat_mode(self) -> None:
        ch = SlackChannel(
            _make_agent(), _make_sessions(),
            _make_config(session_scope_mode="per_chat"),
        )
        key = ch.make_session_scope_key(
            raw_user_id="U1",
            raw_chat_id="C1",
            scope_mode="per_chat",
        )
        assert key == "chat:C1"
