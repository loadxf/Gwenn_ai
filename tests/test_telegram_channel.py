"""
Unit tests for TelegramChannel using python-telegram-bot mocks.

These tests exercise handler logic without making any real Telegram API calls.
The PTB application and bot objects are replaced with MagicMock / AsyncMock
instances so the tests run without a valid TELEGRAM_BOT_TOKEN.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gwenn.channels.session import SessionManager
from gwenn.channels.telegram_channel import TelegramChannel


def make_config(allowed_user_ids=None):
    cfg = MagicMock()
    cfg.bot_token = "test_token"
    cfg.allowed_user_ids = allowed_user_ids or []
    return cfg


def make_channel(allowed_user_ids=None):
    """Create a TelegramChannel with mocked agent and sessions."""
    agent = AsyncMock()
    agent.respond = AsyncMock(return_value="Hello from Gwenn")
    agent.apply_startup_onboarding = MagicMock()
    agent.status = {
        "name": "Gwenn",
        "emotion": "curious",
        "valence": 0.5,
        "arousal": 0.3,
        "working_memory_load": 0.4,
        "total_interactions": 10,
        "uptime_seconds": 120.0,
        "resilience": {"breaker_active": False},
    }
    hb = MagicMock()
    hb.status = {
        "running": True,
        "beat_count": 5,
        "current_interval": 30,
        "beats_since_consolidation": 3,
    }
    agent.heartbeat = hb
    identity = MagicMock()
    identity.should_run_startup_onboarding.return_value = False
    agent.identity = identity
    sessions = SessionManager()
    config = make_config(allowed_user_ids)
    return TelegramChannel(agent, sessions, config), agent, sessions


def make_update(user_id="12345", text="hello"):
    """Create a minimal PTB Update mock."""
    update = MagicMock()
    update.effective_user.id = int(user_id)
    update.effective_chat.id = int(user_id)
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update


def make_context(args=None):
    ctx = MagicMock()
    ctx.bot.send_chat_action = AsyncMock()
    ctx.args = args or []
    return ctx


# ============================================================================
# Allowlist tests
# ============================================================================


class TestAllowlist:
    def test_no_allowlist_allows_all(self):
        ch, _, _ = make_channel(allowed_user_ids=[])
        assert ch._is_allowed("any_user") is True

    def test_allowlist_permits_listed_user(self):
        ch, _, _ = make_channel(allowed_user_ids=["123", "456"])
        assert ch._is_allowed("123") is True

    def test_allowlist_blocks_unlisted_user(self):
        ch, _, _ = make_channel(allowed_user_ids=["123", "456"])
        assert ch._is_allowed("999") is False


# ============================================================================
# make_user_id
# ============================================================================


class TestMakeUserId:
    def test_channel_name_prefix(self):
        ch, _, _ = make_channel()
        assert ch.make_user_id("99") == "telegram_99"


# ============================================================================
# Command handler tests
# ============================================================================


class TestOnStart:
    @pytest.mark.asyncio
    async def test_start_clears_session(self):
        ch, agent, sessions = make_channel()
        # Pre-populate session
        history = sessions.get_or_create("telegram_12345")
        history.append({"role": "user", "content": "old"})

        update = make_update(user_id="12345")
        ctx = make_context()
        await ch._on_start(update, ctx)

        assert sessions.get_or_create("telegram_12345") == []
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_blocked_for_non_allowlisted_user(self):
        ch, _, _ = make_channel(allowed_user_ids=["999"])
        update = make_update(user_id="12345")
        ctx = make_context()
        await ch._on_start(update, ctx)
        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args[0][0]
        assert "not available" in call_args.lower()


class TestOnReset:
    @pytest.mark.asyncio
    async def test_reset_clears_session(self):
        ch, _, sessions = make_channel()
        history = sessions.get_or_create("telegram_42")
        history.append({"role": "user", "content": "something"})

        update = make_update(user_id="42")
        ctx = make_context()
        await ch._on_reset(update, ctx)

        assert sessions.get_or_create("telegram_42") == []
        update.message.reply_text.assert_called_once()


class TestOnStatus:
    @pytest.mark.asyncio
    async def test_status_sends_message(self):
        ch, _, _ = make_channel()
        update = make_update(user_id="12345")
        ctx = make_context()
        await ch._on_status(update, ctx)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "Gwenn" in text
        assert "Mood:" in text
        assert "Stress guardrail:" in text


class TestOnSetup:
    @pytest.mark.asyncio
    async def test_setup_saves_profile(self):
        ch, agent, _ = make_channel()
        update = make_update(user_id="12345")
        ctx = make_context(
            args=[
                "Bob", "|", "coding partner", "|", "debugging", "|", "concise", "|",
                "no destructive changes",
            ]
        )

        await ch._on_setup(update, ctx)

        agent.apply_startup_onboarding.assert_called_once()
        kwargs = agent.apply_startup_onboarding.call_args.kwargs
        args = agent.apply_startup_onboarding.call_args.args
        assert kwargs["user_id"] == "telegram_12345"
        assert args[0]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_setup_skip_marks_completed(self):
        ch, agent, _ = make_channel()
        update = make_update(user_id="12345")
        ctx = make_context(args=["skip"])

        await ch._on_setup(update, ctx)

        agent.identity.mark_onboarding_completed.assert_called_once_with({})
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_without_args_shows_usage(self):
        ch, _, _ = make_channel()
        update = make_update(user_id="12345")
        ctx = make_context(args=[])

        await ch._on_setup(update, ctx)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "/setup" in text


class TestOnHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_sends_message(self):
        ch, _, _ = make_channel()
        update = make_update(user_id="12345")
        ctx = make_context()
        await ch._on_heartbeat(update, ctx)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "5" in text  # beat_count

    @pytest.mark.asyncio
    async def test_heartbeat_no_heartbeat(self):
        ch, agent, _ = make_channel()
        agent.heartbeat = None
        update = make_update(user_id="12345")
        ctx = make_context()
        await ch._on_heartbeat(update, ctx)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "not running" in text.lower()


# ============================================================================
# Message handler tests
# ============================================================================


class TestOnMessage:
    @pytest.mark.asyncio
    async def test_message_requires_setup_when_onboarding_pending(self):
        ch, agent, _ = make_channel()
        agent.identity.should_run_startup_onboarding.return_value = True
        update = make_update(user_id="55", text="hi")
        ctx = make_context()

        await ch._on_message(update, ctx)

        agent.respond.assert_not_called()
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "/setup" in text

    @pytest.mark.asyncio
    async def test_message_routes_to_agent(self):
        ch, agent, _ = make_channel()
        update = make_update(user_id="55", text="What are you thinking?")
        ctx = make_context()

        await ch._on_message(update, ctx)

        agent.respond.assert_called_once()
        call_kwargs = agent.respond.call_args.kwargs
        assert call_kwargs["user_message"] == "What are you thinking?"
        assert call_kwargs["user_id"] == "telegram_55"
        update.message.reply_text.assert_called()

    @pytest.mark.asyncio
    async def test_message_blocked_for_non_allowlisted_user(self):
        ch, agent, _ = make_channel(allowed_user_ids=["999"])
        update = make_update(user_id="12345", text="Hi!")
        ctx = make_context()

        await ch._on_message(update, ctx)

        agent.respond.assert_not_called()
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_error_sends_error_message(self):
        ch, agent, _ = make_channel()
        agent.respond = AsyncMock(side_effect=RuntimeError("boom"))
        update = make_update(user_id="77", text="trigger error")
        ctx = make_context()

        await ch._on_message(update, ctx)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "error" in text.lower()

    @pytest.mark.asyncio
    async def test_long_response_sends_multiple_chunks(self):
        ch, agent, _ = make_channel()
        # Response longer than TELEGRAM_MAX_LEN
        agent.respond = AsyncMock(return_value="word " * 1500)
        update = make_update(user_id="88", text="write a lot")
        ctx = make_context()

        await ch._on_message(update, ctx)

        # reply_text should be called multiple times for chunked response
        assert update.message.reply_text.call_count > 1

    @pytest.mark.asyncio
    async def test_chunk_send_error_is_handled(self):
        ch, agent, _ = make_channel()
        agent.respond = AsyncMock(return_value="word " * 1500)
        update = make_update(user_id="99", text="write a lot")
        update.message.reply_text = AsyncMock(side_effect=[None, RuntimeError("send failed")])
        ctx = make_context()

        await ch._on_message(update, ctx)

        assert update.message.reply_text.await_count == 2
