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


def make_config(allowed_user_ids=None, owner_user_ids=None):
    cfg = MagicMock()
    cfg.bot_token = "test_token"
    cfg.allowed_user_ids = allowed_user_ids or []
    cfg.owner_user_ids = owner_user_ids or []
    cfg.session_scope_mode = "per_user"
    cfg.user_lock_cache_size = 512
    return cfg


def make_channel(
    allowed_user_ids=None,
    owner_user_ids=None,
    session_scope_mode="per_user",
    user_lock_cache_size=512,
):
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
    config = make_config(allowed_user_ids, owner_user_ids=owner_user_ids)
    config.session_scope_mode = session_scope_mode
    config.user_lock_cache_size = user_lock_cache_size
    return TelegramChannel(agent, sessions, config), agent, sessions


def make_update(user_id="12345", text="hello", chat_id=None, thread_id=None):
    """Create a minimal PTB Update mock."""
    update = MagicMock()
    update.effective_user.id = int(user_id)
    update.effective_chat.id = int(chat_id if chat_id is not None else user_id)
    update.message.text = text
    update.message.message_thread_id = thread_id
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
        # Pre-populate session (per_user scope → "telegram_user:12345")
        history = sessions.get_or_create("telegram_user:12345")
        history.append({"role": "user", "content": "old"})

        update = make_update(user_id="12345")
        ctx = make_context()
        await ch._on_start(update, ctx)

        assert sessions.get_or_create("telegram_user:12345") == []
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
        history = sessions.get_or_create("telegram_user:42")
        history.append({"role": "user", "content": "something"})

        update = make_update(user_id="42")
        ctx = make_context()
        await ch._on_reset(update, ctx)

        assert sessions.get_or_create("telegram_user:42") == []
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_uses_scoped_session_key(self):
        ch, _, sessions = make_channel(session_scope_mode="per_chat")
        history = sessions.get_or_create("telegram_chat:777")
        history.append({"role": "user", "content": "something"})

        update = make_update(user_id="42", chat_id="777")
        ctx = make_context()
        await ch._on_reset(update, ctx)

        assert sessions.get_or_create("telegram_chat:777") == []
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
                "Bob",
                "|",
                "coding partner",
                "|",
                "debugging",
                "|",
                "concise",
                "|",
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
    async def test_message_uses_per_chat_scope(self):
        ch, agent, sessions = make_channel(session_scope_mode="per_chat")
        ctx = make_context()
        update_a = make_update(user_id="55", chat_id="9001", text="Hi from user A")
        update_b = make_update(user_id="66", chat_id="9001", text="Hi from user B")

        await ch._on_message(update_a, ctx)
        await ch._on_message(update_b, ctx)

        assert agent.respond.await_count == 2
        first = agent.respond.call_args_list[0].kwargs
        second = agent.respond.call_args_list[1].kwargs
        assert first["user_id"] == "telegram_55"
        assert second["user_id"] == "telegram_66"
        assert first["conversation_history"] is second["conversation_history"]
        assert first["conversation_history"] is sessions.get_or_create("telegram_chat:9001")

    @pytest.mark.asyncio
    async def test_chunk_send_error_is_handled(self):
        ch, agent, _ = make_channel()
        agent.respond = AsyncMock(return_value="word " * 1500)
        update = make_update(user_id="99", text="write a lot")
        # First chunk: HTML succeeds.
        # Second chunk: HTML fails, plain-text fallback also fails → break.
        update.message.reply_text = AsyncMock(
            side_effect=[None, RuntimeError("html failed"), RuntimeError("plain failed")]
        )
        ctx = make_context()

        await ch._on_message(update, ctx)

        # 3 calls: chunk1 HTML ok, chunk2 HTML fail, chunk2 plain fail
        assert update.message.reply_text.await_count == 3

    @pytest.mark.asyncio
    async def test_chunk_html_fallback_to_plain_text(self):
        ch, agent, _ = make_channel()
        agent.respond = AsyncMock(return_value="word " * 1500)
        update = make_update(user_id="99", text="write a lot")
        # First chunk: HTML succeeds.
        # Second chunk: HTML fails, plain-text fallback succeeds.
        update.message.reply_text = AsyncMock(side_effect=[None, RuntimeError("html failed"), None])
        ctx = make_context()

        await ch._on_message(update, ctx)

        assert update.message.reply_text.await_count >= 3


class TestLockCache:
    def test_lock_cache_eviction_bounds_memory(self):
        ch, _, _ = make_channel(user_lock_cache_size=2)
        ch._get_user_lock("u1")
        ch._get_user_lock("u2")
        ch._get_user_lock("u3")
        assert len(ch._user_locks) <= 2


# ============================================================================
# _on_unsupported_media tests
# ============================================================================


class TestOnUnsupportedMedia:
    @pytest.mark.asyncio
    async def test_unsupported_media_sends_reply(self):
        ch, _, _ = make_channel()
        update = make_update(user_id="12345", text="ignored")
        ctx = make_context()
        await ch._on_unsupported_media(update, ctx)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "text" in text.lower()

    @pytest.mark.asyncio
    async def test_unsupported_media_blocked_for_non_allowed_user(self):
        ch, _, _ = make_channel(allowed_user_ids=["999"])
        update = make_update(user_id="12345", text="ignored")
        ctx = make_context()
        await ch._on_unsupported_media(update, ctx)
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsupported_media_no_message(self):
        ch, _, _ = make_channel()
        update = MagicMock()
        update.message = None
        update.effective_user = MagicMock()
        update.effective_user.id = 12345
        ctx = make_context()
        # Should not raise
        await ch._on_unsupported_media(update, ctx)

    @pytest.mark.asyncio
    async def test_unsupported_media_no_user(self):
        ch, _, _ = make_channel()
        update = MagicMock()
        update.message = MagicMock()
        update.effective_user = None
        ctx = make_context()
        await ch._on_unsupported_media(update, ctx)


# ============================================================================
# _parse_setup_payload edge case tests
# ============================================================================


class TestParseSetupPayload:
    def test_basic_parsing(self):
        from gwenn.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._parse_setup_payload("Bob | dev | debugging | concise | safe")
        assert result["name"] == "Bob"
        assert result["role"] == "dev"
        assert result["needs"] == "debugging"
        assert result["communication_style"] == "concise"
        assert result["boundaries"] == "safe"

    def test_missing_trailing_fields(self):
        from gwenn.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._parse_setup_payload("Bob | dev")
        assert result["name"] == "Bob"
        assert result["role"] == "dev"
        assert result["needs"] == ""
        assert result["communication_style"] == ""
        assert result["boundaries"] == ""

    def test_extra_pipes_ignored(self):
        from gwenn.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._parse_setup_payload("a|b|c|d|e|f|g")
        assert result["name"] == "a"
        assert result["boundaries"] == "e"

    def test_only_pipes(self):
        from gwenn.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._parse_setup_payload("||||")
        assert all(v == "" for v in result.values())

    def test_unicode_input(self):
        from gwenn.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._parse_setup_payload("Ren\u00e9 | d\u00e9veloppeur")
        assert result["name"] == "Ren\u00e9"
        assert result["role"] == "d\u00e9veloppeur"

    def test_whitespace_only_fields(self):
        from gwenn.channels.telegram_channel import TelegramChannel

        result = TelegramChannel._parse_setup_payload("   |   |   ")
        assert all(v == "" for v in result.values())


# ============================================================================
# _on_message null text guard tests
# ============================================================================


class TestOnMessageNullGuard:
    @pytest.mark.asyncio
    async def test_none_text_returns_early(self):
        ch, agent, _ = make_channel()
        update = make_update(user_id="55", text="hello")
        update.message.text = None
        ctx = make_context()
        await ch._on_message(update, ctx)
        agent.respond.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_message_returns_early(self):
        ch, agent, _ = make_channel()
        update = MagicMock()
        update.message = None
        update.effective_user = MagicMock()
        update.effective_user.id = 55
        ctx = make_context()
        await ch._on_message(update, ctx)
        agent.respond.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_user_returns_early(self):
        ch, agent, _ = make_channel()
        update = make_update(user_id="55", text="hello")
        update.effective_user = None
        ctx = make_context()
        await ch._on_message(update, ctx)
        agent.respond.assert_not_called()


# ============================================================================
# send_proactive tests
# ============================================================================


class TestSendProactive:
    @pytest.mark.asyncio
    async def test_send_proactive_sends_to_owners(self):
        ch, _, _ = make_channel(owner_user_ids=["111", "222"])
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()
        await ch.send_proactive("Hello owners!")
        # Should have sent to both owners
        assert ch._app.bot.send_message.call_count >= 2

    @pytest.mark.asyncio
    async def test_send_proactive_no_owners_is_noop(self):
        ch, _, _ = make_channel(owner_user_ids=[])
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()
        await ch.send_proactive("Hello?")
        ch._app.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_proactive_does_not_fallback_to_allowed(self):
        ch, _, _ = make_channel(allowed_user_ids=["333"], owner_user_ids=[])
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()
        await ch.send_proactive("Should not send")
        ch._app.bot.send_message.assert_not_called()


# ============================================================================
# send_message validation tests
# ============================================================================


class TestSendMessageValidation:
    @pytest.mark.asyncio
    async def test_send_message_invalid_id_skipped(self):
        ch, _, _ = make_channel()
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()
        await ch.send_message("not_a_number", "Hello")
        ch._app.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_message_html_fallback(self):
        ch, _, _ = make_channel()
        ch._app = MagicMock()
        call_count = [0]

        async def mock_send(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("HTML parse error")

        ch._app.bot.send_message = AsyncMock(side_effect=mock_send)
        await ch.send_message("12345", "Short message")
        # Two calls: first with HTML parse mode (fails), second without (succeeds)
        assert ch._app.bot.send_message.call_count == 2


# ============================================================================
# _on_edited_message tests
# ============================================================================


class TestOnEditedMessage:
    @pytest.mark.asyncio
    async def test_edited_message_acknowledged(self):
        ch, _, _ = make_channel()
        update = MagicMock()
        update.edited_message = MagicMock()
        update.edited_message.reply_text = AsyncMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 12345
        ctx = make_context()
        await ch._on_edited_message(update, ctx)
        update.edited_message.reply_text.assert_called_once()
        text = update.edited_message.reply_text.call_args[0][0]
        assert "edited" in text.lower()

    @pytest.mark.asyncio
    async def test_edited_message_no_message_is_noop(self):
        ch, _, _ = make_channel()
        update = MagicMock()
        update.edited_message = None
        update.effective_user = MagicMock()
        ctx = make_context()
        await ch._on_edited_message(update, ctx)  # should not raise

    @pytest.mark.asyncio
    async def test_edited_message_blocked_for_non_allowed(self):
        ch, _, _ = make_channel(allowed_user_ids=["999"])
        update = MagicMock()
        update.edited_message = MagicMock()
        update.edited_message.reply_text = AsyncMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 12345
        ctx = make_context()
        await ch._on_edited_message(update, ctx)
        update.edited_message.reply_text.assert_not_called()


# ============================================================================
# _on_error tests
# ============================================================================


class TestOnError:
    @pytest.mark.asyncio
    async def test_error_handler_does_not_raise(self):
        from gwenn.channels.telegram_channel import TelegramChannel

        update = MagicMock()
        update.update_id = 42
        context = MagicMock()
        context.error = RuntimeError("test error")
        # Should not raise
        await TelegramChannel._on_error(update, context)

    @pytest.mark.asyncio
    async def test_error_handler_with_none_update(self):
        from gwenn.channels.telegram_channel import TelegramChannel

        context = MagicMock()
        context.error = RuntimeError("test error")
        await TelegramChannel._on_error(None, context)


# ============================================================================
# Rate limiting in _on_message
# ============================================================================


class TestRateLimitInMessage:
    @pytest.mark.asyncio
    async def test_rate_limited_message_silently_dropped(self):
        ch, agent, _ = make_channel()
        # Set a very long rate limit interval to guarantee throttling
        ch._rate_limit_interval = 60.0
        ctx = make_context()

        update1 = make_update(user_id="55", text="first")
        await ch._on_message(update1, ctx)
        agent.respond.assert_called_once()

        update2 = make_update(user_id="55", text="second")
        await ch._on_message(update2, ctx)
        # Should still be 1 — second message was throttled
        assert agent.respond.call_count == 1
