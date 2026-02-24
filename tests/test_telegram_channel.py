"""
Unit tests for TelegramChannel using python-telegram-bot mocks.

These tests exercise handler logic without making any real Telegram API calls.
The PTB application and bot objects are replaced with MagicMock / AsyncMock
instances so the tests run without a valid TELEGRAM_BOT_TOKEN.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import gwenn.channels.telegram_channel as tg_mod
from gwenn.channels.session import SessionManager
from gwenn.channels.telegram_channel import TelegramChannel
from gwenn.types import AgentResponse, ButtonSpec


def make_config(
    allowed_user_ids=None,
    owner_user_ids=None,
    enable_media=False,
    concurrent_updates=0,
):
    cfg = MagicMock()
    cfg.bot_token = "test_token"
    cfg.allowed_user_ids = allowed_user_ids or []
    cfg.owner_user_ids = owner_user_ids or []
    cfg.session_scope_mode = "per_user"
    cfg.user_lock_cache_size = 512
    cfg.concurrent_updates = concurrent_updates
    cfg.enable_media = enable_media
    return cfg


def make_channel(
    allowed_user_ids=None,
    owner_user_ids=None,
    session_scope_mode="per_user",
    user_lock_cache_size=512,
    enable_media=False,
    concurrent_updates=0,
):
    """Create a TelegramChannel with mocked agent and sessions."""
    agent = AsyncMock()
    agent.respond = AsyncMock(return_value=AgentResponse(text="Hello from Gwenn"))
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
    config = make_config(
        allowed_user_ids,
        owner_user_ids=owner_user_ids,
        enable_media=enable_media,
        concurrent_updates=concurrent_updates,
    )
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
    update.message.caption = None
    update.message.photo = None
    update.message.document = None
    update.message.voice = None
    update.message.reply_text = AsyncMock()
    return update


def make_context(args=None):
    ctx = MagicMock()
    ctx.bot.send_chat_action = AsyncMock()
    ctx.args = args or []
    return ctx


@pytest.fixture(autouse=True)
def reset_ptb_install_attempt(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tg_mod, "_PTB_INSTALL_ATTEMPTED", False)


# ============================================================================
# Dependency bootstrap tests
# ============================================================================


class TestDependencyBootstrap:
    @pytest.mark.asyncio
    async def test_ensure_application_builder_no_install_when_present(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(tg_mod, "_import_application_builder", lambda: sentinel)
        attempt = AsyncMock(return_value=True)
        monkeypatch.setattr(tg_mod, "_attempt_auto_install_ptb", attempt)

        result = await tg_mod._ensure_application_builder()

        assert result is sentinel
        attempt.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_application_builder_retries_after_auto_install(self, monkeypatch):
        sentinel = object()
        calls = {"count": 0}

        def fake_import():
            calls["count"] += 1
            if calls["count"] == 1:
                raise ImportError("missing telegram")
            return sentinel

        monkeypatch.setattr(tg_mod, "_import_application_builder", fake_import)
        monkeypatch.setattr(tg_mod, "_attempt_auto_install_ptb", AsyncMock(return_value=True))

        result = await tg_mod._ensure_application_builder()

        assert result is sentinel
        assert calls["count"] == 2

    @pytest.mark.asyncio
    async def test_ensure_application_builder_raises_when_auto_install_fails(self, monkeypatch):
        monkeypatch.setattr(
            tg_mod,
            "_import_application_builder",
            MagicMock(side_effect=ImportError("missing telegram")),
        )
        monkeypatch.setattr(tg_mod, "_attempt_auto_install_ptb", AsyncMock(return_value=False))

        with pytest.raises(ImportError, match="Install/refresh dependencies"):
            await tg_mod._ensure_application_builder()

    @pytest.mark.asyncio
    async def test_attempt_auto_install_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("GWENN_AUTO_INSTALL_TELEGRAM", "false")
        install = MagicMock(return_value=True)
        monkeypatch.setattr(tg_mod, "_pip_install_requirement", install)

        result = await tg_mod._attempt_auto_install_ptb()

        assert result is False
        install.assert_not_called()

    @pytest.mark.asyncio
    async def test_attempt_auto_install_runs_only_once(self, monkeypatch):
        monkeypatch.setenv("GWENN_AUTO_INSTALL_TELEGRAM", "true")
        install = MagicMock(return_value=False)
        monkeypatch.setattr(tg_mod, "_pip_install_requirement", install)

        first = await tg_mod._attempt_auto_install_ptb()
        second = await tg_mod._attempt_auto_install_ptb()

        assert first is False
        assert second is False
        install.assert_called_once()


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
# _get_owner_ids — fallback behavior (#6)
# ============================================================================


class TestGetOwnerIds:
    def test_explicit_owners_used(self):
        ch, _, _ = make_channel(allowed_user_ids=["333"], owner_user_ids=["111"])
        assert ch._get_owner_ids() == {"111"}

    def test_falls_back_to_allowed_when_no_owners(self):
        ch, _, _ = make_channel(allowed_user_ids=["333", "444"], owner_user_ids=[])
        assert ch._get_owner_ids() == {"333", "444"}

    def test_empty_owners_and_allowed_returns_empty(self):
        ch, _, _ = make_channel(allowed_user_ids=[], owner_user_ids=[])
        assert ch._get_owner_ids() == set()


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
        text = update.message.reply_text.call_args[0][0]
        assert "/help" in text

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
    async def test_setup_saves_profile_from_raw_text(self):
        """Setup now parses from raw message text, not context.args (#16)."""
        ch, agent, _ = make_channel()
        update = make_update(user_id="12345")
        update.message.text = "/setup Bob | coding partner | debugging | concise | no destructive changes"
        ctx = make_context()

        await ch._on_setup(update, ctx)

        agent.apply_startup_onboarding.assert_called_once()
        kwargs = agent.apply_startup_onboarding.call_args.kwargs
        args = agent.apply_startup_onboarding.call_args.args
        assert kwargs["user_id"] == "telegram_12345"
        assert args[0]["name"] == "Bob"
        assert args[0]["role"] == "coding partner"

    @pytest.mark.asyncio
    async def test_setup_skip_marks_completed(self):
        ch, agent, _ = make_channel()
        update = make_update(user_id="12345")
        update.message.text = "/setup skip"
        ctx = make_context()

        await ch._on_setup(update, ctx)

        agent.identity.mark_onboarding_completed.assert_called_once_with({})
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_without_args_shows_usage(self):
        ch, _, _ = make_channel()
        update = make_update(user_id="12345")
        update.message.text = "/setup"
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
# /cancel command tests (#14)
# ============================================================================


class TestOnCancel:
    @pytest.mark.asyncio
    async def test_cancel_sets_flag(self):
        ch, _, _ = make_channel()
        update = make_update(user_id="55")
        ctx = make_context()
        await ch._on_cancel(update, ctx)
        assert ch._cancel_flags.get("55") is True
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "cancel" in text.lower()

    @pytest.mark.asyncio
    async def test_cancel_blocked_for_non_allowed(self):
        ch, _, _ = make_channel(allowed_user_ids=["999"])
        update = make_update(user_id="55")
        ctx = make_context()
        await ch._on_cancel(update, ctx)
        assert "55" not in ch._cancel_flags
        update.message.reply_text.assert_not_called()


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
        from gwenn.types import UserMessage
        assert isinstance(call_kwargs["user_message"], UserMessage)
        assert call_kwargs["user_message"].text == "What are you thinking?"
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
        agent.respond = AsyncMock(return_value=AgentResponse(text="word " * 1500))
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
        agent.respond = AsyncMock(return_value=AgentResponse(text="word " * 1500))
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
        agent.respond = AsyncMock(return_value=AgentResponse(text="word " * 1500))
        update = make_update(user_id="99", text="write a lot")
        # First chunk: HTML succeeds.
        # Second chunk: HTML fails, plain-text fallback succeeds.
        update.message.reply_text = AsyncMock(
            side_effect=[None, RuntimeError("html failed"), None]
        )
        ctx = make_context()

        await ch._on_message(update, ctx)

        assert update.message.reply_text.await_count >= 3

    @pytest.mark.asyncio
    async def test_empty_response_sends_fallback(self):
        """Empty agent responses should produce a user-visible fallback (#24)."""
        ch, agent, _ = make_channel()
        agent.respond = AsyncMock(return_value=AgentResponse(text=""))
        update = make_update(user_id="55", text="hi")
        ctx = make_context()

        await ch._on_message(update, ctx)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "rephras" in text.lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_response_sends_fallback(self):
        ch, agent, _ = make_channel()
        agent.respond = AsyncMock(return_value=AgentResponse(text="   \n  "))
        update = make_update(user_id="55", text="hi")
        ctx = make_context()

        await ch._on_message(update, ctx)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "rephras" in text.lower()

    @pytest.mark.asyncio
    async def test_cancel_flag_stops_response_delivery(self):
        """If /cancel was issued during processing, response is discarded (#14)."""
        ch, agent, _ = make_channel()

        async def slow_respond(**kwargs):
            # Simulate cancel being set during processing.
            ch._cancel_flags["55"] = True
            return AgentResponse(text="This should not be delivered")

        agent.respond = AsyncMock(side_effect=slow_respond)
        update = make_update(user_id="55", text="hi")
        ctx = make_context()

        await ch._on_message(update, ctx)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "cancel" in text.lower()


class TestLockCache:
    def test_lock_cache_eviction_bounds_memory(self):
        ch, _, _ = make_channel(user_lock_cache_size=2)
        ch._get_user_lock("u1")
        ch._get_user_lock("u2")
        ch._get_user_lock("u3")
        assert len(ch._user_locks) <= 2


# ============================================================================
# Media handler tests (#12)
# ============================================================================


class TestMediaHandlers:
    @pytest.mark.asyncio
    async def test_photo_handler_routes_to_agent(self):
        ch, agent, _ = make_channel(enable_media=True)
        update = make_update(user_id="55", text="hello")
        update.message.caption = "Check this out"
        update.message.photo = [MagicMock()]
        ctx = make_context()

        await ch._on_photo(update, ctx)

        agent.respond.assert_called_once()
        call_kwargs = agent.respond.call_args.kwargs
        um = call_kwargs["user_message"]
        # Download fails on mock objects — falls back to text-only description.
        assert "[The user sent a photo" in um.text
        assert "Check this out" in um.text

    @pytest.mark.asyncio
    async def test_photo_handler_no_caption(self):
        ch, agent, _ = make_channel(enable_media=True)
        update = make_update(user_id="55", text="hello")
        update.message.caption = None
        update.message.photo = [MagicMock()]
        ctx = make_context()

        await ch._on_photo(update, ctx)

        agent.respond.assert_called_once()
        call_kwargs = agent.respond.call_args.kwargs
        assert "[The user sent a photo]" in call_kwargs["user_message"].text

    @pytest.mark.asyncio
    async def test_document_handler_includes_filename(self):
        ch, agent, _ = make_channel(enable_media=True)
        update = make_update(user_id="55", text="hello")
        update.message.caption = None
        doc = MagicMock()
        doc.file_name = "report.pdf"
        update.message.document = doc
        ctx = make_context()

        await ch._on_document(update, ctx)

        agent.respond.assert_called_once()
        call_kwargs = agent.respond.call_args.kwargs
        assert "report.pdf" in call_kwargs["user_message"].text

    @pytest.mark.asyncio
    async def test_voice_handler_routes_to_agent(self):
        ch, agent, _ = make_channel(enable_media=True)
        update = make_update(user_id="55", text="hello")
        ctx = make_context()

        await ch._on_voice(update, ctx)

        agent.respond.assert_called_once()
        call_kwargs = agent.respond.call_args.kwargs
        assert "voice" in call_kwargs["user_message"].text.lower()

    @pytest.mark.asyncio
    async def test_media_blocked_for_non_allowed(self):
        ch, agent, _ = make_channel(
            enable_media=True, allowed_user_ids=["999"]
        )
        update = make_update(user_id="55", text="hello")
        update.message.photo = [MagicMock()]
        ctx = make_context()

        await ch._on_photo(update, ctx)
        agent.respond.assert_not_called()


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
    async def test_unsupported_media_with_media_enabled(self):
        ch, _, _ = make_channel(enable_media=True)
        update = make_update(user_id="12345", text="ignored")
        ctx = make_context()
        await ch._on_unsupported_media(update, ctx)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "photos" in text.lower() or "support" in text.lower()

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
        result = TelegramChannel._parse_setup_payload(
            "Bob | dev | debugging | concise | safe"
        )
        assert result["name"] == "Bob"
        assert result["role"] == "dev"
        assert result["needs"] == "debugging"
        assert result["communication_style"] == "concise"
        assert result["boundaries"] == "safe"

    def test_missing_trailing_fields(self):
        result = TelegramChannel._parse_setup_payload("Bob | dev")
        assert result["name"] == "Bob"
        assert result["role"] == "dev"
        assert result["needs"] == ""
        assert result["communication_style"] == ""
        assert result["boundaries"] == ""

    def test_extra_pipes_ignored(self):
        result = TelegramChannel._parse_setup_payload("a|b|c|d|e|f|g")
        assert result["name"] == "a"
        assert result["boundaries"] == "e"

    def test_only_pipes(self):
        result = TelegramChannel._parse_setup_payload("||||")
        assert all(v == "" for v in result.values())

    def test_unicode_input(self):
        result = TelegramChannel._parse_setup_payload(
            "Ren\u00e9 | d\u00e9veloppeur"
        )
        assert result["name"] == "Ren\u00e9"
        assert result["role"] == "d\u00e9veloppeur"

    def test_whitespace_only_fields(self):
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
# send_proactive tests — now with owner fallback (#6, #19, #25)
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
    async def test_send_proactive_falls_back_to_allowed(self):
        """When no owners configured, proactive falls back to allowed_user_ids (#6)."""
        ch, _, _ = make_channel(allowed_user_ids=["333"], owner_user_ids=[])
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()
        await ch.send_proactive("Hello via fallback!")
        assert ch._app.bot.send_message.call_count >= 1

    @pytest.mark.asyncio
    async def test_send_proactive_empty_both_is_noop(self):
        ch, _, _ = make_channel(allowed_user_ids=[], owner_user_ids=[])
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()
        await ch.send_proactive("Hello?")
        ch._app.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_proactive_queued_before_startup(self):
        """Proactive messages before bot start should be queued (#19)."""
        ch, _, _ = make_channel(owner_user_ids=["111"])
        # _app is None before start()
        assert ch._app is None
        await ch.send_proactive("Early thought")
        assert len(ch._proactive_queue) == 1
        assert ch._proactive_queue[0] == "Early thought"


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

    @pytest.mark.asyncio
    async def test_send_message_html_fallback_strips_tags(self):
        """Plain-text fallback should have HTML tags stripped (#10)."""
        ch, _, _ = make_channel()
        ch._app = MagicMock()
        calls = []

        async def mock_send(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise Exception("HTML parse error")

        ch._app.bot.send_message = AsyncMock(side_effect=mock_send)
        await ch.send_message("12345", "**bold text**")
        assert len(calls) == 2
        # Second call (fallback) should not have parse_mode
        assert "parse_mode" not in calls[1]
        # And should not contain raw HTML tags
        assert "<b>" not in calls[1]["text"]


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
# _on_error tests — now with differentiated handling (#2)
# ============================================================================


class TestOnError:
    @pytest.mark.asyncio
    async def test_error_handler_does_not_raise(self):
        update = MagicMock()
        update.update_id = 42
        context = MagicMock()
        context.error = RuntimeError("test error")
        # Should not raise
        await TelegramChannel._on_error(update, context)

    @pytest.mark.asyncio
    async def test_error_handler_with_none_update(self):
        context = MagicMock()
        context.error = RuntimeError("test error")
        await TelegramChannel._on_error(None, context)

    @pytest.mark.asyncio
    async def test_error_handler_retry_after(self):
        """RetryAfter errors should be handled without raising (#2)."""
        try:
            from telegram.error import RetryAfter
            error = RetryAfter(0.1)
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        context = MagicMock()
        context.error = error
        update = MagicMock()
        update.update_id = 1
        # Should not raise; internally sleeps for retry_after + 0.5
        await TelegramChannel._on_error(update, context)

    @pytest.mark.asyncio
    async def test_error_handler_forbidden(self):
        """Forbidden errors (user blocked bot) should be handled gracefully (#2)."""
        try:
            from telegram.error import Forbidden
            error = Forbidden("Forbidden: bot was blocked by the user")
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        context = MagicMock()
        context.error = error
        update = MagicMock()
        update.update_id = 2
        await TelegramChannel._on_error(update, context)

    @pytest.mark.asyncio
    async def test_error_handler_network_error(self):
        """Network errors should be logged as warnings, not errors (#2)."""
        try:
            from telegram.error import NetworkError
            error = NetworkError("Connection reset")
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        context = MagicMock()
        context.error = error
        update = MagicMock()
        update.update_id = 3
        await TelegramChannel._on_error(update, context)


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


# ============================================================================
# Inline button (present_choices) tests
# ============================================================================


def _make_callback_query(user_id="12345", data="btn:opt1", chat_id=None):
    """Create a minimal CallbackQuery mock for button callback tests."""
    query = MagicMock()
    query.from_user.id = int(user_id)
    query.data = data
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    msg = MagicMock()
    msg.chat_id = int(chat_id if chat_id is not None else user_id)
    msg.message_thread_id = None
    msg.text = "Pick one"
    msg.reply_text = AsyncMock()
    query.message = msg
    return query


def _make_callback_update(query):
    """Wrap a CallbackQuery mock into an Update mock."""
    update = MagicMock()
    update.callback_query = query
    update.effective_user = query.from_user
    update.effective_chat = MagicMock()
    update.effective_chat.id = query.message.chat_id
    return update


class TestInlineButtons:
    @pytest.mark.asyncio
    async def test_response_with_buttons_attaches_keyboard(self):
        """reply_markup should be set when AgentResponse has buttons."""
        ch, agent, _ = make_channel()
        buttons = [[ButtonSpec(label="Yes"), ButtonSpec(label="No")]]
        agent.respond = AsyncMock(
            return_value=AgentResponse(text="Pick one", buttons=buttons)
        )
        update = make_update(user_id="55", text="choose")
        ctx = make_context()

        await ch._on_message(update, ctx)

        # The last reply_text call should have reply_markup set.
        last_call = update.message.reply_text.call_args
        assert last_call.kwargs.get("reply_markup") is not None or (
            len(last_call.args) > 1 and last_call.args[1] is not None
        )

    @pytest.mark.asyncio
    async def test_response_without_buttons_no_keyboard(self):
        """No reply_markup when buttons=None."""
        ch, agent, _ = make_channel()
        agent.respond = AsyncMock(
            return_value=AgentResponse(text="Just text", buttons=None)
        )
        update = make_update(user_id="55", text="hello")
        ctx = make_context()

        await ch._on_message(update, ctx)

        last_call = update.message.reply_text.call_args
        markup = last_call.kwargs.get("reply_markup")
        assert markup is None

    @pytest.mark.asyncio
    async def test_button_callback_routes_selection(self):
        """Clicking a button sends [Selected: ...] to the agent."""
        ch, agent, _ = make_channel()
        query = _make_callback_query(user_id="55", data="btn:option_a")
        update = _make_callback_update(query)
        ctx = make_context()

        await ch._on_button_callback(update, ctx)

        # query.answer() must be called to dismiss the spinner.
        query.answer.assert_called_once()
        # The keyboard should be removed.
        query.edit_message_reply_markup.assert_called_once()
        # Agent should have been called with the selected value.
        agent.respond.assert_called_once()
        call_kwargs = agent.respond.call_args.kwargs
        from gwenn.types import UserMessage
        assert isinstance(call_kwargs["user_message"], UserMessage)
        assert "[Selected: option_a]" in call_kwargs["user_message"].text

    @pytest.mark.asyncio
    async def test_button_callback_unauthorized_user(self):
        """Non-allowed user's button press is rejected."""
        ch, agent, _ = make_channel(allowed_user_ids=["999"])
        query = _make_callback_query(user_id="55", data="btn:x")
        update = _make_callback_update(query)
        ctx = make_context()

        await ch._on_button_callback(update, ctx)

        query.answer.assert_called_once()
        assert "not authorized" in query.answer.call_args.args[0].lower()
        agent.respond.assert_not_called()

    @pytest.mark.asyncio
    async def test_approval_flow_still_works(self):
        """The approval handler should still match approve:/deny: prefixes."""
        ch, agent, _ = make_channel(owner_user_ids=["55"])
        query = MagicMock()
        query.from_user.id = 55
        query.data = "approve:abc123"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock()
        query.message.text = "Tool approval required"
        update = MagicMock()
        update.callback_query = query

        await ch._on_approval_callback(update, MagicMock())

        # approval_id "abc123" doesn't exist, so it should answer with "expired".
        query.answer.assert_called_once()
        assert "expired" in query.answer.call_args.args[0].lower() or \
               "handled" in query.answer.call_args.args[0].lower()

    def test_build_inline_keyboard_truncates_long_values(self):
        """Callback data must not exceed 64 bytes."""
        long_value = "x" * 200
        rows = [[ButtonSpec(label="Click", value=long_value)]]
        markup = TelegramChannel._build_inline_keyboard(rows)
        assert markup is not None
        # Extract the callback_data from the first button.
        button = markup.inline_keyboard[0][0]
        assert len(button.callback_data.encode("utf-8")) <= 64

    def test_build_inline_keyboard_truncates_multibyte(self):
        """Multi-byte UTF-8 values must still respect the 64-byte limit."""
        # Each emoji is 4 bytes UTF-8; 20 emojis + "btn:" prefix > 64 bytes.
        long_value = "\U0001f600" * 20
        rows = [[ButtonSpec(label="Emoji", value=long_value)]]
        markup = TelegramChannel._build_inline_keyboard(rows)
        assert markup is not None
        button = markup.inline_keyboard[0][0]
        assert len(button.callback_data.encode("utf-8")) <= 64
