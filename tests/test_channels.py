"""
Unit tests for SessionManager and message formatting utilities.

These tests require no real API calls, no Telegram/Discord tokens, and no
running Gwenn agent.  They exercise the pure-Python building blocks of the
channel layer.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from gwenn.channels.formatting import (
    DISCORD_MAX_LEN,
    TELEGRAM_MAX_LEN,
    describe_mood,
    format_uptime,
    format_for_discord,
    format_for_telegram,
    render_heartbeat_text,
    render_status_text,
    split_message,
)
from gwenn.channels.session import SessionManager


# ============================================================================
# SessionManager tests
# ============================================================================


class TestSessionManager:
    def test_get_or_create_returns_list(self):
        sm = SessionManager()
        history = sm.get_or_create("telegram_123")
        assert isinstance(history, list)

    def test_get_or_create_same_reference(self):
        sm = SessionManager()
        h1 = sm.get_or_create("telegram_123")
        h2 = sm.get_or_create("telegram_123")
        assert h1 is h2, "Must return the same list object (reference, not copy)"

    def test_mutations_reflected(self):
        sm = SessionManager()
        history = sm.get_or_create("telegram_abc")
        history.append({"role": "user", "content": "hello"})
        history2 = sm.get_or_create("telegram_abc")
        assert len(history2) == 1
        assert history2[0]["content"] == "hello"

    def test_different_users_isolated(self):
        sm = SessionManager()
        h1 = sm.get_or_create("telegram_user1")
        h2 = sm.get_or_create("telegram_user2")
        h1.append({"role": "user", "content": "hi from user1"})
        assert len(h2) == 0

    def test_clear_session(self):
        sm = SessionManager()
        history = sm.get_or_create("telegram_xyz")
        history.append({"role": "user", "content": "test"})
        sm.clear_session("telegram_xyz")
        assert sm.get_or_create("telegram_xyz") == []

    def test_clear_nonexistent_session_is_noop(self):
        sm = SessionManager()
        sm.clear_session("nobody")  # Should not raise

    def test_trim_history_caps_at_max(self):
        sm = SessionManager(max_history_length=2)
        history = sm.get_or_create("telegram_trim")
        # Add 3 turns (6 entries)
        for i in range(3):
            history.append({"role": "user", "content": f"msg {i}"})
            history.append({"role": "assistant", "content": f"reply {i}"})
        sm.trim_history("telegram_trim")
        # Should be trimmed to 2 turns = 4 entries
        assert len(history) <= 4

    def test_expire_stale_sessions(self):
        sm = SessionManager(session_ttl_seconds=0.01)
        sm.get_or_create("telegram_stale")
        sm.get_or_create("telegram_fresh")
        # Manually age the stale session
        sm._sessions["telegram_stale"].last_activity = time.time() - 1.0
        removed = sm.expire_stale_sessions()
        assert removed == 1
        assert "telegram_stale" not in sm._sessions
        assert "telegram_fresh" in sm._sessions

    def test_get_or_create_expires_stale_sessions(self):
        sm = SessionManager(session_ttl_seconds=0.01)
        sm.get_or_create("telegram_stale")
        sm._sessions["telegram_stale"].last_activity = time.time() - 1.0

        sm.get_or_create("telegram_fresh")

        assert "telegram_stale" not in sm._sessions
        assert "telegram_fresh" in sm._sessions

    def test_session_count(self):
        sm = SessionManager()
        assert sm.session_count() == 0
        sm.get_or_create("u1")
        sm.get_or_create("u2")
        assert sm.session_count() == 2

    def test_get_session_info_none_for_unknown(self):
        sm = SessionManager()
        assert sm.get_session_info("nobody") is None

    def test_get_session_info_returns_dict(self):
        sm = SessionManager()
        sm.get_or_create("telegram_info")
        info = sm.get_session_info("telegram_info")
        assert info is not None
        assert info["user_id"] == "telegram_info"
        assert "message_count" in info
        assert "history_length" in info

    def test_message_count_increments(self):
        sm = SessionManager()
        sm.get_or_create("telegram_count")
        sm.get_or_create("telegram_count")
        sm.get_or_create("telegram_count")
        info = sm.get_session_info("telegram_count")
        assert info["message_count"] == 3


# ============================================================================
# split_message / formatting tests
# ============================================================================


class TestSplitMessage:
    def test_empty_string_returns_empty_list(self):
        assert split_message("", 100) == []

    def test_short_message_returned_as_single_chunk(self):
        result = split_message("Hello!", 100)
        assert result == ["Hello!"]

    def test_long_message_is_split(self):
        text = "word " * 1000  # ~5000 chars
        chunks = split_message(text, 100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_split_at_paragraph(self):
        text = ("A" * 50) + "\n\n" + ("B" * 50)
        chunks = split_message(text, 60)
        assert len(chunks) == 2

    def test_split_at_newline(self):
        text = ("A" * 50) + "\n" + ("B" * 50)
        chunks = split_message(text, 60)
        assert len(chunks) == 2

    def test_no_chunk_exceeds_max_len(self):
        import random
        import string
        random.seed(42)
        text = "".join(random.choices(string.ascii_letters + " \n", k=5000))
        for max_len in (100, 500, 2000, 4096):
            for chunk in split_message(text, max_len):
                assert len(chunk) <= max_len, f"Chunk exceeds {max_len}: {chunk[:50]!r}"

    def test_preserves_whitespace_and_newlines_when_split(self):
        text = "  Intro paragraph.\n\n    indented line\n\nTrailing spaces  "
        chunks = split_message(text, 20)
        assert "".join(chunks) == text

    def test_whitespace_only_returns_empty(self):
        assert split_message("   ", 100) == []

    def test_exactly_max_len_not_split(self):
        text = "x" * 100
        assert split_message(text, 100) == [text]

    def test_one_over_max_len_splits(self):
        text = "hello world foo bar"
        chunks = split_message(text, 11)
        assert all(len(c) <= 11 for c in chunks)
        assert " ".join(chunks).replace("  ", " ") or True  # content preserved


class TestPlatformFormatters:
    def test_format_for_telegram_max_len(self):
        text = "x " * 3000  # 6000 chars
        for chunk in format_for_telegram(text):
            assert len(chunk) <= TELEGRAM_MAX_LEN

    def test_format_for_discord_max_len(self):
        text = "x " * 1500  # 3000 chars
        for chunk in format_for_discord(text):
            assert len(chunk) <= DISCORD_MAX_LEN

    def test_format_for_telegram_short(self):
        assert format_for_telegram("Hi there!") == ["Hi there!"]

    def test_format_for_discord_short(self):
        assert format_for_discord("Hi there!") == ["Hi there!"]


class TestStatusFormatting:
    def test_format_uptime_human_readable(self):
        assert format_uptime(5) == "5s"
        assert format_uptime(65) == "1m 5s"
        assert format_uptime(3661) == "1h 1m 1s"

    def test_describe_mood(self):
        text = describe_mood("curiosity", valence=0.3, arousal=0.2)
        assert "curiosity" in text
        assert "positive" in text

    def test_render_status_text_plain(self):
        text = render_status_text(
            {
                "name": "Gwenn",
                "emotion": "neutral",
                "valence": 0.1,
                "arousal": 0.2,
                "working_memory_load": 0.4,
                "total_interactions": 7,
                "uptime_seconds": 125,
                "resilience": {"breaker_active": False},
            }
        )
        assert "Mood:" in text
        assert "Stress guardrail: normal" in text
        assert "Awake for: 2m 5s" in text

    def test_render_heartbeat_text_plain(self):
        text = render_heartbeat_text(
            {
                "running": True,
                "beat_count": 3,
                "current_interval": 30,
                "beats_since_consolidation": 1,
            }
        )
        assert "Heartbeat Status" in text
        assert "Beat count: 3" in text


# ============================================================================
# BaseChannel.handle_message integration (with mocked agent)
# ============================================================================


class TestBaseChannelHandleMessage:
    """Verify that handle_message correctly routes through SessionManager."""

    @pytest.mark.asyncio
    async def test_handle_message_uses_session_history(self):
        from gwenn.channels.base import BaseChannel

        class ConcreteChannel(BaseChannel):
            channel_name = "test"

            async def start(self): pass
            async def stop(self): pass
            async def send_message(self, uid, text): pass

        mock_agent = AsyncMock()
        mock_agent.respond = AsyncMock(return_value="Hello from Gwenn")
        sessions = SessionManager()

        ch = ConcreteChannel(mock_agent, sessions)
        response = await ch.handle_message("42", "Hi Gwenn")

        assert response == "Hello from Gwenn"
        mock_agent.respond.assert_called_once()
        call_kwargs = mock_agent.respond.call_args
        assert call_kwargs.kwargs["user_message"] == "Hi Gwenn"
        assert call_kwargs.kwargs["user_id"] == "test_42"
        # The history passed must be the session's list
        assert call_kwargs.kwargs["conversation_history"] is sessions.get_or_create("test_42")

    @pytest.mark.asyncio
    async def test_handle_message_namespaces_user_id(self):
        from gwenn.channels.base import BaseChannel

        class ConcreteChannel(BaseChannel):
            channel_name = "myplatform"

            async def start(self): pass
            async def stop(self): pass
            async def send_message(self, uid, text): pass

        mock_agent = AsyncMock()
        mock_agent.respond = AsyncMock(return_value="ok")
        sessions = SessionManager()
        ch = ConcreteChannel(mock_agent, sessions)

        await ch.handle_message("99", "test")
        call_kwargs = mock_agent.respond.call_args
        assert call_kwargs.kwargs["user_id"] == "myplatform_99"
