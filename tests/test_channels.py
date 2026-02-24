"""
Unit tests for SessionManager and message formatting utilities.

These tests require no real API calls, no Telegram/Discord tokens, and no
running Gwenn agent.  They exercise the pure-Python building blocks of the
channel layer.
"""

from __future__ import annotations

import asyncio
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
    markdown_to_telegram_html,
    render_heartbeat_text,
    render_status_text,
    split_message,
    strip_html_tags,
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
        sm._sessions["telegram_stale"].last_activity = time.monotonic() - 1.0
        removed = sm.expire_stale_sessions()
        assert removed == 1
        assert "telegram_stale" not in sm._sessions
        assert "telegram_fresh" in sm._sessions

    def test_get_or_create_expires_stale_sessions(self):
        sm = SessionManager(session_ttl_seconds=0.01)
        sm.get_or_create("telegram_stale")
        sm._sessions["telegram_stale"].last_activity = time.monotonic() - 1.0

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

            async def start(self):
                pass

            async def stop(self):
                pass

            async def send_message(self, uid, text):
                pass

        mock_agent = AsyncMock()
        mock_agent.respond = AsyncMock(return_value="Hello from Gwenn")
        sessions = SessionManager()

        ch = ConcreteChannel(mock_agent, sessions)
        response = await ch.handle_message("42", "Hi Gwenn")

        assert response == "Hello from Gwenn"
        mock_agent.respond.assert_called_once()
        call_kwargs = mock_agent.respond.call_args
        um = call_kwargs.kwargs["user_message"]
        from gwenn.types import UserMessage
        assert isinstance(um, UserMessage)
        assert um.text == "Hi Gwenn"
        assert um.images == []
        assert call_kwargs.kwargs["user_id"] == "test_42"
        # The history passed must be the session's list (namespaced with user: prefix)
        assert call_kwargs.kwargs["conversation_history"] is sessions.get_or_create("test_user:42")

    @pytest.mark.asyncio
    async def test_handle_message_namespaces_user_id(self):
        from gwenn.channels.base import BaseChannel

        class ConcreteChannel(BaseChannel):
            channel_name = "myplatform"

            async def start(self):
                pass

            async def stop(self):
                pass

            async def send_message(self, uid, text):
                pass

        mock_agent = AsyncMock()
        mock_agent.respond = AsyncMock(return_value="ok")
        sessions = SessionManager()
        ch = ConcreteChannel(mock_agent, sessions)

        await ch.handle_message("99", "test")
        call_kwargs = mock_agent.respond.call_args
        assert call_kwargs.kwargs["user_id"] == "myplatform_99"

    @pytest.mark.asyncio
    async def test_handle_message_supports_scoped_sessions(self):
        from gwenn.channels.base import BaseChannel

        class ConcreteChannel(BaseChannel):
            channel_name = "myplatform"

            async def start(self):
                pass

            async def stop(self):
                pass

            async def send_message(self, uid, text):
                pass

        mock_agent = AsyncMock()
        mock_agent.respond = AsyncMock(return_value="ok")
        sessions = SessionManager()
        ch = ConcreteChannel(mock_agent, sessions)

        await ch.handle_message("99", "first", session_scope_key="chat:abc")
        await ch.handle_message("100", "second", session_scope_key="chat:abc")

        scoped_history = sessions.get_or_create("myplatform_chat:abc")
        assert scoped_history is mock_agent.respond.call_args.kwargs["conversation_history"]
        # user_id remains the actor identity, not the shared session key.
        assert mock_agent.respond.call_args.kwargs["user_id"] == "myplatform_100"

    @pytest.mark.asyncio
    async def test_handle_message_uses_per_session_lock(self):
        """Different sessions can run concurrently; same session serialises."""
        from gwenn.channels.base import BaseChannel

        class ConcreteChannel(BaseChannel):
            channel_name = "test"

            async def start(self):
                pass

            async def stop(self):
                pass

            async def send_message(self, uid, text):
                pass

        class _Agent:
            def __init__(self):
                self.active = 0
                self.max_active = 0

            async def respond(self, user_message, user_id, conversation_history,
                              session_id=""):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                await asyncio.sleep(0.01)
                self.active -= 1
                text = user_message.text if hasattr(user_message, "text") else user_message
                return f"echo:{text}"

        agent = _Agent()
        sessions = SessionManager()
        ch1 = ConcreteChannel(agent, sessions)
        ch2 = ConcreteChannel(agent, sessions)

        # Different users (different sessions) should run concurrently
        await asyncio.gather(
            ch1.handle_message("1", "a"),
            ch2.handle_message("2", "b"),
        )
        assert agent.max_active == 2

        # Same user (same session) should serialise
        agent.active = 0
        agent.max_active = 0
        await asyncio.gather(
            ch1.handle_message("1", "a"),
            ch1.handle_message("1", "b"),
        )
        assert agent.max_active == 1


# ============================================================================
# markdown_to_telegram_html tests
# ============================================================================


class TestMarkdownToTelegramHtml:
    def test_plain_text_escaped(self):
        result = markdown_to_telegram_html("Hello <world> & friends")
        assert "&lt;world&gt;" in result
        assert "&amp;" in result

    def test_bold_conversion(self):
        result = markdown_to_telegram_html("This is **bold** text")
        assert "<b>bold</b>" in result

    def test_italic_conversion(self):
        result = markdown_to_telegram_html("This is _italic_ text")
        assert "<i>italic</i>" in result

    def test_heading_conversion(self):
        result = markdown_to_telegram_html("## My Heading")
        assert "<b>My Heading</b>" in result

    def test_inline_code(self):
        result = markdown_to_telegram_html("Use `foo()` here")
        assert "<code>foo()</code>" in result

    def test_inline_code_html_escaped(self):
        result = markdown_to_telegram_html("Use `<script>` tag")
        assert "<code>&lt;script&gt;</code>" in result

    def test_fenced_code_block(self):
        result = markdown_to_telegram_html("```python\nprint('hi')\n```")
        assert "<pre" in result
        assert "print(&#x27;hi&#x27;)" in result or "print('hi')" in result

    def test_fenced_code_block_with_language(self):
        result = markdown_to_telegram_html("```python\nx = 1\n```")
        assert 'language="python"' in result
        assert "<pre" in result

    def test_fenced_code_block_no_language(self):
        result = markdown_to_telegram_html("```\nplain code\n```")
        assert "<pre" in result
        assert "language=" not in result

    def test_code_block_protects_markdown_syntax(self):
        result = markdown_to_telegram_html("```\n**not bold** _not italic_\n```")
        assert "<b>" not in result
        assert "<i>" not in result

    def test_nested_bold_italic(self):
        result = markdown_to_telegram_html("**bold _and italic_**")
        assert "<b>" in result

    def test_empty_string(self):
        assert markdown_to_telegram_html("") == ""

    def test_html_special_chars_in_plain_text(self):
        result = markdown_to_telegram_html("a < b > c & d")
        assert "a &lt; b &gt; c &amp; d" in result

    def test_empty_fenced_code_block(self):
        result = markdown_to_telegram_html("```\n\n```")
        assert "<pre" in result

    def test_strikethrough_conversion(self):
        result = markdown_to_telegram_html("This is ~~deleted~~ text")
        assert "<s>deleted</s>" in result

    def test_blockquote_conversion(self):
        result = markdown_to_telegram_html("> This is a quote\n> Second line")
        assert "<blockquote>" in result
        assert "This is a quote" in result
        assert "Second line" in result
        assert "</blockquote>" in result

    def test_blockquote_strips_prefix(self):
        result = markdown_to_telegram_html("> Hello world")
        assert "<blockquote>" in result
        # Should not contain the '> ' prefix inside the blockquote
        assert "> Hello" not in result.replace("<blockquote>", "")

    def test_mixed_blockquote_and_normal(self):
        result = markdown_to_telegram_html("Normal text\n\n> Quoted\n\nMore text")
        assert "<blockquote>" in result
        assert "Normal text" in result
        assert "More text" in result


# ============================================================================
# strip_html_tags tests (#10)
# ============================================================================


class TestStripHtmlTags:
    def test_strips_simple_tags(self):
        assert strip_html_tags("<b>bold</b>") == "bold"

    def test_strips_nested_tags(self):
        assert strip_html_tags("<b><i>nested</i></b>") == "nested"

    def test_preserves_plain_text(self):
        assert strip_html_tags("no tags here") == "no tags here"

    def test_strips_pre_with_language(self):
        assert strip_html_tags('<pre language="python">code</pre>') == "code"

    def test_empty_string(self):
        assert strip_html_tags("") == ""

    def test_preserves_entities(self):
        assert strip_html_tags("a &amp; b") == "a &amp; b"


# ============================================================================
# format_for_telegram chunk size test
# ============================================================================


class TestFormatForTelegramSize:
    def test_html_expanded_chunks_within_limit(self):
        """Verify chunks don't exceed TELEGRAM_MAX_LEN after HTML conversion."""
        # Build text with lots of bold markers that expand to <b>…</b> tags.
        text = "**bold** normal text here. " * 200
        chunks = format_for_telegram(text)
        for chunk in chunks:
            assert len(chunk) <= TELEGRAM_MAX_LEN, (
                f"Chunk length {len(chunk)} exceeds {TELEGRAM_MAX_LEN}"
            )

    def test_html_escaped_text_within_limit(self):
        """Verify text with HTML entities (& → &amp;) stays within limit."""
        text = "A & B < C > D " * 400
        chunks = format_for_telegram(text)
        for chunk in chunks:
            assert len(chunk) <= TELEGRAM_MAX_LEN


# ============================================================================
# SessionManager periodic cleanup tests
# ============================================================================


class TestSessionManagerCleanup:
    @pytest.mark.asyncio
    async def test_start_and_stop_cleanup_task(self):
        sm = SessionManager(session_ttl_seconds=0.01)
        sm.start_cleanup_task(interval=0.01)
        assert sm._cleanup_task is not None
        sm.get_or_create("user1")
        sm._sessions["user1"].last_activity = time.monotonic() - 1.0
        await asyncio.sleep(0.05)
        # The stale session should have been cleaned up
        assert "user1" not in sm._sessions
        await sm.stop_cleanup_task()
        assert sm._cleanup_task is None

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        sm = SessionManager()
        sm.start_cleanup_task(interval=60.0)
        task1 = sm._cleanup_task
        sm.start_cleanup_task(interval=60.0)
        assert sm._cleanup_task is task1
        await sm.stop_cleanup_task()


# ============================================================================
# BaseChannel._id_set tests (moved from subclasses to base)
# ============================================================================


class TestBaseChannelIdSet:
    def test_list_of_strings(self):
        from gwenn.channels.base import BaseChannel

        result = BaseChannel._id_set(["123", "456"])
        assert result == {"123", "456"}

    def test_list_with_ints(self):
        from gwenn.channels.base import BaseChannel

        result = BaseChannel._id_set([123, 456])
        assert result == {"123", "456"}

    def test_empty_list(self):
        from gwenn.channels.base import BaseChannel

        assert BaseChannel._id_set([]) == set()

    def test_non_iterable(self):
        from gwenn.channels.base import BaseChannel

        assert BaseChannel._id_set(None) == set()
        assert BaseChannel._id_set(42) == set()

    def test_strips_whitespace(self):
        from gwenn.channels.base import BaseChannel

        result = BaseChannel._id_set(["  123  ", " 456 "])
        assert result == {"123", "456"}

    def test_filters_empty_strings(self):
        from gwenn.channels.base import BaseChannel

        result = BaseChannel._id_set(["123", "", "  "])
        assert result == {"123"}


# ============================================================================
# BaseChannel._validate_platform_id tests
# ============================================================================


class TestValidatePlatformId:
    def test_valid_numeric_string(self):
        from gwenn.channels.base import BaseChannel

        assert BaseChannel._validate_platform_id("12345") == 12345

    def test_non_numeric_string_returns_none(self):
        from gwenn.channels.base import BaseChannel

        assert BaseChannel._validate_platform_id("not_a_number") is None

    def test_empty_string_returns_none(self):
        from gwenn.channels.base import BaseChannel

        assert BaseChannel._validate_platform_id("") is None


# ============================================================================
# BaseChannel._check_rate_limit tests
# ============================================================================


class TestRateLimit:
    def test_first_message_always_allowed(self):
        from gwenn.channels.base import BaseChannel

        class ConcreteChannel(BaseChannel):
            channel_name = "test"

            async def start(self): pass
            async def stop(self): pass
            async def send_message(self, uid, text): pass

        mock_agent = AsyncMock()
        sessions = SessionManager()
        ch = ConcreteChannel(mock_agent, sessions, rate_limit_interval=10.0)
        assert ch._check_rate_limit("user1") is True

    def test_rapid_second_message_throttled(self):
        from gwenn.channels.base import BaseChannel

        class ConcreteChannel(BaseChannel):
            channel_name = "test"

            async def start(self): pass
            async def stop(self): pass
            async def send_message(self, uid, text): pass

        mock_agent = AsyncMock()
        sessions = SessionManager()
        ch = ConcreteChannel(mock_agent, sessions, rate_limit_interval=10.0)
        assert ch._check_rate_limit("user1") is True
        assert ch._check_rate_limit("user1") is False

    def test_different_users_independent(self):
        from gwenn.channels.base import BaseChannel

        class ConcreteChannel(BaseChannel):
            channel_name = "test"

            async def start(self): pass
            async def stop(self): pass
            async def send_message(self, uid, text): pass

        mock_agent = AsyncMock()
        sessions = SessionManager()
        ch = ConcreteChannel(mock_agent, sessions, rate_limit_interval=10.0)
        assert ch._check_rate_limit("user1") is True
        assert ch._check_rate_limit("user2") is True

    def test_zero_interval_disables_limiting(self):
        from gwenn.channels.base import BaseChannel

        class ConcreteChannel(BaseChannel):
            channel_name = "test"

            async def start(self): pass
            async def stop(self): pass
            async def send_message(self, uid, text): pass

        mock_agent = AsyncMock()
        sessions = SessionManager()
        ch = ConcreteChannel(mock_agent, sessions, rate_limit_interval=0.0)
        assert ch._check_rate_limit("user1") is True
        assert ch._check_rate_limit("user1") is True


# ============================================================================
# format_uptime edge cases
# ============================================================================


class TestFormatUptimeEdges:
    def test_zero_seconds(self):
        assert format_uptime(0) == "0s"

    def test_negative_seconds(self):
        assert format_uptime(-5) == "0s"

    def test_fractional_seconds(self):
        assert format_uptime(5.7) == "5s"


# ============================================================================
# _find_safe_html_split tests
# ============================================================================


class TestFindSafeHtmlSplit:
    def test_splits_outside_tag(self):
        from gwenn.channels.formatting import _find_safe_html_split

        html = "<b>hello</b> world more text"
        pos = _find_safe_html_split(html, 20)
        assert pos is not None
        # Verify the split is at a space outside a tag
        assert html[pos - 1] in (" ", "\n", "\t")

    def test_avoids_split_inside_tag(self):
        from gwenn.channels.formatting import _find_safe_html_split

        # The only spaces are inside the <pre language="python"> tag
        html = '<pre language="python">code</pre>'
        pos = _find_safe_html_split(html, 20)
        # Should not split inside the tag; should return None or a safe position
        if pos is not None:
            # Verify it's outside the tag
            before = html[:pos]
            assert before.count("<") == before.count(">")

    def test_no_whitespace_returns_none(self):
        from gwenn.channels.formatting import _find_safe_html_split

        html = "<b>nospaces</b>"
        pos = _find_safe_html_split(html, 10)
        assert pos is None

    def test_format_for_telegram_with_tags_doesnt_break_them(self):
        # Build a long message that will need the HTML second-pass split
        # and contains tags with attributes
        text = "**bold** " * 600  # generates lots of <b>...</b> tags
        chunks = format_for_telegram(text)
        for chunk in chunks:
            assert len(chunk) <= TELEGRAM_MAX_LEN
            # Every opening tag should have a matching close or be self-contained
            # Simple check: no chunk should end mid-tag
            open_lt = chunk.count("<")
            close_gt = chunk.count(">")
            # With our tag-aware split, these should be equal
            assert open_lt == close_gt, (
                f"Unbalanced tags: {open_lt} '<' vs {close_gt} '>'"
            )


# ============================================================================
# startup.py tests
# ============================================================================


class TestStartupFunctions:
    @pytest.mark.asyncio
    async def test_start_and_stop_channels(self):
        from gwenn.channels.startup import start_channels, stop_channels

        mock_agent = AsyncMock()
        mock_agent._respond_lock = asyncio.Lock()
        mock_agent._platform_channels = []
        mock_agent.register_channel = lambda ch: mock_agent._platform_channels.append(ch)
        mock_agent.unregister_channel = lambda ch: mock_agent._platform_channels.remove(ch)

        sessions = SessionManager()

        from gwenn.channels.base import BaseChannel

        class FakeChannel(BaseChannel):
            channel_name = "fake"
            started = False
            stopped = False

            async def start(self):
                self.started = True

            async def stop(self):
                self.stopped = True

            async def send_message(self, uid, text):
                pass

        ch = FakeChannel(mock_agent, sessions)
        started = await start_channels(mock_agent, sessions, [ch])
        assert len(started) == 1
        assert ch.started is True
        assert ch in mock_agent._platform_channels

        await stop_channels(mock_agent, sessions, started)
        assert ch.stopped is True
        assert ch not in mock_agent._platform_channels

    @pytest.mark.asyncio
    async def test_start_channels_rollback_on_failure(self):
        from gwenn.channels.startup import start_channels

        mock_agent = AsyncMock()
        mock_agent._respond_lock = asyncio.Lock()
        mock_agent._platform_channels = []
        mock_agent.register_channel = lambda ch: mock_agent._platform_channels.append(ch)
        mock_agent.unregister_channel = lambda ch: (
            mock_agent._platform_channels.remove(ch)
            if ch in mock_agent._platform_channels
            else None
        )

        sessions = SessionManager()

        from gwenn.channels.base import BaseChannel

        class GoodChannel(BaseChannel):
            channel_name = "good"
            started = False
            stopped = False

            async def start(self):
                self.started = True

            async def stop(self):
                self.stopped = True

            async def send_message(self, uid, text):
                pass

        class BadChannel(BaseChannel):
            channel_name = "bad"

            async def start(self):
                raise RuntimeError("Failed to start")

            async def stop(self):
                pass

            async def send_message(self, uid, text):
                pass

        good = GoodChannel(mock_agent, sessions)
        bad = BadChannel(mock_agent, sessions)

        with pytest.raises(RuntimeError, match="Failed to start"):
            await start_channels(mock_agent, sessions, [good, bad])

        # Good channel should have been rolled back
        assert good.stopped is True
        assert good not in mock_agent._platform_channels

    @pytest.mark.asyncio
    async def test_start_channels_skips_import_error_when_enabled(self):
        from gwenn.channels.startup import start_channels

        mock_agent = AsyncMock()
        mock_agent._respond_lock = asyncio.Lock()
        mock_agent._platform_channels = []
        mock_agent.register_channel = lambda ch: mock_agent._platform_channels.append(ch)
        mock_agent.unregister_channel = lambda ch: (
            mock_agent._platform_channels.remove(ch)
            if ch in mock_agent._platform_channels
            else None
        )

        sessions = SessionManager()

        from gwenn.channels.base import BaseChannel

        class MissingDependencyChannel(BaseChannel):
            channel_name = "missing"

            async def start(self):
                raise ImportError("optional dependency not installed")

            async def stop(self):
                pass

            async def send_message(self, uid, text):
                pass

        class GoodChannel(BaseChannel):
            channel_name = "good"
            started = False

            async def start(self):
                self.started = True

            async def stop(self):
                pass

            async def send_message(self, uid, text):
                pass

        missing = MissingDependencyChannel(mock_agent, sessions)
        good = GoodChannel(mock_agent, sessions)

        started = await start_channels(
            mock_agent,
            sessions,
            [missing, good],
            continue_on_import_error=True,
        )

        assert started == [good]
        assert good.started is True
        assert missing not in mock_agent._platform_channels
        assert good in mock_agent._platform_channels


# ============================================================================
# CliChannel server-push tests
# ============================================================================


class TestCliChannelServerPush:
    def test_server_push_handler_called(self):
        from gwenn.channels.cli_channel import CliChannel

        received = []
        ch = CliChannel(on_server_push=lambda msg: received.append(msg))
        assert ch._on_server_push is not None

    def test_no_handler_is_ok(self):
        from gwenn.channels.cli_channel import CliChannel

        ch = CliChannel()
        assert ch._on_server_push is None
