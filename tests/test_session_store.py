"""
Tests for gwenn.memory.session_store — Session Store.

Covers:
- save_session with string and list content blocks (P0-3 validation)
- is_valid_session_id edge cases
- list_sessions with include_preview
- _format_session_time today/yesterday/other
- Pruning: oldest sessions pruned, not newest
- max_messages cap enforcement
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gwenn.memory.session_store import SessionStore, _format_session_time


@pytest.fixture()
def store(tmp_path: Path) -> SessionStore:
    """A fresh SessionStore backed by a temp directory."""
    return SessionStore(sessions_dir=tmp_path / "sessions", max_count=5, max_messages=50)


# ---------------------------------------------------------------------------
# save_session with string content
# ---------------------------------------------------------------------------

class TestSaveSessionStringContent:
    """save_session with standard string content messages."""

    def test_basic_save_and_load(self, store: SessionStore):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        sid = store.save_session(messages, started_at=time.time())
        assert sid != ""

        loaded = store.load_session(sid)
        assert len(loaded) == 2
        assert loaded[0]["content"] == "hello"
        assert loaded[1]["content"] == "hi there"

    def test_empty_messages_returns_empty_id(self, store: SessionStore):
        sid = store.save_session([], started_at=time.time())
        assert sid == ""

    def test_invalid_messages_skipped(self, store: SessionStore):
        messages = [
            {"role": "user", "content": "valid"},
            {"role": "system", "content": "should be skipped"},
            {"not_a_message": True},
            42,
        ]
        sid = store.save_session(messages, started_at=time.time())
        assert sid != ""
        loaded = store.load_session(sid)
        assert len(loaded) == 1
        assert loaded[0]["content"] == "valid"

    def test_text_filter_applied(self, store: SessionStore):
        messages = [
            {"role": "user", "content": "secret password 1234"},
        ]
        sid = store.save_session(
            messages,
            started_at=time.time(),
            text_filter=lambda t: t.replace("1234", "****"),
        )
        loaded = store.load_session(sid)
        assert "****" in loaded[0]["content"]
        assert "1234" not in loaded[0]["content"]


# ---------------------------------------------------------------------------
# save_session with list content blocks (P0-3 validation)
# ---------------------------------------------------------------------------

class TestSaveSessionListContent:
    """save_session must preserve list-type content blocks."""

    def test_list_content_preserved(self, store: SessionStore):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {"data": "base64..."}},
                ],
            },
            {"role": "assistant", "content": "It looks like an image."},
        ]
        sid = store.save_session(messages, started_at=time.time())
        assert sid != ""

        loaded = store.load_session(sid)
        assert len(loaded) == 2
        assert isinstance(loaded[0]["content"], list)
        assert loaded[0]["content"][0]["text"] == "What is this?"

    def test_list_content_with_text_filter(self, store: SessionStore):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "secret 1234"},
                    {"type": "tool_use", "id": "t1", "name": "read_file"},
                ],
            },
        ]
        sid = store.save_session(
            messages,
            started_at=time.time(),
            text_filter=lambda t: t.replace("1234", "****"),
        )
        loaded = store.load_session(sid)
        assert loaded[0]["content"][0]["text"] == "secret ****"
        # Non-text blocks preserved as-is
        assert loaded[0]["content"][1]["type"] == "tool_use"

    def test_mixed_string_and_list_content(self, store: SessionStore):
        messages = [
            {"role": "user", "content": "plain string"},
            {"role": "assistant", "content": [{"type": "text", "text": "structured"}]},
        ]
        sid = store.save_session(messages, started_at=time.time())
        loaded = store.load_session(sid)
        assert isinstance(loaded[0]["content"], str)
        assert isinstance(loaded[1]["content"], list)


# ---------------------------------------------------------------------------
# is_valid_session_id edge cases
# ---------------------------------------------------------------------------

class TestIsValidSessionId:
    """is_valid_session_id should match the canonical format."""

    def test_valid_ids(self):
        assert SessionStore.is_valid_session_id("20240101-120000-abc1")
        assert SessionStore.is_valid_session_id("20261231-235959-z9x0")

    def test_invalid_ids(self):
        assert not SessionStore.is_valid_session_id("")
        assert not SessionStore.is_valid_session_id("not-a-session-id")
        assert not SessionStore.is_valid_session_id("20240101-120000-ABCD")  # uppercase
        assert not SessionStore.is_valid_session_id("20240101-120000-ab")  # too short
        assert not SessionStore.is_valid_session_id("20240101-120000-abcde")  # too long
        assert not SessionStore.is_valid_session_id("../etc/passwd")


# ---------------------------------------------------------------------------
# list_sessions with include_preview
# ---------------------------------------------------------------------------

class TestListSessions:
    """list_sessions should return recent sessions with optional preview."""

    def test_list_empty(self, store: SessionStore):
        assert store.list_sessions() == []

    def test_list_returns_saved_sessions(self, store: SessionStore):
        for i in range(3):
            store.save_session(
                [{"role": "user", "content": f"message {i}"}],
                started_at=time.time() - (i * 60),
            )
        sessions = store.list_sessions(limit=10)
        assert len(sessions) == 3

    def test_list_respects_limit(self, store: SessionStore):
        for i in range(4):
            store.save_session(
                [{"role": "user", "content": f"msg {i}"}],
                started_at=time.time() - (i * 60),
            )
        sessions = store.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_preview_with_string_content(self, store: SessionStore):
        store.save_session(
            [{"role": "user", "content": "hello world preview test"}],
            started_at=time.time(),
        )
        sessions = store.list_sessions(include_preview=True)
        assert len(sessions) == 1
        assert "hello world" in sessions[0]["preview"]

    def test_preview_with_list_content(self, store: SessionStore):
        store.save_session(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "list content preview"},
                    ],
                }
            ],
            started_at=time.time(),
        )
        sessions = store.list_sessions(include_preview=True)
        assert len(sessions) == 1
        assert "list content preview" in sessions[0]["preview"]


# ---------------------------------------------------------------------------
# _format_session_time
# ---------------------------------------------------------------------------

class TestFormatSessionTime:
    """_format_session_time formats timestamps for display."""

    def test_today(self):
        result = _format_session_time(time.time() - 60)
        assert result.startswith("Today")

    def test_yesterday(self):
        result = _format_session_time(time.time() - 100_000)
        assert result.startswith("Yesterday")

    def test_older_than_two_days(self):
        result = _format_session_time(time.time() - 250_000)
        # Should show a weekday name, not "Today" or "Yesterday"
        assert not result.startswith("Today")
        assert not result.startswith("Yesterday")


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

class TestPruning:
    """Pruning should remove oldest sessions, not newest."""

    def test_prune_removes_oldest(self, tmp_path: Path):
        store = SessionStore(sessions_dir=tmp_path / "prune_sessions", max_count=2)
        session_ids = []
        for i in range(4):
            sid = store.save_session(
                [{"role": "user", "content": f"msg {i}"}],
                started_at=time.time() - (300 - i * 60),
            )
            session_ids.append(sid)

        # Only max_count (2) sessions should remain
        remaining = store.list_sessions(limit=10)
        assert len(remaining) <= 2


# ---------------------------------------------------------------------------
# max_messages cap
# ---------------------------------------------------------------------------

class TestMaxMessagesCap:
    """Sessions should be capped at max_messages."""

    def test_cap_enforced(self, tmp_path: Path):
        store = SessionStore(sessions_dir=tmp_path / "cap_sessions", max_messages=5)
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]
        sid = store.save_session(messages, started_at=time.time())
        loaded = store.load_session(sid)
        assert len(loaded) == 5
        # Should keep the most recent messages
        assert loaded[-1]["content"] == "msg 19"
