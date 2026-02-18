"""
Session Store — Persistent CLI conversation history for /resume.

Saves and loads raw conversation histories (list of role/content dicts) to
gwenn_data/sessions/.  This is complementary to episodic memory: episodic
memory stores structured events semantically; session_store saves the raw
Claude API message list so the context window can be fully restored.

Only uses: pathlib, json, time, random, string, structlog — no gwenn imports.
"""

from __future__ import annotations

import json
import re
import secrets
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)
SESSION_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[a-z0-9]{4}$")


class SessionStore:
    """
    Persistent store for CLI conversation sessions.

    Each session is saved as a JSON file in sessions_dir.  The store prunes
    old sessions to keep at most max_count files.
    """

    def __init__(
        self,
        sessions_dir: Path,
        max_count: int = 20,
        max_messages: int = 200,
    ) -> None:
        self.sessions_dir = sessions_dir
        self._sessions_dir_resolved = sessions_dir.resolve()
        self.max_count = max_count
        self.max_messages = max_messages
        # Create directory on first use
        sessions_dir.mkdir(parents=True, exist_ok=True)
        self._best_effort_chmod(sessions_dir, 0o700)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_session(
        self,
        messages: list[dict],
        started_at: float,
        text_filter: Callable[[str], str] | None = None,
    ) -> str:
        """
        Persist a conversation history to disk.

        Validates messages, caps at max_messages, writes JSON file, prunes
        old sessions.  Returns the session_id string.
        """
        # Validate: only keep well-formed role/content pairs
        valid: list[dict] = []
        for m in messages:
            if (
                not isinstance(m, dict)
                or m.get("role") not in ("user", "assistant")
                or not isinstance(m.get("content"), str)
            ):
                continue

            content = m["content"]
            if text_filter is not None:
                content = text_filter(content)

            valid.append({"role": m["role"], "content": content})
        if not valid:
            logger.debug("session_store.save_skipped_empty")
            return ""

        # Cap to recent max_messages
        if len(valid) > self.max_messages:
            valid = valid[-self.max_messages:]

        # Build session_id: YYYYMMDD-HHMMSS-xxxx
        ts_str = datetime.fromtimestamp(started_at).strftime("%Y%m%d-%H%M%S")
        alphabet = string.ascii_lowercase + string.digits
        suffix = "".join(secrets.choice(alphabet) for _ in range(4))
        session_id = f"{ts_str}-{suffix}"

        ended_at = time.time()

        payload = {
            "id": session_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "message_count": len(valid),
            "messages": valid,
        }

        out_path = self.sessions_dir / f"{session_id}.json"
        try:
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._best_effort_chmod(out_path, 0o600)
        except OSError as e:
            logger.error("session_store.write_failed", path=str(out_path), error=str(e))
            return ""

        logger.info(
            "session_store.saved",
            session_id=session_id,
            message_count=len(valid),
        )
        self._prune()
        return session_id

    def list_sessions(self, limit: int = 10, include_preview: bool = False) -> list[dict]:
        """
        Return recent sessions, newest first.

        Each entry: {id, started_at, ended_at, message_count, preview?}
        """
        files = sorted(
            self.sessions_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        results = []
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                entry = {
                    "id": f.stem,
                    "started_at": data.get("started_at", 0.0),
                    "ended_at": data.get("ended_at", 0.0),
                    "message_count": data.get("message_count", 0),
                }
                if include_preview:
                    preview = ""
                    for m in data.get("messages", []):
                        if m.get("role") == "user":
                            preview = m.get("content", "")[:80]
                            break
                    entry["preview"] = preview
                results.append(entry)
            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.warning("session_store.list_read_error", file=str(f), error=str(e))
        return results

    def load_session(self, session_id: str) -> list[dict]:
        """
        Load a session by ID and return validated message list.

        Raises FileNotFoundError if session does not exist.
        Skips malformed entries (logs warning).
        """
        if not self.is_valid_session_id(session_id):
            raise FileNotFoundError(f"Session not found: {session_id}")

        path = (self.sessions_dir / f"{session_id}.json").resolve()
        try:
            path.relative_to(self._sessions_dir_resolved)
        except ValueError as e:
            raise FileNotFoundError(f"Session not found: {session_id}") from e

        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise FileNotFoundError(f"Cannot read session {session_id}: {e}") from e

        raw = data.get("messages", [])
        valid = [
            m for m in raw
            if isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
        ]
        if len(valid) != len(raw):
            logger.warning(
                "session_store.corrupted_entries_skipped",
                session_id=session_id,
                skipped=len(raw) - len(valid),
            )
        logger.info("session_store.loaded", session_id=session_id, message_count=len(valid))
        return valid

    @staticmethod
    def is_valid_session_id(session_id: str) -> bool:
        """Return True when session_id matches the canonical filename format."""
        return bool(SESSION_ID_PATTERN.fullmatch(session_id))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Delete oldest session files beyond max_count."""
        files = sorted(
            self.sessions_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(files) - self.max_count
        if excess <= 0:
            return
        for f in files[:excess]:
            try:
                f.unlink()
                logger.debug("session_store.pruned", file=str(f))
            except OSError as e:
                logger.warning("session_store.prune_failed", file=str(f), error=str(e))

    @staticmethod
    def _best_effort_chmod(path: Path, mode: int) -> None:
        """Attempt to harden permissions without failing on unsupported filesystems."""
        try:
            path.chmod(mode)
        except OSError:
            logger.debug("session_store.chmod_skipped", path=str(path), mode=oct(mode))


def _format_session_time(ts: float) -> str:
    """Format a Unix timestamp for /resume display using local time."""
    now = time.time()
    diff = now - ts
    dt = datetime.fromtimestamp(ts)
    hour = dt.strftime("%I").lstrip("0") or "12"
    time_text = f"{hour}:{dt.strftime('%M %p')}"

    if diff < 86400:  # today
        return "Today " + time_text
    elif diff < 172800:  # yesterday
        return "Yesterday " + time_text
    else:
        return f"{dt.strftime('%A')} {time_text}"
