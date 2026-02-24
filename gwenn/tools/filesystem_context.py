"""
Filesystem Context — Per-Task Path Scoping.

A ``ContextVar`` controls which filesystem paths the current asyncio task may
access through the ``read_file`` / ``write_file`` tools.

Semantics:
  ``None``  (default)       — **unrestricted**.  The main agent runs with the
                               default and can read/write any path the OS
                               process has permission to touch.
  ``(Path(...), ...)``      — **scoped**.  Subagent runners set this to the
                               resolved ``filesystem_access`` directories so
                               the subagent can only operate within those
                               directories.
  ``()``  (empty tuple)     — **denied**.  A subagent spawned without any
                               ``filesystem_paths`` has no access.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path

ALLOWED_FS_PATHS: ContextVar[tuple[Path, ...] | None] = ContextVar(
    "ALLOWED_FS_PATHS", default=None
)


def validate_path(
    requested: str,
    *,
    require_exists: bool = False,
) -> tuple[Path, str | None]:
    """Validate a requested path against the current task's access scope.

    Returns ``(resolved_path, None)`` on success or ``(Path(), error_msg)``
    on failure.
    """
    allowed = ALLOWED_FS_PATHS.get()

    # --- Unrestricted (main agent) ---
    if allowed is None:
        try:
            resolved = Path(os.path.realpath(requested))
        except (OSError, ValueError) as exc:
            return Path(), f"Invalid path: {exc}"
        if require_exists and not resolved.exists():
            return Path(), f"File not found: '{resolved}'."
        return resolved, None

    # --- Denied (empty tuple — no paths granted) ---
    if not allowed:
        return Path(), "Filesystem access denied: no allowed paths configured."

    # --- Scoped (subagent with explicit allowed directories) ---
    try:
        resolved = Path(requested).resolve()
    except (OSError, ValueError) as exc:
        return Path(), f"Invalid path: {exc}"

    # Check that the resolved path falls under at least one allowed directory
    if not any(
        resolved == adir or _is_relative_to(resolved, adir) for adir in allowed
    ):
        return Path(), (
            f"Access denied: '{resolved}' is not within any allowed path."
        )

    # Re-resolve via os.path.realpath to catch symlink escapes
    real = Path(os.path.realpath(resolved))
    if not any(
        real == adir or _is_relative_to(real, adir) for adir in allowed
    ):
        return Path(), (
            "Access denied: symlink resolves outside allowed paths."
        )

    if require_exists and not real.exists():
        return Path(), f"File not found: '{real}'."

    return real, None


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return True if *path* is relative to *parent*."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
