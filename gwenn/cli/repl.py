"""REPL launcher — wraps GwennSession from main.py.

The interactive REPL is the default mode when `gwenn` is invoked without
a subcommand. This module provides the bridge between the Click CLI and
the existing GwennSession class.
"""

from __future__ import annotations

import asyncio
from typing import Any


def run_repl(ctx_obj: dict[str, Any] | None = None) -> None:
    """Launch the interactive REPL.

    Delegates to the existing GwennSession in gwenn.main. The GwennSession
    handles onboarding, slash commands, daemon connection, and all
    interactive features.
    """
    from gwenn.main import GwennSession, configure_logging

    configure_logging()

    session = GwennSession()
    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        pass
