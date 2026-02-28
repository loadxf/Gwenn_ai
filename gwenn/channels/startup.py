"""
Shared channel startup/shutdown helper.

Centralises the logic for creating a SessionManager, instantiating Telegram
and Discord channel adapters, starting them, and tearing them down.  Both
``gwenn/main.py`` and ``gwenn/daemon.py`` delegate to this module so there
is exactly one source of truth for the channel lifecycle.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent
    from gwenn.channels.base import BaseChannel
    from gwenn.channels.session import SessionManager

logger = structlog.get_logger(__name__)


def build_channels(
    agent: "SentientAgent",
    *,
    channel_list: list[str],
) -> tuple["SessionManager", list["BaseChannel"]]:
    """
    Build a SessionManager and channel adapters for the requested platforms.

    Channel configs are loaded lazily so that missing optional dependencies
    (python-telegram-bot, discord.py) only raise errors when actually needed.

    Returns ``(sessions, channels)`` where *channels* may be empty if no
    valid configs could be loaded.
    """
    from gwenn.channels.session import SessionManager

    tg_config: Any = None
    dc_config: Any = None

    if "telegram" in channel_list:
        try:
            from gwenn.config import TelegramConfig

            tg_config = TelegramConfig()
        except Exception as e:
            logger.error("channels.telegram_config_failed", error=str(e))

    if "discord" in channel_list:
        try:
            from gwenn.config import DiscordConfig

            dc_config = DiscordConfig()
        except Exception as e:
            logger.error("channels.discord_config_failed", error=str(e))

    # Determine SessionManager params from actual channel config values.
    # When running both channels, use the stricter (smaller) history cap
    # and the longer of the two TTLs so sessions aren't evicted prematurely
    # for either platform's users.
    if tg_config and dc_config:
        max_history = min(tg_config.max_history_length, dc_config.max_history_length)
        session_ttl = max(tg_config.session_ttl_seconds, dc_config.session_ttl_seconds)
    elif tg_config:
        max_history = tg_config.max_history_length
        session_ttl = tg_config.session_ttl_seconds
    elif dc_config:
        max_history = dc_config.max_history_length
        session_ttl = dc_config.session_ttl_seconds
    else:
        return SessionManager(), []

    sessions = SessionManager(
        max_history_length=max_history,
        session_ttl_seconds=session_ttl,
    )
    channels: list[BaseChannel] = []

    if tg_config is not None:
        from gwenn.channels.telegram_channel import TelegramChannel

        channels.append(TelegramChannel(agent, sessions, tg_config))

    if dc_config is not None:
        from gwenn.channels.discord_channel import DiscordChannel

        channels.append(DiscordChannel(agent, sessions, dc_config))

    return sessions, channels


async def start_channels(
    agent: "SentientAgent",
    sessions: "SessionManager",
    channels: list["BaseChannel"],
    *,
    continue_on_start_error: bool = False,
) -> list["BaseChannel"]:
    """
    Start channel adapters, register them on the agent, and begin session cleanup.

    Returns the list of successfully started channels.  On partial failure the
    already-started channels are rolled back.

    When ``continue_on_start_error`` is True, channels that fail during
    startup are skipped so other channels can still start.
    """
    sessions.start_cleanup_task()
    started: list[BaseChannel] = []
    try:
        for ch in channels:
            try:
                await ch.start()
            except Exception as exc:
                if not continue_on_start_error:
                    raise
                logger.error(
                    "channels.start_skipped",
                    channel=ch.channel_name,
                    error=str(exc),
                )
                continue
            started.append(ch)
            agent.register_channel(ch)
    except Exception:
        for ch in reversed(started):
            agent.unregister_channel(ch)
            try:
                await ch.stop()
            except Exception:
                logger.exception("channels.start_rollback_stop_failed")
        await sessions.stop_cleanup_task()
        raise
    if not started:
        await sessions.stop_cleanup_task()
    return started


async def stop_channels(
    agent: "SentientAgent",
    sessions: "SessionManager",
    started_channels: list["BaseChannel"],
) -> None:
    """Stop all started channels, unregister them, and cancel session cleanup."""
    await sessions.stop_cleanup_task()
    for ch in reversed(started_channels):
        agent.unregister_channel(ch)
        try:
            await ch.stop()
        except Exception:
            logger.exception("channels.stop_failed")


async def run_channels_until_shutdown(
    agent: "SentientAgent",
    sessions: "SessionManager",
    channels: list["BaseChannel"],
    shutdown_event: asyncio.Event,
    *,
    continue_on_start_error: bool = False,
) -> None:
    """Start channels, wait for *shutdown_event*, then tear everything down."""
    started = await start_channels(
        agent,
        sessions,
        channels,
        continue_on_start_error=continue_on_start_error,
    )
    if not started:
        return
    try:
        await shutdown_event.wait()
    finally:
        await stop_channels(agent, sessions, started)
