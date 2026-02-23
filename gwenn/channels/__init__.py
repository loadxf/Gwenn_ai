"""
Gwenn channel adapters — platform-specific bots that front the shared SentientAgent.

Usage (from gwenn/main.py or gwenn/daemon.py):
    from gwenn.channels.startup import build_channels, run_channels_until_shutdown

Telegram and discord.py are optional dependencies; importing this package does
not trigger those imports.  They are loaded lazily inside their respective
channel modules and the startup helper.
"""

from gwenn.channels.base import BaseChannel
from gwenn.channels.cli_channel import CliChannel, DaemonNotRunningError
from gwenn.channels.session import SessionManager, UserSession
from gwenn.channels.formatting import (
    format_for_discord,
    format_for_telegram,
    markdown_to_telegram_html,
    split_message,
    strip_html_tags,
)
from gwenn.channels.startup import (
    build_channels,
    run_channels_until_shutdown,
    start_channels,
    stop_channels,
)

__all__ = [
    "BaseChannel",
    "CliChannel",
    "DaemonNotRunningError",
    "SessionManager",
    "UserSession",
    "split_message",
    "format_for_telegram",
    "format_for_discord",
    "markdown_to_telegram_html",
    "strip_html_tags",
    "build_channels",
    "run_channels_until_shutdown",
    "start_channels",
    "stop_channels",
]
