"""
Gwenn channel adapters — platform-specific bots that front the shared SentientAgent.

Usage (from gwenn/main.py):
    from gwenn.channels.session import SessionManager
    from gwenn.channels.telegram_channel import TelegramChannel
    from gwenn.channels.discord_channel import DiscordChannel

Telegram and discord.py are optional dependencies; importing this package does
not trigger those imports.  They are loaded lazily inside their respective
channel modules.
"""

from gwenn.channels.base import BaseChannel
from gwenn.channels.session import SessionManager, UserSession
from gwenn.channels.formatting import (
    format_for_discord,
    format_for_telegram,
    split_message,
)

__all__ = [
    "BaseChannel",
    "SessionManager",
    "UserSession",
    "split_message",
    "format_for_telegram",
    "format_for_discord",
]
