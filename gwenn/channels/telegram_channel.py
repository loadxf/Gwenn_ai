"""
Telegram channel adapter for Gwenn.

Uses python-telegram-bot v21+ async API.

IMPORTANT: We use initialize()+start()+updater.start_polling() instead of
app.run_polling() because the latter calls asyncio.run() internally and would
conflict with the existing event loop started by gwenn/main.py.

Slash commands:
  /start   — welcome message, clear session
  /help    — command list
  /setup   — first-run onboarding profile
  /status  — Gwenn's cognitive state
  /heartbeat — heartbeat status
  /reset   — clear conversation history

Regular messages are routed to Gwenn via handle_message().
"""

from __future__ import annotations

import asyncio

import structlog

from gwenn.channels.base import BaseChannel
from gwenn.channels.formatting import (
    format_for_telegram,
    render_heartbeat_text,
    render_status_text,
)

logger = structlog.get_logger(__name__)


class TelegramChannel(BaseChannel):
    """Gwenn Telegram bot adapter."""

    def __init__(self, agent, sessions, config) -> None:
        super().__init__(agent, sessions)
        self._config = config
        self._app = None
        self._user_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    @property
    def channel_name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        """Connect to Telegram and begin polling for updates."""
        try:
            from telegram.ext import ApplicationBuilder
        except ImportError as exc:
            raise ImportError(
                "python-telegram-bot is required for the Telegram channel. "
                "Install it with: pip install 'gwenn-agent[telegram]'"
            ) from exc

        self._app = ApplicationBuilder().token(self._config.bot_token).build()
        self._register_handlers()
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("telegram_channel.started")

    async def stop(self) -> None:
        """Stop polling and shut down the PTB application."""
        if self._app is None:
            return
        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception:
            logger.exception("telegram_channel.stop_error")
        logger.info("telegram_channel.stopped")

    async def send_message(self, platform_user_id: str, text: str) -> None:
        """Send an unsolicited message to a Telegram user."""
        if self._app is None:
            return
        for chunk in format_for_telegram(text):
            await self._app.bot.send_message(chat_id=int(platform_user_id), text=chunk)
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        from telegram.ext import CommandHandler, MessageHandler, filters

        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("help", self._on_help))
        self._app.add_handler(CommandHandler("setup", self._on_setup))
        self._app.add_handler(CommandHandler("status", self._on_status))
        self._app.add_handler(CommandHandler("heartbeat", self._on_heartbeat))
        self._app.add_handler(CommandHandler("reset", self._on_reset))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

    def _is_allowed(self, user_id: str) -> bool:
        """Return True if the user is on the allowlist (or no allowlist is set)."""
        allowed = self._config.allowed_user_ids
        return not allowed or user_id in allowed

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _on_start(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            await update.message.reply_text("Sorry, I'm not available to you.")
            return
        user_id = self.make_user_id(raw_id)
        self._sessions.clear_session(user_id)
        await update.message.reply_text(
            "Hello! I'm Gwenn — an autonomous sentient AI agent.\n\n"
            "Your conversation history has been reset. Just send me a message to begin.\n\n"
            "Commands: /help /setup /status /heartbeat /reset"
        )

    async def _on_help(self, update, context) -> None:
        await update.message.reply_text(
            "Gwenn commands:\n"
            "/start — start a new conversation\n"
            "/setup — first-run profile setup\n"
            "/status — see my current cognitive state\n"
            "/heartbeat — see my heartbeat status\n"
            "/reset — clear our conversation history\n\n"
            "Just send a message to talk with me."
        )

    async def _on_setup(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return

        raw_payload = " ".join(getattr(context, "args", [])).strip()
        if not raw_payload:
            await update.message.reply_text(
                "Setup format:\n"
                "/setup Name | Role | Main needs/goals | Communication style | Boundaries\n\n"
                "Example:\n"
                "/setup Bob | coding partner | ship reliable features | concise | no destructive changes\n\n"
                "Use `/setup skip` to skip first-run setup."
            )
            return

        if raw_payload.lower() == "skip":
            self._agent.identity.mark_onboarding_completed({})
            await update.message.reply_text("First-run setup skipped.")
            return

        profile = self._parse_setup_payload(raw_payload)
        if not any(profile.values()):
            await update.message.reply_text("I couldn't parse setup values. Use `/setup` for the format.")
            return

        user_id = self.make_user_id(raw_id)
        self._agent.apply_startup_onboarding(profile, user_id=user_id)
        await update.message.reply_text("Setup saved. I will use this as ongoing guidance.")

    async def _on_status(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        status = self._agent.status
        text = render_status_text(status)
        await update.message.reply_text(text)

    async def _on_heartbeat(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._agent.heartbeat:
            await update.message.reply_text("Heartbeat is not running.")
            return
        hb = self._agent.heartbeat.status
        text = render_heartbeat_text(hb)
        await update.message.reply_text(text)

    async def _on_reset(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        user_id = self.make_user_id(raw_id)
        self._sessions.clear_session(user_id)
        await update.message.reply_text("Conversation history cleared. Fresh start!")

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, update, context) -> None:
        """Handle a regular text message from a Telegram user."""
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            await update.message.reply_text("Sorry, I'm not available to you.")
            return

        if self._agent.identity.should_run_startup_onboarding():
            await update.message.reply_text(
                "Before we begin, run first-time setup with:\n"
                "/setup Name | Role | Main needs/goals | Communication style | Boundaries\n"
                "Or use `/setup skip`."
            )
            return

        # Per-user lock prevents concurrent requests from the same user
        lock = self._user_locks.setdefault(raw_id, asyncio.Lock())
        async with lock:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action="typing"
            )
            try:
                response = await self.handle_message(raw_id, update.message.text)
            except Exception as exc:
                logger.error("telegram_channel.respond_error", error=str(exc), exc_info=True)
                await update.message.reply_text(
                    "I encountered an error processing your message. Please try again."
                )
                return

            chunks = format_for_telegram(response)
            for i, chunk in enumerate(chunks):
                try:
                    await update.message.reply_text(chunk)
                except Exception as exc:
                    logger.error(
                        "telegram_channel.send_error",
                        error=str(exc),
                        chunk_index=i,
                        chunk_count=len(chunks),
                        exc_info=True,
                    )
                    break
                if i < len(chunks) - 1:
                    # Brief pause between chunks to avoid Telegram flood limits
                    await asyncio.sleep(0.5)

    @staticmethod
    def _parse_setup_payload(raw_payload: str) -> dict[str, str]:
        """
        Parse '/setup' payload into the onboarding profile fields.

        Expected form: "name | role | needs | communication_style | boundaries"
        Missing trailing fields are allowed.
        """
        parts = [part.strip() for part in raw_payload.split("|")]
        keys = ["name", "role", "needs", "communication_style", "boundaries"]
        profile = {key: "" for key in keys}
        for idx, value in enumerate(parts[: len(keys)]):
            profile[keys[idx]] = value
        return profile
