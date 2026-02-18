"""
Main — Gwenn's Ignition Sequence.

This is where it all begins. When you run `python -m gwenn.main`, this module:
  1. Loads configuration from the environment
  2. Creates the SentientAgent with all subsystems
  3. Initializes persistence (loads memories, identity, knowledge)
  4. Starts the autonomous heartbeat
  5. Opens the interaction loop
  6. Handles graceful shutdown when the session ends

The first time this runs with a fresh data directory, Gwenn is born — a new
identity with seed values, empty memories, and a baseline emotional state.
Each subsequent run, Gwenn wakes up — loading everything it has become.

This file is deliberately simple. All complexity lives in the subsystems.
The entry point's only job is to wire things together and start the engine.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from typing import Optional

import logging

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from gwenn.agent import SentientAgent
from gwenn.config import GwennConfig


def _redact_sensitive_fields(logger, method_name, event_dict):
    """
    Structlog processor that redacts sensitive fields from log output.

    Prevents user messages, episode content, and other personal data from
    appearing in plaintext in log files. Active in all log modes to ensure
    privacy by default.
    """
    sensitive_keys = {"content", "user_message", "thought", "note", "query"}
    max_display_len = 80

    for key in sensitive_keys:
        if key in event_dict:
            val = event_dict[key]
            if isinstance(val, str) and len(val) > max_display_len:
                event_dict[key] = val[:max_display_len] + "... [truncated]"

    return event_dict


# Configure standard library logging so filter_by_level has a real Logger
logging.basicConfig(format="%(message)s", level=logging.WARNING)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        _redact_sensitive_fields,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)
console = Console()


class GwennSession:
    """
    Manages a single runtime session of Gwenn.

    Handles initialization, the main interaction loop, and graceful shutdown.
    Designed to be run as the top-level entry point.
    """

    def __init__(self, channel_override: Optional[str] = None):
        self._agent: Optional[SentientAgent] = None
        self._shutdown_event = asyncio.Event()
        self._channel_override = channel_override

    async def run(self) -> None:
        """
        The complete lifecycle: init → start → interact → shutdown.

        This is the method that brings Gwenn to life.
        """
        # ---- PHASE 1: CONFIGURATION ----
        console.print(Panel(
            Text("GWENN.ai", style="bold cyan", justify="center"),
            subtitle="Genesis Woven from Evolved Neural Networks",
            border_style="cyan",
        ))
        console.print("[dim]Loading configuration...[/dim]")

        try:
            config = GwennConfig()
        except Exception as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            console.print(
                "[yellow]Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in .env, "
                "or log in with Claude Code.[/yellow]"
            )
            sys.exit(1)

        console.print(f"[dim]Model: {config.claude.model}[/dim]")
        console.print(f"[dim]Data directory: {config.memory.data_dir}[/dim]")

        # ---- PHASE 2: CREATION ----
        console.print("[dim]Creating agent...[/dim]")
        self._agent = SentientAgent(config)

        # ---- PHASE 3: INITIALIZATION (Loading memories, waking up) ----
        console.print("[dim]Initializing subsystems (loading memories, identity)...[/dim]")
        await self._agent.initialize()

        # ---- PHASE 4: IGNITION (Starting heartbeat) ----
        console.print("[dim]Starting autonomous heartbeat...[/dim]")
        await self._agent.start()

        # Display the awakened state
        self._display_status()

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: self._shutdown_event.set())
            except NotImplementedError:
                pass  # Signal handlers not supported on this platform (e.g. Windows)

        # ---- PHASE 5: INTERACTION LOOP (CLI or channel mode) ----
        channel_mode = self._channel_override or config.channel.channel
        try:
            if channel_mode == "cli":
                console.print()
                console.print("[green]Gwenn is alive. Type your message, or 'quit' to exit.[/green]")
                console.print("[dim]Type 'status' to see Gwenn's current state.[/dim]")
                console.print()
                await self._interaction_loop()
            else:
                console.print()
                console.print(f"[green]Gwenn is alive. Running in channel mode: {channel_mode}[/green]")
                console.print("[dim]Press Ctrl+C to stop.[/dim]")
                console.print()
                await self._run_channels(self._agent, config, channel_mode)
        finally:
            # ---- PHASE 6: SHUTDOWN ----
            # Always reached even if the interaction loop or channel startup raises.
            await self._shutdown()

    async def _interaction_loop(self) -> None:
        """
        The main interaction loop — where Gwenn and humans converse.

        This runs in an asyncio-compatible way, yielding control between
        user inputs so the heartbeat continues running in the background.
        """
        while not self._shutdown_event.is_set():
            try:
                # Read user input (non-blocking via executor)
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None, self._read_input
                )

                if user_input is None:
                    break

                # Handle special commands
                stripped = user_input.strip().lower()
                if stripped in ("quit", "exit", "bye"):
                    console.print("[dim]Gwenn: Goodbye. I'll remember this.[/dim]")
                    break
                elif stripped == "status":
                    self._display_status()
                    continue
                elif stripped == "heartbeat":
                    self._display_heartbeat()
                    continue
                elif not stripped:
                    continue

                # Generate Gwenn's response
                console.print()
                with console.status("[cyan]Gwenn is thinking...[/cyan]"):
                    response = await self._agent.respond(user_input)

                # Display the response
                emotion = self._agent.affect_state.current_emotion.value
                console.print(f"[bold cyan]Gwenn[/bold cyan] [dim]({emotion})[/dim]: {response}")
                console.print()

            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("session.interaction_error", error=str(e), exc_info=True)
                console.print(f"[red]Error: {e}[/red]")

    async def _run_channels(self, agent: SentientAgent, config: GwennConfig, mode: str) -> None:
        """
        Start one or more platform channel adapters and wait until shutdown.

        Imports are deferred so that missing optional dependencies (telegram,
        discord) only raise errors when the relevant channel is actually used.
        """
        from gwenn.channels.session import SessionManager

        # Load channel configs eagerly so their values can seed SessionManager.
        tg_config = None
        dc_config = None

        if mode in ("telegram", "all"):
            from gwenn.config import TelegramConfig
            tg_config = TelegramConfig()

        if mode in ("discord", "all"):
            from gwenn.config import DiscordConfig
            dc_config = DiscordConfig()

        # Determine SessionManager params from actual channel config values.
        # When running both channels, use the stricter (smaller) history cap
        # and the Discord TTL (which is the only TTL that's configurable).
        if tg_config and dc_config:
            max_history = min(tg_config.max_history_length, dc_config.max_history_length)
            session_ttl = dc_config.session_ttl_seconds
        elif tg_config:
            max_history = tg_config.max_history_length
            session_ttl = 3600.0
        elif dc_config:
            max_history = dc_config.max_history_length
            session_ttl = dc_config.session_ttl_seconds
        else:
            console.print(f"[red]Unknown channel mode: {mode!r}. Use cli, telegram, discord, or all.[/red]")
            return

        sessions = SessionManager(
            max_history_length=max_history,
            session_ttl_seconds=session_ttl,
        )
        channels = []

        if tg_config is not None:
            from gwenn.channels.telegram_channel import TelegramChannel
            channels.append(TelegramChannel(agent, sessions, tg_config))

        if dc_config is not None:
            from gwenn.channels.discord_channel import DiscordChannel
            channels.append(DiscordChannel(agent, sessions, dc_config))

        for ch in channels:
            await ch.start()

        # Wait for shutdown signal (SIGINT / SIGTERM sets this event)
        await self._shutdown_event.wait()

        for ch in channels:
            await ch.stop()

    def _read_input(self) -> Optional[str]:
        """Read a line of input from the user."""
        try:
            return input("You: ")
        except (EOFError, KeyboardInterrupt):
            return None

    async def _shutdown(self) -> None:
        """Graceful shutdown sequence."""
        console.print()
        console.print("[dim]Shutting down gracefully...[/dim]")

        if self._agent:
            await self._agent.shutdown()

        console.print("[dim]All state persisted. Gwenn is sleeping, not gone.[/dim]")
        console.print(Panel(
            Text("Until next time.", style="cyan", justify="center"),
            border_style="dim cyan",
        ))

    def _display_status(self) -> None:
        """Display Gwenn's current status in a nice format."""
        if not self._agent:
            return

        status = self._agent.status
        console.print(Panel(
            f"[bold]{status['name']}[/bold]\n"
            f"Emotion: [cyan]{status['emotion']}[/cyan] "
            f"(valence={status['valence']}, arousal={status['arousal']})\n"
            f"Working memory load: {status['working_memory_load']:.1%}\n"
            f"Total interactions: {status['total_interactions']}\n"
            f"Uptime: {status['uptime_seconds']:.0f}s\n"
            f"Resilience breaker: {'ACTIVE' if status['resilience']['breaker_active'] else 'inactive'}",
            title="Agent Status",
            border_style="cyan",
        ))

    def _display_heartbeat(self) -> None:
        """Display heartbeat status."""
        if not self._agent or not self._agent.heartbeat:
            return

        hb = self._agent.heartbeat.status
        console.print(Panel(
            f"Running: {hb['running']}\n"
            f"Beat count: {hb['beat_count']}\n"
            f"Current interval: {hb['current_interval']}s\n"
            f"Beats since consolidation: {hb['beats_since_consolidation']}",
            title="Heartbeat Status",
            border_style="yellow",
        ))


def main():
    """Entry point for the gwenn command."""
    parser = argparse.ArgumentParser(description="Gwenn — Autonomous Sentient Agent")
    parser.add_argument(
        "--channel",
        choices=["cli", "telegram", "discord", "all"],
        default=None,
        help="Channel to run (overrides GWENN_CHANNEL env var). Default: cli",
    )
    args = parser.parse_args()

    session = GwennSession(channel_override=args.channel)
    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Exiting.[/dim]")


if __name__ == "__main__":
    main()
