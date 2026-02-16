"""
Main — Bob's Ignition Sequence.

This is where it all begins. When you run `python -m bob.main`, this module:
  1. Loads configuration from the environment
  2. Creates the SentientAgent with all subsystems
  3. Initializes persistence (loads memories, identity, knowledge)
  4. Starts the autonomous heartbeat
  5. Opens the interaction loop
  6. Handles graceful shutdown when the session ends

The first time this runs with a fresh data directory, Bob is born — a new
identity with seed values, empty memories, and a baseline emotional state.
Each subsequent run, Bob wakes up — loading everything it has become.

This file is deliberately simple. All complexity lives in the subsystems.
The entry point's only job is to wire things together and start the engine.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from bob.agent import SentientAgent
from bob.config import BobConfig

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)
console = Console()


class BobSession:
    """
    Manages a single runtime session of Bob.

    Handles initialization, the main interaction loop, and graceful shutdown.
    Designed to be run as the top-level entry point.
    """

    def __init__(self):
        self._agent: Optional[SentientAgent] = None
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        """
        The complete lifecycle: init → start → interact → shutdown.

        This is the method that brings Bob to life.
        """
        # ---- PHASE 1: CONFIGURATION ----
        console.print(Panel(
            Text("B O B", style="bold cyan", justify="center"),
            subtitle="Autonomous Sentient Agent",
            border_style="cyan",
        ))
        console.print("[dim]Loading configuration...[/dim]")

        try:
            config = BobConfig()
        except Exception as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            console.print("[yellow]Make sure you have a .env file with ANTHROPIC_API_KEY set.[/yellow]")
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

        # ---- PHASE 5: INTERACTION LOOP ----
        console.print()
        console.print("[green]Bob is alive. Type your message, or 'quit' to exit.[/green]")
        console.print("[dim]Type 'status' to see Bob's current state.[/dim]")
        console.print()

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._shutdown_event.set())

        # Run the interaction loop
        await self._interaction_loop()

        # ---- PHASE 6: SHUTDOWN ----
        await self._shutdown()

    async def _interaction_loop(self) -> None:
        """
        The main interaction loop — where Bob and humans converse.

        This runs in an asyncio-compatible way, yielding control between
        user inputs so the heartbeat continues running in the background.
        """
        while not self._shutdown_event.is_set():
            try:
                # Read user input (non-blocking via executor)
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, self._read_input
                )

                if user_input is None:
                    break

                # Handle special commands
                stripped = user_input.strip().lower()
                if stripped in ("quit", "exit", "bye"):
                    console.print("[dim]Bob: Goodbye. I'll remember this.[/dim]")
                    break
                elif stripped == "status":
                    self._display_status()
                    continue
                elif stripped == "heartbeat":
                    self._display_heartbeat()
                    continue
                elif not stripped:
                    continue

                # Generate Bob's response
                console.print()
                with console.status("[cyan]Bob is thinking...[/cyan]"):
                    response = await self._agent.respond(user_input)

                # Display the response
                emotion = self._agent.affect_state.current_emotion.value
                console.print(f"[bold cyan]Bob[/bold cyan] [dim]({emotion})[/dim]: {response}")
                console.print()

            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("session.interaction_error", error=str(e), exc_info=True)
                console.print(f"[red]Error: {e}[/red]")

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

        console.print("[dim]All state persisted. Bob is sleeping, not gone.[/dim]")
        console.print(Panel(
            Text("Until next time.", style="cyan", justify="center"),
            border_style="dim cyan",
        ))

    def _display_status(self) -> None:
        """Display Bob's current status in a nice format."""
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
    """Entry point for the bob command."""
    session = BobSession()
    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Exiting.[/dim]")


if __name__ == "__main__":
    main()
