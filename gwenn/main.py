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
import time
from typing import Optional

import anthropic
import logging

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from gwenn.agent import SentientAgent
from gwenn.api.claude import CognitiveEngineInitError
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
        self._last_sigint_at = 0.0
        self._sigint_confirm_window_seconds = 1.25

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
                "[yellow]Set ANTHROPIC_API_KEY or "
                "ANTHROPIC_AUTH_TOKEN/CLAUDE_CODE_OAUTH_TOKEN in .env, "
                "or log in with Claude Code.[/yellow]"
            )
            sys.exit(1)

        console.print(f"[dim]Model: {config.claude.model}[/dim]")
        console.print(f"[dim]Data directory: {config.memory.data_dir}[/dim]")
        channel_mode = self._channel_override or config.channel.channel

        # ---- PHASE 2: CREATION ----
        console.print("[dim]Creating agent...[/dim]")
        try:
            self._agent = SentientAgent(config)
        except CognitiveEngineInitError as e:
            console.print(f"[red]Startup error: {e}[/red]")
            sys.exit(1)

        # ---- PHASE 3: INITIALIZATION (Loading memories, waking up) ----
        console.print("[dim]Initializing subsystems (loading memories, identity)...[/dim]")
        await self._agent.initialize()
        await self._run_first_startup_onboarding_if_needed(channel_mode)

        # ---- PHASE 4: IGNITION (Starting heartbeat) ----
        console.print("[dim]Starting autonomous heartbeat...[/dim]")
        await self._agent.start()

        # Display the awakened state
        self._display_status()

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self._handle_sigint)
            loop.add_signal_handler(signal.SIGTERM, self._request_shutdown)
        except NotImplementedError:
            pass  # Signal handlers not supported on this platform (e.g. Windows)

        # ---- PHASE 5: INTERACTION LOOP (CLI or channel mode) ----
        try:
            if channel_mode == "cli":
                console.print()
                console.print(
                    "[green]Gwenn is alive. Type your message, or 'quit' to exit.[/green]"
                )
                console.print(
                    "[dim]Press Ctrl+C twice quickly to close gracefully.[/dim]"
                )
                console.print("[dim]Type 'status' to see Gwenn's current state.[/dim]")
                console.print()
                await self._interaction_loop()
            else:
                console.print()
                console.print(f"[green]Gwenn is alive. Running in channel mode: {channel_mode}[/green]")
                console.print("[dim]Press Ctrl+C twice quickly to stop.[/dim]")
                console.print()
                await self._run_channels(self._agent, config, channel_mode)
        finally:
            # ---- PHASE 6: SHUTDOWN ----
            # Always reached even if the interaction loop or channel startup raises.
            await self._shutdown()

    async def _run_first_startup_onboarding_if_needed(self, channel_mode: str) -> None:
        """
        Run first-time onboarding to capture the primary user's needs.

        This runs once for fresh identities when an interactive terminal is available.
        """
        if not self._agent:
            return
        if not self._agent.identity.should_run_startup_onboarding():
            return
        if not sys.stdin.isatty():
            logger.info(
                "session.onboarding_skipped_non_interactive",
                channel_mode=channel_mode,
            )
            return

        console.print()
        console.print(Panel(
            "First-time setup so I can tailor how I help you.\n"
            "You can keep answers short. Press Enter to skip any question.",
            title="Welcome",
            border_style="cyan",
        ))

        name = await self._prompt_startup_input("Your name (or what I should call you): ")
        role = await self._prompt_startup_input(
            "My primary role for you (e.g., coding partner, assistant, coach): "
        )
        needs = await self._prompt_startup_input(
            "Top things you want help with right now: "
        )
        communication_style = await self._prompt_startup_input(
            "Preferred communication style (brief, detailed, etc.): "
        )
        boundaries = await self._prompt_startup_input(
            "Any boundaries or preferences I should always respect: "
        )

        profile = {
            "name": name,
            "role": role,
            "needs": needs,
            "communication_style": communication_style,
            "boundaries": boundaries,
        }

        if not any(value.strip() for value in profile.values()):
            console.print("[dim]First-time setup skipped. I can learn your preferences over time.[/dim]")
            return

        self._agent.apply_startup_onboarding(profile, user_id="default_user")
        console.print("[green]Setup saved. I will use this as ongoing guidance.[/green]")
        console.print()

    async def _prompt_startup_input(self, prompt: str) -> str:
        """Prompt for one startup onboarding field without blocking heartbeat."""
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None, lambda: input(prompt)
            )
        except (EOFError, KeyboardInterrupt):
            return ""
        return response.strip()

    async def _interaction_loop(self) -> None:
        """
        The main interaction loop — where Gwenn and humans converse.

        This runs in an asyncio-compatible way, yielding control between
        user inputs so the heartbeat continues running in the background.
        """
        while not self._shutdown_event.is_set():
            try:
                # Read user input without blocking background loops or shutdown signals.
                user_input = await self._read_input()

                if user_input is None:
                    break

                self._last_sigint_at = 0.0

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
                self._handle_sigint()
                if self._shutdown_event.is_set():
                    break
            except anthropic.AuthenticationError as e:
                logger.error(
                    "session.auth_error",
                    error=str(e),
                    status=getattr(e, "status_code", None),
                )
                console.print(
                    "[red]Authentication failed.[/red] "
                    "[dim]Use ANTHROPIC_API_KEY (preferred) or a reachable "
                    "OAuth-compatible endpoint/token.[/dim]"
                )
            except anthropic.APIConnectionError as e:
                logger.warning("session.api_unreachable", error=str(e))
                console.print(
                    "[red]Cannot reach Claude API.[/red] "
                    "[dim]Check DNS/network access for the configured endpoint, "
                    "or switch auth method.[/dim]"
                )
            except anthropic.APIError as e:
                logger.error(
                    "session.api_error",
                    error=str(e),
                    status=getattr(e, "status_code", None),
                )
                console.print(
                    "[red]Claude API request failed.[/red] "
                    "[dim]See logs for details.[/dim]"
                )
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

    async def _read_input(self) -> Optional[str]:
        """
        Read a line of input from the user.

        Uses loop reader callbacks when available so SIGINT/SIGTERM can interrupt
        an active prompt immediately without waiting for Enter.
        """
        loop = asyncio.get_running_loop()
        try:
            fileno = sys.stdin.fileno()
            add_reader = loop.add_reader
        except (AttributeError, OSError):
            return await loop.run_in_executor(None, self._read_input_blocking)

        line_future = loop.create_future()
        shutdown_wait = asyncio.create_task(self._shutdown_event.wait())

        def _on_input_ready() -> None:
            if line_future.done():
                return
            try:
                line = sys.stdin.readline()
            except Exception as exc:
                line_future.set_exception(exc)
                return
            line_future.set_result(line)

        try:
            sys.stdout.write("You: ")
            sys.stdout.flush()
            add_reader(fileno, _on_input_ready)
        except (NotImplementedError, RuntimeError):
            shutdown_wait.cancel()
            await asyncio.gather(shutdown_wait, return_exceptions=True)
            return await loop.run_in_executor(None, self._read_input_blocking)

        try:
            done, _ = await asyncio.wait(
                {line_future, shutdown_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if shutdown_wait in done:
                return None

            line = line_future.result()
            if line == "":
                return None
            return line.rstrip("\n")
        finally:
            loop.remove_reader(fileno)
            if not line_future.done():
                line_future.cancel()
            shutdown_wait.cancel()
            await asyncio.gather(shutdown_wait, return_exceptions=True)

    def _read_input_blocking(self) -> Optional[str]:
        """Fallback input path for loops/platforms that don't support add_reader."""
        try:
            return input("You: ")
        except EOFError:
            return None

    def _request_shutdown(self) -> None:
        """Request graceful shutdown on SIGTERM or confirmed SIGINT."""
        self._shutdown_event.set()

    def _handle_sigint(self) -> None:
        """
        Require a quick double Ctrl+C to avoid accidental termination.

        First press warns; second press within the confirmation window triggers
        graceful shutdown.
        """
        if self._shutdown_event.is_set():
            return

        now = time.monotonic()
        if now - self._last_sigint_at <= self._sigint_confirm_window_seconds:
            console.print("\n[dim]Gwenn: Goodbye. I'll remember this.[/dim]")
            self._request_shutdown()
            self._last_sigint_at = 0.0
            return

        self._last_sigint_at = now
        console.print(
            "\n[dim]Press Ctrl+C again quickly to close Gwenn gracefully.[/dim]"
        )

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
        mood_summary = self._describe_mood(status["emotion"], status["valence"], status["arousal"])
        focus_summary = self._describe_focus_load(status["working_memory_load"])
        stress_summary = self._describe_stress_guardrail(status["resilience"])
        console.print(Panel(
            f"[bold]{status['name']}[/bold]\n"
            f"Mood: [cyan]{mood_summary}[/cyan]\n"
            f"Focus right now: {focus_summary}\n"
            f"Conversations handled: {status['total_interactions']}\n"
            f"Awake for: {self._format_uptime(status['uptime_seconds'])}\n"
            f"Stress guardrail: {stress_summary}",
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

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime in a human-friendly way."""
        total = max(0, int(seconds))
        mins, secs = divmod(total, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins}m {secs}s"
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _describe_mood(emotion: str, valence: float, arousal: float) -> str:
        """Convert emotion metrics into plain-language mood text."""
        if valence >= 0.25:
            tone = "positive"
        elif valence <= -0.25:
            tone = "low"
        else:
            tone = "steady"

        if arousal >= 0.7:
            energy = "high energy"
        elif arousal >= 0.4:
            energy = "moderate energy"
        else:
            energy = "calm energy"

        return f"{emotion} ({tone}, {energy})"

    @staticmethod
    def _describe_focus_load(load: float) -> str:
        """Describe working-memory load in plain terms."""
        if load < 0.3:
            level = "light"
        elif load < 0.7:
            level = "moderate"
        else:
            level = "heavy"
        return f"{level} ({load:.1%} of active memory in use)"

    def _describe_stress_guardrail(self, resilience: dict) -> str:
        """Describe resilience circuit state in plain language."""
        if resilience.get("breaker_active"):
            duration = self._format_uptime(float(resilience.get("distress_duration", 0.0)))
            return f"ACTIVE (recovering from sustained stress for {duration})"
        return "normal"


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
