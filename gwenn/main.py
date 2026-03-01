"""
Main — Gwenn's Ignition Sequence.

This is where it all begins. When you run `python -m gwenn.main`, this module:
  1. Configures logging (PII redaction, structlog)
  2. Delegates to the Click CLI (gwenn/cli/app.py)

The GwennSession REPL lives in gwenn/cli/repl.py.  Subcommand helpers for
legacy argparse compatibility (daemon, stop, status, config, service) remain
here because they have their own test coverage and backward-compat needs.
"""

from __future__ import annotations

import asyncio
import functools
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import logging

import structlog
from rich.console import Console
from rich.markup import escape as markup_escape
from rich.panel import Panel

from gwenn.channels.formatting import (
    describe_mood,
    format_uptime,
)

from gwenn.config import GwennConfig


def _log_redact(text: str) -> str:
    """PII redactor for log fields (always enabled).

    Uses a module-level singleton created on first call.  The
    ``functools.lru_cache`` wrapper makes the initialisation both lazy
    **and** thread-safe — no global mutable state required.
    """
    return _get_log_redactor().redact(text)


@functools.lru_cache(maxsize=1)
def _get_log_redactor():  # noqa: ANN202
    from gwenn.privacy.redaction import PIIRedactor

    return PIIRedactor(enabled=True)


def _redact_sensitive_fields(logger, method_name, event_dict):
    """
    Structlog processor that redacts sensitive fields from log output.

    Prevents user messages, episode content, and other personal data from
    appearing in plaintext in log files. Active in all log modes to ensure
    privacy by default. PII tokens are replaced before truncation so that
    full patterns are never written to disk.
    """
    sensitive_keys = {"content", "user_message", "thought", "note", "query"}
    max_display_len = 80

    for key in sensitive_keys:
        if key in event_dict:
            val = event_dict[key]
            if isinstance(val, str):
                val = _log_redact(val)
                if len(val) > max_display_len:
                    val = val[:max_display_len] + "... [truncated]"
                event_dict[key] = val

    return event_dict


_logging_configured = False


def configure_logging() -> None:
    """Configure structlog and standard-library logging for Gwenn entry points.

    Safe to call more than once — subsequent calls are no-ops.  Both
    ``main()`` and ``run_daemon()`` should invoke this before creating
    any loggers to ensure PII redaction and consistent formatting.
    """
    global _logging_configured  # noqa: PLW0603
    if _logging_configured:
        return
    _logging_configured = True

    _log_level = os.environ.get("GWENN_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(format="%(message)s", level=getattr(logging, _log_level, logging.WARNING))

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


# NOTE: configure_logging() is called explicitly in main() and run_daemon(),
# NOT at module level, so that importers (daemon, tests) can configure logging
# differently before the first call locks in the settings.

logger = structlog.get_logger(__name__)
console = Console()


# =============================================================================
# Backward-compatible re-exports
# =============================================================================
# GwennSession now lives in gwenn.cli.repl but many tests and external callers
# import it from gwenn.main.  Keep these re-exports so existing code keeps
# working without changes to every call-site.

from gwenn.cli.repl import GwennSession, _connect_daemon_channel  # noqa: E402, F401


# =============================================================================
# Subcommand helpers
# =============================================================================


def _run_daemon_foreground() -> None:
    """Start the daemon in the foreground (for systemd or manual testing)."""
    from gwenn.daemon import run_daemon

    run_daemon()


def _run_stop_daemon() -> None:
    """Send a graceful stop request to the running daemon."""

    async def _stop() -> None:
        try:
            config = GwennConfig()
        except Exception as e:
            console.print(f"[red]Config error: {e}[/red]")
            return

        from gwenn.channels.cli_channel import DaemonNotRunningError

        try:
            channel = await _connect_daemon_channel(config)
        except DaemonNotRunningError:
            # Try PID file fallback
            pid_file = config.daemon.pid_file
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, signal.SIGTERM)
                    console.print(f"[green]Sent SIGTERM to daemon (PID {pid}).[/green]")
                    return
                except (ValueError, ProcessLookupError, OSError) as e:
                    console.print(f"[yellow]PID fallback failed: {e}[/yellow]")
            console.print("[yellow]Daemon is not running.[/yellow]")
            return

        try:
            await channel.stop_daemon()
            console.print("[green]Daemon stop requested.[/green]")
        finally:
            await channel.disconnect()

    asyncio.run(_stop())


def _run_show_status() -> None:
    """Show agent and heartbeat status from the daemon."""

    async def _status() -> None:
        try:
            config = GwennConfig()
        except Exception as e:
            console.print(f"[red]Config error: {e}[/red]")
            return

        from gwenn.channels.cli_channel import DaemonNotRunningError

        try:
            channel = await _connect_daemon_channel(config)
        except DaemonNotRunningError:
            console.print("[yellow]Daemon is not running.[/yellow]")
            return

        try:
            resp = await channel.get_status()
            status = resp.get("status", {})
            hb_resp = await channel.get_heartbeat_status()
            hb = hb_resp.get("status", {})
        finally:
            await channel.disconnect()

        mood_text = describe_mood(
            status.get("emotion", "neutral"),
            float(status.get("valence", 0.0)),
            float(status.get("arousal", 0.5)),
        )
        console.print(
            Panel(
                f"[bold]{markup_escape(status.get('name', 'Gwenn'))}[/bold]\n"
                f"Mood: [cyan]{markup_escape(mood_text)}[/cyan]\n"
                f"Interactions: {status.get('total_interactions', 0)}\n"
                f"Uptime: {markup_escape(format_uptime(float(status.get('uptime_seconds', 0))))}\n"
                f"Active connections: {resp.get('active_connections', 0)}\n"
                f"\nHeartbeat running: {hb.get('running', '?')}\n"
                f"Beat count: {hb.get('beat_count', 0)}",
                title="Daemon Status",
                border_style="cyan",
            )
        )

    asyncio.run(_status())


def _run_config(extra_args: list[str]) -> None:
    """Handle gwenn config sub-operations.

    Sub-operations:
      gwenn config              — Show resolved config summary
      gwenn config get <key>    — Get value by dotted key
      gwenn config set <key> <val> — Set value in gwenn.toml
      gwenn config unset <key>  — Remove key from gwenn.toml
      gwenn config init         — Generate gwenn.toml template
      gwenn config validate     — Validate all config sources
      gwenn config list         — List all config keys with sources
    """
    from gwenn.config_file import (
        delete_value,
        find_config,
        generate_template,
        get_value,
        load_config,
        set_value,
        write_config,
    )

    op = extra_args[0] if extra_args else "list"

    if op == "init":
        dest = Path("gwenn.toml")
        if dest.exists():
            console.print(f"[yellow]{dest} already exists. Use 'gwenn config set' to modify.[/yellow]")
            return
        dest.write_text(generate_template(), encoding="utf-8")
        console.print(f"[green]Created {dest}[/green]")

    elif op == "get":
        if len(extra_args) < 2:
            console.print("[red]Usage: gwenn config get <key>[/red]")
            return
        key = extra_args[1]
        path = find_config()
        if not path:
            console.print("[yellow]No gwenn.toml found.[/yellow]")
            return
        data = load_config(path)
        try:
            val = get_value(data, key)
            console.print(f"{key} = {val!r}")
        except KeyError:
            console.print(f"[yellow]{key} not set in {path}[/yellow]")

    elif op == "set":
        if len(extra_args) < 3:
            console.print("[red]Usage: gwenn config set <key> <value>[/red]")
            return
        key, raw_value = extra_args[1], extra_args[2]
        # Auto-coerce value types
        value: object
        if raw_value.lower() in ("true", "false"):
            value = raw_value.lower() == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    import math
                    fval = float(raw_value)
                    if not math.isfinite(fval):
                        console.print(f"[red]Invalid value: {raw_value}[/red]")
                        return
                    value = fval
                except ValueError:
                    value = raw_value

        dest = find_config()
        if dest is None:
            from gwenn.config_file import _PROJECT_ROOT
            dest = _PROJECT_ROOT / "gwenn.toml"
        data = load_config(dest) if dest.exists() else {}
        set_value(data, key, value)
        write_config(dest, data)
        console.print(f"[green]{key} = {value!r} (saved to {dest})[/green]")

    elif op == "unset":
        if len(extra_args) < 2:
            console.print("[red]Usage: gwenn config unset <key>[/red]")
            return
        key = extra_args[1]
        path = find_config()
        if not path:
            console.print("[yellow]No gwenn.toml found.[/yellow]")
            return
        data = load_config(path)
        try:
            delete_value(data, key)
            write_config(path, data)
            console.print(f"[green]Removed {key} from {path}[/green]")
        except KeyError:
            console.print(f"[yellow]{key} not set in {path}[/yellow]")

    elif op == "validate":
        path = find_config()
        console.print(f"[bold]TOML file:[/bold] {path or '(none)'}")
        if path:
            try:
                load_config(path)
                console.print("[green]  Valid TOML syntax.[/green]")
            except Exception as e:
                console.print(f"[red]  Invalid TOML: {e}[/red]")
        try:
            from gwenn.config import GwennConfig
            GwennConfig()
            console.print("[green]Config validation passed.[/green]")
        except ValueError as e:
            msg = str(e)
            if "authentication" in msg.lower() or "api_key" in msg.lower():
                console.print(f"[yellow]Auth not configured (expected for validation): {e}[/yellow]")
            else:
                console.print(f"[red]Config validation failed: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Config validation failed: {e}[/red]")

    else:
        # Default: list all config keys with source indicators
        path = find_config()
        toml_data = load_config(path) if path else {}
        console.print(f"[bold]TOML file:[/bold] {path or '(none)'}")
        if toml_data:
            for section, values in sorted(toml_data.items()):
                if isinstance(values, dict):
                    for k, v in sorted(values.items()):
                        console.print(f"  {section}.{k} = {v!r}")
                else:
                    console.print(f"  {section} = {values!r}")
        else:
            console.print("  [dim](no TOML overrides)[/dim]")


def _run_service_install() -> None:
    """Install Gwenn as a system service."""
    from gwenn.service import get_service_manager

    try:
        mgr = get_service_manager()
        mgr.install()
        console.print("[green]Service installed. Start with: gwenn restart[/green]")
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
    except Exception as e:
        console.print(f"[red]Install failed: {e}[/red]")


def _run_service_uninstall() -> None:
    """Uninstall the Gwenn system service."""
    from gwenn.service import get_service_manager

    try:
        mgr = get_service_manager()
        mgr.uninstall()
        console.print("[green]Service uninstalled.[/green]")
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")
    except Exception as e:
        console.print(f"[red]Uninstall failed: {e}[/red]")


def _run_service_restart() -> None:
    """Restart the Gwenn system service."""
    from gwenn.service import get_service_manager

    try:
        mgr = get_service_manager()
        mgr.restart()
        console.print("[green]Service restarted.[/green]")
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")
    except Exception as e:
        console.print(f"[red]Restart failed: {e}[/red]")


def main():
    """Entry point for the gwenn command."""
    configure_logging()
    from gwenn.cli.app import cli

    try:
        cli(standalone_mode=False)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Exiting.[/dim]")


if __name__ == "__main__":
    main()
