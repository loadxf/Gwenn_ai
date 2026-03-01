"""Live TUI dashboard — multi-panel real-time view of Gwenn's state."""

from __future__ import annotations

import asyncio
import json as json_mod
from typing import Any

import click

from gwenn.cli.app import async_cmd
from gwenn.cli.formatters import format_duration


@click.command("dashboard")
@click.option("--interval", default=1000, type=int, help="Refresh rate in ms")
@click.pass_context
@async_cmd
async def dashboard_cmd(ctx: click.Context, interval: int) -> None:
    """Live TUI: multi-panel real-time view of Gwenn's state."""
    from gwenn.cli.connection import DaemonConnection, DaemonNotRunning

    from gwenn.config import DaemonConfig

    try:
        config = DaemonConfig()
    except Exception:
        config = None

    host = getattr(config, "gateway_host", "127.0.0.1") if config else "127.0.0.1"
    port = getattr(config, "gateway_port", 18900) if config else 18900

    conn = DaemonConnection(host=str(host), port=int(port))
    try:
        await conn.connect()
    except DaemonNotRunning:
        click.echo("Gwenn daemon is not running. Start with: gwenn daemon")
        return

    try:
        from rich.console import Console
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.text import Text

        console = Console(no_color=ctx.obj.get("no_color", False))

        # Subscribe to all events for dashboard
        await conn.subscribe(["*"])

        state: dict[str, Any] = {
            "heartbeat": {},
            "events": [],
            "connections": 0,
        }

        # Fetch initial state
        try:
            hb = await conn.rpc("heartbeat_status")
            state["heartbeat"] = hb.get("status", {})
        except Exception:
            pass

        try:
            st = await conn.rpc("status")
            state["connections"] = st.get("active_connections", 0)
        except Exception:
            pass

        def make_layout() -> Layout:
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=1),
                Layout(name="body"),
                Layout(name="footer", size=1),
            )
            return layout

        def render_dashboard() -> str:
            hb = state.get("heartbeat", {})
            lines = []
            lines.append(f"  HEARTBEAT  Beat #{hb.get('beat_count', 0)}")
            lines.append(f"  Emotion: {hb.get('emotion', 'unknown')}")
            lines.append(f"  Interval: {hb.get('interval', 0)}s")
            lines.append(f"  Running: {hb.get('running', False)}")
            lines.append("")
            lines.append(f"  CONNECTIONS: {state.get('connections', 0)}")
            lines.append("")
            lines.append("  RECENT EVENTS")
            for evt in state.get("events", [])[-10:]:
                etype = evt.get("event_type", evt.get("method", ""))
                lines.append(f"    [{etype}]")
            return "\n".join(lines)

        with Live(
            Panel(render_dashboard(), title="Gwenn Dashboard"),
            console=console,
            refresh_per_second=max(1, 1000 // max(1, interval)),
        ) as live:
            update_task = asyncio.create_task(
                _update_state(conn, state, live, render_dashboard, interval)
            )
            try:
                await update_task
            except KeyboardInterrupt:
                pass
            except asyncio.CancelledError:
                pass
            finally:
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass

    except KeyboardInterrupt:
        pass
    finally:
        await conn.disconnect()


async def _update_state(
    conn: Any,
    state: dict[str, Any],
    live: Any,
    render_fn: Any,
    interval_ms: int,
) -> None:
    """Background task: consume events and update dashboard state."""
    from rich.panel import Panel

    async for event in conn.events():
        params = event.get("params", {})
        etype = params.get("event_type", "")

        # Update heartbeat state
        if "heartbeat.beat" in etype:
            state["heartbeat"].update({
                "beat_count": params.get("beat_count"),
                "emotion": params.get("emotion"),
            })

        # Store recent events
        events = state.setdefault("events", [])
        events.append(params)
        if len(events) > 50:
            state["events"] = events[-50:]

        live.update(Panel(render_fn(), title="Gwenn Dashboard"))
