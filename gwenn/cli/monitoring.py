"""Monitoring commands — status, feed, logs, costs."""

from __future__ import annotations

import json as json_mod

import click

from gwenn.cli.app import async_cmd
from gwenn.cli.formatters import format_duration, get_console, status_indicator


@click.command("status")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.option("--verbose", "-v", is_flag=True, help="Extended details")
@click.pass_context
@async_cmd
async def status_cmd(ctx: click.Context, json_output: bool, verbose: bool) -> None:
    """Show daemon status: heartbeat, channels, connections."""
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
        if json_output:
            click.echo(json_mod.dumps({"status": "not_running"}))
        else:
            click.echo("Gwenn daemon is not running. Start with: gwenn daemon")
        return

    try:
        # Fetch status and heartbeat info
        status_result = await conn.rpc("status")
        hb_result = await conn.rpc("heartbeat_status")

        if json_output:
            click.echo(json_mod.dumps({
                "status": status_result,
                "heartbeat": hb_result.get("status", {}),
            }, indent=2))
            return

        console = get_console(no_color=ctx.obj.get("no_color", False))
        hb = hb_result.get("status", {})

        console.print("[bold]Gwenn Status[/bold]")
        console.print()

        # Heartbeat info
        console.print("[bold]Heartbeat[/bold]")
        console.print(f"  Beat: #{hb.get('beat_count', 0)}")
        console.print(f"  Running: {hb.get('running', False)}")
        console.print(f"  Interval: {hb.get('interval', 0)}s")

        emotion = hb.get("emotion", "unknown")
        console.print(f"  Emotion: {emotion}")

        if verbose:
            console.print(f"  Last error: {hb.get('last_error', 'none')}")
            console.print(f"  Consecutive failures: {hb.get('consecutive_failures', 0)}")

        # Connection info
        active = status_result.get("active_connections", 0)
        console.print()
        console.print(f"[bold]Connections:[/bold] {active}")

    finally:
        await conn.disconnect()


@click.command("feed")
@click.option("--follow", "-f", is_flag=True, help="Continuous streaming")
@click.option("--type", "event_type", default="*", help="Event pattern filter")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.pass_context
@async_cmd
async def feed_cmd(
    ctx: click.Context,
    follow: bool,
    event_type: str,
    json_output: bool,
) -> None:
    """Real-time event stream from the event bus."""
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
        await conn.subscribe([event_type])

        console = get_console(no_color=ctx.obj.get("no_color", False))

        if not follow:
            console.print("[dim]Listening for events (press Ctrl+C to stop)...[/dim]")

        count = 0
        async for event in conn.events():
            method = event.get("method", "")
            params = event.get("params", {})

            if json_output:
                click.echo(json_mod.dumps(params))
            else:
                etype = params.get("event_type", method)
                console.print(f"  [{etype}] {_summarize_event(params)}")

            count += 1
            if not follow and count >= 50:
                break

    except KeyboardInterrupt:
        pass
    finally:
        await conn.disconnect()


def _summarize_event(params: dict) -> str:
    """Create a one-line summary from event params."""
    etype = params.get("event_type", "")
    if "heartbeat.beat" in etype:
        return f"beat #{params.get('beat_count', '?')} emotion={params.get('emotion', '?')}"
    if "emotion.changed" in etype:
        return f"{params.get('previous', '?')} -> {params.get('current', '?')}"
    if "heartbeat.thought" in etype:
        return params.get("thought_summary", "")[:80]
    if "interoceptive" in etype:
        return f"fatigue={params.get('fatigue', 0):.2f} flow={params.get('flow', 0):.2f}"
    # Fallback: show first few fields
    items = [f"{k}={v}" for k, v in list(params.items())[:3] if k != "event_type"]
    return " ".join(items)
