"""Daemon start/stop commands."""

from __future__ import annotations

import click

from gwenn.cli.app import async_cmd


@click.command("daemon")
@click.option(
    "--channel",
    type=click.Choice(["cli", "telegram", "discord", "all"]),
    default=None,
    help="Channel to run (overrides config). Default: all configured.",
)
def daemon_cmd(channel: str | None) -> None:
    """Start Gwenn's heartbeat daemon (foreground)."""
    from gwenn.main import configure_logging

    configure_logging()

    from gwenn.config import GwennConfig
    from gwenn.heartbeat import Heartbeat

    config = GwennConfig()
    hb = Heartbeat(config)

    import asyncio

    try:
        asyncio.run(hb.run())
    except KeyboardInterrupt:
        pass


@click.command("stop")
@async_cmd
async def stop_cmd() -> None:
    """Stop the running Gwenn daemon gracefully."""
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
        await conn.rpc("stop")
        click.echo("Gwenn daemon stopped.")
    except DaemonNotRunning:
        click.echo("Gwenn daemon is not running.")
    finally:
        await conn.disconnect()
