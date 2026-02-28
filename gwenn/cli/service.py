"""Service management commands — install, uninstall, restart."""

from __future__ import annotations

import click


@click.command("install")
def install_cmd() -> None:
    """Install Gwenn as a system service."""
    from gwenn.service import get_service_manager

    try:
        mgr = get_service_manager()
        mgr.install()
        click.echo("Service installed. Start with: gwenn restart")
    except NotImplementedError as e:
        raise click.ClickException(str(e))
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Install failed: {e}")


@click.command("uninstall")
def uninstall_cmd() -> None:
    """Remove the Gwenn system service."""
    from gwenn.service import get_service_manager

    try:
        mgr = get_service_manager()
        mgr.uninstall()
        click.echo("Service uninstalled.")
    except NotImplementedError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Uninstall failed: {e}")


@click.command("restart")
def restart_cmd() -> None:
    """Restart the Gwenn system service."""
    from gwenn.service import get_service_manager

    try:
        mgr = get_service_manager()
        mgr.restart()
        click.echo("Service restarted.")
    except NotImplementedError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Restart failed: {e}")
