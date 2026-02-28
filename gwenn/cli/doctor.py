"""Health check command — run diagnostics on the Gwenn installation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click


@click.command("doctor")
@click.option("--fix", is_flag=True, help="Auto-fix issues where possible")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
def doctor_cmd(fix: bool, json_output: bool) -> None:
    """Run health checks on the Gwenn installation."""
    checks: list[dict] = []

    # 1. Python version
    py_ver = sys.version_info
    checks.append({
        "name": "Python version",
        "status": "ok" if py_ver >= (3, 11) else "error",
        "detail": f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}",
    })

    # 2. Required dependencies
    for pkg in ["anthropic", "structlog", "pydantic", "aiohttp", "rich", "click"]:
        try:
            __import__(pkg)
            checks.append({"name": f"Package: {pkg}", "status": "ok", "detail": "installed"})
        except ImportError:
            checks.append({"name": f"Package: {pkg}", "status": "error", "detail": "missing"})

    # 3. API key
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GWENN_API_KEY"))
    checks.append({
        "name": "API key",
        "status": "ok" if has_key else "warning",
        "detail": "set" if has_key else "ANTHROPIC_API_KEY not set",
    })

    # 4. Data directory
    from gwenn.config import MemoryConfig

    try:
        mem_cfg = MemoryConfig()
        data_dir = Path(mem_cfg.data_dir)
        if data_dir.exists():
            checks.append({"name": "Data directory", "status": "ok", "detail": str(data_dir)})
        else:
            if fix:
                data_dir.mkdir(parents=True, exist_ok=True)
                checks.append({
                    "name": "Data directory",
                    "status": "fixed",
                    "detail": f"Created {data_dir}",
                })
            else:
                checks.append({
                    "name": "Data directory",
                    "status": "warning",
                    "detail": f"{data_dir} does not exist (will be created on first run)",
                })
    except Exception as e:
        checks.append({"name": "Data directory", "status": "error", "detail": str(e)})

    # 5. Config file
    from gwenn.config_file import find_config

    config_path = find_config()
    checks.append({
        "name": "Config file",
        "status": "ok" if config_path else "info",
        "detail": str(config_path) if config_path else "No gwenn.toml (using defaults)",
    })

    # 6. Daemon status
    import asyncio

    daemon_ok = asyncio.run(_check_daemon())
    checks.append({
        "name": "Daemon",
        "status": "ok" if daemon_ok else "info",
        "detail": "running" if daemon_ok else "not running",
    })

    # Output
    if json_output:
        import json as json_mod

        click.echo(json_mod.dumps({"checks": checks}, indent=2))
        return

    for check in checks:
        status = check["status"]
        icon = {
            "ok": "\u2713",  # checkmark
            "warning": "!",
            "error": "\u2717",  # cross
            "info": "\u2022",  # bullet
            "fixed": "\u2713",
        }.get(status, "?")
        color = {
            "ok": "green",
            "warning": "yellow",
            "error": "red",
            "info": "white",
            "fixed": "cyan",
        }.get(status, None)
        click.echo(click.style(f"  {icon} {check['name']}: {check['detail']}", fg=color))

    errors = sum(1 for c in checks if c["status"] == "error")
    warnings = sum(1 for c in checks if c["status"] == "warning")
    if errors:
        click.echo(f"\n{errors} error(s) found.")
    elif warnings:
        click.echo(f"\n{warnings} warning(s). Gwenn should work but check the above.")
    else:
        click.echo("\nAll checks passed.")


async def _check_daemon() -> bool:
    """Check if the daemon is reachable."""
    from gwenn.cli.connection import DaemonConnection

    conn = DaemonConnection()
    try:
        await conn.connect()
        await conn.rpc("ping")
        return True
    except Exception:
        return False
    finally:
        await conn.disconnect()
