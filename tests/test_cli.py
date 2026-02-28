"""Tests for gwenn/cli/ — Click-based CLI commands."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch

from gwenn.cli.app import cli, async_cmd
from gwenn.cli.formatters import (
    build_table,
    format_duration,
    get_console,
    status_indicator,
)
from gwenn.cli.connection import DaemonConnection, DaemonNotRunning


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_seconds(self) -> None:
        assert format_duration(45) == "45s"

    def test_minutes(self) -> None:
        assert format_duration(125) == "2m05s"

    def test_hours(self) -> None:
        assert format_duration(7200) == "2h 00m"

    def test_zero(self) -> None:
        assert format_duration(0) == "0s"


class TestStatusIndicator:
    def test_known_statuses(self) -> None:
        for status in ("connected", "active", "running", "idle", "down", "error"):
            result = status_indicator(status)
            assert result is not None

    def test_unknown_status(self) -> None:
        result = status_indicator("something_else")
        assert result is not None


class TestGetConsole:
    def test_default(self) -> None:
        c = get_console()
        assert c is not None

    def test_no_color(self) -> None:
        c = get_console(no_color=True)
        assert c is not None


class TestBuildTable:
    def test_creates_table(self) -> None:
        table = build_table("Test", ["A", "B"], [["1", "2"]])
        assert table is not None
        assert table.title == "Test"


# ---------------------------------------------------------------------------
# async_cmd decorator
# ---------------------------------------------------------------------------


class TestAsyncCmd:
    def test_wraps_async_function(self) -> None:
        async def my_func():
            return 42

        wrapped = async_cmd(my_func)
        result = wrapped()
        assert result == 42


# ---------------------------------------------------------------------------
# CLI group — help and structure
# ---------------------------------------------------------------------------


class TestCliGroup:
    def test_help_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Gwenn" in result.output

    def test_subcommands_registered(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        for cmd in ("daemon", "stop", "status", "feed", "dashboard",
                     "agents", "channels", "config", "doctor",
                     "install", "uninstall", "restart"):
            assert cmd in result.output

    def test_global_flags(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "--json" in result.output
        assert "--quiet" in result.output
        assert "--verbose" in result.output
        assert "--no-color" in result.output


# ---------------------------------------------------------------------------
# Config commands
# ---------------------------------------------------------------------------


class TestConfigCmd:
    def test_config_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "get" in result.output
        assert "set" in result.output

    def test_config_init(self, tmp_path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["config", "init"])
            assert result.exit_code == 0
            assert "Created gwenn.toml" in result.output

    def test_config_init_already_exists(self, tmp_path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create file first
            open("gwenn.toml", "w").close()
            result = runner.invoke(cli, ["config", "init"])
            assert result.exit_code != 0

    def test_config_validate(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "validate"])
        # Should either succeed or warn about API key
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Doctor command
# ---------------------------------------------------------------------------


class TestDoctorCmd:
    def test_doctor_runs(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "Python version" in result.output

    def test_doctor_json(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "checks" in data
        assert len(data["checks"]) > 0


# ---------------------------------------------------------------------------
# Service commands
# ---------------------------------------------------------------------------


class TestServiceCmds:
    def test_install_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["install", "--help"])
        assert result.exit_code == 0

    def test_uninstall_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["uninstall", "--help"])
        assert result.exit_code == 0

    def test_restart_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["restart", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# DaemonConnection
# ---------------------------------------------------------------------------


class TestDaemonConnection:
    def test_init_defaults(self) -> None:
        conn = DaemonConnection()
        assert conn._host == "127.0.0.1"
        assert conn._port == 18900

    def test_init_custom(self) -> None:
        conn = DaemonConnection(host="0.0.0.0", port=9999)
        assert conn._host == "0.0.0.0"
        assert conn._port == 9999

    @pytest.mark.asyncio
    async def test_connect_fails_raises(self) -> None:
        conn = DaemonConnection(host="127.0.0.1", port=1)
        with pytest.raises(DaemonNotRunning):
            await conn.connect()


class TestDaemonNotRunning:
    def test_message(self) -> None:
        err = DaemonNotRunning()
        assert "daemon" in str(err.message).lower()
        assert "gwenn daemon" in str(err.message)


# ---------------------------------------------------------------------------
# Status command (with mock connection)
# ---------------------------------------------------------------------------


class TestStatusCmd:
    def test_status_not_running(self) -> None:
        """status should report not running when daemon unreachable."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])
        # The daemon is not running in test, so we expect the error message
        assert "not running" in result.output.lower() or result.exit_code != 0

    def test_status_json_not_running(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--json"])
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert data.get("status") == "not_running"


# ---------------------------------------------------------------------------
# Feed command help
# ---------------------------------------------------------------------------


class TestFeedCmd:
    def test_feed_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["feed", "--help"])
        assert result.exit_code == 0
        assert "--follow" in result.output
        assert "--type" in result.output


# ---------------------------------------------------------------------------
# Dashboard command help
# ---------------------------------------------------------------------------


class TestDashboardCmd:
    def test_dashboard_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "--interval" in result.output


# ---------------------------------------------------------------------------
# Agents command help
# ---------------------------------------------------------------------------


class TestAgentsCmd:
    def test_agents_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["agents", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Channels command help
# ---------------------------------------------------------------------------


class TestChannelsCmd:
    def test_channels_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["channels", "--help"])
        assert result.exit_code == 0
