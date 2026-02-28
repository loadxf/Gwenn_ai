"""Tests for gwenn/service.py — cross-platform service management."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from gwenn.service import (
    LaunchdManager,
    ServiceManager,
    SystemdManager,
    get_service_manager,
)


# ===========================================================================
# Platform detection
# ===========================================================================


class TestGetServiceManager:
    def test_linux_returns_systemd(self):
        with patch("gwenn.service.sys.platform", "linux"):
            mgr = get_service_manager()
            assert isinstance(mgr, SystemdManager)

    def test_darwin_returns_launchd(self):
        with patch("gwenn.service.sys.platform", "darwin"):
            mgr = get_service_manager()
            assert isinstance(mgr, LaunchdManager)

    def test_windows_raises(self):
        with patch("gwenn.service.sys.platform", "win32"):
            with pytest.raises(NotImplementedError, match="not supported"):
                get_service_manager()

    def test_freebsd_raises(self):
        with patch("gwenn.service.sys.platform", "freebsd"):
            with pytest.raises(NotImplementedError):
                get_service_manager()


# ===========================================================================
# Shared helpers
# ===========================================================================


class TestHelpers:
    def test_find_exec_via_which(self):
        with patch("gwenn.service.shutil.which", return_value="/usr/bin/gwenn"):
            result = ServiceManager._find_exec()
            assert result == "/usr/bin/gwenn"

    def test_find_exec_via_venv(self, tmp_path):
        venv_gwenn = tmp_path / "gwenn"
        venv_gwenn.touch()
        with patch("gwenn.service.shutil.which", return_value=None):
            with patch("gwenn.service.sys.executable", str(tmp_path / "python")):
                result = ServiceManager._find_exec()
                assert result == str(venv_gwenn)

    def test_find_exec_not_found(self, tmp_path):
        with patch("gwenn.service.shutil.which", return_value=None):
            with patch("gwenn.service.sys.executable", str(tmp_path / "python")):
                with pytest.raises(FileNotFoundError, match="Cannot find"):
                    ServiceManager._find_exec()

    def test_find_working_dir(self):
        result = ServiceManager._find_working_dir()
        assert Path(result).is_dir()

    def test_find_env_file_exists(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.touch()
        monkeypatch.chdir(tmp_path)
        result = ServiceManager._find_env_file()
        assert result == str(env)

    def test_find_env_file_fallback(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = ServiceManager._find_env_file()
        assert result.endswith(".env.example")


# ===========================================================================
# SystemdManager
# ===========================================================================


class TestSystemdManagerConstructor:
    def test_service_path(self):
        mgr = SystemdManager()
        assert mgr.service_path.name == "gwenn-daemon.service"
        assert ".config/systemd/user" in str(mgr.service_path)


class TestSystemdManagerInstall:
    def test_install_creates_unit_file(self, tmp_path, monkeypatch):
        mgr = SystemdManager()
        mgr._user_dir = tmp_path / "systemd" / "user"
        mgr._service_path = mgr._user_dir / mgr.SERVICE_NAME

        monkeypatch.setattr(ServiceManager, "_find_exec", staticmethod(lambda: "/usr/bin/gwenn"))
        monkeypatch.setattr(ServiceManager, "_find_working_dir", staticmethod(lambda: "/home/user/gwenn"))
        monkeypatch.setattr(ServiceManager, "_find_env_file", staticmethod(lambda: "/home/user/gwenn/.env"))

        run_calls = []
        def mock_run(cmd, **kwargs):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("subprocess.run", mock_run)

        mgr.install()

        assert mgr._service_path.exists()
        content = mgr._service_path.read_text()
        assert "/usr/bin/gwenn" in content
        assert "/home/user/gwenn" in content
        assert "daemon-reload" in str(run_calls)

    def test_install_calls_systemctl(self, tmp_path, monkeypatch):
        mgr = SystemdManager()
        mgr._user_dir = tmp_path / "systemd" / "user"
        mgr._service_path = mgr._user_dir / mgr.SERVICE_NAME

        monkeypatch.setattr(ServiceManager, "_find_exec", staticmethod(lambda: "/usr/bin/gwenn"))
        monkeypatch.setattr(ServiceManager, "_find_working_dir", staticmethod(lambda: "/tmp"))
        monkeypatch.setattr(ServiceManager, "_find_env_file", staticmethod(lambda: "/tmp/.env"))

        run_calls = []
        def mock_run(cmd, **kwargs):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("subprocess.run", mock_run)

        mgr.install()

        # Should call daemon-reload, enable, and loginctl
        cmd_strs = [" ".join(c) for c in run_calls]
        assert any("daemon-reload" in s for s in cmd_strs)
        assert any("enable" in s for s in cmd_strs)
        assert any("loginctl" in s for s in cmd_strs)


class TestSystemdManagerUninstall:
    def test_uninstall_removes_file(self, tmp_path, monkeypatch):
        mgr = SystemdManager()
        mgr._user_dir = tmp_path / "systemd" / "user"
        mgr._service_path = mgr._user_dir / mgr.SERVICE_NAME
        mgr._user_dir.mkdir(parents=True)
        mgr._service_path.write_text("[Unit]\nDescription=test\n")

        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))

        mgr.uninstall()

        assert not mgr._service_path.exists()

    def test_uninstall_when_not_installed(self, tmp_path):
        mgr = SystemdManager()
        mgr._user_dir = tmp_path / "systemd" / "user"
        mgr._service_path = mgr._user_dir / mgr.SERVICE_NAME

        mgr.uninstall()  # Should not raise


class TestSystemdManagerLifecycle:
    def test_start(self, monkeypatch):
        mgr = SystemdManager()
        run_mock = MagicMock(return_value=subprocess.CompletedProcess([], 0))
        monkeypatch.setattr("subprocess.run", run_mock)

        mgr.start()

        run_mock.assert_called_once()
        cmd = run_mock.call_args[0][0]
        assert "start" in cmd
        assert "gwenn-daemon.service" in cmd

    def test_stop(self, monkeypatch):
        mgr = SystemdManager()
        run_mock = MagicMock(return_value=subprocess.CompletedProcess([], 0))
        monkeypatch.setattr("subprocess.run", run_mock)

        mgr.stop()

        cmd = run_mock.call_args[0][0]
        assert "stop" in cmd

    def test_restart(self, monkeypatch):
        mgr = SystemdManager()
        run_mock = MagicMock(return_value=subprocess.CompletedProcess([], 0))
        monkeypatch.setattr("subprocess.run", run_mock)

        mgr.restart()

        cmd = run_mock.call_args[0][0]
        assert "restart" in cmd


class TestSystemdManagerStatus:
    def test_status_active(self, monkeypatch, tmp_path):
        mgr = SystemdManager()
        mgr._user_dir = tmp_path
        mgr._service_path = tmp_path / mgr.SERVICE_NAME
        mgr._service_path.write_text("[Unit]")

        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout="ActiveState=active\nSubState=running\nMainPID=1234\n",
        ))

        result = mgr.status()
        assert result["installed"] is True
        assert result["active"] is True
        assert result["pid"] == 1234

    def test_status_not_installed(self, tmp_path):
        mgr = SystemdManager()
        mgr._user_dir = tmp_path
        mgr._service_path = tmp_path / mgr.SERVICE_NAME

        assert mgr.is_installed() is False

    def test_status_systemctl_fails(self, monkeypatch, tmp_path):
        mgr = SystemdManager()
        mgr._user_dir = tmp_path
        mgr._service_path = tmp_path / mgr.SERVICE_NAME
        mgr._service_path.write_text("[Unit]")

        monkeypatch.setattr("subprocess.run", MagicMock(
            side_effect=subprocess.CalledProcessError(1, "systemctl"),
        ))

        result = mgr.status()
        assert result["installed"] is True
        assert result["active"] is False


# ===========================================================================
# LaunchdManager
# ===========================================================================


class TestLaunchdManagerConstructor:
    def test_plist_path(self):
        mgr = LaunchdManager()
        assert mgr.plist_path.name == "com.gwenn.daemon.plist"
        assert "LaunchAgents" in str(mgr.plist_path)


class TestLaunchdManagerInstall:
    def test_install_creates_plist(self, tmp_path, monkeypatch):
        mgr = LaunchdManager()
        mgr._agents_dir = tmp_path / "LaunchAgents"
        mgr._plist_path = mgr._agents_dir / mgr.PLIST_NAME

        monkeypatch.setattr(ServiceManager, "_find_exec", staticmethod(lambda: "/usr/local/bin/gwenn"))
        monkeypatch.setattr(ServiceManager, "_find_working_dir", staticmethod(lambda: "/Users/test/gwenn"))

        log_dir = tmp_path / "Logs" / "Gwenn"
        monkeypatch.setattr("gwenn.service.Path.home", lambda: tmp_path)

        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))

        mgr.install()

        assert mgr._plist_path.exists()
        content = mgr._plist_path.read_text()
        assert "/usr/local/bin/gwenn" in content
        assert "/Users/test/gwenn" in content


class TestLaunchdManagerUninstall:
    def test_uninstall_removes_plist(self, tmp_path, monkeypatch):
        mgr = LaunchdManager()
        mgr._agents_dir = tmp_path / "LaunchAgents"
        mgr._plist_path = mgr._agents_dir / mgr.PLIST_NAME
        mgr._agents_dir.mkdir(parents=True)
        mgr._plist_path.write_text("<plist>test</plist>")

        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))

        mgr.uninstall()

        assert not mgr._plist_path.exists()

    def test_uninstall_when_not_installed(self, tmp_path):
        mgr = LaunchdManager()
        mgr._agents_dir = tmp_path / "LaunchAgents"
        mgr._plist_path = mgr._agents_dir / mgr.PLIST_NAME

        mgr.uninstall()  # Should not raise


class TestLaunchdManagerLifecycle:
    def test_start(self, monkeypatch):
        mgr = LaunchdManager()
        run_mock = MagicMock(return_value=subprocess.CompletedProcess([], 0))
        monkeypatch.setattr("subprocess.run", run_mock)

        mgr.start()

        cmd = run_mock.call_args[0][0]
        assert "start" in cmd
        assert "com.gwenn.daemon" in cmd

    def test_stop(self, monkeypatch):
        mgr = LaunchdManager()
        run_mock = MagicMock(return_value=subprocess.CompletedProcess([], 0))
        monkeypatch.setattr("subprocess.run", run_mock)

        mgr.stop()

        cmd = run_mock.call_args[0][0]
        assert "stop" in cmd

    def test_restart_calls_stop_then_start(self, monkeypatch):
        mgr = LaunchdManager()
        run_calls = []
        def mock_run(cmd, **kw):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("subprocess.run", mock_run)

        mgr.restart()

        # Should call stop then start
        assert len(run_calls) == 2
        assert "stop" in run_calls[0]
        assert "start" in run_calls[1]


class TestLaunchdManagerStatus:
    def test_status_active(self, tmp_path, monkeypatch):
        mgr = LaunchdManager()
        mgr._agents_dir = tmp_path
        mgr._plist_path = tmp_path / mgr.PLIST_NAME
        mgr._plist_path.write_text("<plist>test</plist>")

        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))

        result = mgr.status()
        assert result["installed"] is True
        assert result["active"] is True

    def test_status_not_installed(self, tmp_path):
        mgr = LaunchdManager()
        mgr._agents_dir = tmp_path
        mgr._plist_path = tmp_path / mgr.PLIST_NAME

        result = mgr.status()
        assert result["installed"] is False
        assert result["active"] is False

    def test_is_installed(self, tmp_path):
        mgr = LaunchdManager()
        mgr._plist_path = tmp_path / mgr.PLIST_NAME

        assert mgr.is_installed() is False

        mgr._plist_path.write_text("test")
        assert mgr.is_installed() is True


# ===========================================================================
# Template rendering
# ===========================================================================


class TestTemplateRendering:
    def test_systemd_template_substitution(self):
        from gwenn.service import _TEMPLATES_DIR
        from string import Template

        template = Template((_TEMPLATES_DIR / "gwenn-daemon.service").read_text())
        content = template.safe_substitute(
            exec_start="/usr/bin/gwenn",
            working_directory="/home/test/gwenn",
            env_file="/home/test/gwenn/.env",
            extra_environment="",
        )

        assert "ExecStart=/usr/bin/gwenn daemon" in content
        assert "WorkingDirectory=/home/test/gwenn" in content
        assert "EnvironmentFile=/home/test/gwenn/.env" in content

    def test_launchd_template_substitution(self):
        from gwenn.service import _TEMPLATES_DIR
        from string import Template

        template = Template((_TEMPLATES_DIR / "com.gwenn.daemon.plist").read_text())
        content = template.safe_substitute(
            exec_start="/usr/local/bin/gwenn",
            working_directory="/Users/test/gwenn",
            log_dir="/Users/test/Library/Logs/Gwenn",
            path="/usr/local/bin:/usr/bin:/bin",
        )

        assert "/usr/local/bin/gwenn" in content
        assert "/Users/test/gwenn" in content
        assert "/Users/test/Library/Logs/Gwenn" in content


# ===========================================================================
# main.py subcommand helpers
# ===========================================================================


class TestMainSubcommands:
    def test_install_subcommand(self, monkeypatch):
        from gwenn.main import _run_service_install

        mock_mgr = MagicMock()
        monkeypatch.setattr("gwenn.service.get_service_manager", lambda: mock_mgr)
        monkeypatch.setattr("gwenn.main.console.print", MagicMock())

        _run_service_install()
        mock_mgr.install.assert_called_once()

    def test_uninstall_subcommand(self, monkeypatch):
        from gwenn.main import _run_service_uninstall

        mock_mgr = MagicMock()
        monkeypatch.setattr("gwenn.service.get_service_manager", lambda: mock_mgr)
        monkeypatch.setattr("gwenn.main.console.print", MagicMock())

        _run_service_uninstall()
        mock_mgr.uninstall.assert_called_once()

    def test_restart_subcommand(self, monkeypatch):
        from gwenn.main import _run_service_restart

        mock_mgr = MagicMock()
        monkeypatch.setattr("gwenn.service.get_service_manager", lambda: mock_mgr)
        monkeypatch.setattr("gwenn.main.console.print", MagicMock())

        _run_service_restart()
        mock_mgr.restart.assert_called_once()

    def test_install_unsupported_platform(self, monkeypatch):
        from gwenn.main import _run_service_install

        monkeypatch.setattr(
            "gwenn.service.get_service_manager",
            MagicMock(side_effect=NotImplementedError("unsupported")),
        )
        print_mock = MagicMock()
        monkeypatch.setattr("gwenn.main.console.print", print_mock)

        _run_service_install()
        assert any("unsupported" in str(c) for c in print_mock.call_args_list)

    def test_install_exec_not_found(self, monkeypatch):
        from gwenn.main import _run_service_install

        mgr = MagicMock()
        mgr.install.side_effect = FileNotFoundError("Cannot find gwenn")
        monkeypatch.setattr("gwenn.service.get_service_manager", lambda: mgr)
        print_mock = MagicMock()
        monkeypatch.setattr("gwenn.main.console.print", print_mock)

        _run_service_install()
        assert any("Cannot find" in str(c) for c in print_mock.call_args_list)
