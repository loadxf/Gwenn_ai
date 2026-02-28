"""
Cross-Platform Service Management.

Provides an abstract ``ServiceManager`` with concrete implementations for
systemd (Linux) and launchd (macOS).  Replaces the shell scripts in
``scripts/`` with Python equivalents invoked via ``gwenn install`` /
``gwenn uninstall`` / ``gwenn restart``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from string import Template
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class ServiceManager(ABC):
    """Abstract base for OS service management."""

    @abstractmethod
    def install(self) -> None:
        """Install the service (create unit file, enable on boot)."""

    @abstractmethod
    def uninstall(self) -> None:
        """Uninstall the service (stop, disable, remove unit file)."""

    @abstractmethod
    def start(self) -> None:
        """Start the service."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the service."""

    @abstractmethod
    def restart(self) -> None:
        """Restart the service."""

    @abstractmethod
    def status(self) -> dict[str, Any]:
        """Return service status as a dict."""

    @abstractmethod
    def is_installed(self) -> bool:
        """Check if the service is currently installed."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_exec() -> str:
        """Locate the ``gwenn`` executable."""
        exe = shutil.which("gwenn")
        if exe:
            return exe
        # Fall back to the running interpreter's sibling
        venv_bin = Path(sys.executable).parent / "gwenn"
        if venv_bin.exists():
            return str(venv_bin)
        raise FileNotFoundError(
            "Cannot find 'gwenn' executable. "
            "Ensure it is installed (pip install -e .) and on your PATH."
        )

    @staticmethod
    def _find_working_dir() -> str:
        """Detect the working directory (project root)."""
        return str(Path.cwd())

    @staticmethod
    def _find_env_file() -> str:
        """Locate the .env file."""
        env = Path.cwd() / ".env"
        if env.exists():
            return str(env)
        return str(Path.cwd() / ".env.example")


class SystemdManager(ServiceManager):
    """systemd user service manager (Linux).

    Installs to ``~/.config/systemd/user/gwenn-daemon.service``.
    """

    SERVICE_NAME = "gwenn-daemon.service"

    def __init__(self) -> None:
        self._user_dir = Path.home() / ".config" / "systemd" / "user"
        self._service_path = self._user_dir / self.SERVICE_NAME

    @property
    def service_path(self) -> Path:
        return self._service_path

    def install(self) -> None:
        exec_start = self._find_exec()
        working_dir = self._find_working_dir()
        env_file = self._find_env_file()

        template_path = _TEMPLATES_DIR / "gwenn-daemon.service"
        template = Template(template_path.read_text())
        content = template.safe_substitute(
            exec_start=exec_start,
            working_directory=working_dir,
            env_file=env_file,
            extra_environment="",
        )

        self._user_dir.mkdir(parents=True, exist_ok=True)
        self._service_path.write_text(content)
        logger.info("service.installed", path=str(self._service_path))

        self._systemctl("daemon-reload")
        self._systemctl("enable", self.SERVICE_NAME)

        # Enable linger so the service persists after logout
        try:
            subprocess.run(
                ["loginctl", "enable-linger", os.environ.get("USER", "")],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("service.linger_failed")

    def uninstall(self) -> None:
        if not self.is_installed():
            logger.info("service.not_installed")
            return
        self._systemctl("stop", self.SERVICE_NAME, check=False)
        self._systemctl("disable", self.SERVICE_NAME, check=False)
        self._service_path.unlink(missing_ok=True)
        self._systemctl("daemon-reload")
        logger.info("service.uninstalled")

    def start(self) -> None:
        self._systemctl("start", self.SERVICE_NAME)

    def stop(self) -> None:
        self._systemctl("stop", self.SERVICE_NAME)

    def restart(self) -> None:
        self._systemctl("restart", self.SERVICE_NAME)

    def status(self) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["systemctl", "--user", "show", self.SERVICE_NAME, "--no-pager"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"installed": self.is_installed(), "active": False}

        props: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                props[k] = v

        return {
            "installed": self.is_installed(),
            "active": props.get("ActiveState") == "active",
            "state": props.get("ActiveState", "unknown"),
            "sub_state": props.get("SubState", "unknown"),
            "pid": int(props.get("MainPID", "0")),
        }

    def is_installed(self) -> bool:
        return self._service_path.exists()

    @staticmethod
    def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = ["systemctl", "--user", *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=check)


class LaunchdManager(ServiceManager):
    """launchd service manager (macOS).

    Installs to ``~/Library/LaunchAgents/com.gwenn.daemon.plist``.
    """

    PLIST_NAME = "com.gwenn.daemon.plist"
    LABEL = "com.gwenn.daemon"

    def __init__(self) -> None:
        self._agents_dir = Path.home() / "Library" / "LaunchAgents"
        self._plist_path = self._agents_dir / self.PLIST_NAME

    @property
    def plist_path(self) -> Path:
        return self._plist_path

    def install(self) -> None:
        exec_start = self._find_exec()
        working_dir = self._find_working_dir()
        log_dir = str(Path.home() / "Library" / "Logs" / "Gwenn")
        path_env = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")

        template_path = _TEMPLATES_DIR / "com.gwenn.daemon.plist"
        template = Template(template_path.read_text())
        content = template.safe_substitute(
            exec_start=exec_start,
            working_directory=working_dir,
            log_dir=log_dir,
            path=path_env,
        )

        self._agents_dir.mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._plist_path.write_text(content)
        logger.info("service.installed", path=str(self._plist_path))

        self._launchctl("load", str(self._plist_path))

    def uninstall(self) -> None:
        if not self.is_installed():
            logger.info("service.not_installed")
            return
        self._launchctl("unload", str(self._plist_path), check=False)
        self._plist_path.unlink(missing_ok=True)
        logger.info("service.uninstalled")

    def start(self) -> None:
        self._launchctl("start", self.LABEL)

    def stop(self) -> None:
        self._launchctl("stop", self.LABEL)

    def restart(self) -> None:
        self.stop()
        self.start()

    def status(self) -> dict[str, Any]:
        if not self.is_installed():
            return {"installed": False, "active": False}
        try:
            result = subprocess.run(
                ["launchctl", "list", self.LABEL],
                capture_output=True,
                text=True,
            )
            active = result.returncode == 0
        except FileNotFoundError:
            active = False

        return {
            "installed": True,
            "active": active,
        }

    def is_installed(self) -> bool:
        return self._plist_path.exists()

    @staticmethod
    def _launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = ["launchctl", *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_service_manager() -> ServiceManager:
    """Auto-detect platform and return the appropriate service manager.

    Raises ``NotImplementedError`` on unsupported platforms.
    """
    if sys.platform.startswith("linux"):
        return SystemdManager()
    if sys.platform == "darwin":
        return LaunchdManager()
    raise NotImplementedError(
        f"Service management is not supported on {sys.platform}. "
        "Gwenn uses Unix-specific features (Unix sockets, signals)."
    )
