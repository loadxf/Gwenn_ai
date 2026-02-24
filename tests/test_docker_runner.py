"""Tests for gwenn.orchestration.docker_manager — Docker container lifecycle.

These tests mock Docker CLI calls so they run without Docker installed.
For true integration tests, Docker must be available on the host.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from gwenn.orchestration.docker_manager import DockerManager, DENIED_MOUNT_PATTERNS


class TestDockerManager:
    @pytest.mark.asyncio
    async def test_check_available_with_docker(self):
        """Docker available when docker version succeeds."""
        manager = DockerManager()
        manager._available = None  # reset cache

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"24.0.0", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await manager.check_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_available_without_docker(self):
        """Docker unavailable when docker version fails."""
        manager = DockerManager()
        manager._available = None

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError,
        ):
            result = await manager.check_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_available_cached(self):
        """Second call uses cached result."""
        manager = DockerManager()
        manager._available = True
        result = await manager.check_available()
        assert result is True

    def test_mount_denied_patterns(self):
        manager = DockerManager()

        # These should be denied
        assert not manager._is_mount_allowed(Path("/home/user/gwenn_data/episodic.db"))
        assert not manager._is_mount_allowed(Path("/app/.env"))
        assert not manager._is_mount_allowed(Path("/home/user/.claude/config.json"))
        assert not manager._is_mount_allowed(Path("/etc/credentials/api.key"))
        assert not manager._is_mount_allowed(Path("/app/secrets/token"))

        # These should be allowed
        assert manager._is_mount_allowed(Path("/home/user/documents/research.txt"))
        assert manager._is_mount_allowed(Path("/tmp/data.csv"))
        assert manager._is_mount_allowed(Path("/app/src/main.py"))

    @pytest.mark.asyncio
    async def test_ensure_image_exists(self):
        manager = DockerManager()

        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=None)
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await manager.ensure_image()
        assert result is True

    @pytest.mark.asyncio
    async def test_ensure_image_builds_when_missing(self, tmp_path):
        """When image doesn't exist but Dockerfile does, build it."""
        # Create a fake Dockerfile
        docker_dir = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile.subagent").write_text("FROM python:3.11-slim")

        manager = DockerManager(project_root=tmp_path)

        call_count = 0

        async def mock_create_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            if call_count == 1:
                # docker image inspect — image not found
                proc.wait = AsyncMock(return_value=None)
                proc.returncode = 1
            else:
                # docker build — success
                proc.communicate = AsyncMock(return_value=(b"built", b""))
                proc.returncode = 0
            return proc

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=mock_create_subprocess,
        ):
            result = await manager.ensure_image()
        assert result is True

    @pytest.mark.asyncio
    async def test_kill_container(self):
        manager = DockerManager()

        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=None)
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await manager.kill_container("test-container")
        # Should not raise


class TestDeniedMountPatterns:
    def test_all_patterns_exist(self):
        assert "gwenn_data" in DENIED_MOUNT_PATTERNS
        assert ".env" in DENIED_MOUNT_PATTERNS
        assert ".claude" in DENIED_MOUNT_PATTERNS
        assert "credentials" in DENIED_MOUNT_PATTERNS
        assert "secrets" in DENIED_MOUNT_PATTERNS
        assert ".git" in DENIED_MOUNT_PATTERNS
