"""
Docker Manager — Container Lifecycle for Subagents.

Handles building the subagent image, launching containers with proper isolation
settings, monitoring their status, and cleaning up on shutdown.

Security:
  - API keys are injected via temporary secret file mounts (not env vars)
  - Containers run with --network=none by default
  - Memory and CPU limits are enforced
  - Sensitive paths (gwenn_data/, .env, credentials) are never mounted
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

import structlog

from gwenn.orchestration.models import SubagentSpec

logger = structlog.get_logger(__name__)

# Paths that are never mounted into subagent containers
DENIED_MOUNT_PATTERNS = frozenset(
    {
        "gwenn_data",
        ".env",
        ".claude",
        "credentials",
        "secrets",
        ".git",
    }
)


class DockerManager:
    """Manages Docker container lifecycle for subagent execution."""

    def __init__(
        self,
        image: str = "gwenn-subagent:latest",
        network: str = "none",
        memory_limit: str = "256m",
        cpu_limit: float = 0.5,
        project_root: Optional[Path] = None,
    ):
        self._image = image
        self._network = network
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        if project_root is not None:
            self._project_root = project_root
        else:
            # Resolve relative to the package root, not the working directory.
            self._project_root = Path(__file__).resolve().parent.parent.parent
        self._available: Optional[bool] = None

    async def check_available(self) -> bool:
        """Check if Docker is available on this system."""
        if self._available is not None:
            return self._available

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "version",
                "--format",
                "{{.Server.Version}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            self._available = proc.returncode == 0
            if self._available:
                version = stdout.decode().strip()
                logger.info("docker.available", version=version)
            else:
                logger.warning("docker.unavailable")
        except (FileNotFoundError, asyncio.TimeoutError):
            self._available = False
            logger.warning("docker.not_found")

        return self._available

    async def ensure_image(self) -> bool:
        """Check if the subagent image exists, build if missing."""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            self._image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

        if proc.returncode == 0:
            return True

        # Image not found — try to build
        dockerfile = self._project_root / "docker" / "Dockerfile.subagent"
        if not dockerfile.exists():
            logger.error("docker.dockerfile_missing", path=str(dockerfile))
            return False

        logger.info("docker.building_image", image=self._image)
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "-t",
            self._image,
            "-f",
            str(dockerfile),
            str(self._project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error(
                "docker.build_failed",
                stderr=stderr.decode()[:500],
            )
            return False

        logger.info("docker.image_built", image=self._image)
        return True

    async def run_container(
        self,
        spec: SubagentSpec,
        api_key: str,
    ) -> tuple[str, asyncio.subprocess.Process]:
        """Launch a Docker container for a subagent.

        Returns (container_id_prefix, subprocess_process).
        """
        container_name = f"gwenn-sub-{spec.task_id}"

        # Create temporary secret file for API key with restricted permissions
        secret_path = os.path.join(
            tempfile.gettempdir(), f"gwenn_secret_{spec.task_id}.key"
        )
        fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as secret_file:
                secret_file.write(api_key)

            # Build docker run command
            cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--network",
                self._network,
                f"--memory={self._memory_limit}",
                f"--cpus={self._cpu_limit}",
                "--label",
                "gwenn-subagent=true",
                "--label",
                f"gwenn-pid={os.getpid()}",
                "-i",  # interactive (stdin open)
                "-v",
                f"{secret_path}:/run/secrets/api_key:ro",
            ]

            # Add filesystem mounts (validated)
            for path_str in spec.filesystem_access:
                path = Path(path_str).resolve()
                if self._is_mount_allowed(path):
                    cmd.extend(["-v", f"{path}:{path}:rw"])
                else:
                    logger.warning(
                        "docker.mount_denied",
                        path=str(path),
                        task_id=spec.task_id,
                    )

            cmd.extend([self._image])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            logger.info(
                "docker.container_started",
                container=container_name,
                task_id=spec.task_id,
                network=self._network,
            )

            # Schedule cleanup after container process completes
            async def _cleanup_after_exit() -> None:
                try:
                    await proc.wait()
                finally:
                    self._cleanup_secret(secret_path)
            asyncio.ensure_future(_cleanup_after_exit())

            return container_name, proc

        except Exception:
            # If container launch failed, clean up the secret file immediately
            self._cleanup_secret(secret_path)
            raise

    async def kill_container(self, container_name: str) -> None:
        """Kill a running container."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "kill",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10.0)
            logger.info("docker.container_killed", container=container_name)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(
                "docker.kill_failed",
                container=container_name,
                error=str(exc),
            )

    async def cleanup_orphans(self) -> int:
        """Kill orphaned subagent containers whose parent PID no longer exists."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "-q",
                "--filter",
                "label=gwenn-subagent=true",
                "--format",
                "{{.ID}}\t{{.Labels}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()

            if not stdout:
                return 0

            killed = 0
            for line in stdout.decode().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                container_id = parts[0]

                # Try to get the parent PID from container labels
                try:
                    inspect_proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "inspect",
                        "--format",
                        '{{index .Config.Labels "gwenn-pid"}}',
                        container_id,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    inspect_out, _ = await inspect_proc.communicate()
                    parent_pid_str = inspect_out.decode().strip()

                    if parent_pid_str and int(parent_pid_str) != os.getpid():
                        # Check if parent PID is still alive
                        try:
                            os.kill(int(parent_pid_str), 0)
                        except (OSError, ValueError):
                            # Parent is dead — kill the orphan
                            await self.kill_container(container_id)
                            killed += 1
                except Exception:
                    pass

            if killed:
                logger.info("docker.orphans_cleaned", count=killed)
            return killed

        except Exception as exc:
            logger.warning("docker.orphan_cleanup_failed", error=str(exc))
            return 0

    async def cleanup_all(self) -> None:
        """Kill all subagent containers owned by this process."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "-q",
                "--filter",
                "label=gwenn-subagent=true",
                "--filter",
                f"label=gwenn-pid={os.getpid()}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()

            if stdout:
                container_ids = stdout.decode().strip().split("\n")
                for cid in container_ids:
                    if cid.strip():
                        await self.kill_container(cid.strip())

        except Exception as exc:
            logger.warning("docker.cleanup_all_failed", error=str(exc))

    def _is_mount_allowed(self, path: Path) -> bool:
        """Check if a path is safe to mount into a container.

        Uses path component matching instead of substring matching to avoid
        false positives (e.g. ``.git`` matching ``.github/``).
        """
        parts = [p.lower() for p in path.parts]
        return not any(pattern in parts for pattern in DENIED_MOUNT_PATTERNS)

    @staticmethod
    def _cleanup_secret(path: str) -> None:
        """Remove a temporary secret file."""
        try:
            os.unlink(path)
        except OSError:
            pass
