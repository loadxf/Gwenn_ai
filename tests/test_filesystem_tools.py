"""
Tests for gwenn.tools.filesystem_context and the read_file / write_file tool
handlers.

Covers:
  - validate_path: unrestricted (main agent), scoped (subagent), denied (empty)
  - Symlink escape detection for scoped mode
  - Handler integration: read success, write success, offset/max_lines,
    append mode, access denied
  - Contextvar isolation: two asyncio tasks don't leak paths
  - Main agent unrestricted access (default contextvar)
  - Registration: tools present in registry with correct schemas
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from gwenn.tools.filesystem_context import (
    ALLOWED_FS_PATHS,
    validate_path,
)
from gwenn.tools.registry import ToolRegistry
from gwenn.tools.builtin import register_builtin_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def scoped_dir(tmp_path: Path):
    """Set tmp_path as the sole allowed path (simulating a subagent)."""
    token = ALLOWED_FS_PATHS.set((tmp_path,))
    yield tmp_path
    ALLOWED_FS_PATHS.reset(token)


@pytest.fixture()
def unrestricted():
    """Ensure the contextvar is at its default (None → main agent)."""
    token = ALLOWED_FS_PATHS.set(None)
    yield
    ALLOWED_FS_PATHS.reset(token)


@pytest.fixture()
def sample_file(scoped_dir: Path) -> Path:
    """Create a small sample text file inside the scoped directory."""
    f = scoped_dir / "sample.txt"
    f.write_text("line0\nline1\nline2\nline3\nline4\n", encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# validate_path — unit tests
# ---------------------------------------------------------------------------

class TestValidatePathUnrestricted:
    """Main agent (default=None) has unrestricted access."""

    def test_default_contextvar_is_none(self):
        """The factory default is None (unrestricted)."""
        # In a fresh context the default should be None
        token = ALLOWED_FS_PATHS.set(None)
        try:
            assert ALLOWED_FS_PATHS.get() is None
        finally:
            ALLOWED_FS_PATHS.reset(token)

    def test_any_path_allowed(self, unrestricted, tmp_path: Path):
        f = tmp_path / "main_agent.txt"
        f.write_text("full access")
        resolved, err = validate_path(str(f), require_exists=True)
        assert err is None
        assert resolved == f.resolve()

    def test_system_paths_allowed(self, unrestricted):
        """Main agent can read system files like /etc/hostname."""
        resolved, err = validate_path("/etc/hostname", require_exists=True)
        if Path("/etc/hostname").exists():
            assert err is None
        else:
            assert "not found" in err.lower()

    def test_require_exists_missing(self, unrestricted):
        _, err = validate_path("/nonexistent/path/file.txt", require_exists=True)
        assert err is not None
        assert "not found" in err.lower()

    def test_no_restriction_on_sensitive_names(self, unrestricted, tmp_path: Path):
        """Main agent can access paths with .git, .env, etc."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        f = git_dir / "config"
        f.write_text("ok")
        resolved, err = validate_path(str(f), require_exists=True)
        assert err is None


class TestValidatePathScoped:
    """Subagent with explicit allowed directories."""

    def test_empty_tuple_denies(self):
        """Empty tuple (subagent with no filesystem_paths) denies all."""
        token = ALLOWED_FS_PATHS.set(())
        try:
            _, err = validate_path("/tmp/anything")
            assert err is not None
            assert "no allowed paths" in err
        finally:
            ALLOWED_FS_PATHS.reset(token)

    def test_valid_path_succeeds(self, scoped_dir: Path):
        target = scoped_dir / "hello.txt"
        target.write_text("hi")
        resolved, err = validate_path(str(target), require_exists=True)
        assert err is None
        assert resolved == target.resolve()

    def test_path_outside_allowed_denied(self, scoped_dir: Path):
        _, err = validate_path("/etc/passwd")
        assert err is not None
        assert "not within any allowed path" in err

    def test_require_exists_missing(self, scoped_dir: Path):
        _, err = validate_path(str(scoped_dir / "nope.txt"), require_exists=True)
        assert err is not None
        assert "not found" in err.lower()

    def test_require_exists_false_allows_missing(self, scoped_dir: Path):
        resolved, err = validate_path(
            str(scoped_dir / "nope.txt"), require_exists=False
        )
        assert err is None

    def test_subdirectory_access(self, scoped_dir: Path):
        sub = scoped_dir / "sub" / "deep"
        sub.mkdir(parents=True)
        f = sub / "data.txt"
        f.write_text("ok")
        resolved, err = validate_path(str(f), require_exists=True)
        assert err is None
        assert resolved == f.resolve()

    def test_sensitive_names_allowed_within_scope(self, scoped_dir: Path):
        """Subagents have full access within their dirs — .git, .env, etc."""
        git_dir = scoped_dir / ".git"
        git_dir.mkdir()
        f = git_dir / "config"
        f.write_text("gitconfig")
        resolved, err = validate_path(str(f), require_exists=True)
        assert err is None

        env_file = scoped_dir / ".env"
        env_file.write_text("SECRET=yes")
        resolved, err = validate_path(str(env_file), require_exists=True)
        assert err is None

    def test_multiple_allowed_paths(self, tmp_path: Path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        f = dir_b / "ok.txt"
        f.write_text("yes")
        token = ALLOWED_FS_PATHS.set((dir_a, dir_b))
        try:
            resolved, err = validate_path(str(f), require_exists=True)
            assert err is None
            assert resolved == f.resolve()
        finally:
            ALLOWED_FS_PATHS.reset(token)

    def test_symlink_escape_denied(self, scoped_dir: Path):
        """A symlink that resolves outside allowed dirs is rejected."""
        link = scoped_dir / "escape_link"
        link.symlink_to("/tmp")
        _, err = validate_path(str(link / "file.txt"))
        assert err is not None
        assert "denied" in err.lower()


# ---------------------------------------------------------------------------
# Handler tests — read_file / write_file
# ---------------------------------------------------------------------------

def _wire_read_handler(registry: ToolRegistry):
    """Wire read_file handler inline (mirrors agent._wire_filesystem_tool_handlers)."""
    from gwenn.tools.filesystem_context import validate_path as _vp

    tool = registry.get("read_file")

    async def handle_read_file(
        path: str, max_lines: int = 500, offset: int = 0
    ) -> str:
        resolved, err = _vp(path, require_exists=True)
        if err:
            return err
        if not resolved.is_file():
            return f"Not a regular file: '{resolved}'."
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error reading file: {exc}"
        lines = text.splitlines(keepends=True)
        total = len(lines)
        selected = lines[offset : offset + max_lines]
        content = "".join(selected)
        if len(content) > 100_000:
            content = content[:100_000] + "\n... [truncated at 100 000 chars]"
        return (
            f"# {resolved}  (lines {offset}\u2013{offset + len(selected) - 1}"
            f" of {total})\n{content}"
        )

    tool.handler = handle_read_file


def _wire_write_handler(registry: ToolRegistry):
    """Wire write_file handler inline."""
    from gwenn.tools.filesystem_context import validate_path as _vp

    tool = registry.get("write_file")

    async def handle_write_file(
        path: str, content: str, mode: str = "write"
    ) -> str:
        resolved, err = _vp(path)
        if err:
            return err
        if mode not in ("write", "append"):
            return f"Invalid mode '{mode}'. Must be 'write' or 'append'."
        if len(content) > 100_000:
            content = content[:100_000]
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with resolved.open("a", encoding="utf-8") as f:
                    f.write(content)
            else:
                resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"Error writing file: {exc}"
        byte_count = len(content.encode("utf-8"))
        verb = "Appended to" if mode == "append" else "Wrote"
        return f"{verb} {resolved} ({byte_count} bytes)."

    tool.handler = handle_write_file


class TestReadFileHandler:
    """Tests for the read_file handler (scoped subagent)."""

    @pytest.fixture(autouse=True)
    def _registry(self, scoped_dir):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)
        _wire_read_handler(self.registry)
        self.scoped_dir = scoped_dir

    @pytest.mark.asyncio
    async def test_read_success(self, sample_file: Path):
        handler = self.registry.get("read_file").handler
        result = await handler(path=str(sample_file))
        assert "line0" in result
        assert "line4" in result

    @pytest.mark.asyncio
    async def test_read_with_offset(self, sample_file: Path):
        handler = self.registry.get("read_file").handler
        result = await handler(path=str(sample_file), offset=2, max_lines=2)
        assert "line2" in result
        assert "line3" in result
        assert "line0" not in result

    @pytest.mark.asyncio
    async def test_read_denied_path(self):
        handler = self.registry.get("read_file").handler
        result = await handler(path="/etc/passwd")
        assert "denied" in result.lower() or "not within" in result.lower()

    @pytest.mark.asyncio
    async def test_read_not_a_file(self):
        handler = self.registry.get("read_file").handler
        result = await handler(path=str(self.scoped_dir))
        assert "not a regular file" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_read_missing_file(self):
        handler = self.registry.get("read_file").handler
        result = await handler(path=str(self.scoped_dir / "ghost.txt"))
        assert "not found" in result.lower()


class TestReadFileHandlerUnrestricted:
    """read_file handler with main-agent (unrestricted) context."""

    @pytest.fixture(autouse=True)
    def _registry(self, unrestricted, tmp_path):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)
        _wire_read_handler(self.registry)
        self.tmp_path = tmp_path

    @pytest.mark.asyncio
    async def test_read_any_file(self):
        f = self.tmp_path / "anywhere.txt"
        f.write_text("main agent data")
        handler = self.registry.get("read_file").handler
        result = await handler(path=str(f))
        assert "main agent data" in result


class TestWriteFileHandler:
    """Tests for the write_file handler (scoped subagent)."""

    @pytest.fixture(autouse=True)
    def _registry(self, scoped_dir):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)
        _wire_write_handler(self.registry)
        self.scoped_dir = scoped_dir

    @pytest.mark.asyncio
    async def test_write_success(self):
        handler = self.registry.get("write_file").handler
        target = self.scoped_dir / "out.txt"
        result = await handler(path=str(target), content="hello world")
        assert "wrote" in result.lower()
        assert target.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_parents(self):
        handler = self.registry.get("write_file").handler
        target = self.scoped_dir / "a" / "b" / "deep.txt"
        result = await handler(path=str(target), content="deep")
        assert "wrote" in result.lower()
        assert target.read_text() == "deep"

    @pytest.mark.asyncio
    async def test_append_mode(self):
        handler = self.registry.get("write_file").handler
        target = self.scoped_dir / "log.txt"
        target.write_text("first\n")
        result = await handler(
            path=str(target), content="second\n", mode="append"
        )
        assert "appended" in result.lower()
        assert target.read_text() == "first\nsecond\n"

    @pytest.mark.asyncio
    async def test_write_denied_path(self):
        handler = self.registry.get("write_file").handler
        result = await handler(path="/etc/nope.txt", content="hax")
        assert "denied" in result.lower() or "not within" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_mode(self):
        handler = self.registry.get("write_file").handler
        target = self.scoped_dir / "x.txt"
        result = await handler(path=str(target), content="x", mode="truncate")
        assert "invalid mode" in result.lower()


class TestWriteFileHandlerUnrestricted:
    """write_file handler with main-agent (unrestricted) context."""

    @pytest.fixture(autouse=True)
    def _registry(self, unrestricted, tmp_path):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)
        _wire_write_handler(self.registry)
        self.tmp_path = tmp_path

    @pytest.mark.asyncio
    async def test_write_any_file(self):
        target = self.tmp_path / "anywhere.txt"
        handler = self.registry.get("write_file").handler
        result = await handler(path=str(target), content="main agent write")
        assert "wrote" in result.lower()
        assert target.read_text() == "main agent write"


# ---------------------------------------------------------------------------
# Contextvar isolation — asyncio tasks
# ---------------------------------------------------------------------------

class TestContextvarIsolation:
    """Verify that two concurrent asyncio tasks each see their own paths."""

    @pytest.mark.asyncio
    async def test_tasks_do_not_leak(self, tmp_path: Path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "fa.txt").write_text("A")
        (dir_b / "fb.txt").write_text("B")

        results: dict[str, str | None] = {}

        async def worker(name: str, allowed: Path, target: str):
            ALLOWED_FS_PATHS.set((allowed,))
            await asyncio.sleep(0.01)
            _, err = validate_path(target, require_exists=True)
            results[name] = err

        task_a = asyncio.create_task(
            worker("a", dir_a, str(dir_a / "fa.txt"))
        )
        task_b = asyncio.create_task(
            worker("b", dir_b, str(dir_b / "fb.txt"))
        )
        await asyncio.gather(task_a, task_b)

        assert results["a"] is None
        assert results["b"] is None

    @pytest.mark.asyncio
    async def test_cross_access_denied(self, tmp_path: Path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_b / "secret.txt").write_text("secret")

        result_err: str | None = None

        async def worker():
            ALLOWED_FS_PATHS.set((dir_a,))
            _, err = validate_path(str(dir_b / "secret.txt"))
            nonlocal result_err
            result_err = err

        await asyncio.create_task(worker())
        assert result_err is not None
        assert "not within" in result_err

    @pytest.mark.asyncio
    async def test_unrestricted_does_not_leak_to_scoped(self, tmp_path: Path):
        """A task that sets scoped paths is still scoped even if parent is unrestricted."""
        f = tmp_path / "outside.txt"
        f.write_text("outside")

        scoped_dir = tmp_path / "scoped"
        scoped_dir.mkdir()

        result_err: str | None = None

        async def scoped_worker():
            ALLOWED_FS_PATHS.set((scoped_dir,))
            _, err = validate_path(str(f))
            nonlocal result_err
            result_err = err

        # Parent context is unrestricted (default)
        await asyncio.create_task(scoped_worker())
        assert result_err is not None
        assert "not within" in result_err


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestRegistration:
    """Verify read_file / write_file appear in the registry correctly."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)

    def test_read_file_registered(self):
        tool = self.registry.get("read_file")
        assert tool is not None
        assert tool.is_builtin is True
        assert tool.risk_level == "medium"
        assert tool.category == "filesystem"
        schema = tool.input_schema
        assert "path" in schema["properties"]
        assert "path" in schema["required"]

    def test_write_file_registered(self):
        tool = self.registry.get("write_file")
        assert tool is not None
        assert tool.is_builtin is True
        assert tool.risk_level == "medium"
        assert tool.category == "filesystem"
        schema = tool.input_schema
        assert "path" in schema["properties"]
        assert "content" in schema["properties"]
        assert set(schema["required"]) == {"path", "content"}

    def test_read_file_schema_defaults(self):
        tool = self.registry.get("read_file")
        props = tool.input_schema["properties"]
        assert props["max_lines"]["default"] == 500
        assert props["offset"]["default"] == 0

    def test_write_file_mode_enum(self):
        tool = self.registry.get("write_file")
        mode_prop = tool.input_schema["properties"]["mode"]
        assert set(mode_prop["enum"]) == {"write", "append"}
