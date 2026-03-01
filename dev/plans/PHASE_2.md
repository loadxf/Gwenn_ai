# Phase 2: Cross-Platform Service Management

**Risk: LOW** — Self-contained addition. No changes to core logic.

**Prerequisites:** Phase 1 complete (daemon calls `heartbeat.run()`).

---

## Goal

Replace shell scripts with Python service management. The service runs `heartbeat.run()` via the daemon.

---

## New Files

### `gwenn/service.py` — Platform Abstraction

```python
class ServiceManager(ABC):
    """Abstract base for OS service management."""
    @abstractmethod
    def install(self) -> None: ...
    @abstractmethod
    def uninstall(self) -> None: ...
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def restart(self) -> None: ...
    @abstractmethod
    def status(self) -> dict: ...
    @abstractmethod
    def is_installed(self) -> bool: ...

class SystemdManager(ServiceManager):
    """Full implementation — replaces scripts/install_service.sh"""
    # Renders gwenn-daemon.service template with:
    #   - ExecStart pointing to `gwenn daemon`
    #   - WorkingDirectory, User, Group from config or detection
    #   - Restart=on-failure, RestartSec=5
    # Installs to ~/.config/systemd/user/ (user mode) or /etc/systemd/system/ (system mode)
    # Calls systemctl daemon-reload, enable, start

class LaunchdManager(ServiceManager):
    """Basic macOS implementation"""
    # Renders com.gwenn.daemon.plist template
    # Installs to ~/Library/LaunchAgents/
    # Calls launchctl load/unload

# No Windows stub — Gwenn uses Unix-specific features (Unix sockets,
# signal handling, /tmp paths). Windows is out of scope.

def get_service_manager() -> ServiceManager:
    """Auto-detect OS: Linux → SystemdManager, macOS → LaunchdManager.
    Raises NotImplementedError on unsupported platforms."""
```

**Implementation notes:**
- Template rendering: use `string.Template` or plain f-strings (no Jinja2 dependency needed)
- Path detection: use `shutil.which("gwenn")` for ExecStart, `Path.home()` for WorkingDirectory
- Validation: check that `gwenn daemon` is callable before installing
- Permissions: user-mode systemd by default (no sudo), with option for system-mode

### `gwenn/templates/gwenn-daemon.service` — systemd template

Move from current location at `scripts/gwenn-daemon.service`. Template with placeholders:

```ini
[Unit]
Description=Gwenn AI Daemon
After=network.target

[Service]
Type=simple
ExecStart=${exec_start}
WorkingDirectory=${working_directory}
Restart=on-failure
RestartSec=5
Environment=PATH=${path}
${environment_lines}

[Install]
WantedBy=default.target
```

### `gwenn/templates/com.gwenn.daemon.plist` — launchd template

New file for macOS:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "...">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.gwenn.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>${exec_start}</string>
        <string>daemon</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${working_directory}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${log_dir}/gwenn.out.log</string>
    <key>StandardErrorPath</key>
    <string>${log_dir}/gwenn.err.log</string>
</dict>
</plist>
```

### `tests/test_service.py`

- Test template rendering with various config combinations
- Test `is_installed()` detection (mock filesystem)
- Test `get_service_manager()` platform detection (mock `sys.platform`)
- Test install/uninstall file creation/deletion (mock subprocess for systemctl/launchctl)

---

## Modified Files

### `gwenn/main.py`

Add `"install"`, `"uninstall"`, `"restart"` subcommands to argparse:

```python
parser.add_argument(
    "subcommand", nargs="?",
    choices=["daemon", "stop", "status", "install", "uninstall", "restart"],
    default=None,
)
```

Route to:
- `install` → `get_service_manager().install()`
- `uninstall` → `get_service_manager().uninstall()`
- `restart` → `get_service_manager().restart()`

### `scripts/install_service.sh` / `scripts/uninstall_service.sh`

Add deprecation notice at top:
```bash
echo "WARNING: This script is deprecated. Use 'gwenn install' / 'gwenn uninstall' instead."
```

### `pyproject.toml`

Add `[tool.setuptools.package-data]` to include templates:
```toml
[tool.setuptools.package-data]
gwenn = ["templates/*"]
```

---

## Implementation Sub-Steps

```
2a. Create gwenn/service.py + templates (gwenn-daemon.service, com.gwenn.daemon.plist)
2b. Add install/uninstall/restart subcommands to main.py
2c. Add deprecation notices to scripts/install_service.sh and uninstall_service.sh
2d. Add package-data config to pyproject.toml
2e. Write tests
```

All in a single commit.

---

## Verification

- `gwenn install` → systemd service installed → `gwenn status` shows running
- `gwenn restart` → service restarts cleanly
- `gwenn stop` → service stops
- `gwenn uninstall` → service removed
- On macOS: `gwenn install` → launchd plist installed → `launchctl list | grep gwenn`
- Old scripts still work but show deprecation warning
