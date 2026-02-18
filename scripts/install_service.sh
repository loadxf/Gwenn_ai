#!/usr/bin/env bash
# install_service.sh — Install Gwenn as a systemd user service
#
# Usage: bash scripts/install_service.sh
#
# Installs the daemon service for the current user, enables it on boot,
# and injects the absolute socket path into .env so the CLI works from
# any directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_TEMPLATE="$SCRIPT_DIR/gwenn-daemon.service"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="gwenn-daemon.service"
ENV_FILE="$PROJECT_DIR/.env"

echo "=== Gwenn Daemon Service Installer ==="
echo "Project directory: $PROJECT_DIR"

# ---- 1. Verify prerequisites ----
if [ ! -f "$SERVICE_TEMPLATE" ]; then
    echo "ERROR: Service template not found at $SERVICE_TEMPLATE" >&2
    exit 1
fi

if [ ! -f "$PROJECT_DIR/.venv/bin/gwenn" ]; then
    echo "ERROR: .venv/bin/gwenn not found. Run: uv sync && uv pip install -e ." >&2
    exit 1
fi

# ---- 2. Inject absolute socket path into .env ----
ABS_SOCKET="$PROJECT_DIR/gwenn_data/gwenn.sock"
ABS_PID="$PROJECT_DIR/gwenn_data/gwenn.pid"
ABS_SESSIONS="$PROJECT_DIR/gwenn_data/sessions"

if [ ! -f "$ENV_FILE" ]; then
    echo "INFO: .env not found — creating from .env.example"
    install -m 600 "$PROJECT_DIR/.env.example" "$ENV_FILE"
fi

_set_env() {
    local key="$1"
    local value="$2"
    if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

echo "Injecting absolute paths into .env..."
_set_env "GWENN_DAEMON_SOCKET" "$ABS_SOCKET"
_set_env "GWENN_DAEMON_PID_FILE" "$ABS_PID"
_set_env "GWENN_DAEMON_SESSIONS_DIR" "$ABS_SESSIONS"
chmod 600 "$ENV_FILE" 2>/dev/null || true

# ---- 3. Install systemd unit ----
mkdir -p "$SYSTEMD_USER_DIR"
SERVICE_DEST="$SYSTEMD_USER_DIR/$SERVICE_NAME"

sed "s|INSTALL_DIR|$PROJECT_DIR|g" "$SERVICE_TEMPLATE" > "$SERVICE_DEST"
echo "Service installed: $SERVICE_DEST"

# ---- 4. Reload, enable, linger ----
systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
echo "Service enabled."

# Allow service to start at login (linger)
loginctl enable-linger "$USER" 2>/dev/null || true
echo "Linger enabled for $USER."

echo ""
echo "=== Installation complete ==="
echo ""
echo "Commands:"
echo "  systemctl --user start $SERVICE_NAME        # start now"
echo "  systemctl --user stop $SERVICE_NAME         # stop"
echo "  systemctl --user status $SERVICE_NAME       # check status"
echo "  journalctl --user -u gwenn-daemon -f        # live logs"
echo "  gwenn                                       # connect CLI to daemon"
echo "  gwenn status                                # show daemon status"
echo "  gwenn stop                                  # graceful stop"
