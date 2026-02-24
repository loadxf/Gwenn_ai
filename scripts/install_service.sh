#!/usr/bin/env bash
# install_service.sh — Install Gwenn as a systemd user service
#
# Usage: bash scripts/install_service.sh [--dry-run] [--help]
#
# Installs the daemon service for the current user, enables it on boot,
# and injects the absolute socket path into .env so the CLI works from
# any directory.

set -euo pipefail

# ---- Parse flags ----
DRY_RUN=false

usage() {
    echo "Usage: bash scripts/install_service.sh [--dry-run] [--help]"
    echo ""
    echo "Install Gwenn as a systemd user service."
    echo ""
    echo "Options:"
    echo "  --dry-run   Print what would happen without making changes"
    echo "  --help      Show this help message"
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help)    usage; exit 0 ;;
        *)
            echo "ERROR: Unknown flag: $arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_TEMPLATE="$SCRIPT_DIR/gwenn-daemon.service"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="gwenn-daemon.service"
SERVICE_DEST="$SYSTEMD_USER_DIR/$SERVICE_NAME"
ENV_FILE="$PROJECT_DIR/.env"

if $DRY_RUN; then
    echo "=== Gwenn Daemon Service Installer (DRY RUN) ==="
else
    echo "=== Gwenn Daemon Service Installer ==="
fi
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

# ---- 2. Idempotency check ----
if [ -f "$SERVICE_DEST" ]; then
    echo "WARNING: Service file already exists at $SERVICE_DEST"
    if $DRY_RUN; then
        echo "  (dry-run: would prompt for confirmation)"
    else
        read -rp "Overwrite and reinstall? [y/N] " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
fi

# ---- 3. Pre-create gwenn_data/ ----
if $DRY_RUN; then
    echo "Would create: $PROJECT_DIR/gwenn_data/"
else
    mkdir -p "$PROJECT_DIR/gwenn_data"
fi

# ---- 4. Inject absolute socket path into .env ----
ABS_SOCKET="$PROJECT_DIR/gwenn_data/gwenn.sock"
ABS_PID="$PROJECT_DIR/gwenn_data/gwenn.pid"
ABS_SESSIONS="$PROJECT_DIR/gwenn_data/sessions"

if [ ! -f "$ENV_FILE" ]; then
    echo "INFO: .env not found — creating from .env.example"
    if $DRY_RUN; then
        echo "Would copy: .env.example -> .env"
    else
        install -m 600 "$PROJECT_DIR/.env.example" "$ENV_FILE"
    fi
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

if $DRY_RUN; then
    echo "Would set in .env:"
    echo "  GWENN_DAEMON_SOCKET=$ABS_SOCKET"
    echo "  GWENN_DAEMON_PID_FILE=$ABS_PID"
    echo "  GWENN_DAEMON_SESSIONS_DIR=$ABS_SESSIONS"
else
    echo "Injecting absolute paths into .env..."
    _set_env "GWENN_DAEMON_SOCKET" "$ABS_SOCKET"
    _set_env "GWENN_DAEMON_PID_FILE" "$ABS_PID"
    _set_env "GWENN_DAEMON_SESSIONS_DIR" "$ABS_SESSIONS"

    if ! chmod 600 "$ENV_FILE"; then
        echo "WARNING: Could not set .env permissions to 600. Run manually:" >&2
        echo "  chmod 600 $ENV_FILE" >&2
    fi
fi

# ---- 5. Install systemd unit ----
if $DRY_RUN; then
    echo "Would install service: $SERVICE_DEST"
else
    mkdir -p "$SYSTEMD_USER_DIR"
    sed "s|INSTALL_DIR|$PROJECT_DIR|g" "$SERVICE_TEMPLATE" > "$SERVICE_DEST"
    echo "Service installed: $SERVICE_DEST"
fi

# ---- 6. Reload, enable, linger ----
if $DRY_RUN; then
    echo "Would run: systemctl --user daemon-reload"
    echo "Would run: systemctl --user enable $SERVICE_NAME"
    echo "Would run: loginctl enable-linger $USER"
else
    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE_NAME"
    echo "Service enabled."

    if ! loginctl enable-linger "$USER"; then
        echo "WARNING: Could not enable linger. Run manually:" >&2
        echo "  sudo loginctl enable-linger $USER" >&2
    else
        echo "Linger enabled for $USER."
    fi
fi

# ---- 7. Validate .env auth ----
if [ -f "$ENV_FILE" ]; then
    has_auth=false
    for key in ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN; do
        value=$(grep "^${key}=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2- || true)
        if [ -n "$value" ] && [ "$value" != "your_api_key_here" ]; then
            has_auth=true
            break
        fi
    done
    if ! $has_auth; then
        echo ""
        echo "WARNING: No auth key found in .env (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,"
        echo "  or CLAUDE_CODE_OAUTH_TOKEN). The daemon will fail to start without auth."
        echo "  Gwenn can also auto-detect OAuth from ~/.claude/.credentials.json."
    fi
fi

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
