#!/usr/bin/env bash
# uninstall_service.sh — Remove Gwenn systemd user service
#
# DEPRECATED: Use 'gwenn uninstall' instead. This script will be removed in a future version.
#
# Usage: bash scripts/uninstall_service.sh [--keep-env] [--help]
#
# Stops and disables the daemon service, removes the unit file, and
# (optionally) reverts .env daemon paths to relative defaults.
#
# Does NOT remove gwenn_data/ (contains sessions and memory).
# Does NOT disable linger (other services may depend on it).

echo "WARNING: This script is deprecated. Use 'gwenn uninstall' instead." >&2
echo "" >&2

set -euo pipefail

# ---- Parse flags ----
KEEP_ENV=false

usage() {
    echo "Usage: bash scripts/uninstall_service.sh [--keep-env] [--help]"
    echo ""
    echo "Remove the Gwenn systemd user service."
    echo ""
    echo "Options:"
    echo "  --keep-env  Skip reverting .env daemon paths to relative defaults"
    echo "  --help      Show this help message"
}

for arg in "$@"; do
    case "$arg" in
        --keep-env) KEEP_ENV=true ;;
        --help)     usage; exit 0 ;;
        *)
            echo "ERROR: Unknown flag: $arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="gwenn-daemon.service"
SERVICE_DEST="$SYSTEMD_USER_DIR/$SERVICE_NAME"
ENV_FILE="$PROJECT_DIR/.env"

echo "=== Gwenn Daemon Service Uninstaller ==="
echo "Project directory: $PROJECT_DIR"

# ---- 1. Check if service is installed ----
if [ ! -f "$SERVICE_DEST" ]; then
    echo "Service file not found at $SERVICE_DEST — nothing to uninstall."
    exit 0
fi

# ---- 2. Stop and disable ----
echo "Stopping service..."
systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true

echo "Disabling service..."
systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true

# ---- 3. Remove unit file ----
rm -f "$SERVICE_DEST"
echo "Removed: $SERVICE_DEST"

# ---- 4. Reload systemd ----
systemctl --user daemon-reload
echo "Systemd daemon reloaded."

# ---- 5. Revert .env daemon paths to relative defaults ----
if $KEEP_ENV; then
    echo "Skipping .env revert (--keep-env)."
else
    if [ -f "$ENV_FILE" ]; then
        _revert_env() {
            local key="$1"
            local default_value="$2"
            if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
                sed -i "s|^${key}=.*|${key}=${default_value}|" "$ENV_FILE"
            fi
        }

        echo "Reverting .env daemon paths to relative defaults..."
        _revert_env "GWENN_DAEMON_SOCKET" "./gwenn_data/gwenn.sock"
        _revert_env "GWENN_DAEMON_PID_FILE" "./gwenn_data/gwenn.pid"
        _revert_env "GWENN_DAEMON_SESSIONS_DIR" "./gwenn_data/sessions"
        echo "Done."
    else
        echo "No .env found — skipping revert."
    fi
fi

echo ""
echo "=== Uninstallation complete ==="
echo ""
echo "Notes:"
echo "  - gwenn_data/ was NOT removed (contains sessions and memory)"
echo "  - Linger was NOT disabled (other services may depend on it)"
echo "  - To reinstall: bash scripts/install_service.sh"
