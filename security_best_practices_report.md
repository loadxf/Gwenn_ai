# Security Best Practices Report

## Executive Summary

Initial security audit date: **February 18, 2026**.
Last updated: **February 19, 2026**.

This report was updated after remediation work and verification.

Verification performed after fixes:
- `ruff check gwenn tests` (clean)
- `pytest -q` (`1371 passed`)
- Second-pass pattern scan over execution/auth/storage paths

### Additional hardening (2026-02-19)

- **SP-07 addendum:** Daemon now disconnects clients after 3 consecutive auth
  failures to prevent brute-force attempts (`daemon.py:_MAX_AUTH_FAILURES`).
- **Identity deserialization safety:** `_safe_dataclass_init()` in `identity.py`
  ignores unknown JSON keys, preventing `TypeError` on load from newer/older
  versions.
- **Thread-safe log redaction:** PII log redactor uses `@functools.lru_cache`
  singleton instead of mutable global state — no race conditions.
- **Heartbeat circuit breaker:** Exponential backoff (60s base, 15-minute cap)
  instead of fixed 60s cooldown. Resets on success.
- **Shared respond lock:** Daemon and channel adapters share the agent's canonical
  `_respond_lock` instead of maintaining redundant locks.

## Finding Status

### SP-01 Sandbox control gap
- Status: **Resolved**
- Evidence:
  - `gwenn/agent.py:167`
  - `gwenn/tools/executor.py:112`
  - `gwenn/tools/executor.py:190`
- Notes:
  - `GWENN_SANDBOX_ENABLED` is now wired into `ToolExecutor`.
  - In sandbox mode, non-builtin tools are blocked unless explicitly allowlisted.

### SP-02 `calculate` used `eval`
- Status: **Resolved**
- Evidence:
  - `gwenn/agent.py:1142`
  - `gwenn/agent.py:1235`
- Notes:
  - Replaced with AST-based evaluator with strict node/operator/function allowlist and complexity bounds.

### SP-03 Timed-out sync tools continued in unbounded threads
- Status: **Mitigated**
- Evidence:
  - `gwenn/tools/executor.py:114`
  - `gwenn/tools/executor.py:122`
  - `gwenn/tools/executor.py:308`
- Notes:
  - Added bounded sync concurrency (`max_concurrent_sync`) and saturation fail-closed behavior.
  - Residual: Python cannot forcibly kill arbitrary running threads; long-running sync handlers are bounded, not hard-killed.

### SP-04 DB permission hardening missing
- Status: **Resolved**
- Evidence:
  - `gwenn/memory/store.py:139`
  - `gwenn/memory/store.py:159`
  - `gwenn/memory/store.py:610`
- Notes:
  - Memory store now applies best-effort `0700` directory and `0600` file permissions for SQLite artifacts.

### SP-05 Installer created `.env` without restrictive perms
- Status: **Resolved**
- Evidence:
  - `scripts/install_service.sh:40`
  - `scripts/install_service.sh:57`
- Notes:
  - Installer now creates `.env` with `0600` and re-applies `chmod 600` after updates.

### SP-06 Discord slash commands bypassed guild allowlist
- Status: **Resolved**
- Evidence:
  - `gwenn/channels/discord_channel.py:179`
  - `gwenn/channels/discord_channel.py:193`
- Notes:
  - Slash commands now enforce guild allowlist checks consistently.

### SP-07 Daemon protocol lacked app-layer auth
- Status: **Resolved**
- Evidence:
  - `gwenn/config.py:305`
  - `gwenn/channels/cli_channel.py:39`
  - `gwenn/daemon.py:350`
- Notes:
  - Added optional `GWENN_DAEMON_AUTH_TOKEN`.
  - Client sends token; daemon validates using constant-time compare.

### SP-08 `fetch_url` read full bodies before truncation
- Status: **Resolved**
- Evidence:
  - `gwenn/agent.py:1337`
  - `gwenn/agent.py:1344`
- Notes:
  - Switched to bounded chunked reads with byte cap and truncation marker.

### SP-09 Session persistence leaked raw previews/content by default
- Status: **Mitigated**
- Evidence:
  - `gwenn/memory/session_store.py:59`
  - `gwenn/memory/session_store.py:121`
  - `gwenn/daemon.py:56`
  - `gwenn/config.py:312`
  - `gwenn/config.py:313`
- Notes:
  - Session list previews are now disabled by default unless explicitly enabled.
  - Daemon session transcripts are redacted by default before persistence.
  - Residual: session history is still intentionally persisted for `/resume`.

### SP-10 Vector-store permissions not hardened
- Status: **Resolved**
- Evidence:
  - `gwenn/memory/store.py:215`
  - `gwenn/memory/store.py:613`
- Notes:
  - Vector store directory is now hardened with best-effort restrictive permissions.

### SP-11 Privacy redaction defaults disabled globally
- Status: **Informational (unchanged)**
- Evidence:
  - `.env.example:61`
  - `gwenn/privacy/redaction.py:79`
- Notes:
  - Global privacy flags remain opt-in.
  - Session persistence now has separate secure defaults in daemon config.
