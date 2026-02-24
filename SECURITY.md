# Security Policy

## Supported Versions

Gwenn AI is currently in active early development. The `main` branch always
contains the latest supported code. There are no tagged releases yet — once
versioned releases begin, this table will be updated.

| Version     | Supported          |
| ----------- | ------------------ |
| `main` HEAD | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Gwenn AI, **please do not open a
public issue.** Instead, report it privately so we can address it before it's
disclosed.

### How to report

Send an email to **[loadxf](https://github.com/loadxf)** via GitHub private
contact, or use
[GitHub's private vulnerability reporting](https://github.com/loadxf/Gwenn_ai/security/advisories/new)
to submit a report directly on this repository.

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- The component affected (e.g., `gwenn/harness/safety.py`, `gwenn/tools/executor.py`, `gwenn/privacy/redaction.py`)
- Your suggested severity (low / medium / high / critical)

### What to expect

- **Acknowledgement** within **48 hours** of your report.
- **Status update** within **7 days** with an initial assessment.
- If the vulnerability is accepted, we will work on a fix and coordinate
disclosure with you. You will be credited (unless you prefer to remain
anonymous).
- If the report is declined, we will explain why.

## Security Architecture

Gwenn AI has several built-in security layers — see the README for full
details. Key components:

- **Input validation & action filtering** — all user input is validated before
  processing.
- **Tool risk tiers** — every tool is classified as low / medium / high /
  critical. MCP-sourced tools are deny-by-default.
- **Sandbox policy enforcement** — with `GWENN_SANDBOX_ENABLED=True`, non-builtin
  tools are blocked unless explicitly allowlisted.
- **Rate limits & budget tracking** — API call budgets are enforced with a
  hard kill switch.
- **PII redaction** — configurable redaction of emails, phone numbers, SSNs,
  credit card numbers, and IP addresses in logs (disabled by default, enable
  via `GWENN_REDACTION_ENABLED`).
- **Daemon auth + local ACLs** — Unix socket permissions are owner-only, and
  optional protocol auth is available via `GWENN_DAEMON_AUTH_TOKEN`. Clients
  are disconnected after 3 consecutive auth failures to prevent brute-force.
- **Session privacy defaults** — daemon session previews are disabled by default
  and session content redaction defaults to enabled.
- **Media handling** — image downloads from Telegram/Discord are size-capped
  (20 MB), format-restricted (JPEG/PNG/GIF/WebP), and gated behind opt-in config
  flags (`TELEGRAM_ENABLE_MEDIA`, `DISCORD_ENABLE_MEDIA`). Image content blocks
  are preserved through redaction (no base64 corruption) and stripped from old
  messages during context compaction.
- **Subagent isolation** — subagents run with independent budgets, iteration limits,
  and optional Docker containerization (`GWENN_SUBAGENT_RUNTIME=docker`). Nesting
  depth is capped at 3 levels. Session-wide API call limits prevent runaway costs.
- **Data-at-rest hardening** — memory and session persistence applies restrictive
  file permissions on supported filesystems.
- **Identity deserialization safety** — crash-safe loading that gracefully
  ignores unknown or extra JSON keys from newer/older versions.
- **Heartbeat circuit breaker** — exponential backoff (60s base, 15-minute cap)
  prevents cascading failures; resets on success.
- **Thread-safe logging** — PII redaction in log fields uses a `lru_cache`
  singleton shared across entry points (main + daemon).

## Scope

The following are **in scope** for security reports:

- Bypassing safety guardrails (`gwenn/harness/safety.py`)
- Escaping the tool sandbox or escalating tool risk tiers
- PII leakage when redaction is enabled
- Prompt injection that causes Gwenn to violate her ethical constraints
- Memory poisoning (corrupting episodic or semantic memory stores)
- Unauthorized access to local data (SQLite databases, ChromaDB stores)

The following are **out of scope**:

- Vulnerabilities in upstream dependencies (report those to the relevant
  project, but feel free to let us know)
- Issues that require physical access to the machine running Gwenn
- Social engineering of the AI's personality (this is by design — her identity
  is emergent)

## Disclosure Policy

We follow coordinated disclosure. We ask that you give us a reasonable window
(typically 90 days) to address the issue before any public disclosure. We will
work with you on timing.
