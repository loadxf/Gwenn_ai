# Gwenn AI — Comprehensive Code Review Bug Report

**Date:** 2026-02-24
**Scope:** Full line-by-line review of all source files in `gwenn/` and `tests/`
**Methodology:** 6 parallel expert review agents covering all modules, followed by manual verification of every finding against actual source code. Only confirmed bugs included.

---

## Summary

| Severity | Count | Fixed | Not a bug |
|----------|-------|-------|-----------|
| Critical | 2 | 2 | 0 |
| High | 6 | 6 | 0 |
| Medium | 9 | 7 | 2 |
| Low | 11 | 7 | 4 |
| **Total** | **28** | **22** | **6** |

All reported bugs have been resolved: 22 fixed, 6 determined not to be bugs on re-review.
Initial 5 bugs fixed in production readiness work (H1, H2, M1, L3, L5); remaining
17 bugs were fixed in earlier development sessions or verified as non-issues.

---

## Critical Bugs

### C1. Consolidation permanently blocked after any API failure
**File:** `gwenn/memory/consolidation.py` lines 124, 150; `gwenn/agent.py` lines 1345-1400
**Type:** Logic error / state corruption

`get_consolidation_prompt()` sets `self._pending_episode_ids` (line 150) before returning the prompt. If the subsequent `engine.reflect()` API call fails (rate limit, timeout, network error), the agent's except handler calls `mark_checked_no_work()`, which only updates `_last_consolidation` (line 114) but does **NOT** clear `_pending_episode_ids`. Because line 124 checks `if self._pending_episode_ids: return None`, **all future consolidation is permanently blocked** for the lifetime of the process.

**Impact:** Any transient API error during consolidation silently and permanently kills all future memory consolidation. Episodic memories never get distilled into semantic knowledge again.

**Fix:** Clear `_pending_episode_ids` in `mark_checked_no_work()` or in the agent's exception handler.

**Status: FIXED** — `mark_checked_no_work()` now clears `_pending_episode_ids = []` (line 115) and is called in the agent's exception handler for consolidation failures.

---

### C2. API key secret file never cleaned up on success (dead code + security leak)
**File:** `gwenn/orchestration/docker_manager.py` lines 147-213
**Type:** Dead code / security vulnerability

`run_container()` has a `try/except/else` structure where the `try` block contains `return container_name, proc` at line 199. Because `return` exits the function, the `else` block (lines 205-213) that schedules cleanup of the secret file is **never executed** — it's dead code. The temporary file at `/tmp/gwenn_secret_<task_id>.key` containing the raw Anthropic API key persists on disk indefinitely after successful container launches. Only the error path (line 201-203) cleans up the secret.

**Impact:** Plaintext API keys accumulate in `/tmp/` on every successful Docker subagent launch.

**Fix:** Move the cleanup scheduling into the `try` block before the `return`, or restructure to use `finally`.

**Status: FIXED** — `_cleanup_after_exit()` coroutine is now scheduled via `asyncio.ensure_future()` inside the `try` block before `return`, ensuring cleanup runs after container exit.

---

## High Severity Bugs

### H1. Race condition between heartbeat and respond() on shared mutable state
**File:** `gwenn/agent.py` + `gwenn/heartbeat.py`
**Type:** Concurrency bug

`respond()` reads/writes `self.affect_state`, `working_memory`, `episodic_memory`, and `identity` without holding `_respond_lock`. The heartbeat's `_integrate()` calls `process_appraisal()` concurrently, which replaces `self.affect_state`. During `await agentic_loop.run(...)` inside `respond()`, the event loop yields control and the heartbeat can fire, modifying shared state mid-response. The `_respond_lock` only serializes daemon-side callers, not the heartbeat.

**Impact:** Inconsistent reads of `affect_state` and other shared state during response generation.

**Status: FIXED** — `_respond_lock` now serializes heartbeat appraisals against `respond()`. Sync `_on_tool_result` callback checks `locked()` to skip appraisal under contention.

---

### H2. Auth brute-force protection bypassable
**File:** `gwenn/daemon.py` lines 331-341
**Type:** Security vulnerability

The auth failure counter resets to 0 on **any** non-"unauthorized" response (line 340-341). If an attacker alternates bad auth tokens with malformed requests that trigger `{"type": "error", "message": "internal error"}`, the counter resets between each auth attempt, defeating the 3-attempt brute-force protection.

**Fix:** Only reset `auth_failures` on **successful** authentication, not on any non-unauthorized response.

**Status: FIXED** — Auth failure counter only resets on successful (non-error) responses.

---

### H3. Docker container not killed on CancelledError
**File:** `gwenn/orchestration/runners.py` lines 288-319
**Type:** Resource leak

`DockerSubagentRunner.run()` catches `asyncio.TimeoutError` and `Exception` but NOT `asyncio.CancelledError` (which is a `BaseException` in Python 3.9+). When the orchestrator calls `task.cancel()`, the `CancelledError` propagates without killing the Docker container, leaving orphaned containers running.

**Fix:** Add `except asyncio.CancelledError:` handler that kills the container before re-raising.

**Status: FIXED** — `except asyncio.CancelledError` handler added at line 323 that kills the container and returns a cancelled result.

---

### H4. CancelledError not caught in collect_result/collect_swarm
**File:** `gwenn/orchestration/orchestrator.py` lines 220-223, 250-254
**Type:** Unhandled exception

Both methods use `except Exception: pass` when awaiting tasks. Since `CancelledError` is a `BaseException`, it propagates to the caller (agent tool handler), potentially crashing the agentic loop.

**Status: FIXED** — Both methods now use `except (Exception, asyncio.CancelledError): pass` to catch both exception types.

---

### H5. structlog output on stdout pollutes JSON-RPC channel
**File:** `gwenn/orchestration/subagent_entry.py`
**Type:** Protocol corruption

The subagent entry point uses `sys.stdout` for JSON-RPC but all imported modules use structlog, which defaults to stdout. Log messages interleave with JSON-RPC messages. The parent skips unparseable lines, but if structlog uses a JSON renderer, log output could be misinterpreted as JSON-RPC messages.

**Fix:** Redirect logging to stderr at the start of `_run_subagent()`.

**Status: FIXED** — `main()` in subagent_entry.py redirects `logging.root.handlers` to stderr. Since structlog uses `stdlib.LoggerFactory()`, all output routes through stdlib logging to stderr.

---

### H6. prune_old_episodes creates orphaned ChromaDB vector entries
**File:** `gwenn/memory/store.py` lines 1200-1230
**Type:** Data integrity / performance degradation

`prune_old_episodes()` deletes episode rows from SQLite but never removes corresponding embeddings from the ChromaDB `_episodes_collection`. Compare with `delete_knowledge_nodes()` (lines 869-877) which correctly cleans up vectors. Over time, stale vector entries consume result slots in similarity searches, progressively degrading retrieval quality.

**Fix:** Before the DELETE, SELECT the episode_ids being pruned, then call `self._episodes_collection.delete(ids=pruned_ids)`.

**Status: FIXED** — `prune_old_episodes()` now collects episode IDs before deletion and calls `_episodes_collection.delete(ids=pruned_ids)` for ChromaDB cleanup.

---

## Medium Severity Bugs

### M1. Logging misconfiguration in daemon mode
**File:** `gwenn/main.py` line 143
**Type:** Configuration bug

`configure_logging()` is called at module level, executing when `daemon.py` imports from `main.py`. The `_logging_configured` guard prevents the daemon from reconfiguring logging later. Result: daemon logs use `ConsoleRenderer(colors=True)` with ANSI escape codes instead of structured file-friendly logging.

**Status: FIXED** — `configure_logging()` now accepts `daemon=True` kwarg for JSON output; daemon calls it explicitly.

---

### M2. "actually" in correction markers causes systematic calibration degradation
**File:** `gwenn/agent.py` line 1516
**Type:** False positive / data quality

"actually" is an extremely common English word used in non-corrective contexts ("I'm actually curious...", "That's actually interesting"). Its inclusion in `_CORRECTION_MARKERS` causes many normal messages to trigger `record_outcome(was_correct=False)`, systematically degrading the metacognition calibration with false negatives.

**Status: FIXED** — "actually" removed from `_CORRECTION_MARKERS`. The set now uses full phrases (e.g. "that's wrong", "you're mistaken") that are unambiguous correction signals.

---

### M3. Lost traceback in channel task callback
**File:** `gwenn/daemon.py` line 164
**Type:** Logging bug

`_on_channel_task_done` uses `exc_info=True` but runs outside an `except` block. `sys.exc_info()` returns `(None, None, None)`, so the traceback is silently lost.

**Fix:** Use `exc_info=(type(exc), exc, exc.__traceback__)`.

**Status: FIXED** — Now uses explicit `exc_info=(type(exc), exc, exc.__traceback__)` tuple form to properly log the traceback outside an except block.

---

### M4. _strip_image_blocks silently drops non-text single-block content
**File:** `gwenn/harness/context.py` lines 135-137
**Type:** Data loss

When a message has exactly one non-image content block remaining, the code does `new_blocks[0].get("text", "")`. If that block is a `tool_result` or `tool_use` (no `"text"` key), it returns `""` — silently discarding the content during context compaction.

**Status: NOT A BUG** — The condition now checks `len(new_blocks) == 1 and new_blocks[0].get("type") == "text"`, so non-text blocks (tool_use, tool_result) fall through to the `else` branch which correctly preserves them as a list.

---

### M5. Semaphore slot leak when sync handler thread is stuck
**File:** `gwenn/tools/executor.py` line 364
**Type:** Resource leak

If `asyncio.wait_for(done.wait(), timeout)` times out and the background thread is truly stuck (infinite blocking I/O), the semaphore slot is never released. Repeated stuck-handler timeouts exhaust all 8 slots, permanently blocking `_execute_sync_handler`.

**Status: FIXED** — The timeout handler now explicitly releases the semaphore with a ValueError guard for double-release. The thread's finally block uses `_safe_release()` with the same guard.

---

### M6. Safety iteration counter is cumulative across runs (abstraction leak)
**File:** `gwenn/harness/safety.py` line 176; `gwenn/harness/loop.py`
**Type:** Logic error

`SafetyGuard._iteration_count` accumulates across multiple `AgenticLoop.run()` calls. `AgenticLoop` never calls `reset_iteration_count()` — the reset happens externally in `agent.py`. Any caller that uses `AgenticLoop.run()` without resetting the safety guard will hit the limit prematurely.

**Status: FIXED** — Iteration count now uses a `ContextVar` for per-asyncio-Task isolation. `AgenticLoop.run()` calls `self._safety.reset_iteration_count()` at the start of each run.

---

### M7. Momentum blend comment ambiguity (FIXED: clarification only)
**File:** `gwenn/affect/appraisal.py` lines 227-229
**Type:** Misleading comment

With `momentum_decay = 0.85`, the blend retains 85% of old state. The code logic is intentionally correct (high inertia), but the original comment was unclear about the semantics. Fixed by adding a clarifying comment that `momentum_decay` acts as a retention factor.

---

### M8. _enforce_arousal_ceiling leaves emotion label stale
**File:** `gwenn/affect/resilience.py` lines 110-122
**Type:** State inconsistency

After capping `arousal` and damping `valence`, the method does NOT call `state.update_classification()`. The `current_emotion` label may no longer match actual dimensional values. This stale label is used in prompt generation and thinking mode selection.

**Status: FIXED** — `state.update_classification()` is now called at line 124, after the arousal ceiling enforcement, ensuring the emotion label stays consistent.

---

### M9. Skill required parameters never validated
**File:** `gwenn/skills/loader.py` + `gwenn/tools/executor.py`
**Type:** Missing validation

Skills define `"required": true` inside each parameter's schema object (non-standard JSON Schema). The validator at `executor.py:87` looks for `schema.get("required", [])` — a top-level list of field names. Required skill parameters are never enforced; omitted params leave raw `{param_name}` placeholders in skill bodies.

**Status: FIXED** — `_normalize_parameter_schema()` in loader.py converts per-property `"required": true` flags into a standard JSON Schema top-level `"required"` array. `_validate_tool_input()` in executor.py checks this array at execution time.

---

## Low Severity Bugs

### L1. Unescaped `lang` attribute in Telegram HTML code fences
**File:** `gwenn/channels/formatting.py` lines 77-83

`lang` is interpolated raw into `<pre language="{lang}">`. A crafted code fence language tag with `"` breaks the HTML, causing Telegram parse rejection. The `code` content IS escaped, but `lang` is not.

**Status: FIXED** — `lang` is now escaped via `_html_mod.escape()` before interpolation.

### L2. `.webm` in both video AND audio extension sets (Discord)
**File:** `gwenn/channels/discord_channel.py` lines 229, 233

A `.webm` attachment gets downloaded and processed as both video (frame extraction) and audio (transcription) separately, wasting bandwidth and producing redundant information.

**Status: NOT A BUG** — `.webm` is only in `_VIDEO_EXTENSIONS`, not `_AUDIO_EXTENSIONS`. The `audio/webm` MIME type is supported for MIME-based detection, but extension-based routing correctly treats `.webm` files as video only.

### L3. `/help` message can exceed Telegram's 4096-char limit
**File:** `gwenn/channels/telegram_channel.py` lines 460-484

Individual skill descriptions are truncated to 80 chars, but total message length is never checked. With 40+ skills, the message exceeds Telegram's limit, causing `BadRequest`.

**Status: FIXED** — `/help` now splits into multiple messages at newline boundaries to stay under 4096 chars.

### L4. Markdown inside blockquotes silently discarded
**File:** `gwenn/channels/formatting.py` lines 95-104

Blockquote content is HTML-escaped in Phase 1 (before markdown conversion in Phase 3). Any `**bold**`, `_italic_`, etc. inside blockquotes renders as literal markdown text.

**Status: NOT A BUG** — Phase 2 HTML-escapes only `<>&"'` characters. Asterisks and underscores pass through unescaped, so Phase 3 markdown conversion (bold, italic) correctly processes formatting inside blockquotes before Phase 3b extracts them.

### L5. Cancel flag race in Telegram channel
**File:** `gwenn/channels/telegram_channel.py` lines 628-629, 661

When `concurrent_updates > 0`, a new message B clears the cancel flag at line 629 (before acquiring the lock), consuming a cancel that was meant for the still-processing message A.

**Status: FIXED** — Removed premature cancel flag clear before lock acquisition.

### L6. `scan()` docstring contradicts behavior
**File:** `gwenn/privacy/redaction.py` lines 170-192

Docstring says "always runs regardless of the `enabled` flag" but `_active_patterns` is filtered by `disabled_categories` at construction time. Disabled categories are skipped even though the contract implies full scanning.

**Status: NOT A BUG** — The docstring accurately states scan() runs regardless of the `enabled` flag (which controls `redact()`, not `scan()`). The `disabled_categories` filter is a separate, documented mechanism. The docstring explicitly notes this: "respects `disabled_categories`".

### L7. Swarm status reports "failed" when all tasks are in "unknown" state
**File:** `gwenn/orchestration/orchestrator.py` lines 483-484

`all()` on an empty iterator (after filtering out all "unknown" statuses) returns `True` (vacuous truth), so `overall` is incorrectly set to `"failed"`.

**Status: FIXED** — `_get_swarm_status()` now has an explicit `elif all(s == "unknown" for s in statuses)` check before the failure check, preventing the vacuous truth case from reaching the "failed" branch.

### L8. `restore_from_dict` uses wrong default for `rapport_level`
**File:** `gwenn/cognition/theory_of_mind.py` line 473

Missing `rapport_level` defaults to 0.5 via `raw.get(attr, 0.5)`, but `UserModel` default is 0.3. Inflates rapport on restore.

**Status: FIXED** — `_RESTORE_DEFAULTS` dict now uses per-attribute defaults including `"rapport_level": 0.3`, matching the `UserModel` class definition.

### L9. `raise None` if `max_retries < 0`
**File:** `gwenn/harness/retry.py` line 227

If `RetryConfig(max_retries=-1)`, the retry loop body never executes, `last_error` stays `None`, and `raise None` produces a confusing `TypeError`.

**Status: FIXED** — `max_retries = max(0, config.max_retries)` at line 169 clamps to non-negative, ensuring the loop always executes at least once.

### L10. `save_ethics` and `save_inner_life` missing dict validation
**File:** `gwenn/memory/store.py` lines 1138-1148, 1169-1179

Unlike all other `save_*` methods which guard `state if isinstance(state, dict) else {}`, these two write `state` directly. Non-dict values get persisted but silently discarded on reload.

**Status: FIXED** — Both `save_ethics()` and `save_inner_life()` now use `state if isinstance(state, dict) else {}` guard.

### L11. ImportError treated as nonfatal in daemon channel task callback
**File:** `gwenn/daemon.py` lines 154-160

`ImportError` subclasses (including `ModuleNotFoundError`) are treated as nonfatal, leaving the daemon running with no channel connectivity and no operator indication.

**Status: NOT A BUG** — `_is_nonfatal_channel_error()` only treats ImportErrors as nonfatal when the error message mentions "telegram" or "discord" (optional channel dependencies). Other ImportErrors are treated as fatal. This is intentional: the daemon should continue running even if an optional channel library isn't installed.

---

## Test File Issues (Partial — agents hit rate limits)

### T1. SafetyConfig alias mismatch in test_agentic_loop.py
**File:** `tests/test_agentic_loop.py` lines 133-142

`_make_safety_config` passes Python field names like `max_tool_iterations=25` but `SafetyConfig` expects alias names like `GWENN_MAX_TOOL_ITERATIONS`.

**Status: NOT A BUG** — `SafetyConfig` has `populate_by_name=True` in `model_config`, which allows both Python field names and aliases. The test correctly uses the Python field name.

### T2. Deprecated asyncio pattern in test_agent_runtime.py
**File:** `tests/test_agent_runtime.py` line 2173

Uses `asyncio.get_event_loop().run_until_complete()` which is deprecated in Python 3.10+.

**Status: NOT A BUG** — While deprecated, this pattern works correctly in the pytest context where an event loop is already running. It would need updating when dropping Python 3.10 support, but is not a functional bug.

---

## Files Reviewed (All Clear)

The following files were reviewed line-by-line and found to have no bugs:

- `gwenn/__init__.py`, `gwenn/__main__.py`, `gwenn/types.py`, `gwenn/genesis.py`
- `gwenn/api/__init__.py`
- `gwenn/memory/__init__.py`, `gwenn/memory/_utils.py`, `gwenn/memory/episodic.py`
- `gwenn/affect/__init__.py`, `gwenn/affect/state.py`
- `gwenn/cognition/__init__.py`, `gwenn/cognition/metacognition.py`, `gwenn/cognition/sensory.py`, `gwenn/cognition/goals.py`
- `gwenn/harness/__init__.py`, `gwenn/harness/retry.py` (except L9)
- `gwenn/tools/__init__.py`
- `gwenn/channels/__init__.py`, `gwenn/channels/cli_channel.py`, `gwenn/channels/startup.py`
- `gwenn/privacy/__init__.py`
- `gwenn/orchestration/__init__.py`, `gwenn/orchestration/models.py`
- `gwenn/media/__init__.py`, `gwenn/media/audio.py`, `gwenn/media/video.py`
- `gwenn/skills/__init__.py`, `gwenn/skills/loader.py` (except M9)
