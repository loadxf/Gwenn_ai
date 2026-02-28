"""
Orchestrator — The Central Coordination Engine.

Manages the full lifecycle of subagent tasks: spawning, monitoring, collecting
results, and aggregating swarm outputs. The Orchestrator is per-agent
(daemon-global) — all CLI sessions share the same instance, budget tracker,
and concurrency semaphore.

Key responsibilities:
  - Enforce concurrency limits (max parallel subagents)
  - Track session-wide API call budgets
  - Handle nested spawn requests from child subagents
  - Aggregate swarm results (concatenate / synthesize / vote)
  - Cancel running subagents and propagate cancellation to children
  - Clean up resources on shutdown
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import Any, Optional

import structlog

from gwenn.orchestration.models import (
    SubagentProgress,
    SubagentResult,
    SubagentSpec,
    SwarmResult,
    SwarmSpec,
)
from gwenn.orchestration.runners import SubagentRunnerBase

# Lazy imports for event types to avoid circular dependencies.
# These are imported inside methods that emit events.

logger = structlog.get_logger(__name__)


class Orchestrator:
    """Manages subagent lifecycle: spawn, monitor, collect, aggregate."""

    def __init__(
        self,
        config: Any,  # OrchestrationConfig
        runner: Optional[SubagentRunnerBase] = None,
        engine: Any = None,  # CognitiveEngine — for synthesize aggregation
        redactor: Any = None,  # PIIRedactor — for result redaction before storage
        event_bus: Any = None,  # EventBus — for emitting swarm events
        bot_pool: Any = None,  # TelegramBotPool — for swarm visualization
    ):
        self._config = config
        self._runner = runner
        self._engine = engine
        self._redactor = redactor
        self._event_bus = event_bus
        self._bot_pool = bot_pool

        # Active tasks: task_id -> asyncio.Task
        self._active_tasks: dict[str, asyncio.Task] = {}
        # Completed results: task_id -> SubagentResult
        self._completed_results: dict[str, SubagentResult] = {}
        # Progress tracking: task_id -> SubagentProgress
        self._progress: dict[str, SubagentProgress] = {}

        # Swarm tracking: swarm_id -> SwarmSpec
        self._active_swarms: dict[str, SwarmSpec] = {}
        self._swarm_tasks: dict[str, list[str]] = {}  # swarm_id -> [task_ids]

        # Origin session tracking: task_id -> session_id (for routing results back)
        self._origin_sessions: dict[str, str] = {}

        # Concurrency control
        self._concurrency_semaphore = asyncio.Semaphore(config.max_concurrent_subagents)

        # Session-wide budget tracker
        self._total_api_calls = 0
        self._autonomous_spawn_times: list[float] = []

        logger.info(
            "orchestrator.initialized",
            max_concurrent=config.max_concurrent_subagents,
            max_depth=config.max_nesting_depth,
            max_api_calls=config.max_total_api_calls,
        )

    @property
    def active_count(self) -> int:
        return len(self._active_tasks)

    @property
    def completed_count(self) -> int:
        return len(self._completed_results)

    async def spawn(self, spec: SubagentSpec) -> str:
        """Spawn a subagent for a single task. Returns the task_id."""
        if not self._config.enabled:
            raise RuntimeError("Orchestration is disabled")

        if self._runner is None:
            raise RuntimeError("No subagent runner configured")

        # Validate nesting depth
        if spec.depth >= self._config.max_nesting_depth:
            raise ValueError(
                f"Maximum nesting depth ({self._config.max_nesting_depth}) exceeded "
                f"at depth {spec.depth}"
            )

        # Check budget
        if self._total_api_calls >= self._config.max_total_api_calls:
            raise RuntimeError(
                f"Subagent API call budget exhausted "
                f"({self._total_api_calls}/{self._config.max_total_api_calls})"
            )

        task_id = spec.task_id

        # Emit TaskDispatchMessage for swarm visualization.
        if spec.parent_task_id or spec.persona:
            try:
                from gwenn.events import TaskDispatchMessage
                self._emit_event(TaskDispatchMessage(
                    sender_task_id="coordinator",
                    recipient_task_id=task_id,
                    task_description=spec.task_description,
                    assigned_persona_name=spec.persona.name if spec.persona else None,
                ))
            except Exception:
                pass

        # Apply defaults from config
        if not spec.model and self._config.subagent_model:
            spec.model = self._config.subagent_model
        if spec.timeout_seconds <= 0:
            spec.timeout_seconds = self._config.default_timeout
        if spec.max_iterations <= 0:
            spec.max_iterations = self._config.default_max_iterations
        if not spec.tools and self._config.default_tools:
            spec.tools = list(self._config.default_tools)

        # Record progress
        self._progress[task_id] = SubagentProgress(
            task_id=task_id,
            status="pending",
            started_at=time.time(),
            parent_task_id=spec.parent_task_id,
            runtime_tier=spec.runtime_tier,
        )

        # Track origin session for routing results back to the right chat/topic
        if spec.origin_session_id:
            self._origin_sessions[task_id] = spec.origin_session_id

        # Launch as asyncio.Task
        task = asyncio.create_task(self._run_with_semaphore(spec))
        self._active_tasks[task_id] = task
        task.add_done_callback(lambda t: self._on_task_done(task_id, t))

        logger.info(
            "orchestration.spawn",
            task_id=task_id,
            depth=spec.depth,
            runtime=spec.runtime_tier,
            tools=spec.tools,
        )

        return task_id

    async def spawn_swarm(self, swarm: SwarmSpec) -> str:
        """Spawn multiple subagents in parallel. Returns swarm_id."""
        if not self._config.enabled:
            raise RuntimeError("Orchestration is disabled")

        if len(self._active_swarms) >= self._config.max_active_swarms:
            raise RuntimeError(f"Maximum active swarms ({self._config.max_active_swarms}) reached")

        swarm_id = swarm.swarm_id
        self._active_swarms[swarm_id] = swarm
        task_ids: list[str] = []

        try:
            for agent_spec in swarm.agents:
                # Inherit swarm timeout if agent doesn't specify
                if agent_spec.timeout_seconds <= 0:
                    agent_spec.timeout_seconds = swarm.timeout_seconds

                task_id = await self.spawn(agent_spec)
                task_ids.append(task_id)
        except Exception:
            # Cancel any already-spawned agents on partial failure
            for tid in task_ids:
                await self.cancel(tid)
            self._active_swarms.pop(swarm_id, None)
            raise

        self._swarm_tasks[swarm_id] = task_ids

        # Acquire bots from pool for swarm visualization (Phase 6).
        await self._acquire_swarm_bots(swarm_id, swarm.agents)

        logger.info(
            "orchestration.spawn_swarm",
            swarm_id=swarm_id,
            agent_count=len(swarm.agents),
            strategy=swarm.aggregation_strategy,
        )

        return swarm_id

    async def check_status(self, task_id: str) -> dict[str, Any]:
        """Check the status of a running or completed subagent."""
        # Check completed results first
        if task_id in self._completed_results:
            result = self._completed_results[task_id]
            return {
                "task_id": task_id,
                "status": result.status,
                "elapsed_seconds": result.elapsed_seconds,
                "result_preview": result.result_text[:200] if result.result_text else "",
            }

        # Check active tasks
        if task_id in self._active_tasks:
            progress = self._progress.get(task_id)
            elapsed = time.time() - progress.started_at if progress else 0.0
            return {
                "task_id": task_id,
                "status": "running",
                "elapsed_seconds": round(elapsed, 2),
            }

        # Check if it's a swarm
        if task_id in self._active_swarms:
            return self._get_swarm_status(task_id)

        return {"task_id": task_id, "status": "not_found"}

    async def collect_result(
        self,
        task_id: str,
        full: bool = False,
    ) -> Optional[SubagentResult]:
        """Collect result from a completed subagent. Returns None if still running."""
        # If still running, wait for it
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            try:
                await task
            except (Exception, asyncio.CancelledError):
                pass
            # Yield so the done-callback can populate _completed_results
            await asyncio.sleep(0)

        result = self._completed_results.get(task_id)
        if result is None:
            return None

        if not full and result.result_text and len(result.result_text) > 2000:
            result = result.model_copy(
                update={"result_text": result.result_text[:2000] + "\n[truncated]"}
            )
        return result

    async def collect_swarm(
        self,
        swarm_id: str,
        full: bool = False,
    ) -> Optional[SwarmResult]:
        """Collect aggregated results from a completed swarm."""
        if swarm_id not in self._swarm_tasks:
            return None

        task_ids = self._swarm_tasks[swarm_id]

        # Wait for all tasks to complete
        for tid in task_ids:
            if tid in self._active_tasks:
                try:
                    await self._active_tasks[tid]
                except (Exception, asyncio.CancelledError):
                    pass
        # Yield so done-callbacks can populate _completed_results
        await asyncio.sleep(0)

        # Gather results
        individual: list[SubagentResult] = []
        total_tokens = 0
        total_elapsed = 0.0

        for tid in task_ids:
            result = self._completed_results.get(tid)
            if result:
                individual.append(result)
                total_tokens += result.tokens_used
                total_elapsed = max(total_elapsed, result.elapsed_seconds)

        # Determine overall status
        statuses = [r.status for r in individual]
        if all(s == "completed" for s in statuses):
            overall = "completed"
        elif all(s in {"failed", "timeout", "cancelled"} for s in statuses):
            overall = "failed"
        elif any(s == "cancelled" for s in statuses):
            overall = "cancelled"
        else:
            overall = "partial"

        # Aggregate results
        swarm_spec = self._active_swarms.get(swarm_id)
        strategy = swarm_spec.aggregation_strategy if swarm_spec else "concatenate"
        aggregated = await self._aggregate_results(individual, strategy)

        swarm_result = SwarmResult(
            swarm_id=swarm_id,
            status=overall,
            individual_results=individual,
            aggregated_result=aggregated,
            total_elapsed_seconds=round(total_elapsed, 2),
            total_tokens_used=total_tokens,
        )

        # Release swarm bots back to pool (Phase 6).
        await self._release_swarm_bots(swarm_id, task_ids)

        # Clean up
        self._active_swarms.pop(swarm_id, None)
        self._swarm_tasks.pop(swarm_id, None)

        return swarm_result

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running subagent and all its children."""
        cancelled = False

        # Cancel children first (depth-first)
        children = [
            tid
            for tid, p in self._progress.items()
            if p.parent_task_id == task_id and tid in self._active_tasks
        ]
        for child_id in children:
            await self.cancel(child_id)

        # Cancel the task itself
        if task_id in self._active_tasks:
            self._active_tasks[task_id].cancel()
            cancelled = True
            logger.info("orchestration.cancel", task_id=task_id)

        # Cancel swarm
        if task_id in self._swarm_tasks:
            for tid in self._swarm_tasks[task_id]:
                await self.cancel(tid)
            cancelled = True

        return cancelled

    def collect_completed(self) -> list[SubagentResult]:
        """Return all completed results (used by heartbeat integration).

        Unlike collect_result(), this does NOT remove entries from the results
        dict, so user-triggered collect_result() calls still work afterwards.
        """
        return list(self._completed_results.values())

    def get_origin_session(self, task_id: str) -> str | None:
        """Return the originating session_id for a task, or None."""
        return self._origin_sessions.get(task_id)

    async def handle_nested_spawn(
        self,
        parent_task_id: str,
        spec: SubagentSpec,
    ) -> str:
        """Handle a nested spawn request from a child subagent."""
        spec.parent_task_id = parent_task_id
        # Depth is set by the caller, but validate it
        if spec.depth >= self._config.max_nesting_depth:
            raise ValueError(f"Maximum nesting depth ({self._config.max_nesting_depth}) exceeded")

        return await self.spawn(spec)

    def can_autonomous_spawn(self) -> bool:
        """Check if autonomous spawning is allowed (rate-limited)."""
        if not self._config.autonomous_spawn_enabled:
            return False

        now = time.time()

        # Check cooldown
        if self._autonomous_spawn_times:
            last_spawn = self._autonomous_spawn_times[-1]
            if now - last_spawn < self._config.autonomous_spawn_cooldown:
                return False

        # Check hourly limit
        one_hour_ago = now - 3600
        recent = [t for t in self._autonomous_spawn_times if t > one_hour_ago]
        if len(recent) >= self._config.autonomous_spawn_max_per_hour:
            return False

        return True

    def record_autonomous_spawn(self) -> None:
        """Record that an autonomous spawn occurred (for rate limiting)."""
        self._autonomous_spawn_times.append(time.time())
        # Prune old entries
        one_hour_ago = time.time() - 3600
        self._autonomous_spawn_times = [t for t in self._autonomous_spawn_times if t > one_hour_ago]

    async def shutdown(self) -> None:
        """Cancel all active subagents and clean up resources."""
        logger.info(
            "orchestrator.shutting_down",
            active_tasks=len(self._active_tasks),
        )

        # Cancel all active tasks
        for task_id in list(self._active_tasks.keys()):
            await self.cancel(task_id)

        # Wait briefly for cancellations to propagate
        if self._active_tasks:
            tasks = list(self._active_tasks.values())
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.warning("orchestrator.shutdown_timeout", remaining=len(self._active_tasks))

        # Release all swarm bots.
        if self._bot_pool is not None:
            try:
                await self._bot_pool.release_all()
            except Exception:
                logger.debug("orchestrator.bot_pool_release_error", exc_info=True)

        self._active_tasks.clear()
        self._progress.clear()
        self._active_swarms.clear()
        self._swarm_tasks.clear()

        logger.info("orchestrator.shutdown_complete")

    # ---- Swarm Bot Pool Helpers (Phase 6) ----

    def _emit_event(self, event: Any) -> None:
        """Emit an event on the bus if available. Fails silently."""
        if self._event_bus is not None:
            try:
                self._event_bus.emit(event)
            except Exception:
                logger.debug("orchestrator.emit_event_failed", exc_info=True)

    async def _acquire_swarm_bots(
        self,
        swarm_id: str,
        agents: list[SubagentSpec],
    ) -> None:
        """Acquire bots from the pool for each agent with a persona."""
        if self._bot_pool is None:
            return
        for spec in agents:
            if spec.persona is None:
                continue
            try:
                slot = await self._bot_pool.acquire(spec.persona, spec.task_id)
                if slot is not None:
                    try:
                        from gwenn.events import SwarmBotAcquiredEvent
                        self._emit_event(SwarmBotAcquiredEvent(
                            swarm_id=swarm_id,
                            task_id=spec.task_id,
                            bot_name=slot.bot_username,
                            persona_name=spec.persona.name,
                        ))
                    except Exception:
                        pass
            except Exception:
                logger.warning(
                    "orchestrator.bot_acquire_failed",
                    task_id=spec.task_id,
                    exc_info=True,
                )

    async def _release_swarm_bots(
        self,
        swarm_id: str,
        task_ids: list[str],
    ) -> None:
        """Release all bots assigned to swarm tasks back to the pool."""
        if self._bot_pool is None:
            return
        for task_id in task_ids:
            slot = self._bot_pool.get_slot_for_task(task_id)
            if slot is not None:
                try:
                    await self._bot_pool.release(slot)
                    try:
                        from gwenn.events import SwarmBotReleasedEvent
                        self._emit_event(SwarmBotReleasedEvent(
                            swarm_id=swarm_id,
                            task_id=task_id,
                            bot_name=slot.bot_username,
                        ))
                    except Exception:
                        pass
                except Exception:
                    logger.warning(
                        "orchestrator.bot_release_failed",
                        task_id=task_id,
                        exc_info=True,
                    )

    async def send_swarm_message(
        self,
        swarm_id: str,
        task_id: str,
        text: str,
    ) -> bool:
        """Send a visible message through the swarm bot assigned to task_id.

        Returns True if sent successfully, False if no bot pool or no slot.
        Emits a SwarmTurnEvent on success.
        """
        if self._bot_pool is None:
            return False
        slot = self._bot_pool.get_slot_for_task(task_id)
        if slot is None:
            return False

        # Resolve session for this task to find the Telegram chat/thread.
        session_id = self._origin_sessions.get(task_id)
        if not session_id or not session_id.startswith("telegram_"):
            return False

        scope = session_id[len("telegram_"):]
        chat_id: int | None = None
        thread_id: int | None = None
        if scope.startswith("thread:"):
            # Thread routing requires the channel's _thread_to_chat map,
            # which we don't have here. Use the bot pool's send_as directly
            # with the thread and chat IDs encoded in the session_id.
            # For thread sessions, the chat_id is stored by TelegramChannel.
            # We'll handle this through the channel's send_as_swarm_bot instead.
            return False
        elif scope.startswith("chat:"):
            try:
                chat_id = int(scope[len("chat:"):])
            except ValueError:
                return False

        if chat_id is None:
            return False

        await self._bot_pool.send_as(slot, chat_id, thread_id, text)

        try:
            from gwenn.events import SwarmTurnEvent
            self._emit_event(SwarmTurnEvent(
                swarm_id=swarm_id,
                task_id=task_id,
                bot_name=slot.bot_username,
                message_preview=text[:100],
            ))
        except Exception:
            pass

        return True

    # ---- Internal Methods ----

    async def _run_with_semaphore(self, spec: SubagentSpec) -> SubagentResult:
        """Run a subagent while respecting concurrency limits."""
        async with self._concurrency_semaphore:
            progress = self._progress.get(spec.task_id)
            if progress:
                progress.status = "running"

            result = await self._runner.run(spec)

            self._total_api_calls += result.iterations
            return result

    def _on_task_done(self, task_id: str, task: asyncio.Task) -> None:
        """Callback when an asyncio.Task completes."""
        self._active_tasks.pop(task_id, None)

        try:
            result = task.result()
        except asyncio.CancelledError:
            result = SubagentResult(
                task_id=task_id,
                status="cancelled",
            )
        except Exception as exc:
            result = SubagentResult(
                task_id=task_id,
                status="failed",
                error=str(exc),
            )

        # Apply PII redaction before storing
        if self._redactor and hasattr(self._redactor, "redact") and result.result_text:
            if getattr(self._redactor, "enabled", False):
                try:
                    result.result_text = self._redactor.redact(result.result_text)
                except Exception:
                    logger.warning("orchestration.redaction_failed", task_id=task_id, exc_info=True)

        self._completed_results[task_id] = result

        # Emit CompletionMessage for swarm visualization.
        try:
            from gwenn.events import CompletionMessage
            self._emit_event(CompletionMessage(
                sender_task_id=task_id,
                recipient_task_id="coordinator",
                result_text=result.result_text[:500] if result.result_text else "",
                files_modified=result.files_modified,
                success=result.status == "completed",
            ))
        except Exception:
            pass

        # Evict oldest completed results to prevent unbounded growth
        max_completed = max(50, self._config.max_total_api_calls)
        if len(self._completed_results) > max_completed:
            oldest_key = next(iter(self._completed_results))
            del self._completed_results[oldest_key]
            self._origin_sessions.pop(oldest_key, None)

        # Update progress
        progress = self._progress.get(task_id)
        if progress:
            progress.status = result.status
            progress.elapsed_seconds = result.elapsed_seconds
            progress.iterations = result.iterations

        logger.info(
            "orchestration.complete",
            task_id=task_id,
            status=result.status,
            elapsed=result.elapsed_seconds,
        )

    def _get_swarm_status(self, swarm_id: str) -> dict[str, Any]:
        """Get status of a swarm."""
        task_ids = self._swarm_tasks.get(swarm_id, [])
        statuses: dict[str, int] = Counter()

        for tid in task_ids:
            if tid in self._completed_results:
                statuses[self._completed_results[tid].status] += 1
            elif tid in self._active_tasks:
                statuses["running"] += 1
            else:
                statuses["unknown"] += 1

        if statuses.get("running", 0) > 0:
            overall = "running"
        elif all(s == "unknown" for s in statuses):
            overall = "unknown"
        elif all(s in {"failed", "timeout", "cancelled"} for s in statuses if s != "unknown"):
            overall = "failed"
        elif any(s in {"failed", "timeout", "cancelled"} for s in statuses):
            overall = "partial"
        else:
            overall = "completed"

        return {
            "task_id": swarm_id,
            "type": "swarm",
            "status": overall,
            "agent_statuses": dict(statuses),
            "total_agents": len(task_ids),
        }

    async def _aggregate_results(
        self,
        results: list[SubagentResult],
        strategy: str,
    ) -> str:
        """Aggregate individual subagent results according to strategy."""
        # Filter to successful results
        successful = [r for r in results if r.status == "completed" and r.result_text]

        if not successful:
            return "[No successful results to aggregate]"

        if strategy == "concatenate":
            parts = []
            for i, r in enumerate(successful, 1):
                parts.append(f"--- Result {i} (task: {r.task_id}) ---\n{r.result_text}")
            return "\n\n".join(parts)

        elif strategy == "synthesize":
            if not self._engine:
                # Fall back to concatenation if no engine available
                return await self._aggregate_results(results, "concatenate")

            concat = "\n\n".join(
                f"[Agent {i + 1}]: {r.result_text}" for i, r in enumerate(successful)
            )
            synthesis_prompt = (
                "You are synthesizing results from multiple parallel research agents. "
                "Combine their findings into a coherent, unified response. "
                "Identify common themes, resolve contradictions, and produce a "
                "clear summary. Keep it under 1000 tokens.\n\n"
                f"Agent results:\n{concat}"
            )
            try:
                response = await self._engine.think(
                    system_prompt="You are a synthesis assistant.",
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    tools=[],
                )
                return response.text if hasattr(response, "text") else str(response)
            except Exception as exc:
                logger.warning("orchestration.synthesize_failed", error=str(exc))
                return await self._aggregate_results(results, "concatenate")

        elif strategy == "vote":
            # Simple majority — best for categorical outputs
            texts = [r.result_text.strip() for r in successful]
            counts = Counter(texts)
            winner, count = counts.most_common(1)[0]
            return f"[Majority result ({count}/{len(texts)} agents)]: {winner}"

        return await self._aggregate_results(results, "concatenate")
