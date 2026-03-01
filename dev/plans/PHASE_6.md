# Phase 6: Telegram Swarm Visualization (Visible Subagents)

**Risk: MEDIUM** — Telegram API constraints (rate limits, multi-bot polling). Well-isolated from core.

**Prerequisites:** Phase 5 complete (CLI `gwenn agents` can show swarm status).

---

## Goal

Make Gwenn's orchestration visible. When Gwenn spawns a swarm, each subagent appears as a **separate Telegram bot** with its own name, avatar, and persona — chatting as a visible participant in the group topic.

---

## Concept

- A pool of pre-registered Telegram bots (each with its own bot token, name, and avatar)
- When Gwenn spawns a swarm, each subagent is assigned a bot from the pool
- Gwenn dynamically sets each bot's display name and persona via the Telegram API
- Subagents chat within the group topic as separate visible users
- When the swarm completes, bots reset to idle until the next swarm
- Gwenn orchestrates the entire conversation — deciding when subagents speak

---

## New Files

### `gwenn/channels/telegram_bot_pool.py` — Bot Pool Manager

```python
class TelegramBotSlot:
    """A pre-registered Telegram bot available for subagent assignment."""
    bot_token: str
    bot_id: int
    current_persona: SubagentPersona | None  # None = idle
    current_task_id: str | None
    application: Application | None           # python-telegram-bot instance
    is_active: bool

class TelegramBotPool:
    """Manages a pool of Telegram bots for swarm visualization."""

    async def acquire(self, persona: SubagentPersona) -> TelegramBotSlot:
        """Assign an idle bot to a subagent, set its name/photo/bio.

        Steps:
        1. Pick idle bot from pool
        2. Set display name via bot.set_my_name(persona.name) — only if changed
        3. Set description via bot.set_my_description(persona.role) — only if changed
        4. Set profile photo via bot.set_my_photo(persona.avatar) — optional, only if changed
        5. Start polling on this bot (listening for messages in the group)
        6. Return slot

        Rate limit mitigation:
        - Cache current persona per slot. Skip API calls when persona unchanged.
        - asyncio.sleep(1.0) between set_my_name/set_my_description/set_my_photo calls
          when acquiring multiple bots simultaneously.
        """

    async def release(self, slot: TelegramBotSlot) -> None:
        """Return bot to pool. Reset to placeholder persona.

        Stops active polling to reduce API calls while idle.
        Resets name to "Gwenn Agent (Available)" only if different from current.
        """

    async def send_as(self, slot: TelegramBotSlot, chat_id: int,
                      thread_id: int, text: str) -> None:
        """Send a message to the group as this subagent bot."""
```

### `gwenn/orchestration/models.py` — SubagentPersona

`SubagentPersona` is defined here (NOT in `telegram_bot_pool.py`) because personas are channel-agnostic — the same concept applies whether visualization is via Telegram bots, Discord webhooks, or CLI output.

```python
class SubagentPersona(BaseModel):
    """Persona assigned to a subagent by Gwenn."""
    name: str            # Display name (e.g., "Researcher", "Devil's Advocate")
    role: str            # Role description for system prompt
    style: str           # Communication style (formal, casual, provocative, etc.)
    avatar_url: str | None = None  # Optional profile photo URL
```

Extend `SubagentSpec` with optional persona:
```python
class SubagentSpec(BaseModel):
    # ... existing fields ...
    persona: SubagentPersona | None = None  # For swarm visualization
```

### `tests/test_telegram_bot_pool.py`

- Test acquire/release lifecycle (mock Telegram API)
- Test persona caching — skip API calls when persona unchanged
- Test pool exhaustion — raise error when all bots busy
- Test `send_as` message routing
- Test rate limit spacing between API calls

---

## Modified Files

### `gwenn/config.py` — Swarm Bot Config

Add to `TelegramConfig`:
```python
swarm_bot_tokens: StrList = Field([], alias="GWENN_TELEGRAM_SWARM_BOT_TOKENS")
swarm_visible: bool = Field(True, alias="GWENN_TELEGRAM_SWARM_VISIBLE")
```

**Note:** The original plan had `swarm_bot_pool_size` as a separate field. This is **redundant** — the pool size is determined by `len(swarm_bot_tokens)`. If you have 3 tokens, you have 3 bots. Removed to avoid confusion.

### `gwenn/channels/telegram_channel.py` — Integrate Bot Pool

- `TelegramChannel` gains `_bot_pool: TelegramBotPool | None` attribute
- Initialized when `config.telegram.swarm_bot_tokens` is non-empty AND `swarm_visible` is True
- On swarm spawn: acquire bots, route subagent output through assigned bot
- On swarm complete: post synthesis from Gwenn's primary bot, release pool bots
- Uses existing `send_to_session()` pattern but routes through the subagent's bot
- Uses existing `_thread_to_chat` mapping for forum topic routing

### `gwenn/orchestration/orchestrator.py` — Wire Bot Pool into Swarm Lifecycle

- `spawn_swarm()` flow changes:
  1. Create swarm spec with personas
  2. If Telegram bot pool is available: acquire bots before spawning
  3. Wrap each subagent's output callback to route through assigned bot
  4. Spawn subagents
  5. Orchestrator mediates turn-taking: decides when subagents speak
  6. On completion: post synthesis, release bots

- Orchestrator turn-taking logic:
  - After spawning, orchestrator decides execution order
  - Each subagent's intermediate output is routed through its assigned bot
  - If a subagent produces no output for >30s, post status update ("Still researching...")
  - Subagents can "@mention" each other — orchestrator mediates these interactions

### `gwenn/events.py` — Swarm Visualization Events + Typed Inter-Agent Messages

**Swarm visualization events:**

```python
class SwarmBotAcquiredEvent(GwennEvent):
    swarm_id: str; task_id: str; bot_name: str; persona_name: str

class SwarmBotReleasedEvent(GwennEvent):
    swarm_id: str; task_id: str; bot_name: str

class SwarmTurnEvent(GwennEvent):
    swarm_id: str; task_id: str; bot_name: str; message_preview: str
```

**Typed inter-agent messaging protocol:**

The swarm orchestration needs structured communication between the coordinator (Gwenn) and subagents. Without typed messages, Phase 6's turn-taking and output routing would rely on ad-hoc string passing. These message types flow through the event bus and are persisted for replay/debugging.

```python
class AgentMessage(GwennEvent):
    """Base class for all inter-agent messages."""
    sender_task_id: str        # Task ID of the sending agent ("coordinator" for Gwenn)
    recipient_task_id: str     # Target agent or "all" / "@builders" / "@reviewers"
    swarm_id: str | None = None

class TaskDispatchMessage(AgentMessage):
    """Coordinator assigns task to subagent."""
    event_type: str = "agent.task.dispatch"
    task_description: str
    assigned_persona: SubagentPersona | None = None

class StatusUpdateMessage(AgentMessage):
    """Subagent reports progress."""
    event_type: str = "agent.status.update"
    status: str                # "thinking", "researching", "implementing", "blocked"
    progress_pct: float | None = None   # 0.0-1.0 if estimable
    detail: str = ""

class CompletionMessage(AgentMessage):
    """Subagent finished, here are results."""
    event_type: str = "agent.completion"
    result_text: str
    files_modified: list[str] = []
    success: bool = True

class EscalationMessage(AgentMessage):
    """Subagent stuck, needs help or human input."""
    event_type: str = "agent.escalation"
    reason: str                # "blocked", "confused", "needs_approval", "conflict"
    detail: str
    suggested_action: str | None = None

class RequestHelpMessage(AgentMessage):
    """Subagent asks another subagent for assistance."""
    event_type: str = "agent.request.help"
    request: str               # What help is needed
    context: str = ""          # Context for the request

class BroadcastMessage(AgentMessage):
    """Message to all agents or a role group."""
    event_type: str = "agent.broadcast"
    recipient_task_id: str = "all"  # "all", "@builders", "@reviewers", etc.
    content: str
```

**How this integrates with swarm visualization:**
- When the orchestrator dispatches a task, it emits `TaskDispatchMessage` → the Telegram bot pool routes this as a visible "assignment" message
- Subagent `StatusUpdateMessage` → routed through the assigned bot as a visible status update ("Still researching...")
- `CompletionMessage` → routed as the subagent's final visible response before Gwenn synthesizes
- `EscalationMessage` → displayed as the subagent asking for help, visible to users in the topic
- `RequestHelpMessage` → routed to the target subagent's bot as a visible peer-to-peer request; orchestrator mediates the response
- `BroadcastMessage` → all active bots post a message (e.g., coordinator announcing a direction change)
- All messages persisted via event bus for replay and debugging

### `tests/test_agent_messages.py`

- Test all message types serialize/deserialize correctly
- Test event bus routing: `agent.*` wildcard catches all agent messages
- Test orchestrator emits `TaskDispatchMessage` on spawn
- Test `StatusUpdateMessage` triggers bot output routing
- Test `EscalationMessage` escalation flow (bot posts help request, orchestrator responds)
- Test `RequestHelpMessage` peer-to-peer routing (subagent asks another subagent, orchestrator mediates)
- Test `BroadcastMessage` delivery to all active bots

---

## Telegram API Considerations

### Rate Limits for Profile Updates

`bot.set_my_name()` / `bot.set_my_description()` / `bot.set_chat_photo()` are rate-limited (~1 call per method per minute).

**Mitigations:**
1. **Persona caching**: Each `TelegramBotSlot` stores its current persona. Only call Telegram API when the persona actually changes.
2. **Staggered acquisition**: When acquiring multiple bots simultaneously, space API calls with `asyncio.sleep(1.0)` between each.
3. **Lazy reset**: When releasing a bot, only reset name if it will be idle for an extended period. If another swarm spawns soon, the bot may be reassigned quickly.

### Multi-Bot Polling in Same Group

Each pool bot runs its own `python-telegram-bot` `Application` with `updater.start_polling()`.

**Problem:** All bots receive every message in the group.

**Solution:** Each pool bot's update handler filters strictly:
1. Direct replies to its own messages (`reply_to_message.from_user.id == bot_id`)
2. @mentions of its username
3. Orchestrator-directed turns (the orchestrator tells the bot "it's your turn")

Pool bots do NOT respond to general group messages — only Gwenn's primary bot does that. When a pool bot is idle (released), its polling is stopped.

### Subagent Output Routing

- The orchestrator wraps each subagent's output callback to route through the assigned bot
- Intermediate output (thinking, progress, partial results): routed via `TelegramBotPool.send_as()`
- Final results: also routed through the bot before Gwenn posts synthesis
- Liveness: if no output for >30s, post "Still researching..." through the bot

---

## Flow (End-to-End)

1. User (or Gwenn autonomously) initiates a swarm task in a Telegram forum topic
2. Gwenn creates `SwarmSpec` with subagents, each having a `SubagentPersona`
3. `TelegramBotPool.acquire()` assigns idle bots, sets personas via Telegram API
4. `SwarmBotAcquiredEvent` emitted for each bot
5. Orchestrator spawns subagents (in-process or Docker)
6. Subagent output routed through assigned bot → visible messages in group topic
7. Subagents can "@mention" each other — orchestrator mediates turn-taking
8. `SwarmTurnEvent` emitted for each visible message
9. Gwenn (primary bot) moderates: summarizes, redirects, asks follow-ups
10. Swarm completes → Gwenn posts synthesis
11. `SwarmBotReleasedEvent` emitted → bots released, reset to "Gwenn Agent (Available)"

---

## Setup Requirement

1. Pre-register N bots via @BotFather (e.g., "Gwenn Agent 1", "Gwenn Agent 2", ...)
2. Add all bots to the Telegram group as admins (needed for name-change permissions)
3. Store tokens as JSON array in config: `GWENN_TELEGRAM_SWARM_BOT_TOKENS='["token1","token2","token3"]'`
4. Or in gwenn.toml:
   ```toml
   [telegram]
   swarm_bot_tokens = ["token1", "token2", "token3"]
   swarm_visible = true
   ```

---

## Implementation Sub-Steps

```
6a. Add SubagentPersona to gwenn/orchestration/models.py, extend SubagentSpec
6b. Add swarm config fields to TelegramConfig in config.py
6c. Add swarm visualization events to gwenn/events.py
6d. Add typed inter-agent message types (AgentMessage hierarchy) to gwenn/events.py
6e. Create gwenn/channels/telegram_bot_pool.py — BotPool, BotSlot
6f. Integrate bot pool into telegram_channel.py
6g. Wire bot pool + typed messages into orchestrator.py swarm lifecycle + turn-taking
6h. Wire message routing: orchestrator emits typed messages → bot pool routes to visible bots
6i. Write tests (bot pool, agent messages, message routing, integration)
```

1 commit per sub-step.

---

## Verification

- Swarm spawned in Telegram topic → pool bots acquire personas via Telegram API
- Each subagent posts as a separate visible bot in the group topic
- Turn-taking is orchestrated — subagents speak in sequence, @mention each other
- Gwenn moderates: summarizes, redirects, asks follow-ups
- Swarm completes → Gwenn posts synthesis → bots released back to pool
- `SwarmBotAcquiredEvent` / `SwarmBotReleasedEvent` / `SwarmTurnEvent` on event bus
- `gwenn agents` (Phase 5 CLI) shows which bots are active in swarms
- Pool bots reset to "Gwenn Agent (Available)" when idle
- Rate limits respected — persona changes only when needed, staggered API calls
- Typed inter-agent messages flow through event bus: `TaskDispatchMessage`, `StatusUpdateMessage`, `CompletionMessage`, `EscalationMessage`, `RequestHelpMessage`, `BroadcastMessage`
- Subagent status updates visible in topic via bot ("Still researching...", "Found 3 relevant patterns...")
- Escalation messages visible — subagent asks for help, Gwenn or user can respond
- Peer help requests visible — subagent asks another subagent for assistance, orchestrator mediates
- All inter-agent messages persisted for replay/debugging
