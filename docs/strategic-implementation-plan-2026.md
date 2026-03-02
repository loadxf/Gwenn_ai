# Strategic Implementation Plan: Top 5 Moves for Gwenn AI

**Date:** March 2, 2026
**Status:** Research & Planning
**Purpose:** Detailed, actionable implementation solutions for each of the five strategic priorities identified in the investor appeal research.

---

## Table of Contents

1. [Move 1: Multi-Model LLM Support](#move-1-multi-model-llm-support)
2. [Move 2: Open-Core Pricing & Business Model](#move-2-open-core-pricing--business-model)
3. [Move 3: Developer Productivity Vertical Beachhead](#move-3-developer-productivity-vertical-beachhead)
4. [Move 4: Managed Cloud Service (Gwenn Cloud)](#move-4-managed-cloud-service-gwenn-cloud)
5. [Move 5: Skill Marketplace & Community Flywheel](#move-5-skill-marketplace--community-flywheel)
6. [Implementation Timeline](#implementation-timeline)
7. [Sources](#sources)

---

## Move 1: Multi-Model LLM Support

### Why This Matters

Single-provider LLM dependency is now a top-3 investor red flag. Jake Flomenberg of Wing VC: *"I'm skeptical of moats built purely on model performance or prompting — those advantages erode in months."* Multi-model support addresses three concerns simultaneously:

1. **Investor risk**: Eliminates COGS concentration on Anthropic
2. **User flexibility**: Let users choose cost vs. quality tradeoffs
3. **Resilience**: Automatic failover across providers

### Current State Analysis

Gwenn's `CognitiveEngine` (`gwenn/api/claude.py`) is **tightly coupled to Anthropic's SDK**:

- Directly imports and uses `anthropic.AsyncAnthropic`
- Returns `anthropic.types.Message` from all methods
- Catches Anthropic-specific exceptions (`anthropic.BadRequestError`, `anthropic.AuthenticationError`, etc.)
- Uses Anthropic-specific features: extended thinking (`thinking.type`), prompt caching (`cache_control`), OAuth
- Four cognitive modes (`think`, `reflect`, `appraise`, `compact`) all route through Anthropic

The configuration (`gwenn/config.py`) only has `ClaudeConfig` with Anthropic-specific fields.

### Recommended Solution: Provider Abstraction Layer

Rather than adopting a third-party library like LiteLLM (which adds dependency risk and production-readiness concerns), build a **thin internal abstraction layer** that preserves Gwenn's unique cognitive semantics.

#### Architecture

```
gwenn/api/
├── __init__.py          # Re-export public interface
├── base.py              # Abstract CognitiveEngine protocol
├── claude.py            # Anthropic provider (existing, refactored)
├── openai.py            # OpenAI provider (new)
├── gemini.py            # Google Gemini provider (new)
├── openrouter.py        # OpenRouter meta-provider (new)
├── local.py             # Ollama/vLLM local provider (new)
├── types.py             # Provider-agnostic response types
└── factory.py           # Provider selection & instantiation
```

#### Step 1: Define Provider-Agnostic Types (`gwenn/api/types.py`)

```python
"""Provider-agnostic types for the cognitive engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

class ContentBlockType(Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"
    THINKING = "thinking"

@dataclass
class ContentBlock:
    type: ContentBlockType
    text: str = ""
    tool_use_id: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    thinking: str = ""

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

@dataclass
class CognitiveResponse:
    """Provider-agnostic response from any LLM.

    Maps the union of response semantics across providers into
    a single type that the rest of Gwenn's subsystems consume.
    """
    content: list[ContentBlock]
    usage: Usage
    stop_reason: str = ""
    model: str = ""
    provider: str = ""

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if b.type == ContentBlockType.TEXT)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [
            {"id": b.tool_use_id, "name": b.tool_name, "input": b.tool_input}
            for b in self.content if b.type == ContentBlockType.TOOL_USE
        ]

    @property
    def thinking_text(self) -> Optional[str]:
        for b in self.content:
            if b.type == ContentBlockType.THINKING:
                return b.thinking
        return None
```

#### Step 2: Define the Abstract Provider Protocol (`gwenn/api/base.py`)

```python
"""Abstract cognitive engine protocol."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from gwenn.api.types import CognitiveResponse

class BaseCognitiveEngine(ABC):
    """Contract that all LLM providers must implement."""

    @abstractmethod
    async def think(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[dict[str, Any]] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        cache_system: bool = True,
    ) -> CognitiveResponse: ...

    @abstractmethod
    async def reflect(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> CognitiveResponse: ...

    @abstractmethod
    async def appraise(
        self,
        system_prompt: str,
        content: str,
    ) -> CognitiveResponse: ...

    @abstractmethod
    async def compact(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        compaction_prompt: str,
    ) -> CognitiveResponse: ...

    @abstractmethod
    def extract_text(self, response: CognitiveResponse) -> str: ...

    @abstractmethod
    def extract_tool_calls(self, response: CognitiveResponse) -> list[dict[str, Any]]: ...

    @property
    @abstractmethod
    def telemetry(self) -> dict[str, Any]: ...
```

#### Step 3: Provider Factory (`gwenn/api/factory.py`)

```python
"""Provider factory — instantiates the configured cognitive engine."""
from gwenn.api.base import BaseCognitiveEngine

def create_engine(provider: str, config: dict) -> BaseCognitiveEngine:
    if provider == "anthropic":
        from gwenn.api.claude import CognitiveEngine
        return CognitiveEngine(config)
    elif provider == "openai":
        from gwenn.api.openai import OpenAICognitiveEngine
        return OpenAICognitiveEngine(config)
    elif provider == "gemini":
        from gwenn.api.gemini import GeminiCognitiveEngine
        return GeminiCognitiveEngine(config)
    elif provider == "openrouter":
        from gwenn.api.openrouter import OpenRouterCognitiveEngine
        return OpenRouterCognitiveEngine(config)
    elif provider == "local":
        from gwenn.api.local import LocalCognitiveEngine
        return LocalCognitiveEngine(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

#### Step 4: Configuration Changes

Add to `gwenn/config.py`:

```python
class ProviderConfig(GwennSettingsBase):
    """Multi-provider LLM configuration."""

    primary_provider: str = Field("anthropic", alias="GWENN_PROVIDER")
    fallback_provider: Optional[str] = Field(None, alias="GWENN_FALLBACK_PROVIDER")

    # Per-cognitive-mode provider overrides (cost optimization)
    appraise_provider: Optional[str] = Field(None, alias="GWENN_APPRAISE_PROVIDER")
    appraise_model: Optional[str] = Field(None, alias="GWENN_APPRAISE_MODEL")

    # OpenAI
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", alias="GWENN_OPENAI_MODEL")

    # Google Gemini
    gemini_api_key: Optional[str] = Field(None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field("gemini-2.0-flash", alias="GWENN_GEMINI_MODEL")

    # OpenRouter (meta-provider)
    openrouter_api_key: Optional[str] = Field(None, alias="OPENROUTER_API_KEY")

    # Local (Ollama / vLLM)
    local_base_url: str = Field("http://localhost:11434", alias="GWENN_LOCAL_URL")
    local_model: str = Field("llama3.2", alias="GWENN_LOCAL_MODEL")
```

Add `gwenn.toml` support:

```toml
[providers]
primary = "anthropic"
fallback = "openai"

[providers.anthropic]
model = "claude-sonnet-4-5-20250929"

[providers.openai]
model = "gpt-4o"

[providers.appraise]
# Use cheap/fast model for emotional appraisals
provider = "local"
model = "llama3.2"
```

#### Step 5: Smart Routing (Cost Optimization)

Map cognitive modes to provider tiers:

| Cognitive Mode | Default Provider | Rationale |
|----------------|-----------------|-----------|
| `think()` | Primary (Claude) | Needs best reasoning + tool use |
| `reflect()` | Primary (Claude) | Needs extended thinking |
| `appraise()` | Fast/Cheap (Gemini Flash or local) | Quick evaluation, 512 max tokens |
| `compact()` | Mid-tier (GPT-4o-mini or Gemini) | Summarization doesn't need top reasoning |
| Subagent tasks | Configurable per-spec | Let users optimize cost per task |

This alone could **reduce LLM costs by 40-60%** — appraisals and compactions are frequent operations that don't need frontier reasoning.

#### Step 6: Automatic Failover

```python
class ResilientEngine(BaseCognitiveEngine):
    """Wraps primary + fallback engines with automatic failover."""

    def __init__(self, primary: BaseCognitiveEngine, fallback: BaseCognitiveEngine):
        self._primary = primary
        self._fallback = fallback

    async def think(self, **kwargs) -> CognitiveResponse:
        try:
            return await self._primary.think(**kwargs)
        except (ConnectionError, RateLimitError, TimeoutError):
            logger.warning("engine.failover", from_="primary", to="fallback")
            return await self._fallback.think(**kwargs)
```

#### Migration Strategy

1. **Phase A** (non-breaking): Introduce `CognitiveResponse` and `BaseCognitiveEngine`; wrap existing `CognitiveEngine` to return the new types. All callers continue working.
2. **Phase B**: Add OpenAI provider (most similar API to Anthropic).
3. **Phase C**: Add Gemini, OpenRouter, and local providers.
4. **Phase D**: Add smart routing and failover.
5. **Phase E**: Add per-mode provider configuration.

#### Estimated Effort

| Phase | Duration | Risk |
|-------|----------|------|
| A: Abstraction layer | 3-5 days | Low — additive, not breaking |
| B: OpenAI provider | 2-3 days | Low — well-documented API |
| C: Gemini + OpenRouter + local | 4-6 days | Medium — API differences |
| D: Failover | 2-3 days | Low |
| E: Per-mode routing | 2-3 days | Low |
| **Total** | **~2-3 weeks** | |

---

## Move 2: Open-Core Pricing & Business Model

### Why This Matters

Per Bessemer Venture Partners' AI Pricing Playbook: *"The charge metric you pick isn't just a billing decision. It's a statement about what you believe your AI is worth."* AI-first SaaS economics are fundamentally different — COGS matter again, with 50-60% gross margins vs. 80-90% for traditional SaaS.

### Recommended Pricing Architecture

#### The Three-Tier Model

**Tier 1: Gwenn Community (Free, Open Source)**
- Full self-hosted Gwenn with all cognitive subsystems
- All 26 built-in skills
- CLI + single channel (Telegram OR Discord OR Slack)
- Local memory persistence (SQLite + ChromaDB)
- Single LLM provider
- Community support (GitHub + Discord)
- MPL-2.0 license (unchanged)

**Tier 2: Gwenn Pro ($29-49/month per instance)**
- Everything in Community, plus:
- Gwenn Cloud hosting (always-on, no self-hosting needed)
- Multi-channel simultaneous (CLI + Telegram + Discord + Slack)
- Multi-model support with smart routing
- Cloud memory backup & sync
- Skill marketplace access (install community skills)
- Priority support
- Usage dashboard & analytics
- Mobile companion app access

**Tier 3: Gwenn Enterprise (Custom pricing)**
- Everything in Pro, plus:
- Multi-instance orchestration (team Gwenn agents)
- SSO / SAML authentication
- SOC 2 compliance documentation
- Custom model fine-tuning
- Dedicated infrastructure (single-tenant option)
- API access for integration
- SLA guarantees (99.9% uptime)
- Dedicated support & onboarding

#### Consumption-Based Component

On top of subscription tiers, add usage-based pricing for compute-intensive features:

```
Base subscription: Platform access, memory, channels
+ LLM usage:       Pass-through cost + 20-30% margin
+ Premium skills:   $0.01-0.10 per execution (marketplace creators earn 70%)
+ Subagent swarms:  $0.05 per subagent-minute
```

This hybrid model follows the Bessemer recommendation: *"Light subscription for platform access plus usage for volume while outcomes are uncertain."*

#### Implementation Requirements

1. **Usage Tracking Service** — instrument all LLM calls, skill executions, and subagent operations with per-tenant metering
2. **Billing Integration** — Stripe for subscriptions + usage-based billing
3. **API Gateway** — Rate limiting, quota enforcement, tenant isolation
4. **Account System** — User registration, authentication, tenant management
5. **Admin Dashboard** — Usage monitoring, billing management, analytics

#### Key Files to Add/Modify

```
gwenn/
├── billing/
│   ├── __init__.py
│   ├── metering.py       # Usage tracking & aggregation
│   ├── stripe_client.py  # Stripe integration
│   ├── plans.py          # Plan definitions & entitlements
│   └── quota.py          # Rate limiting & quota enforcement
├── api/claude.py          # Add metering hooks (already has telemetry)
├── config.py              # Add BillingConfig section
└── gateway.py             # Add auth middleware for API keys
```

#### Gross Margin Analysis

| Cost Component | Per Instance/Month | Notes |
|----------------|-------------------|-------|
| Claude API (casual user) | $5-15 | ~100 conversations/month |
| Claude API (power user) | $30-80 | ~500 conversations/month |
| Cloud hosting (compute) | $5-10 | Container on shared infrastructure |
| Memory storage | $1-3 | SQLite + ChromaDB on persistent disk |
| **Total COGS** | **$11-93** | |
| **Pro subscription** | **$29-49** | |
| **Estimated gross margin** | **35-70%** | Improves with smart routing |

With multi-model smart routing (Move 1), COGS drop significantly — using Gemini Flash for appraisals and local models for compaction could cut LLM costs by 40-60%.

---

## Move 3: Developer Productivity Vertical Beachhead

### Why This Matters

From the developer productivity research:
- 84% of developers use AI tools; 51% use them daily
- Tools write 41% of all code in 2026
- The biggest unmet need is **persistent context across sessions**: *"AI coding tools possess broad programming knowledge, but they lack project memory. Each session begins without awareness of prior sessions."*
- Gwenn's 3-layer memory system is the **exact solution** to the #1 pain point

### The Opportunity: "Gwenn Dev" — Your Codebase Companion

Unlike Cursor, Claude Code, or Copilot (which are session-based tools), Gwenn Dev would be a **persistent codebase companion** that:

- Remembers every architectural decision and why it was made
- Knows your coding conventions and enforces them across sessions
- Learns your team's patterns and anti-patterns over time
- Proactively suggests improvements based on accumulated context
- Runs autonomously during idle time (heartbeat) to analyze code quality, security, and documentation

### Differentiation from Existing Tools

| Feature | Cursor/Claude Code | Gwenn Dev |
|---------|-------------------|-----------|
| Memory across sessions | None (starts fresh) | Full episodic + semantic memory |
| Learns your conventions | Manual rules files | Emergent from observation |
| Autonomous analysis | On-demand only | Continuous heartbeat |
| Emotional context | None | Tracks developer frustration, flow state |
| Relationship depth | Tool | Companion that knows your work |
| Codebase knowledge | Per-session context | Accumulated knowledge graph |

### Implementation: Codebase-Aware Skills

Create a set of **developer productivity skills** that leverage Gwenn's cognitive architecture:

#### Skill 1: `codebase_memory` (Autonomous)
```markdown
---
{
  "name": "codebase_memory",
  "description": "Maintains persistent knowledge about the codebase structure, decisions, and patterns",
  "category": "developer",
  "version": "1.0",
  "risk_level": "low",
  "autonomous": true,
  "parameters": {}
}
---
During heartbeat cycles, analyze recently changed files and:
1. Extract architectural decisions and record them in semantic memory
2. Identify coding patterns and conventions
3. Note technical debt and potential improvements
4. Track dependency changes and their implications
5. Update the codebase knowledge graph
```

#### Skill 2: `code_review` (User-invocable)
```markdown
---
{
  "name": "code_review",
  "description": "Reviews code changes with full project context and historical awareness",
  "category": "developer",
  "version": "1.0",
  "risk_level": "low",
  "parameters": {
    "target": { "type": "string", "description": "File, PR URL, or git ref to review", "required": true }
  }
}
---
Review {target} using your accumulated knowledge of this codebase:
1. Check against known conventions (from codebase_memory)
2. Compare with similar patterns in the codebase
3. Identify potential bugs based on historical issues
4. Assess architectural fit with existing design decisions
5. Suggest improvements based on patterns that worked well before
```

#### Skill 3: `explain_decision` (User-invocable)
```markdown
---
{
  "name": "explain_decision",
  "description": "Explains why a particular code decision was made, using episodic memory",
  "category": "developer",
  "version": "1.0",
  "risk_level": "low",
  "parameters": {
    "topic": { "type": "string", "description": "The code, pattern, or decision to explain", "required": true }
  }
}
---
Search your episodic and semantic memory for context about {topic}:
1. Find relevant conversations where this was discussed
2. Identify the decision-makers and their reasoning
3. Surface any tradeoffs that were considered
4. Note if the context has changed since the decision was made
5. Suggest whether the decision should be revisited
```

#### Skill 4: `onboard_developer` (User-invocable)
```markdown
---
{
  "name": "onboard_developer",
  "description": "Generates personalized onboarding based on accumulated codebase knowledge",
  "category": "developer",
  "version": "1.0",
  "risk_level": "low",
  "parameters": {
    "focus_area": { "type": "string", "description": "Which part of the codebase to focus on", "required": false }
  }
}
---
Using your accumulated knowledge of this codebase, create an onboarding guide:
1. Architecture overview with key design decisions and WHY they were made
2. Critical paths and their dependencies
3. Common gotchas and pitfalls (from historical issues)
4. Coding conventions and patterns specific to this project
5. Key contacts/owners for each subsystem (from conversation history)
Focus area: {focus_area}
```

### Integration Points

**MCP Server for IDE Integration:**
Gwenn already supports MCP. Create an MCP server that exposes Gwenn Dev capabilities to any IDE:

```python
# gwenn/mcp/dev_server.py
"""MCP server exposing Gwenn Dev tools to IDEs (VS Code, Cursor, etc.)."""

TOOLS = [
    {
        "name": "ask_gwenn",
        "description": "Ask Gwenn about this codebase with full historical context",
        "parameters": {"question": {"type": "string"}}
    },
    {
        "name": "review_changes",
        "description": "Review current changes with codebase-aware context",
        "parameters": {"diff": {"type": "string"}}
    },
    {
        "name": "recall_decision",
        "description": "Recall why a specific decision was made",
        "parameters": {"topic": {"type": "string"}}
    },
]
```

**Git Hooks Integration:**
- Pre-commit: Gwenn reviews staged changes against known conventions
- Post-commit: Gwenn updates codebase knowledge graph
- Post-merge: Gwenn analyzes incoming changes for conflicts with known patterns

### File-Level Integration

Create a new `gwenn/tools/dev_tools.py` module with tools that leverage the filesystem:

```python
DEVELOPER_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the project directory",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}
    },
    {
        "name": "search_codebase",
        "description": "Search for patterns across the codebase",
        "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "file_glob": {"type": "string"}}}
    },
    {
        "name": "git_log",
        "description": "View recent git history",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}
    },
    {
        "name": "run_tests",
        "description": "Run the project's test suite",
        "input_schema": {"type": "object", "properties": {"target": {"type": "string"}}}
    },
]
```

### Distribution Strategy

1. **VS Code Extension**: MCP server that connects VS Code to a running Gwenn instance
2. **CLI Tool**: `gwenn dev` command for terminal-first developers
3. **GitHub App**: Automatic PR reviews powered by Gwenn's codebase memory
4. **Skill Pack**: "Developer Productivity Pack" in the skill marketplace (Move 5)

### Go-to-Market

- Target: Open-source project maintainers (natural early adopters, influence multiplier)
- Channel: Developer communities (Hacker News, Reddit r/programming, Dev.to)
- Hook: "Your AI assistant forgets you every session. Gwenn remembers your entire codebase."
- Proof point: Use Gwenn Dev to maintain Gwenn itself (dogfooding)

---

## Move 4: Managed Cloud Service (Gwenn Cloud)

### Why This Matters

Self-hosted AI is a niche. Cloud services generate the SaaS metrics investors understand: ARR, NRR, CAC, LTV. Gwenn already has the infrastructure primitives needed — the gateway server, daemon mode, checkpoint/restore, and WebSocket support.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Gwenn Cloud                          │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  API Gateway  │  │  Auth/Billing│  │  Dashboard   │  │
│  │  (nginx/CF)   │  │  (Stripe)    │  │  (React)     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│  ┌──────▼─────────────────▼──────────────────▼───────┐  │
│  │              Orchestration Layer                    │  │
│  │         (Kubernetes / Cloudflare Workers)           │  │
│  └──────────────────┬────────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────▼────────────────────────────────┐  │
│  │              Gwenn Instance Pool                    │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐             │  │
│  │  │ Gwenn-1 │ │ Gwenn-2 │ │ Gwenn-N │  ...         │  │
│  │  │(user A) │ │(user B) │ │(user N) │             │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘             │  │
│  │       │           │           │                    │  │
│  │  ┌────▼───────────▼───────────▼────────────────┐  │  │
│  │  │        Persistent Storage Layer              │  │  │
│  │  │  (PostgreSQL + S3/R2 + ChromaDB cluster)     │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Recommended Platform: Cloudflare Workers + Durable Objects

Based on the research, Cloudflare Agents offer the best fit for Gwenn Cloud:

- **Durable Objects** = stateful micro-servers (one per Gwenn instance)
- Built-in SQL database per instance (replaces SQLite)
- Native WebSocket support (matches Gwenn's gateway pattern)
- Scheduling / cron for heartbeat cycles
- Global edge deployment (low latency worldwide)
- Scales to millions of instances automatically
- Cost-effective: pay-per-request, not per-server

#### Alternative: Kubernetes on AWS/GCP

For enterprise customers requiring single-tenant or on-prem:

- One Gwenn pod per instance with persistent volume
- PostgreSQL + pgvector for memory (replaces SQLite + ChromaDB)
- Redis for session state and pub/sub
- Horizontal scaling via pod autoscaling
- Terraform modules for reproducible deployment

### Key Implementation Steps

#### Step 1: Storage Abstraction

Gwenn currently uses SQLite (aiosqlite) and ChromaDB locally. Abstract storage to support both local and cloud backends:

```python
# gwenn/storage/base.py
class StorageBackend(ABC):
    """Abstract storage interface for episodic and semantic memory."""

    @abstractmethod
    async def store_episode(self, episode: dict) -> str: ...

    @abstractmethod
    async def query_episodes(self, query: str, limit: int) -> list[dict]: ...

    @abstractmethod
    async def store_semantic(self, fact: dict) -> str: ...

    @abstractmethod
    async def checkpoint(self) -> bytes: ...

    @abstractmethod
    async def restore(self, data: bytes) -> None: ...

# gwenn/storage/local.py    — SQLite + ChromaDB (existing, wrapped)
# gwenn/storage/postgres.py — PostgreSQL + pgvector (cloud)
# gwenn/storage/r2.py       — Cloudflare R2 for blob storage
```

#### Step 2: Instance Lifecycle Management

```python
# gwenn/cloud/instance_manager.py
class InstanceManager:
    """Manages Gwenn instance lifecycle for cloud deployment."""

    async def create_instance(self, user_id: str, config: dict) -> str:
        """Provision a new Gwenn instance for a user."""
        ...

    async def hibernate_instance(self, instance_id: str) -> None:
        """Checkpoint and suspend an idle instance to save resources."""
        ...

    async def wake_instance(self, instance_id: str) -> None:
        """Restore a hibernated instance from checkpoint."""
        ...

    async def destroy_instance(self, instance_id: str) -> None:
        """Permanently delete an instance and its data."""
        ...
```

This leverages Gwenn's existing checkpoint/restore system — one of its major architectural advantages. Most cloud AI products can't hibernate and restore cognitive state.

#### Step 3: Multi-Tenant Gateway

Extend the existing `GatewayServer` with tenant routing:

```python
# Modifications to gwenn/gateway.py

async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
    # Extract tenant from auth token
    tenant_id = await self._authenticate_tenant(request)

    # Route to correct Gwenn instance
    instance = await self._instance_manager.get_or_wake(tenant_id)

    # Proxy WebSocket to tenant's Gwenn instance
    ...
```

#### Step 4: Health Monitoring & Auto-Scaling

Gwenn already has:
- Interoception (system self-awareness: CPU, memory, latency)
- Self-healing engine (autonomous recovery)
- Health endpoint (`/health`)
- Dashboard endpoint (`/dashboard`)

Extend these for cloud operations:
- Instance-level health aggregation
- Auto-hibernate idle instances (cost saving)
- Auto-wake on incoming request
- Prometheus metrics export for Kubernetes/Grafana

### Infrastructure Costs (Estimated)

| Component | Cost/Instance/Month | Notes |
|-----------|-------------------|-------|
| Compute (idle) | $0.50-2 | Hibernated instances near-zero |
| Compute (active) | $3-8 | ~4 hours active per day |
| Storage (PostgreSQL) | $1-3 | Managed database, shared cluster |
| Storage (vectors) | $0.50-2 | pgvector on same PostgreSQL |
| CDN/Edge | $0.10-0.50 | Cloudflare |
| **Total infra** | **$5-15** | |
| **+ LLM costs** | **$5-80** | Pass-through + margin |
| **Total COGS** | **$10-95** | |

With $29-49/mo Pro pricing, margins are healthy for casual users and improve with smart routing.

---

## Move 5: Skill Marketplace & Community Flywheel

### Why This Matters

Skill marketplaces create **network effects** — the most powerful moat type. Every skill published makes the platform more valuable for all users. The AI skills marketplace ecosystem is exploding in 2026:

- SkillsMP has 270,000+ skills using the open SKILL.md format
- ClawHub has 3,000+ published skills and 15,000+ daily installations
- Claude Code now has a native plugin marketplace

Gwenn's skill system is architecturally ready for this — skills are markdown files with JSON frontmatter, hot-loadable, and self-creatable via the `skill_builder` tool.

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                Gwenn Skill Marketplace                │
│                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │  Registry   │  │  Discovery  │  │  Reviews   │     │
│  │  (API)      │  │  (Search)   │  │  (Ratings) │     │
│  └─────┬──────┘  └──────┬─────┘  └─────┬──────┘     │
│        │                │               │             │
│  ┌─────▼────────────────▼───────────────▼──────────┐ │
│  │              Skill Store (S3/R2)                  │ │
│  │  ┌──────────────────────────────────────────┐    │ │
│  │  │  skill.md files + metadata + signatures  │    │ │
│  │  └──────────────────────────────────────────┘    │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │              Security Layer                       │ │
│  │  • Static analysis (malicious pattern detection)  │ │
│  │  • Sandboxed execution testing                    │ │
│  │  • Community flagging & review                    │ │
│  │  • Publisher verification                         │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Skill Format (Extend Existing)

Gwenn already uses this format:

```markdown
---
{
  "name": "skill_name",
  "description": "...",
  "category": "information",
  "version": "1.0",
  "risk_level": "low",
  "parameters": { ... }
}
---
Instruction body...
```

Extend the frontmatter for marketplace metadata:

```json
{
  "name": "smart_summarizer",
  "description": "Summarizes documents with context-aware compression",
  "category": "productivity",
  "version": "2.1.0",
  "risk_level": "low",
  "author": "username",
  "license": "MIT",
  "tags": ["summary", "documents", "productivity"],
  "min_gwenn_version": "0.4.0",
  "dependencies": [],
  "parameters": { ... },
  "marketplace": {
    "pricing": "free",
    "downloads": 0,
    "rating": 0.0,
    "verified": false
  }
}
```

### CLI Commands

```bash
# Browse & search marketplace
gwenn skills search "code review"
gwenn skills browse --category developer
gwenn skills trending

# Install & manage
gwenn skills install smart_summarizer
gwenn skills install smart_summarizer@2.1.0
gwenn skills update --all
gwenn skills uninstall smart_summarizer
gwenn skills list --installed

# Publish
gwenn skills publish ./my_skill.md
gwenn skills publish --private ./internal_skill.md
```

### Implementation

#### Step 1: Marketplace API (`gwenn/marketplace/api.py`)

```python
class SkillMarketplace:
    """Client for the Gwenn Skill Marketplace."""

    BASE_URL = "https://marketplace.gwenn.ai/api/v1"

    async def search(self, query: str, category: str = None) -> list[SkillListing]:
        """Search marketplace for skills."""
        ...

    async def install(self, name: str, version: str = "latest") -> SkillDefinition:
        """Download and install a skill from the marketplace."""
        ...

    async def publish(self, skill_path: Path, private: bool = False) -> str:
        """Publish a skill to the marketplace."""
        ...

    async def trending(self, limit: int = 10) -> list[SkillListing]:
        """Get trending skills."""
        ...
```

#### Step 2: Security Scanning

Given that 5.2% of skills in the wild exhibit malicious patterns (per arXiv research), security is critical:

```python
class SkillScanner:
    """Static analysis for marketplace skills."""

    DANGEROUS_PATTERNS = [
        r"rm\s+-rf",
        r"curl.*\|.*sh",
        r"eval\(",
        r"exec\(",
        r"__import__",
        r"subprocess",
        r"os\.system",
    ]

    def scan(self, skill: SkillDefinition) -> ScanResult:
        """Scan a skill for dangerous patterns before installation."""
        ...
```

#### Step 3: SKILL.md Interoperability

Adopt the open SKILL.md standard (established by Anthropic, adopted by OpenAI) so Gwenn skills work across platforms:

```python
class SkillFormatConverter:
    """Convert between Gwenn's native format and SKILL.md standard."""

    def to_skillmd(self, skill: SkillDefinition) -> str:
        """Export a Gwenn skill as SKILL.md for cross-platform use."""
        ...

    def from_skillmd(self, content: str) -> SkillDefinition:
        """Import a SKILL.md format skill into Gwenn's format."""
        ...
```

This is a strategic decision — by supporting the same format as Claude Code and Codex, Gwenn gains access to the 270,000+ existing skills on SkillsMP.

#### Step 4: Revenue Model

| Skill Type | Price | Creator Revenue | Gwenn Revenue |
|-----------|-------|----------------|---------------|
| Free | $0 | — | User acquisition |
| Premium | $1-10 one-time | 70% | 30% |
| Subscription | $1-5/month | 70% | 30% |
| Enterprise | Custom | Negotiated | Negotiated |

#### Step 5: Community Features

- **Ratings & Reviews**: 5-star + written reviews
- **Download counts**: Social proof for quality
- **Verified publishers**: Badge for trusted creators
- **Skill collections**: Curated bundles (e.g., "Developer Pack", "Writer's Toolkit")
- **Fork & improve**: GitHub-style forking for community improvement
- **Usage analytics**: Creators see how their skills are used

### Starter Skill Packs (Created by Gwenn Team)

1. **Developer Productivity Pack** (ties to Move 3)
   - `code_review`, `codebase_memory`, `explain_decision`, `onboard_developer`, `debug_assistant`

2. **Research & Learning Pack**
   - `deep_research`, `summarize_paper`, `explain_concept`, `create_flashcards`, `literature_review`

3. **Writing & Content Pack**
   - `blog_post`, `email_drafter`, `proofread`, `style_guide`, `content_calendar`

4. **Personal Productivity Pack**
   - `daily_briefing`, `goal_tracker`, `meeting_prep`, `decision_matrix`, `weekly_review`

---

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-4)

| Week | Move 1 | Move 2 | Move 3 | Move 4 | Move 5 |
|------|--------|--------|--------|--------|--------|
| 1 | Abstraction layer + types | Pricing model design | Dev skills design | Storage abstraction | Marketplace API design |
| 2 | Wrap existing Claude engine | Usage metering hooks | Implement codebase_memory skill | PostgreSQL backend | Skill format extension |
| 3 | OpenAI provider | Billing config | Implement code_review skill | Instance lifecycle | Security scanner |
| 4 | Testing + failover | Stripe integration | MCP dev server | Cloud gateway routing | CLI commands |

### Phase 2: Launch Prep (Weeks 5-8)

| Week | Focus |
|------|-------|
| 5 | Gemini + local providers; Pricing page; Dev skills testing |
| 6 | Smart routing; Cloud deployment (Cloudflare/K8s); Marketplace backend |
| 7 | Per-mode routing; Dashboard UI; VS Code extension; Skill packs |
| 8 | Integration testing; Beta launch prep; Documentation; Security audit |

### Phase 3: Beta Launch (Weeks 9-12)

| Week | Focus |
|------|-------|
| 9 | Private beta launch (cloud + marketplace) |
| 10 | Beta feedback iteration |
| 11 | Public beta |
| 12 | GA launch; Investor deck update with metrics |

### Key Milestones

- **Week 2**: Multi-model support working (Anthropic + OpenAI)
- **Week 4**: Usage metering in place; Dev skills functional
- **Week 6**: Cloud hosting prototype running
- **Week 8**: Marketplace with 5+ first-party skill packs
- **Week 9**: Private beta with 10-50 users
- **Week 12**: Public launch with pricing

---

## Sources

### AI Pricing & Business Models
- [The AI Pricing and Monetization Playbook — Bessemer Venture Partners](https://www.bvp.com/atlas/the-ai-pricing-and-monetization-playbook)
- [The 2026 Guide to SaaS, AI, and Agentic Pricing Models — Monetizely](https://www.getmonetizely.com/blogs/the-2026-guide-to-saas-ai-and-agentic-pricing-models)
- [The Economics of AI-First B2B SaaS in 2026 — Monetizely](https://www.getmonetizely.com/blogs/the-economics-of-ai-first-b2b-saas-in-2026)
- [From Traditional SaaS-Pricing to AI Agent Seats — AIMultiple](https://research.aimultiple.com/ai-agent-pricing/)
- [B2B SaaS and Agentic AI Pricing Predictions for 2026 — Ibbaka](https://www.ibbaka.com/ibbaka-market-blog/b2b-saas-and-agentic-ai-pricing-predictions-for-2026)
- [2026's Real SaaS Threat Isn't AI. It's Business Model Debt — Chargebee](https://blog.chargebee.com/blog/saas-business-model-ai-monetization/)

### Multi-Model LLM Abstraction
- [Top 5 LiteLLM Alternatives in 2026 — TrueFoundry](https://www.truefoundry.com/blog/litellm-alternatives)
- [Building AI Agent With Multiple AI Model Providers Using an LLM Gateway — Dev.to](https://dev.to/crosspostr/building-ai-agent-with-multiple-ai-model-providers-using-an-llm-gateway-openai-anthropic-gemini-fl2)
- [aisuite — PyPI](https://pypi.org/project/aisuite/)
- [The LLM Abstraction Layer — ProxAI](https://www.proxai.co/blog/archive/llm-abstraction-layer)
- [Unifying 3 LLM APIs in Python — Dev.to](https://dev.to/inozem/unifying-3-llm-apis-in-python-openai-anthropic-google-with-one-sdk-4l2)

### Developer Productivity
- [My Predictions for MCP and AI-Assisted Coding in 2026 — Dev.to](https://dev.to/blackgirlbytes/my-predictions-for-mcp-and-ai-assisted-coding-in-2026-16bm)
- [Top 100 Developer Productivity Statistics with AI Tools 2026 — Index.dev](https://www.index.dev/blog/developer-productivity-statistics-with-ai-tools)
- [Best AI Coding Agents for 2026: Real-World Developer Reviews — Faros AI](https://www.faros.ai/blog/best-ai-coding-agents-2026)
- [Codified Context: Infrastructure for AI Agents in a Complex Codebase — arXiv](https://arxiv.org/abs/2602.20478)
- [Memory for AI Agents: A New Paradigm of Context Engineering — The New Stack](https://thenewstack.io/memory-for-ai-agents-a-new-paradigm-of-context-engineering/)

### Cloud Hosting & Architecture
- [Cloudflare Agents Docs](https://developers.cloudflare.com/agents/)
- [Host AI Agents on Cloud Run — Google Cloud](https://docs.google.com/run/docs/ai-agents)
- [Build a Multi-Tenant Generative AI Environment — AWS](https://aws.amazon.com/blogs/machine-learning/build-a-multi-tenant-generative-ai-environment-for-your-enterprise-on-aws/)
- [10 Best AI Agent Hosting Platforms 2026 — Fast.io](https://fast.io/resources/best-ai-agent-hosting-platforms/)
- [Reference Architecture: OpenClaw — RobotPaper](https://robotpaper.ai/reference-architecture-openclaw-early-feb-2026-edition-opus-4-6/)
- [Top Cloud Platforms for Enterprise AI Deployment — Render](https://render.com/articles/best-cloud-platforms-for-enterprise-ai-deployment)

### Skill Marketplace
- [ClawHub Skills Marketplace: Developer Guide 2026 — DigitalApplied](https://www.digitalapplied.com/blog/clawhub-skills-marketplace-developer-guide-2026)
- [SkillsMP — Agent Skills Marketplace](https://skillsmp.com)
- [Claude Code Plugin Marketplace Walkthrough — Medium](https://medium.com/@markchen69/claude-code-has-a-skills-marketplace-now-a-beginner-friendly-walkthrough-8adeb67cdc89)
- [n-skills: Curated Plugin Marketplace for AI Agents — GitHub](https://github.com/numman-ali/n-skills)

### Investor Context
- [Investors Spill What They Aren't Looking for Anymore in AI SaaS — TechCrunch](https://techcrunch.com/2026/03/01/investors-spill-what-they-arent-looking-for-anymore-in-ai-saas-companies/)
- [Scaling an AI Supernova: Lessons from Anthropic, Cursor, and fal — Bessemer](https://www.bvp.com/atlas/scaling-an-ai-supernova-lessons-from-anthropic-cursor-and-fal)
- [Developer Laws in the AI Era — Bessemer](https://www.bvp.com/atlas/developer-laws-in-the-ai-era)
