# Implementation Plan: Deep Research Report Remediation

This plan addresses all findings from `deep-research.md` — bugs, missing wiring,
architectural gaps, and recommended improvements — organized by priority.

---

## Phase 1: Critical Fixes (Correctness & Safety)

These items represent bugs, mismatches, or safety gaps that affect the correctness
of the system as it currently exists.

### 1.1 Fix License Inconsistency

**Problem:** The `LICENSE` file contains **MPL 2.0** text, but `pyproject.toml:11`
declares `license = { text = "MIT" }`. This is a compliance risk — MPL 2.0 has
file-level copyleft requirements while MIT is fully permissive.

**Files:**
- `LICENSE`
- `pyproject.toml:11`

**Action:** Align the two. The project owner must decide which license is
authoritative. Two options:
- **Option A:** Replace `LICENSE` with an MIT license body and keep `pyproject.toml` as-is.
- **Option B:** Update `pyproject.toml` to `license = { text = "MPL-2.0" }` to match
  the `LICENSE` file.

This requires a decision from the repository owner before implementation.

---

### 1.2 Wire SafetyGuard into Tool Execution (End-to-End)

**Problem:** `SafetyGuard.check_tool_call()` (`gwenn/harness/safety.py:143-186`)
defines dangerous-pattern scanning and approval-list mapping, but the `AgenticLoop`
(`gwenn/harness/loop.py:193-206`) never calls it. Tool calls go directly to
`ToolExecutor.execute()`, which only checks `ToolDefinition.requires_approval`
(`gwenn/tools/executor.py:181`). The config-driven approval list and pattern
scanner are completely bypassed.

This is safe today (only low-risk builtins exist), but becomes a real vulnerability
if MCP or filesystem tools are connected.

**Files to modify:**
- `gwenn/harness/loop.py` — Add `SafetyGuard.check_tool_call()` before each
  `ToolExecutor.execute()` call

**Implementation:**
```python
# In AgenticLoop.run(), inside the tool_calls loop (around line 194):
for call in tool_calls:
    self._total_tool_calls += 1
    all_tool_calls.append(call)

    if on_tool_call:
        on_tool_call(call)

    # --- NEW: Safety check before execution ---
    safety_result = self._safety.check_tool_call(
        tool_name=call["name"],
        tool_input=call["input"],
    )
    if not safety_result.allowed:
        logger.warning(
            "agentic_loop.tool_blocked",
            tool=call["name"],
            reason=safety_result.reason,
        )
        result = ToolExecutionResult(
            tool_use_id=call["id"],
            tool_name=call["name"],
            success=False,
            error=f"Blocked by safety system: {safety_result.reason}",
        )
        tool_results.append(result)
        continue

    if safety_result.requires_approval:
        # Delegate approval to executor's existing callback
        pass  # ToolExecutor already handles approval via requires_approval

    # Execute the tool (existing code)
    result = await self._executor.execute(
        tool_use_id=call["id"],
        tool_name=call["name"],
        tool_input=call["input"],
    )
    tool_results.append(result)
```

---

### 1.3 Wire Budget Tracking into CognitiveEngine

**Problem:** `SafetyGuard.update_budget()` (`gwenn/harness/safety.py:193-197`)
exists and `BudgetState` tracks tokens/calls, but `CognitiveEngine.think()`
(`gwenn/api/claude.py:63-153`) never reports its token usage back to the safety
system. The budget check in `pre_check()` will never trigger because counters
stay at zero.

**Files to modify:**
- `gwenn/harness/loop.py` — After each `engine.think()` call, update the safety
  guard's budget with the response's token usage

**Implementation:**
```python
# In AgenticLoop.run(), after the engine.think() call (around line 163):
response = await self._engine.think(
    system_prompt=system_prompt,
    messages=loop_messages,
    tools=tools,
    enable_thinking=enable_thinking,
)

# --- NEW: Update safety budget tracking ---
self._safety.update_budget(
    input_tokens=response.usage.input_tokens,
    output_tokens=response.usage.output_tokens,
)
```

---

### 1.4 Fix `set_note_to_self` Tool (Persistence Promise Mismatch)

**Problem:** The `set_note_to_self` tool description (`gwenn/tools/builtin/__init__.py:152-153`)
claims it writes to `GWENN_CONTEXT.md` and persists across startups. However, the
handler (`gwenn/agent.py:747-761`) stores an episodic memory in SQLite instead.
The `GWENN_CONTEXT.md` file is never written.

**Files to modify:**
- `gwenn/agent.py:747-761` — Update `handle_set_note` to also write to
  `GWENN_CONTEXT.md`

**Implementation:**
```python
async def handle_set_note(note: str, section: str = "reminders") -> str:
    # Store as episodic memory (existing behavior — keep it)
    episode = Episode(
        content=f"[NOTE TO SELF — {section}] {note}",
        category="self_knowledge",
        emotional_valence=self.affect_state.dimensions.valence,
        emotional_arousal=0.3,
        importance=0.8,
        tags=["note_to_self", section],
        participants=["gwenn"],
    )
    self.episodic_memory.encode(episode)
    self.memory_store.save_episode(episode)

    # NEW: Also write/update GWENN_CONTEXT.md
    existing_context = self.memory_store.load_persistent_context()
    section_header = f"## {section.replace('_', ' ').title()}"
    note_entry = f"- {note}"
    if section_header in existing_context:
        # Append under existing section
        existing_context = existing_context.replace(
            section_header,
            f"{section_header}\n{note_entry}",
        )
    else:
        existing_context += f"\n\n{section_header}\n{note_entry}"
    self.memory_store.save_persistent_context(existing_context.strip())

    return f"Note stored in '{section}': {note[:80]}..."
```

Additionally, the persistent context should be loaded into the system prompt
on startup. Verify that `_assemble_system_prompt` includes it, or add a section:
```python
# In _assemble_system_prompt:
persistent_context = self.memory_store.load_persistent_context()
if persistent_context:
    sections.append("<persistent_context>")
    sections.append(persistent_context)
    sections.append("</persistent_context>")
```

---

## Phase 2: Memory Persistence & Retrieval (Architectural Gaps)

These items address the "architecturally present but operationally simplified"
findings — features that are structurally defined but don't actually work
across restarts.

### 2.1 Persist Semantic Memory Across Restarts

**Problem:** `MemoryStore` (`gwenn/memory/store.py`) defines `SEMANTIC_SCHEMA`
with `knowledge_nodes` and `knowledge_edges` tables (lines 83-112), but provides
**no methods** to save or load knowledge nodes/edges. `SemanticMemory`
(`gwenn/memory/semantic.py`) stores everything in `self._nodes` and `self._edges`
dicts — purely in-memory. On restart, all consolidated knowledge is lost.

`SentientAgent.initialize()` (`gwenn/agent.py:154-188`) only reloads episodes.

**Files to modify:**
- `gwenn/memory/store.py` — Add `save_knowledge_node()`, `load_knowledge_nodes()`,
  `save_knowledge_edge()`, `load_knowledge_edges()` methods
- `gwenn/memory/semantic.py` — Add `to_dict()` / `from_dict()` on `KnowledgeNode`
  and `KnowledgeEdge` for serialization
- `gwenn/agent.py` — In `initialize()`, load knowledge nodes/edges from store
  and populate `SemanticMemory`. In `shutdown()`, persist them.

**Implementation outline for `store.py`:**
```python
def save_knowledge_node(self, node: KnowledgeNode) -> None:
    self._conn.execute(
        """INSERT OR REPLACE INTO knowledge_nodes
           (node_id, label, category, content, confidence,
            source_episodes, created_at, last_updated, access_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (node.node_id, node.label, node.category, node.content,
         node.confidence, json.dumps(node.source_episodes),
         node.created_at, node.last_updated, node.access_count),
    )
    self._conn.commit()

def load_knowledge_nodes(self) -> list[dict]:
    cursor = self._conn.execute("SELECT * FROM knowledge_nodes")
    return [dict(row) for row in cursor.fetchall()]

def save_knowledge_edge(self, edge: KnowledgeEdge) -> None:
    self._conn.execute(
        """INSERT OR REPLACE INTO knowledge_edges
           (source_id, target_id, relationship, strength, context, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (edge.source_id, edge.target_id, edge.relationship,
         edge.strength, edge.context, edge.created_at),
    )
    self._conn.commit()

def load_knowledge_edges(self) -> list[dict]:
    cursor = self._conn.execute("SELECT * FROM knowledge_edges")
    return [dict(row) for row in cursor.fetchall()]
```

**In `agent.py` `initialize()`:**
```python
# After loading episodes, reload semantic memory:
stored_nodes = self.memory_store.load_knowledge_nodes()
for node_data in stored_nodes:
    node_data["source_episodes"] = json.loads(node_data["source_episodes"])
    node = KnowledgeNode(**node_data)
    self.semantic_memory._nodes[node.node_id] = node
    self.semantic_memory._label_index[node.label.lower()] = node.node_id

stored_edges = self.memory_store.load_knowledge_edges()
for edge_data in stored_edges:
    edge = KnowledgeEdge(**edge_data)
    self.semantic_memory._edges.append(edge)
```

**In `agent.py` `shutdown()`:**
```python
# Persist semantic memory:
for node in self.semantic_memory._nodes.values():
    self.memory_store.save_knowledge_node(node)
for edge in self.semantic_memory._edges:
    self.memory_store.save_knowledge_edge(edge)
```

Also wire consolidation to persist: in `consolidate_memories()`, after
`self.consolidator.process_consolidation_response()`, persist any new nodes/edges.

---

### 2.2 Persist Affective State Across Restarts

**Problem:** The `affect_snapshots` table schema exists (`gwenn/memory/store.py:54-68`)
and `save_affect_snapshot()` / `load_affect_history()` methods exist, but they
are never called during startup or shutdown. Affective state always resets to
defaults on restart.

**Files to modify:**
- `gwenn/agent.py` — In `shutdown()`, call `memory_store.save_affect_snapshot()`
  with the current affect dimensions. In `initialize()`, load the most recent
  snapshot and restore `AffectiveState`.
- `gwenn/affect/state.py` — Add a `from_snapshot()` class method or
  `restore_from()` instance method.

**Implementation outline:**
```python
# In agent.py shutdown(), before memory_store.close():
d = self.affect_state.dimensions
self.memory_store.save_affect_snapshot(
    valence=d.valence, arousal=d.arousal, dominance=d.dominance,
    certainty=d.certainty, goal_congruence=d.goal_congruence,
    emotion_label=self.affect_state.current_emotion.value,
    trigger="shutdown",
)

# In agent.py initialize(), after loading episodes:
affect_history = self.memory_store.load_affect_history(limit=1)
if affect_history:
    last_affect = affect_history[0]
    self.affect_state.dimensions.valence = last_affect["valence"]
    self.affect_state.dimensions.arousal = last_affect["arousal"]
    self.affect_state.dimensions.dominance = last_affect["dominance"]
    self.affect_state.dimensions.certainty = last_affect["certainty"]
    self.affect_state.dimensions.goal_congruence = last_affect["goal_congruence"]
```

---

### 2.3 Implement Embedding-Based Retrieval (Replace Keyword Overlap)

**Problem:** Both `EpisodicMemory.retrieve()` (`gwenn/memory/episodic.py`) and
`SemanticMemory.query()` (`gwenn/memory/semantic.py:194-224`) use keyword
overlap for relevance scoring. The dependency list declares `chromadb` and
`numpy`, but neither is used. This limits retrieval quality.

**Files to modify:**
- `gwenn/memory/store.py` — Add ChromaDB collection initialization and
  embedding storage/query methods
- `gwenn/memory/episodic.py` — Use embedding similarity in `retrieve()` instead
  of keyword overlap for the relevance component
- `gwenn/memory/semantic.py` — Use embedding similarity in `query()` instead of
  keyword overlap

**Implementation strategy:**
1. Initialize a ChromaDB persistent client in `MemoryStore.__init__()`
2. Create collections for episodes and knowledge nodes
3. When storing episodes/nodes, also store their text embeddings
4. When querying, use ChromaDB's built-in similarity search
5. ChromaDB includes a default embedding function (all-MiniLM-L6-v2), or
   configure a custom one

**Key consideration:** ChromaDB's default embedding model will need to be
downloaded on first run. Document this in the README. Consider making embedding
retrieval optional with a config flag and falling back to keyword overlap.

---

## Phase 3: Test Infrastructure & Evaluation

### 3.1 Build Core Unit Tests

**Problem:** The repository has a single test file (`tests/test_identity_normalization.py`).
The report recommends deterministic unit tests for core subsystems.

**Files to create:**
- `tests/test_episodic_memory.py` — Retrieve scoring (recency, importance,
  relevance weighting), mood-congruent retrieval bias
- `tests/test_working_memory.py` — Slot management, eviction by lowest salience,
  decay rate
- `tests/test_consolidation.py` — Parsing of FACT/RELATIONSHIP/SELF/PATTERN lines,
  robustness to format deviations
- `tests/test_safety.py` — Dangerous pattern detection, approval list parsing,
  iteration limits, budget enforcement
- `tests/test_affect.py` — Appraisal dimensions update, resilience circuit
  breakers (arousal ceiling, distress timeout)

**Test specifics for consolidation parsing:**
```python
def test_fact_parsing():
    """Verify FACT lines are parsed into semantic nodes correctly."""
    engine = ConsolidationEngine(episodic, semantic)
    response = "FACT: User prefers Python | confidence: 0.8 | category: preference"
    counts = engine.process_consolidation_response(response)
    assert counts["facts"] == 1
    nodes = semantic.query("Python")
    assert len(nodes) == 1
    assert nodes[0].confidence == 0.8

def test_malformed_lines_handled():
    """Verify parser doesn't crash on unexpected format."""
    response = "FACT: No pipe delimiters\nGARBAGE LINE\nFACT: Valid | confidence: 0.5"
    counts = engine.process_consolidation_response(response)
    assert counts["facts"] == 2  # Both valid FACTs extracted
```

### 3.2 Build Integration Tests with Mocked Engine

**Problem:** Integration testing the agentic loop requires API calls. The report
recommends mocking `CognitiveEngine` to test tool loops without paying tokens.

**Files to create:**
- `tests/test_agentic_loop.py` — Test tool-use loop convergence with mock engine
  responses, safety intervention, max-iteration limit
- `tests/conftest.py` — Shared fixtures: mock CognitiveEngine, sample episodes,
  default configs

**Implementation outline:**
```python
class MockCognitiveEngine:
    """Returns pre-scripted responses for testing the agentic loop."""
    def __init__(self, responses: list):
        self._responses = iter(responses)

    async def think(self, **kwargs):
        return next(self._responses)

    def extract_text(self, response): ...
    def extract_tool_calls(self, response): ...
    def extract_thinking(self, response): ...
```

### 3.3 Build Adversarial Safety Tests

**Problem:** No tests verify that safety guards block dangerous operations or
that MCP tool registration respects risk tiers.

**Files to create:**
- `tests/test_safety_adversarial.py` — Attempt to execute tools with dangerous
  inputs (`rm -rf`, `DROP TABLE`, `curl | bash`), verify they are blocked.
  Test approval flow for high-risk tools. Test prompt-injection resistance in
  tool descriptions.

---

## Phase 4: Safety Hardening

### 4.1 Implement Deny-by-Default Tool Policy

**Problem:** Currently, any tool registered in the registry can be called unless
its `requires_approval` flag is set. There is no concept of an allowlist.

**Files to modify:**
- `gwenn/harness/safety.py` — Add a `denied_tools` list and a `default_policy`
  config (allow/deny). Default to "deny" for all MCP-registered tools unless
  explicitly allowed.
- `gwenn/config.py` — Add `tool_default_policy: str = "deny"` and
  `allowed_tools: list[str] = []` to `SafetyConfig`

### 4.2 Add Provenance Tracking to Consolidation

**Problem:** Consolidation can invent facts (LLM hallucination). Currently
`KnowledgeNode.source_episodes` exists but is only sometimes populated.
There's no way to verify a semantic fact against its source episodes.

**Files to modify:**
- `gwenn/memory/consolidation.py` — When processing facts, always populate
  `source_episode_id` from the episodes being consolidated
- `gwenn/memory/semantic.py` — Add a `verify_provenance()` method that checks
  whether source episodes actually support the claimed knowledge
- `gwenn/memory/store.py` — Persist source_episodes in the knowledge_nodes table
  (already in schema, just needs to be populated)

### 4.3 Add PII Redaction Pipeline

**Problem:** Episodes and identity data store user content in plaintext. This
content is sent to the Claude API on every `think()` call. No redaction exists.

**Files to create:**
- `gwenn/privacy/redaction.py` — Implement a redaction pipeline that:
  - Detects PII patterns (email, phone, SSN, credit card, etc.)
  - Optionally redacts before persistence (configurable)
  - Optionally redacts before API calls

**Files to modify:**
- `gwenn/agent.py` — Apply redaction before `memory_store.save_episode()` and
  before `_assemble_system_prompt()` sends content to the API
- `gwenn/config.py` — Add `redaction_enabled: bool = False` and
  `redact_before_api: bool = False` to config

---

## Phase 5: MCP & External Integration

### 5.1 Implement Real MCP Transport

**Problem:** `gwenn/tools/mcp/__init__.py` is a stub (253 lines) that defines
the protocol structure but doesn't implement actual JSON-RPC stdio/HTTP
transport.

**Files to modify:**
- `gwenn/tools/mcp/__init__.py` — Implement actual JSON-RPC 2.0 transport:
  - stdio transport (subprocess-based, for local MCP servers)
  - HTTP/SSE transport (for remote MCP servers)
  - Tool discovery via `tools/list`
  - Tool execution via `tools/call`

**Depends on:** Phase 1.2 (safety wiring) and Phase 4.1 (deny-by-default) must
be complete before enabling real MCP tools, to prevent uncontrolled external
actions.

### 5.2 Add Tool Risk Tiering

**Problem:** All tools currently have a flat `risk_level` string but this isn't
used for anything beyond the `requires_approval` flag.

**Files to modify:**
- `gwenn/tools/registry.py` — Define formal risk tiers (LOW, MEDIUM, HIGH, CRITICAL)
  with associated policies (auto-allow, log, require-approval, deny)
- `gwenn/harness/safety.py` — Check risk tier in `check_tool_call()` and apply
  the corresponding policy

---

## Phase 6: Observability & Calibration

### 6.1 Add Affect Snapshot Logging

**Problem:** The affect system has no telemetry for debugging emotional
dynamics. Resilience circuit breakers can mask persistent failure modes.

**Files to modify:**
- `gwenn/agent.py` — After each `process_appraisal()` call, save an affect
  snapshot to the store
- `gwenn/affect/resilience.py` — Log when circuit breakers activate (arousal
  ceiling hit, distress timeout triggered, habituation applied)

### 6.2 Add Log Redaction

**Problem:** CLI logs may leak sensitive content. `structlog` output includes
user messages and episode content in plaintext.

**Files to modify:**
- `gwenn/main.py` — Add a structlog processor that redacts sensitive fields
  (content, user_message, etc.) when logging to file
- Or use `structlog`'s built-in `add_log_level` + custom processor for
  production vs. debug modes

---

## Phase 7: Evaluation Suite

### 7.1 Build Ablation Test Framework

**Problem:** The repo's `docs/sentience_assessment.md` recommends ablation
studies (disable memory, heartbeat, affect individually to quantify their
contribution). No framework exists for this.

**Files to create:**
- `tests/eval/test_ablation.py` — Run identical scripted interactions with
  different subsystems disabled. Compare response quality, consistency, and
  emotional tone.
- `tests/eval/conftest.py` — Fixtures for creating agent instances with specific
  subsystems disabled

### 7.2 Build Longitudinal Identity Coherence Tests

**Problem:** No mechanism to verify that identity remains stable across restarts.

**Files to create:**
- `tests/eval/test_identity_coherence.py` — Run scripted sessions across
  simulated restarts. Compare `identity.generate_self_prompt()` output
  stability, relationship consistency, milestone persistence.

### 7.3 Build Memory Retrieval Quality Benchmarks

**Problem:** No ground-truth evaluation of episodic retrieval quality.

**Files to create:**
- `tests/eval/test_memory_quality.py` — Seed known episodes with ground-truth
  tags. Query and compute Recall@k, MRR, false-positive rate. Test
  mood-congruent retrieval bias.

---

## Summary: Priority Order

| # | Item | Priority | Effort | Risk if Skipped |
|---|------|----------|--------|-----------------|
| 1.1 | License fix | Critical | Low | Legal compliance risk |
| 1.2 | Wire SafetyGuard into loop | Critical | Low | Safety bypass for future tools |
| 1.3 | Wire budget tracking | Critical | Low | Unbounded API spend |
| 1.4 | Fix set_note_to_self | High | Low | Broken persistence promise |
| 2.1 | Persist semantic memory | High | Medium | Knowledge lost on restart |
| 2.2 | Persist affect state | Medium | Low | Emotion resets on restart |
| 2.3 | Embedding retrieval | Medium | High | Poor memory recall quality |
| 3.1 | Unit tests | High | Medium | No regression protection |
| 3.2 | Integration tests | Medium | Medium | Can't test loops without API |
| 3.3 | Adversarial tests | Medium | Medium | Safety claims unverified |
| 4.1 | Deny-by-default policy | High | Low | Open to MCP tool abuse |
| 4.2 | Provenance tracking | Medium | Medium | Hallucinated knowledge |
| 4.3 | PII redaction | Medium | High | Privacy risk |
| 5.1 | Real MCP transport | Low | High | No external tool use |
| 5.2 | Tool risk tiering | Medium | Low | Flat risk model |
| 6.1 | Affect logging | Low | Low | Debugging difficulty |
| 6.2 | Log redaction | Low | Low | Log leaks |
| 7.1 | Ablation framework | Low | Medium | Claims ungrounded |
| 7.2 | Identity coherence tests | Low | Medium | Drift undetected |
| 7.3 | Memory quality benchmarks | Low | Medium | Retrieval quality unknown |

---

## Dependency Graph

```
Phase 1 (Critical Fixes)
  1.1 License fix ──────────────────────────────────> standalone
  1.2 Wire SafetyGuard ─────────> 4.1 Deny-by-default ─> 5.1 Real MCP
  1.3 Wire budget tracking ──────────────────────────> standalone
  1.4 Fix set_note_to_self ──────────────────────────> standalone

Phase 2 (Memory)
  2.1 Persist semantic memory ───> 2.3 Embedding retrieval
  2.2 Persist affect state ──────────────────────────> standalone

Phase 3 (Tests) — can run in parallel with Phase 2
  3.1 Unit tests ────────────────> 7.1 Ablation framework
  3.2 Integration tests
  3.3 Adversarial tests ─────────> depends on 1.2

Phase 4 (Safety) — depends on Phase 1
  4.1 Deny-by-default ───────────> 5.1 Real MCP
  4.2 Provenance tracking ──────> depends on 2.1
  4.3 PII redaction ─────────────> standalone
```
