# Implementation Plan: Deep Research Report Remediation

This plan addresses all findings from `deep-research.md` — bugs, missing wiring,
architectural gaps, and recommended improvements — organized by priority.

**Last updated:** 2026-02-24

---

## Phase 1: Critical Fixes (Correctness & Safety)

These items represent bugs, mismatches, or safety gaps that affect the correctness
of the system as it currently exists.

### 1.1 Fix License Inconsistency -- DONE

**Status:** Resolved. `pyproject.toml` now declares `license = { text = "MPL-2.0" }`
matching the `LICENSE` file.

---

### 1.2 Wire SafetyGuard into Tool Execution (End-to-End) -- DONE

**Status:** Resolved. `SafetyGuard.check_tool_call()` is now invoked in the
`AgenticLoop` before every `ToolExecutor.execute()` call. Non-builtin tools are
blocked by default via deny-by-default policy.

---

### 1.3 Wire Budget Tracking into CognitiveEngine -- DONE

**Status:** Resolved. `SafetyGuard.update_budget()` is called after each
`engine.think()` in the agentic loop with the response's token usage.

---

### 1.4 Fix `set_note_to_self` Tool (Persistence Promise Mismatch) -- DONE

**Status:** Resolved. `handle_set_note` now writes to both episodic memory and
`GWENN_CONTEXT.md`. The persistent context is loaded into the system prompt on
startup.

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

## Phase 3: Test Infrastructure & Evaluation -- DONE

### 3.1 Build Core Unit Tests -- DONE

**Status:** Resolved. 35+ test files with 2941 tests covering all core subsystems:
- `tests/test_episodic_memory.py` — Retrieve scoring, mood-congruent retrieval
- `tests/test_working_memory.py` — Slot management, eviction, decay
- `tests/test_consolidation.py` — FACT/RELATIONSHIP/SELF/PATTERN parsing, malformed lines
- `tests/test_safety.py` — Dangerous patterns, approval lists, iteration limits, budgets
- `tests/test_affect.py` — Appraisal dimensions, resilience circuit breakers
- `tests/test_config_paths.py` — Config path derivation and resolution
- Plus: agent_runtime, agentic_loop, appraisal, channels, cognitive_engine, daemon,
  discord, ethics, goal_system, heartbeat, identity, inner_life, interagent,
  main_session, mcp_client, memory_store, metacognition, redaction, retry,
  security_regressions, semantic_memory, sensory, session_store, skill_system,
  telegram, theory_of_mind, tool_executor, tool_registry

### 3.2 Build Integration Tests with Mocked Engine -- DONE

**Status:** Resolved. `tests/test_agentic_loop.py` tests tool-use loop convergence
with mock engine responses, safety intervention, and max-iteration limits.
`tests/conftest.py` provides shared fixtures.

### 3.3 Build Adversarial Safety Tests -- DONE

**Status:** Resolved. `tests/test_safety_adversarial.py` tests dangerous inputs
(`rm -rf`, `DROP TABLE`, `curl | bash`), approval flows, and prompt-injection
resistance.

---

## Phase 4: Safety Hardening -- MOSTLY DONE

### 4.1 Implement Deny-by-Default Tool Policy -- DONE

**Status:** Resolved. Non-builtin tools are denied by default via `SafetyGuard`.
`GWENN_SANDBOX_ENABLED=True` is the default. MCP-registered tools require
explicit allowlisting.

### 4.2 Add Provenance Tracking to Consolidation

**Status:** Open. `KnowledgeNode.source_episodes` exists but is only sometimes
populated. No `verify_provenance()` method yet. Depends on 2.1.

### 4.3 Add PII Redaction Pipeline -- DONE

**Status:** Resolved. `gwenn/privacy/redaction.py` implements PII pattern detection
(email, phone, SSN, credit card, IP). Configurable via `GWENN_REDACTION_ENABLED`,
`GWENN_REDACT_BEFORE_API`, `GWENN_REDACT_BEFORE_PERSIST`. Log fields are always
redacted via a shared `configure_logging()` structlog processor. Daemon sessions
are redacted by default.

---

## Phase 5: MCP & External Integration

### 5.1 Implement Real MCP Transport -- DONE

**Status:** Resolved. `gwenn/tools/mcp/__init__.py` implements JSON-RPC 2.0
transport with both `stdio` (subprocess) and `streamable_http` (HTTP POST)
transports. Tool discovery via `tools/list`, execution via `tools/call`,
Content-Length framing for stdio, and optional Bearer auth for HTTP.
MCP tools are registered as proxy tools prefixed with `mcp_<server>_`.

### 5.2 Add Tool Risk Tiering

**Problem:** All tools currently have a flat `risk_level` string but this isn't
used for anything beyond the `requires_approval` flag.

**Files to modify:**
- `gwenn/tools/registry.py` — Define formal risk tiers (LOW, MEDIUM, HIGH, CRITICAL)
  with associated policies (auto-allow, log, require-approval, deny)
- `gwenn/harness/safety.py` — Check risk tier in `check_tool_call()` and apply
  the corresponding policy

---

## Phase 6: Observability & Calibration -- MOSTLY DONE

### 6.1 Add Affect Snapshot Logging

**Status:** Open. Affect snapshots schema exists in `MemoryStore` but is not
called during normal operation.

### 6.2 Add Log Redaction -- DONE

**Status:** Resolved. `gwenn/main.py` includes a `_redact_sensitive_fields`
structlog processor that PII-redacts and truncates sensitive log fields (`content`,
`user_message`, `thought`, `note`, `query`). Thread-safe singleton via
`@functools.lru_cache`. Both `main.py` and `daemon.py` share the same
`configure_logging()` configuration.

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

| # | Item | Priority | Effort | Status |
|---|------|----------|--------|--------|
| 1.1 | License fix | Critical | Low | DONE |
| 1.2 | Wire SafetyGuard into loop | Critical | Low | DONE |
| 1.3 | Wire budget tracking | Critical | Low | DONE |
| 1.4 | Fix set_note_to_self | High | Low | DONE |
| 2.1 | Persist semantic memory | High | Medium | DONE |
| 2.2 | Persist affect state | Medium | Low | Open |
| 2.3 | Embedding retrieval | Medium | High | DONE (keyword/embedding/hybrid) |
| 3.1 | Unit tests | High | Medium | DONE (2941 tests) |
| 3.2 | Integration tests | Medium | Medium | DONE |
| 3.3 | Adversarial tests | Medium | Medium | DONE |
| 4.1 | Deny-by-default policy | High | Low | DONE |
| 4.2 | Provenance tracking | Medium | Medium | Open |
| 4.3 | PII redaction | Medium | High | DONE |
| 5.1 | Real MCP transport | Low | High | DONE |
| 5.2 | Tool risk tiering | Medium | Low | Open |
| 6.1 | Affect logging | Low | Low | Open |
| 6.2 | Log redaction | Low | Low | DONE |
| 7.1 | Ablation framework | Low | Medium | Open |
| 7.2 | Identity coherence tests | Low | Medium | Open |
| 7.3 | Memory quality benchmarks | Low | Medium | Open |

**Completed:** 14/19 items (74%). Remaining items are lower priority or depend on
architectural decisions (affect persistence, provenance tracking, ablation framework).

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
