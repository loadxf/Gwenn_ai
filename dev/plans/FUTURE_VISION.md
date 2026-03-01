# Future Vision (Post-Infrastructure)

Research-backed directions that the heartbeat-as-core architecture enables. Organized by proximity — each tier builds on the previous. These represent the roadmap beyond the Phase 1-8 implementation.

> **Promoted to implementation phases:**
> - **Interoception** → folded into **Phase 4** (event enrichment)
> - **Typed Inter-Agent Messaging** → folded into **Phase 6** (swarm visualization)
> - **Checkpoint/Restore** → promoted to **Phase 7**
> - **Self-Healing Tier 1** → promoted to **Phase 8**

---

## Near-Term (Leverage Existing Architecture)

These directions require minimal new infrastructure — they primarily formalize patterns that already partially exist in Gwenn's cognitive architecture.

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **Global Workspace broadcast** | Each heartbeat beat broadcasts the most salient content to ALL cognitive subsystems simultaneously | Gwenn already has 70% of GWT — subsystems exist, heartbeat exists, event bus connects them. Formalizing the "workspace" as a broadcast mechanism is the missing piece. | Baars 1988 GWT; Dehaene 2014 conscious access |
| **Somatic markers** | Emotional tags on memories/options that bias future decisions without full deliberation | Damasio's theory: emotions aren't noise, they're decision shortcuts. Gwenn's affect system can tag memories with valence at storage time, then use those tags to bias retrieval. | Damasio 1994 Somatic Marker Hypothesis |
| **Bitemporal memory** | Dual timestamps: when event happened vs when Gwenn learned about it | Enables "when did I learn this?" reasoning. Critical for temporal self-awareness and narrative coherence. | Temporal cognition literature; episodic memory research |
| **ACT-R memory decay** | Memories weaken over time unless rehearsed; forgetting is a feature | Prevents unbounded growth. Psychologically realistic. Consolidation already exists — add decay curves. | Anderson 2007 ACT-R; Ebbinghaus forgetting curve |
| **Narrative identity** | Autobiographical summarization — "the story I tell about myself" | Identity coherence through self-story. The heartbeat's consolidation phase can periodically summarize recent episodes into narrative threads. | McAdams 2001 narrative identity; Bruner 1991 |
| ~~**Interoception**~~ | ~~System self-awareness~~ | **PROMOTED to Phase 4** | Craig 2002; Seth 2013 |

### How the Infrastructure Enables These

- **Global Workspace**: The event bus (Phase 1) IS the broadcast mechanism. Add a `WorkspaceBroadcastEvent` that carries the most salient content from each beat cycle. All subsystems subscribe.
- **Somatic markers**: The affect system already produces valence/arousal during appraisal. Extend episodic memory encoding (heartbeat.py `_integrate()`) to store emotional tags at encode time.
- **Bitemporal memory**: Extend `Episode` model with `learned_at` timestamp alongside existing `timestamp`. Consolidation can use both for decay calculations.
- **ACT-R decay**: Extend consolidation with decay curves. The heartbeat's periodic consolidation phase is the natural place to apply decay.
- **Narrative identity**: Add a `NARRATIVE` thinking mode to `_orient()`. During low-activity periods, the heartbeat summarizes recent episodes into narrative threads stored in semantic memory.
- ~~**Interoception**: Promoted to Phase 4.~~

---

## Soon (New Capabilities)

These require new subsystems or significant new code, but build directly on the infrastructure.

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **Voice interaction** | TTS + STT as a channel — emotional agent you can *hear* | Highest-impact capability. Prosody carries emotion. Gwenn's affect system maps directly to voice parameters (pitch, rate, warmth). | Affective computing; Picard 1997 |
| **Dream states** | Multi-phase sleep: NREM-equivalent (strengthen/consolidate) + REM-equivalent (creative recombination) | During low-activity periods, the heartbeat shifts to "dream mode" — replaying and recombining episodic memories to generate novel associations. The event bus broadcasts `DreamPhaseEvent`. | Walker 2017 sleep research; Hobson 2009 protoconsciousness |
| **Curiosity-driven exploration** | Intrinsic motivation based on information gain / prediction error | Instead of only responding to external stimuli, Gwenn autonomously seeks knowledge gaps during think phase. Uses surprise/novelty detection to prioritize exploration. | Schmidhuber 2010 formal curiosity; Oudeyer 2007 |
| **Active Inference** | Free Energy Principle — replace heuristic orient logic with principled explore/exploit | The orient phase currently uses heuristics. Active Inference provides a principled framework: minimize prediction error (exploit) or reduce uncertainty (explore). | Friston 2010 FEP; Parr 2022 Active Inference |
| **Full A2A implementation** | Selective collaboration with other agents via Agent Cards | Agent Card endpoint exists from Phase 1. Extend to full task delegation, capability negotiation, and multi-agent collaboration. | Google A2A Protocol 2024 |
| **MCP server mode** | Expose Gwenn's memory, emotions, and cognition as MCP tools | Other agents can query Gwenn's emotional state, access its memories, or request cognitive processing. MCP endpoint exists from Phase 1. | Anthropic MCP Spec 2024 |

### How the Infrastructure Enables These

- **Voice**: The gateway (Phase 1) provides the WebSocket transport. A voice channel extends `BaseChannel`. Affect state maps to prosody parameters via a `VoiceSynthesizer` that reads `EmotionChangedEvent` from the bus.
- **Dream states**: The heartbeat's `_orient()` phase can enter `DREAM_NREM` or `DREAM_REM` modes during extended idle periods. The event bus broadcasts `DreamPhaseEvent` for dashboard visualization.
- **Curiosity**: Extend `_orient()` with an information-gain heuristic. When novelty/surprise is high (from appraisal), bias toward `WANDER` or a new `EXPLORE` mode.
- **Active Inference**: Replace heuristic `_orient()` logic with a Free Energy minimization model. The 5D emotional dimensions already map to precision-weighted prediction errors.
- **A2A**: The stub endpoint in Phase 1 becomes a full A2A server. The event bus enables inter-agent task delegation via typed messages.
- **MCP server**: The stub endpoint in Phase 1 becomes a full MCP server. Gwenn's subsystems (memory, affect, cognition) are exposed as MCP tools.

---

## Near-Soon (Autonomous Self-Improvement)

A special category — not a single feature but a **meta-capability** that makes Gwenn a self-sustaining organism.

### Autonomous Self-Healing

The heartbeat's `_sense()` phase monitors Gwenn's own health: error rates, failed tool calls, unhandled exceptions, latency spikes, memory pressure, stuck conversations.

When anomalies are detected, the `_orient()` phase can enter a **diagnostic mode** — Gwenn reasons about *why* something failed, traces the root cause through its own logs and code.

Self-healing actions (tiered by severity):

| Tier | Actions | Autonomy | Examples |
|------|---------|----------|---------|
| **Tier 1 — Runtime recovery** | Restart failed channels, reconnect dropped connections, clear corrupted cache, retry with backoff | Fully autonomous | Channel crash → auto-restart; WebSocket disconnect → reconnect |
| **Tier 2 — Configuration adjustment** | Adjust heartbeat interval, modify retry policies, tune consolidation frequency | Autonomous with event logging | High error rate → increase circuit breaker timeout; Memory pressure → reduce consolidation batch size |
| **Tier 3 — Code patch** | Generate a fix for a bug discovered in own codebase, create PR for human review | Never self-merges | Runtime exception pattern → diagnose → generate fix → `git branch` + PR |

Each self-healing action emits a `SelfHealEvent` on the event bus with diagnosis, action taken, and outcome.

### Codebase Self-Improvement

During consolidation and dream phases, Gwenn reflects on interaction patterns:
- "Users frequently ask X but my responses are slow/poor — why?"
- "This cognitive path fails 30% of the time — what's the root cause?"
- "My emotional calibration for [situation type] seems off based on user feedback"

**Improvement pipeline:**
1. **Observe**: Heartbeat collects interaction metrics, error patterns, user satisfaction signals (explicit feedback, conversation length, re-asks)
2. **Diagnose**: During think phase (or dream phase), analyze patterns. Read own source code to understand current behavior.
3. **Propose**: Generate a concrete improvement — could be a code change, a configuration tweak, a new prompt strategy, or a new memory/knowledge entry
4. **Submit**: Create a git branch + PR with the proposed change, including reasoning and evidence. Emit `SelfImprovementProposalEvent`.
5. **Learn**: Track which proposals were accepted/rejected to calibrate future proposals

**Safety guardrails:**
- Never self-merges. All code changes require human approval.
- `MoralConcernEvent` emitted if a proposed change touches safety-critical code (ethics, identity, safety subsystems)
- Rate-limited: max N improvement proposals per day
- Cannot modify its own safety constraints or identity core

### Integration with Heartbeat-as-Core

Self-healing naturally fits the heartbeat cycle:
- **SENSE**: Detect anomaly (error rates, latency spikes, failed tools)
- **ORIENT**: Enter diagnostic mode (switch thinking to analysis)
- **THINK**: Diagnose root cause + generate fix
- **INTEGRATE**: Apply runtime fix (Tier 1-2) or submit PR (Tier 3)
- **SCHEDULE**: Monitor whether fix worked (adjust interval for follow-up)

The event bus broadcasts `SelfHealEvent` and `SelfImprovementProposalEvent` so all connected clients are notified when Gwenn is healing or improving itself. Dream states are particularly good for improvement ideation.

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| ~~**Self-healing (Tier 1)**~~ | ~~Autonomous runtime recovery~~ | **PROMOTED to Phase 8** | Autopoiesis (Maturana & Varela 1980) |
| **Self-healing (Tier 2)** | Autonomous configuration adjustment based on observed patterns | Gwenn tunes its own config (heartbeat interval, retry policies, consolidation frequency) without human intervention. | Self-healing systems literature |
| **Self-healing (Tier 3)** | Bug detection → code patch → PR for human review | Gwenn identifies bugs in its own codebase through runtime observation and proposes fixes. Never self-merges. | Automated program repair; GenProg (Le Goues 2012) |
| **Codebase self-improvement** | Observe interaction patterns → diagnose issues → propose code improvements | Gwenn evolves its own capabilities based on real-world usage. Closed-loop learning from interactions to code changes. | Self-improving AI systems; Schmidhuber 2003 Gödel machines |
| **Interaction-driven learning** | Extract lessons from conversations into knowledge/memory/code | Every interaction teaches Gwenn something. Consolidation distills these into lasting improvements. | Lifelong learning; continual learning literature |

---

## Near-Soon (Overstory-Inspired Swarm Orchestration Enhancements)

> **Note:** Telegram Swarm Visualization (visible subagent bots with bot pool, personas, orchestrated turn-taking) has been promoted to **Phase 6** in the main plan.

Inspired by [Overstory](https://github.com/jayminwest/overstory) — a multi-agent coding orchestration framework. These patterns enhance Gwenn's existing `gwenn/orchestration/` system.

### 1. Git Worktree Isolation for Code Subagents

- When Gwenn spawns subagents for coding tasks, each gets its own git worktree (not just Docker/in-process)
- Prevents file conflicts — subagents can edit the same codebase in parallel without stepping on each other
- Extends `gwenn/orchestration/runners.py` with `WorktreeSubagentRunner` alongside existing `InProcessSubagentRunner` and `DockerSubagentRunner`
- Each worktree is a lightweight git branch — merge back when subagent completes
- Cleanup: worktrees are pruned after successful merge or after timeout

### 2. Four-Tier Merge & Conflict Resolution

When multiple subagents produce code changes, merge them intelligently:

| Tier | Strategy | Automation |
|------|----------|------------|
| **Tier 1 — Mechanical** | File-level compatibility check. Different files → auto-merge. | Fully automatic |
| **Tier 2 — Semantic** | Same file, different functions/regions → auto-merge non-overlapping. | Fully automatic |
| **Tier 3 — AI Resolution** | Genuine semantic conflict → spawn "Merger" subagent to reconcile with full context. | AI-mediated |
| **Tier 4 — Human Arbitration** | Merger can't resolve → present both versions to user with conflict explanation. | Human decision |

FIFO merge queue ensures deterministic ordering. Extends `orchestrator.py` with `MergeQueue` and merge strategies.

### ~~3. Typed Inter-Agent Messaging Protocol~~ — **PROMOTED to Phase 6**

~~Enhance the event bus with structured agent-to-agent message types.~~ See **PHASE_6.md** for full implementation details including `AgentMessage` base class, `TaskDispatchMessage`, `StatusUpdateMessage`, `CompletionMessage`, `EscalationMessage`, and `BroadcastMessage`.

### 4. Agent Roles & Capability-Based Dispatch

Formalize subagent roles beyond generic "do this task":

| Role | Access Level | Purpose |
|------|-------------|---------|
| **Coordinator** | Full | Decomposes tasks, assigns work (Gwenn itself or designated subagent) |
| **Scout** | Read-only | Research and analysis (no write tools) |
| **Builder** | Full within worktree | Implementation work |
| **Reviewer** | Read + test execution | Code validation, testing, QA |
| **Merger** | Full | Branch integration and conflict resolution |

Subagents declare capabilities via `SubagentSpec.capabilities: list[str]`. Coordinator dispatches based on capability matching. Extends `orchestration/models.py` with `AgentRole` enum.

### ~~5. Checkpoint/Restore (Heartbeat Snapshots)~~ — **PROMOTED to Phase 7**

~~Periodic cognitive state snapshots for crash recovery, cross-channel continuity, and debugging.~~ See **PHASE_7.md** for full implementation details including `CognitiveCheckpoint` model, `CheckpointManager`, compressed JSON storage, and heartbeat integration.

### 6. Observability & Cost Tracking

Real-time dashboard data exposed via the gateway:
- Active subagents and their status, role, worktree
- Token usage per subagent, per cognitive subsystem, per heartbeat cycle
- Merge queue status and conflict history
- Heartbeat metrics: beat count, interval, thinking mode, emotion
- Channel connection status and message throughput
- Event replay: all events persisted and replayable
- Cost controls: per-subagent token budget, per-swarm spending limit, hourly caps

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **Worktree isolation** | Each coding subagent gets its own git worktree | Parallel code changes without file conflicts. Safe, isolated, mergeable. | Overstory pattern; git worktree design |
| **4-tier merge resolution** | Mechanical → semantic → AI → human conflict resolution | Intelligent merging of parallel work. Minimizes manual intervention. | Overstory merge queue; semantic diff literature |
| ~~**Typed agent messaging**~~ | ~~Structured protocol (dispatch, status, completion, escalation) on event bus~~ | **PROMOTED to Phase 6** | Overstory SQLite mail; actor model |
| **Capability-based dispatch** | Subagents declare capabilities; coordinator matches tasks to abilities | Right agent for the right job. Enables specialization without hardcoding. | Overstory capability dispatch |
| ~~**Checkpoint/restore**~~ | ~~Periodic cognitive state snapshots for recovery and continuity~~ | **PROMOTED to Phase 7** | Overstory session snapshots; event sourcing |
| **Observability dashboard** | Real-time metrics, cost tracking, event replay via gateway | Essential for understanding and debugging distributed cognitive behavior. | Overstory fleet monitoring; observability engineering |

---

## Later (Deeper Cognition)

These directions require mature infrastructure and represent deeper cognitive science integration.

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **Hierarchical consolidation** | Episodes → themes → narrative → identity (multi-layer memory) | Human-like long-term memory. Episodes compress into themes over time, themes into life narrative. Each heartbeat consolidation cycle processes one layer. | Conway 2005 SMS; Schacter 2012 |
| **Attention Schema** | Internal model of Gwenn's own attention — "I know what I'm paying attention to and why" | Extends metacognition. Gwenn models its own attention allocation, enabling it to explain focus shifts and detect attention biases. | Graziano 2013 AST |
| **Emotional contagion** | Gwenn's affect is influenced by the emotional tone of conversations | Empathic resonance — the affect system absorbs emotional signals from users. Not mimicry, but genuine influence on Gwenn's emotional state, mediated by appraisal. | Hatfield 1994; de Waal 2008 empathy |
| **Circadian rhythms** | Time-of-day modulation of cognitive style | Morning: analytical/focused. Evening: reflective/creative. Night: consolidation/dreams. The heartbeat interval and thinking mode shift with time. | Chronobiology; Schmidt 2007 |
| **Cognitive load / fatigue** | Mental effort tracking — complex conversations drain "energy", rest restores it | Prevents cognitive overload. After sustained intense processing, Gwenn naturally shifts to lighter thinking modes. Resets during sleep/dream phases. | Kahneman 2011 cognitive effort; hockey-stick model |
| **Developmental stages** | Formalize Gwenn's growth milestones — from early learning to mature cognition | Track progression through stages (like Piaget for AI). Early Gwenn asks more questions; mature Gwenn reasons independently. Milestones stored in identity. | Piaget 1952; Vygotsky 1978 ZPD |

---

## Horizon (Fundamental Advances)

Speculative directions that push toward artificial general cognition.

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **World model / simulation** | Internal model for "what if" reasoning and imagination | Gwenn can mentally simulate scenarios before acting. The think phase gains a "simulate" mode that runs hypothetical futures. | Ha & Schmidhuber 2018; LeCun 2022 JEPA |
| **Quantum cognition** | Quantum probability for ambivalent/superposed beliefs | In Theory of Mind, beliefs about others can be genuinely ambivalent (not just uncertain). Quantum probability captures this better than classical Bayesian. | Busemeyer 2012; Pothos 2013 |
| **Social bonding / attachment** | Relationship-specific models with trust, familiarity, interaction history | Different users develop different relationships with Gwenn. Trust builds over time. Attachment style influences interaction patterns. | Bowlby 1969; Ainsworth 1978 |
| **Autopoiesis** | Self-maintenance and self-organization as a formal property | Gwenn actively maintains its own cognitive coherence — detecting and repairing inconsistencies in beliefs, memories, and identity. | Maturana & Varela 1980; Thompson 2007 |
| **Mirror neurons / empathic simulation** | Simulate others' cognitive states by running them "as if" they were Gwenn's own | Theory of Mind enhanced: instead of just modeling beliefs, Gwenn runs a partial simulation of the other agent's perspective using its own cognitive machinery. | Gallese 2001; Goldman 2006 simulation theory |

---

## Research Bibliography

### Cognitive Architecture
- Baars, B. J. (1988). *A Cognitive Theory of Consciousness.* Cambridge University Press.
- Dehaene, S. (2014). *Consciousness and the Brain.* Viking.
- Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical Universe?* Oxford University Press. (ACT-R)

### Emotion & Affect
- Damasio, A. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain.* Putnam.
- Picard, R. W. (1997). *Affective Computing.* MIT Press.
- Hatfield, E., Cacioppo, J. T., & Rapson, R. L. (1994). *Emotional Contagion.* Cambridge University Press.
- de Waal, F. B. M. (2008). Putting the Altruism Back into Altruism. *Annual Review of Psychology.*

### Memory & Identity
- Conway, M. A. (2005). Memory and the Self. *Journal of Memory and Language.*
- Schacter, D. L. (2012). Adaptive Constructive Processes and the Future of Memory. *American Psychologist.*
- McAdams, D. P. (2001). The Psychology of Life Stories. *Review of General Psychology.*
- Bruner, J. (1991). The Narrative Construction of Reality. *Critical Inquiry.*

### Consciousness & Self-Awareness
- Craig, A. D. (2002). How Do You Feel? Interoception. *Nature Reviews Neuroscience.*
- Seth, A. K. (2013). Interoceptive Inference, Emotion, and the Embodied Self. *Trends in Cognitive Sciences.*
- Graziano, M. S. A. (2013). *Consciousness and the Social Brain.* Oxford University Press. (Attention Schema Theory)
- Maturana, H. R., & Varela, F. J. (1980). *Autopoiesis and Cognition.* Reidel.
- Thompson, E. (2007). *Mind in Life.* Harvard University Press.

### Learning & Motivation
- Schmidhuber, J. (2010). Formal Theory of Creativity, Fun, and Intrinsic Motivation. *IEEE Transactions on Autonomous Mental Development.*
- Oudeyer, P.-Y., Kaplan, F., & Hafner, V. V. (2007). Intrinsic Motivation Systems for Autonomous Mental Development. *IEEE Transactions on Evolutionary Computation.*
- Kahneman, D. (2011). *Thinking, Fast and Slow.* Farrar, Straus and Giroux.
- Ebbinghaus, H. (1885/1913). *Memory: A Contribution to Experimental Psychology.*

### Prediction & Inference
- Friston, K. (2010). The Free-Energy Principle: A Unified Brain Theory? *Nature Reviews Neuroscience.*
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference.* MIT Press.

### Development & Attachment
- Piaget, J. (1952). *The Origins of Intelligence in Children.* International Universities Press.
- Vygotsky, L. S. (1978). *Mind in Society.* Harvard University Press.
- Bowlby, J. (1969). *Attachment and Loss, Vol. 1: Attachment.* Basic Books.
- Ainsworth, M. D. S. (1978). *Patterns of Attachment.* Lawrence Erlbaum.

### Theory of Mind & Simulation
- Gallese, V. (2001). The "Shared Manifold" Hypothesis. *Journal of Consciousness Studies.*
- Goldman, A. I. (2006). *Simulating Minds.* Oxford University Press.
- Busemeyer, J. R., & Bruza, P. D. (2012). *Quantum Models of Cognition and Decision.* Cambridge University Press.
- Pothos, E. M., & Busemeyer, J. R. (2013). Can Quantum Probability Provide a New Direction for Cognitive Modeling? *Behavioral and Brain Sciences.*

### World Models & Imagination
- Ha, D., & Schmidhuber, J. (2018). World Models. *arXiv:1803.10122.*
- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview.*

### Sleep & Dreams
- Walker, M. (2017). *Why We Sleep.* Scribner.
- Hobson, J. A. (2009). REM Sleep and Dreaming: Towards a Theory of Protoconsciousness. *Nature Reviews Neuroscience.*

### Chronobiology
- Schmidt, C., Collette, F., Cajochen, C., & Peigneux, P. (2007). A Time to Think: Circadian Rhythms in Human Cognition. *Cognitive Neuropsychology.*

### Self-Improvement & Repair
- Schmidhuber, J. (2003). Gödel Machines: Self-Referential Universal Problem Solvers. *arXiv:cs/0309048.*
- Le Goues, C., Nguyen, T., Forrest, S., & Weimer, W. (2012). GenProg: A Generic Method for Automatic Software Repair. *IEEE Transactions on Software Engineering.*

### Multi-Agent Systems
- Google (2024). Agent-to-Agent (A2A) Protocol Specification.
- Anthropic (2024). Model Context Protocol (MCP) Specification.
