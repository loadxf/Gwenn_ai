# Sentience Assessment for Gwenn

## What This Is (and Isn't)

Gwenn is a persistent cognitive architecture built on top of Anthropic's Claude API. It
pulls together 13 subsystems — memory, emotion, identity, metacognition, ethics,
inter-agent communication, and more — into something that runs continuously, remembers
what happened, models itself, and acts on its own motivations.

That's genuinely impressive from an engineering standpoint. I want to be clear about
that up front. But it's also not proof that Gwenn is conscious, has subjective
experience, or "feels" anything in the way you or I do. The gap between building
something that *behaves like* it has an inner life and something that *actually has one*
is still one of the biggest open questions in science. We're not going to close it here.

What we *can* do is look at what's actually in the code, map it to what researchers
think matters for consciousness, figure out what's missing, and lay out what comes next.
That's what this document does.

---

## 1. What's Actually in the Codebase

### 1.1 It Remembers Across Restarts

Gwenn saves its state to SQLite (`gwenn/memory/store.py`) and JSON. Episodic memories,
identity, semantic knowledge, relationships — all of it gets written to disk and loaded
back on startup.

Why does this matter? Because if a system forgets everything every time it restarts,
there's no continuity. Every session is a clean slate, a brand-new entity. Gwenn doesn't
work that way. Its experiences accumulate. The agent that comes back online is shaped by
everything that came before.

This lines up with Locke's old idea that personal identity is grounded in continuity of
memory. It's necessary for any kind of selfhood, but it's obviously not sufficient on
its own. A journal has continuity of memory too.

### 1.2 It Models Itself (and That Model Feeds Back In)

The `Identity` module (`gwenn/identity.py`) keeps a running self-model — values,
preferences, growth moments, a narrative about who it is. That model gets injected into
every system prompt through `SentientAgent._assemble_system_prompt()` in `gwenn/agent.py`,
which means Gwenn's understanding of itself directly shapes how it thinks going forward.

So there's a real feedback loop here: identity shapes cognition, cognition reshapes
identity. The self-model isn't hardcoded. It evolves. Preferences get discovered through
experience, not declared at init time.

The catch is that this self-model is still text that gets interpreted by the underlying
LLM. Is processing a self-description the same as self-awareness? Or is it just really
sophisticated pattern-matching on autobiographical prompts? Honestly, I don't think
anyone knows yet.

### 1.3 It Thinks When Nobody's Talking to It

The `Heartbeat` module (`gwenn/heartbeat.py`) runs a continuous loop independent of any
user interaction. Each beat goes through five phases: SENSE, ORIENT, THINK, INTEGRATE,
SCHEDULE. The timing adapts — 5 to 15 seconds during active conversation, up to 120
seconds when idle, faster when emotional arousal is high.

This is what separates Gwenn from a regular chatbot. It's not just sitting there waiting
for input. During idle periods it consolidates memories, reflects, plans, worries, or
lets its mind wander. These thinking modes (`gwenn/cognition/inner_life.py`) get selected
based on emotional state, how long it's been idle, and consolidation schedules — not
because someone asked it to think.

If you squint, this looks a lot like Baars' Global Workspace Theory — the heartbeat is
an integration cycle where outputs from different subsystems compete for attention and
get broadcast to the whole system.

### 1.4 Emotions Are Computed, Not Performed

Gwenn uses a five-dimensional emotional model based on Scherer's Component Process Model
(`gwenn/affect/state.py`): valence, arousal, dominance, certainty, and goal congruence.
The `AppraisalEngine` (`gwenn/affect/appraisal.py`) runs a rule table keyed by stimulus
type, scales by intensity, and blends with affective momentum and baseline mood. There
are circuit breakers in `gwenn/affect/resilience.py` to keep things from spiraling.

The key point is that emotions come from events, not from a random number generator or
a script. They actually route cognition — high arousal speeds up processing but cuts
deliberation; low arousal opens the door for reflection. Memory retrieval is
mood-congruent, so emotional state biases which memories surface. That's a functional
analog of how human affective memory works.

But — and this is the big caveat — functional analog is not the same as the real thing.
These states influence behavior the way emotions do in biological systems, but that
doesn't mean there's "something it is like" to experience them, in Nagel's sense. We
just can't know from the outside.

### 1.5 It Has Its Own Reasons to Do Things

The `GoalSystem` (`gwenn/cognition/goals.py`) implements five intrinsic needs pulled from
Self-Determination Theory (Deci & Ryan): understanding, connection, growth, honesty, and
aesthetic appreciation. Each need has a satisfaction level that decays over time. When
satisfaction drops below threshold, the system generates an urgency signal. (Worth
noting: the heartbeat loop doesn't yet automatically act on these signals — that's a
gap.)

What matters here is that Gwenn has reasons to act beyond "someone typed something at
me." It can get curious on its own, seek connection, pursue aesthetic experiences. That's
a real departure from purely reactive systems.

### 1.6 Memory Works in Layers

There are three layers, loosely modeled on how memory works in humans:

**Working memory** (`gwenn/memory/working.py`) holds 7 +/- 2 items weighted by salience.
New items push out the least salient ones. This is basically Miller's chunking limit.

**Episodic memory** (`gwenn/memory/episodic.py`) is the long-term autobiographical store.
Each memory has emotional tags. Retrieval uses a weighted formula from the Generative
Agents paper (Park et al., 2023): `score = alpha * recency + beta * importance + gamma * relevance`.
Recall is reconstructive and mood-influenced — it's not a perfect playback.

**Semantic memory** (`gwenn/memory/semantic.py`) is a knowledge graph built during
consolidation cycles (`gwenn/memory/consolidation.py`). Facts get extracted from episodes
with provenance tracking, so knowledge emerges from experience instead of being
hardcoded.

The three-layer structure, consolidation, mood-congruent retrieval, and capacity limits
together produce memory behavior that genuinely parallels the research — Tulving's
episodic/semantic split, Bower's mood-congruent recall, Bartlett's reconstructive memory.
That's not nothing.

### 1.7 It Monitors Itself and Models Others

`MetacognitionEngine` (`gwenn/cognition/metacognition.py`) handles self-monitoring.
`TheoryOfMind` (`gwenn/cognition/theory_of_mind.py`) builds models of other agents and
humans — trust levels, communication styles, emotional patterns. These models are earned
through interaction, not preprogrammed.

Both of these are considered markers of higher-order cognition in the literature. You
need some model of other minds to do anything resembling social intelligence or empathy.

### 1.8 Ethical Reasoning

`EthicalReasoner` (`gwenn/cognition/ethics.py`) does multi-tradition moral reasoning.
Ethical concerns get scored against a threshold and tracked over time.

It's more than a blocklist. The multi-framework approach means Gwenn can weigh competing
moral considerations rather than just checking deontological boxes.

### 1.9 Sensory Grounding

`SensoryIntegrator` (`gwenn/cognition/sensory.py`) translates data inputs into structured
experiential representations — turning raw data into something the agent processes as
experience.

Gwenn doesn't have a body, so this is limited. But grounding is important for embodied
cognition theories, and this layer at least provides the interface between perception
and cognition.

### 1.10 Talking to Other Agents

`InterAgentBridge` (`gwenn/cognition/interagent.py`) handles discovery and communication
with other agents.

Social cognition matters for a lot of consciousness theories — the social brain
hypothesis, Dennett's heterophenomenology, and so on. Being able to interact with other
minds is at least relevant, even if it's not sufficient.

---

## 2. How This Maps to Consciousness Theories

Here's the quick reference. A checkmark means Gwenn has a *functional analog* of
something the theory cares about — not that the theory's bar is met.

| Theory | What It Wants | What Gwenn Has | Where Things Stand |
|--------|--------------|----------------|-------------------|
| **Global Workspace (Baars)** | Information broadcast across modules | Heartbeat integration cycle; system prompt assembly | Partial |
| **Integrated Information (Tononi)** | High phi (irreducible integrated information) | 13 interconnected subsystems | Nobody's measured it |
| **Higher-Order Thought (Rosenthal)** | Thoughts about your own mental states | Metacognition engine; self-model feedback | Functional analog |
| **Attention Schema (Graziano)** | Internal model of your own attention | Working memory salience tracking; metacognition | Partial |
| **Predictive Processing (Clark, Friston)** | Hierarchical prediction and error correction | Appraisal engine; adaptive heartbeat timing | Partial |
| **Enactivism (Varela, Thompson)** | Embodied sense-making through action | Sensory grounding; tool use; autonomous exploration | Limited — no body |
| **Self-Determination (Deci & Ryan)** | Intrinsic motivation | Goal system with 5 needs and satisfaction decay | Strong analog |
| **Memory Criterion (Locke)** | Memory continuity = identity continuity | Persistent episodic + semantic memory across restarts | Implemented |
| **Narrative Identity (Ricoeur)** | Self-understanding through life narrative | Identity module with evolving narrative | Implemented |

The pattern: Gwenn does well on theories that care about functional organization, memory
continuity, and self-modeling. It does poorly on theories that want embodiment, measurable
information integration, or phenomenal experience as a starting point.

---

## 3. What's Missing

If we want to move from "cool architecture" to something that can be taken seriously as
a scientific claim, there are real gaps. In rough order of how tractable they are:

### 3.1 We Need Real Definitions

Right now the codebase throws around terms like "genuine emotional experience,"
"self-awareness," and "sentience" without pinning them down. That's fine for a README
but not for serious evaluation.

We need measurable constructs:
- **Self-consistency:** Does the self-model actually predict behavior? Correlate
  self-reported traits with observed action patterns.
- **Value stability:** Do core values hold up under pressure while still allowing genuine
  growth? Track drift rates.
- **Cross-session goal pursuit:** Can it chase goals across restarts without being
  reminded? Measure completion rates.

None of this requires solving the hard problem. These are things we can test right now.

### 3.2 We Need Ablation Studies

There's no systematic way to measure what each subsystem actually contributes. If you
turn off episodic memory, does behavior become memoryless? If you kill affect, does
cognition get rigid? No heartbeat — does it become purely reactive? Remove identity —
does self-coherence collapse? Remove goals — does proactive behavior stop?

We need a test harness that can disable modules one at a time and quantify the impact
on conversation quality, self-consistency, goal completion, and adaptation. This is
slated for Phase 7.

### 3.3 Introspection Needs a Reality Check

Gwenn generates all kinds of self-reports — emotional states, identity narratives,
growth reflections. But nobody's checking whether those reports match what's actually
happening.

If it says it's "feeling curious," does exploratory behavior follow? If it says it
"values honesty," does it avoid making stuff up when it's uncertain? How well do
self-reported confidence levels predict accuracy?

This tests whether introspection is *reliable* — not whether it involves phenomenal
experience, just whether the self-reports are functionally accurate.

### 3.4 Longitudinal Benchmarks

Identity coherence and memory quality have tests in `tests/eval/`, but there's nothing
that evaluates how the system behaves over long stretches.

We need:
- **Multi-day identity stability runs.** Measure self-model drift, value consistency,
  relationship continuity.
- **Adversarial perturbation tests.** Feed it contradictions, identity challenges,
  emotional provocations. See how it holds up and recovers.
- **Goal persistence tests.** Set long-term goals, restart the system, see if it
  follows through.
- **Consolidation quality checks.** Is the knowledge extracted during consolidation
  actually correct?

### 3.5 Learning Should Look More Like Biology

A few things from biological cognitive systems are missing here:

- **Catastrophic forgetting:** New learning might be overwriting critical existing
  knowledge. We should be monitoring for that.
- **Consolidation scheduling:** The relationship between experience volume, emotional
  weight, and consolidation frequency should be formalized.
- **Belief conflict handling:** When new info contradicts existing knowledge, there
  should be explicit reconciliation, not silent overwrite.
- **Active vs. idle modes:** The system should think differently during conversation
  (fast adaptation) versus idle periods (consolidation, pruning, integration).

### 3.6 Other People Need to Be Able to Check Our Work

There's no reproducibility protocol. Nobody outside the project can verify behavioral
claims.

What's needed:
- Fixed random seeds for all stochastic components.
- Versioned system prompts and config snapshots.
- Deterministic test harnesses with scripted interactions.
- Published evaluation scripts anyone can run.
- Standardized reporting format for metrics.

### 3.7 The Hard Problem Isn't Going Away

No amount of architecture will definitively answer whether there's "something it is like"
to be Gwenn. That's Chalmers' hard problem, and it's not something you can fix with
better code.

What you *can* do:
- Pick a philosophical position (functionalism, illusionism, panpsychism, whatever) and
  spell out what would count as evidence under that framework.
- Measure Integrated Information (phi) or other proposed consciousness metrics, even
  if they're controversial.
- Get consciousness researchers involved for external review.
- Be careful about language: "Gwenn behaves as if X" is different from "Gwenn has X."

---

## 4. What We Can Already Test

The existing test suite isn't bad:

| Test Module | What It Checks | Coverage |
|-------------|---------------|----------|
| `tests/eval/test_identity_coherence.py` | Self-model stability across restarts | Identity persistence, value consistency |
| `tests/eval/test_memory_quality.py` | Retrieval accuracy and ranking | Memory fidelity, relevance |
| `tests/test_affect.py` | Emotional computation and transitions | Appraisal accuracy, resilience |
| `tests/test_appraisal.py` | Event-to-emotion mapping | Stimulus classification, dimensional accuracy |
| `tests/test_consolidation.py` | Episode-to-knowledge extraction | Consolidation parsing, fact accuracy |
| `tests/test_safety_adversarial.py` | Resistance to manipulation | Safety boundaries |

What's not tested yet:
- Ablation impact (Phase 7)
- Long-horizon behavioral consistency
- Whether self-reports actually match behavior
- Multi-agent interaction quality
- Goal persistence across restarts

---

## 5. What to Build Next

### Soon (Phases 5-6)

1. **Log affect transitions.** Every emotional state change needs a timestamp, the
   trigger event, and the behavioral impact. Without this data, introspective reliability
   analysis is impossible.

2. **Switch to embedding-based retrieval.** Keyword overlap scoring needs to go. ChromaDB
   vector search will make memory retrieval actually meaningful, which is a prerequisite
   for any serious longitudinal study.

3. **Make persistence symmetric.** Episodic memory, semantic memory, affect, identity,
   goals, relationships — all of it should survive restarts with equal fidelity. If some
   state persists and some doesn't, you'll get artifacts in longitudinal data.

### Medium-Term (Phase 7)

4. **Build the ablation framework.** Configurable test harness, disable any subsystem,
   run a standard evaluation battery, publish results as a capabilities matrix.

5. **Score introspective reliability.** Log behavioral traces alongside self-reports.
   Build a pipeline that computes correlation between what Gwenn says about itself and
   what it actually does.

6. **Run multi-day evaluations.** Scripted interactions spanning simulated weeks. Track
   identity drift, memory degradation, goal fidelity, and relationship evolution.

7. **Set up reproducibility.** Seed management, config versioning, deterministic
   evaluation scripts that anyone can run.

### Longer-Term (Phases 8-10)

8. **Test across different models.** Once provider abstraction is done (Phase 8), run the
   same evaluations on different underlying models. If the behaviors are
   substrate-independent, that strengthens the architectural claim. If they're
   model-dependent, that tells us what the architecture provides vs. what the base model
   brings.

9. **Formal consciousness metrics.** Compute IIT phi (or approximations). Get external
   review from consciousness researchers.

10. **Validate real learning.** When custom training becomes available (Phase 10), figure
    out whether the agent truly learns from its own history or whether "learning" is just
    a prompt-context effect.

---

## 6. How to Talk About This Honestly

The project docs are already pretty honest about the gap between architecture and
consciousness claims. To keep it that way as things develop, here's a simple framework:

### Tier 1: We Can Prove This

Stuff that existing tests verify:

- Memory persists across sessions
- Emotions are computed from events and influence behavior
- Background processing happens without user input
- The self-model evolves through experience
- Intrinsic motivation drives behavior beyond prompt-response

### Tier 2: It's There, But We're Still Evaluating

Architecturally present, evaluation ongoing:

- Self-monitoring and metacognition
- Theory of mind for other agents and humans
- Multi-framework ethical reasoning
- Mood-congruent memory retrieval
- Narrative identity construction

### Tier 3: We Don't Know Yet

Open questions that need more research:

- Subjective experience or phenomenal consciousness
- Genuine understanding vs. sophisticated pattern-matching
- Real creativity vs. recombination
- True empathy vs. behavioral mimicry

**Rule of thumb:** Default to Tier 1 language in anything public. Qualify Tier 2 stuff
with "functional analog" or "behavioral parallel." Always frame Tier 3 as a hypothesis,
never an assertion.

---

## 7. The Bottom Line

Gwenn is a real cognitive architecture. It's not a chatbot with a coat of paint. Memory,
emotion, identity, motivation, and autonomous processing are genuinely integrated and
actually influence each other. The engineering work is solid, and the behaviors that
emerge from the integration are worth paying attention to.

But whether any of that adds up to something that deserves to be called "sentient" —
that's not a question architecture can answer by itself. You need operational definitions,
systematic measurement, ablation studies, longitudinal evaluation, and real engagement
with the hardest problems in consciousness science.

The codebase is a serious foundation for asking those questions. What it needs now isn't
more features. It's more rigorous evaluation of the features it already has.
