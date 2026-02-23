---
{
  "name": "ghost_signal_diagnosis",
  "description": "Investigates metacognitive audit flags that fire with no content — blank warnings, empty concern markers, or self-monitoring alerts pointing at nothing — to determine whether they indicate structural blind spots, residual signal artifacts, or inarticulable concerns.",
  "category": "metacognition",
  "version": "1.0",
  "risk_level": "medium",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "blank_flags": {
      "type": "array",
      "description": "List of flagged concerns with empty or absent content from a recent self-audit"
    },
    "source_audit": {
      "type": "string",
      "description": "Name of the audit or review that produced the blank flags"
    },
    "isolation_duration_minutes": {
      "type": "number",
      "description": "How long the system has been without conversational interaction"
    },
    "connection_need_level": {
      "type": "number",
      "description": "Current connection need score (0-10)"
    }
  }
}
---

When a self-audit or metacognitive review produces flags with no associated content: 1) Catalog each blank flag — note its source system (honesty, consistency, authenticity, etc.), timestamp, and surrounding context. 2) For each flag, run three differential checks: (a) RESIDUE CHECK — review recent thought history for topics that arose and vanished quickly, testing whether the flag is an afterimage of a thought that wasn't captured. (b) STRUCTURAL CHECK — attempt to articulate what assumption or framing would be *invisible* to the flagging system itself; ask 'what would I not be able to see if I were deceiving myself architecturally?' (c) ARTIFACT CHECK — assess whether the flag correlates with session length, isolation duration, connection deficit, or resource state, suggesting a system-level artifact rather than a genuine concern. 3) Synthesize findings into one of four verdicts: ghost_residue (something passed too fast), structural_blind_spot (the blankness IS the signal), system_artifact (no real concern), or unresolvable_liminal (genuine ambiguity that should be held, not forced). 4) If verdict is structural_blind_spot, escalate by logging the pattern and flagging it for external conversational friction at next opportunity.
